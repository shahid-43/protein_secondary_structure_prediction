import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================================================
# 1. Data loading: cb513.npy (features) + cb513.csv (labels)
# =========================================================

def load_cb513_npy_and_csv(npy_path="data/cb513.npy", csv_path="data/cb513.csv"):
    """
    Returns:
        X:        (N_use, L, F) float32 features (padded rows zeroed)
        y_q3:     list of length N_use, Q3 label strings
        lengths:  np.ndarray (N_use,), true lengths (from pad-flag AND seq)
        L:        int, sequence length (700)
        F:        int, feature dimension (57)
        seqs:     list of length N_use, AA sequences as strings
    """
    print("Loading numeric features from:", npy_path)
    X_flat = np.load(npy_path, allow_pickle=True)  # (N, 39900)
    N_npy, dim = X_flat.shape
    print(f"npy: {N_npy} proteins, flattened dim = {dim}")

    L = 700
    assert dim % L == 0, "39900 != 700 * F? Unexpected cb513.npy format."
    F = dim // L

    X = X_flat.reshape(N_npy, L, F).astype(np.float32)
    print(f"Reshaped to: X.shape = {X.shape} (N, L, F)")

    # ---- Use last feature as pad flag ----
    # pad_flag = 1 for padded positions, 0 for real residues
    pad_flag = X[:, :, -1]
    # A padded row seems to be [0,0,...,0,1], so row_sum ~= 1
    row_sum = X.sum(axis=-1)
    pad_mask = (pad_flag > 0.5) & (np.isclose(row_sum, 1.0, atol=1e-3))

    # Zero out padded rows so Masking(mask_value=0.0) can detect them
    X[pad_mask] = 0.0

    print("Loading labels from CSV:", csv_path)
    df = pd.read_csv(csv_path)
    seqs = df["input"].astype(str).tolist()
    y_q3 = df["dssp3"].astype(str).tolist()
    N_csv = len(seqs)
    print(f"csv: {N_csv} sequences")

    N_use = min(N_npy, N_csv)
    if N_npy != N_csv:
        print(f"WARNING: npy has {N_npy}, csv has {N_csv}. Using first {N_use} entries.")

    X = X[:N_use]
    y_q3 = y_q3[:N_use]
    seqs = seqs[:N_use]

    # lengths from pad_mask and from seq; take min to be safe
    pad_mask = pad_mask[:N_use]
    len_from_mask = (~pad_mask).sum(axis=1)         # number of non-padded rows
    len_from_seq = np.array([len(s) for s in seqs])  # AA sequence length

    lengths = np.minimum(len_from_mask, len_from_seq).astype(np.int32)

    print(f"Using N = {N_use}, L = {L}, F = {F}")
    print("Example lengths (mask vs seq):", len_from_mask[:5], len_from_seq[:5])
    return X, y_q3, lengths, L, F, seqs


# =========================================================
# 2. Train/val/test split
# =========================================================

def train_val_test_split(N, train=0.7, val=0.15, test=0.15, seed=42):
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(train * N)
    n_val = int(val * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return train_idx, val_idx, test_idx


# =========================================================
# 3. BiLSTM model (features)
# =========================================================

class FeatureBiLSTM_PSSP:
    def __init__(self, max_len, n_features):
        self.max_len = max_len
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        # Now padded rows really are all zeros
        model.add(
            Masking(
                mask_value=0.0,
                input_shape=(self.max_len, self.n_features),
            )
        )

        model.add(
            Bidirectional(
                LSTM(
                    128,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.0,
                )
            )
        )
        model.add(
            Bidirectional(
                LSTM(
                    128,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.0,
                )
            )
        )

        model.add(TimeDistributed(Dense(64, activation="relu")))
        model.add(TimeDistributed(Dense(3, activation="softmax")))  # Q3: H/E/C

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()
        return model

    def _encode_labels(self, labels, lengths):
        map_dict = {"H": 0, "E": 1, "C": 2}
        encoded = []
        weights = []

        for lab, L_real in zip(labels, lengths):
            # ensure we respect both label length & max_len
            L_real = int(min(L_real, self.max_len, len(lab)))
            nums = [map_dict.get(c, 2) for c in lab[:L_real]]

            # pad to max_len
            if L_real < self.max_len:
                nums = nums + [2] * (self.max_len - L_real)

            w_row = [1.0] * L_real + [0.0] * (self.max_len - L_real)

            encoded.append(nums)
            weights.append(w_row)

        y_int = np.array(encoded, dtype=np.int32)
        y_cat = to_categorical(y_int, num_classes=3)
        sample_weight = np.array(weights, dtype=np.float32)
        return y_cat, sample_weight

    def train(self, X_train, y_train, lengths_train, epochs=40):
        y_encoded, sample_weight = self._encode_labels(y_train, lengths_train)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        print("Training Feature-BiLSTM with masking...")
        history = self.model.fit(
            X_train,
            y_encoded,
            epochs=epochs,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
            sample_weight=sample_weight,
        )
        return history

    def predict(self, X_test, lengths_test):
        probs = self.model.predict(X_test)
        preds = []
        inv_map = {0: "H", 1: "E", 2: "C"}

        for i in range(len(X_test)):
            p_indices = np.argmax(probs[i], axis=1)
            real_len = int(min(lengths_test[i], self.max_len))
            valid = p_indices[:real_len]
            preds.append("".join(inv_map[x] for x in valid))

        return preds


# =========================================================
# 4. Plot helper (unchanged)
# =========================================================

def plot_loss_curve(history, out_path):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure()
    plt.plot(loss, label="Train Loss")
    if val_loss:
        plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Feature-BiLSTM Loss Convergence")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss convergence plot to {out_path}")