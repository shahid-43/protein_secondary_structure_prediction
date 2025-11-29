import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""
BiLSTM_PSSP

Assumptions:
- X passed to .train() / .predict() is an integer-encoded, padded array
  of shape (N, max_len), with:
    * 0 used as the PAD index
    * 1..vocab_size-1 used for amino acids
- y passed to .train() is a list of strings, each like "HECCH..."
- original_lengths passed to .predict() is a list of true sequence lengths
"""

class BiLSTM_PSSP:
    def __init__(self, vocab_size, max_len):
        """
        Args:
            vocab_size: size of the integer vocabulary (including PAD=0)
            max_len:    maximum sequence length after padding
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model = self._build_model()

    # -------------------------
    # Model definition
    # -------------------------
    def _build_model(self):
        model = Sequential()

        # Embedding: mask_zero=True tells RNNs to ignore PAD index (0)
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=128,
                input_length=self.max_len,
                mask_zero=True,
            )
        )

        # Two BiLSTM layers
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

        # TimeDistributed Dense → per-residue logits over 3 Q3 classes (H/E/C)
        model.add(TimeDistributed(Dense(64, activation="relu")))
        model.add(TimeDistributed(Dense(3, activation="softmax")))

        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Optional: uncomment if you want to see summary at import
        # model.summary()
        return model

    # -------------------------
    # Label encoding + masking
    # -------------------------
    def _encode_labels(self, labels):
        """
        labels: list of strings, each string is a Q3 label sequence ("H", "E", "C")

        Returns:
            y_cat: (N, max_len, 3) one-hot labels for H/E/C
            mask:  (N, max_len)    1.0 for real residues, 0.0 for padded positions
        """
        map_dict = {"H": 0, "E": 1, "C": 2}
        encoded = []
        mask = []

        for lab in labels:
            # Convert characters to integer class ids (0,1,2)
            nums = [map_dict.get(c, 2) for c in lab]  # default to 'C' if unknown
            real_len = len(nums)

            # Pad labels up to max_len (the actual value for padded positions
            # doesn't matter because we'll weight them 0 in the loss)
            if real_len < self.max_len:
                nums = nums + [2] * (self.max_len - real_len)
            else:
                nums = nums[: self.max_len]
                real_len = self.max_len

            # Mask: 1 for real residues, 0 for padded positions
            mask_row = [1.0] * real_len + [0.0] * (self.max_len - real_len)

            encoded.append(nums)
            mask.append(mask_row)

        y_int = np.array(encoded, dtype=np.int32)
        y_cat = to_categorical(y_int, num_classes=3)
        mask = np.array(mask, dtype=np.float32)
        return y_cat, mask

    # -------------------------
    # Training
    # -------------------------
    def train(self, X_train, y_train, epochs=40):
        """
        Args:
            X_train: np.ndarray (N, max_len), int-encoded sequences with 0 as PAD
            y_train: list of length-N strings (Q3 labels)
            epochs:  max epochs (EarlyStopping will usually stop earlier)

        Returns:
            history: Keras History object (for plotting loss convergence)
        """
        y_encoded, sample_weight = self._encode_labels(y_train)

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

        print("Training BiLSTM with label masking via sample_weight...")
        history = self.model.fit(
            X_train,
            y_encoded,
            epochs=epochs,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
            sample_weight=sample_weight,  # (N, max_len), 0 weight for padded positions
        )

        return history

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X_test, original_lengths):
        """
        Predict Q3 labels and truncate each sequence to its original length.

        Args:
            X_test:            (N, max_len) int-encoded padded sequences
            original_lengths:  list/array of true lengths (no padding)

        Returns:
            preds: list of strings (predicted Q3 labels)
        """
        probs = self.model.predict(X_test)
        preds = []
        map_inv = {0: "H", 1: "E", 2: "C"}

        for i in range(len(X_test)):
            # Argmax over classes for each position
            p_indices = np.argmax(probs[i], axis=1)

            # Truncate back to real length
            real_len = int(original_lengths[i])
            valid_indices = p_indices[:real_len]

            pred_str = "".join(map_inv[x] for x in valid_indices)
            preds.append(pred_str)

        return preds