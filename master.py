import os
import matplotlib.pyplot as plt

from src.preprocessing import load_and_process_data, one_hot_encode
from src.chou_fasman import ChouFasman
from src.gor import GOR
from src.bilstm_updated import BiLSTM_PSSP
from src.evaluation import evaluate_predictions


def plot_loss_curve(history, out_path):
    """
    Plot training vs validation loss from a Keras History object
    and save to out_path (PNG).
    """
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure()
    plt.plot(loss, label="Train Loss")
    if val_loss:
        plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiLSTM Loss Convergence")
    plt.legend()
    plt.tight_layout()

    # Make sure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss convergence plot to {out_path}")


def main():
    # 1. Load Data
    print("Loading and Preprocessing Data...")
    data_path = "data/cb513.csv"
    X_train, X_test, y_train, y_test = load_and_process_data(data_path)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 2. Chou-Fasman
    print("\nRunning Chou-Fasman...")
    cf = ChouFasman()
    cf.train(X_train, y_train)
    cf_preds = [cf.predict(seq) for seq in X_test]
    evaluate_predictions(y_test, cf_preds, "Chou-Fasman")

    # 3. GOR
    print("\nRunning GOR...")
    gor = GOR()
    gor.train(X_train, y_train)
    gor_preds = [gor.predict(seq) for seq in X_test]
    evaluate_predictions(y_test, gor_preds, "GOR")

    # 4. BiLSTM
    print("\nRunning BiLSTM (Enhanced)...")

    # Assumes one_hot_encode() returns:
    #   X_enc:    np.ndarray (N, max_len) of int indices (0=PAD)
    #   max_len:  int
    #   vocab_size: int
    X_train_enc, max_len, vocab_size = one_hot_encode(X_train)
    X_test_enc, _, _ = one_hot_encode(X_test, max_len=max_len)

    # True sequence lengths for truncating predictions
    test_lengths = [len(seq) for seq in X_test]

    bilstm = BiLSTM_PSSP(vocab_size, max_len)

    # Train with up to 40 epochs (EarlyStopping will usually stop earlier)
    history = bilstm.train(X_train_enc, y_train, epochs=40)

    # Plot loss convergence
    plot_loss_curve(history, out_path="reports/bilstm_loss_curve.png")

    # Predict on test set
    dl_preds = bilstm.predict(X_test_enc, original_lengths=test_lengths)

    # Evaluate BiLSTM predictions
    evaluate_predictions(y_test, dl_preds, "BiLSTM")


if __name__ == "__main__":
    main()