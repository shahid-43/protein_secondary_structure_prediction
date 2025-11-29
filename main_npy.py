import os

from src.bilstm_cb513_npy import (
    load_cb513_npy_and_csv,
    train_val_test_split,
    FeatureBiLSTM_PSSP,
    plot_loss_curve,
)
from src.evaluation import evaluate_predictions


def main():
    # 1. Load CB513 features (.npy) + labels (.csv)
    print("Loading CB513 features (npy) and labels (csv)...")
    X, y_q3, lengths, L, F = load_cb513_npy_and_csv(
        npy_path="data/cb513.npy",
        csv_path="data/cb513.csv",
    )

    N = X.shape[0]
    print(f"Total proteins: {N}, sequence length L={L}, feature dim F={F}")

    # 2. Split into train/val/test (by protein)
    train_idx, val_idx, test_idx = train_val_test_split(
        N,
        train=0.7,
        val=0.15,
        test=0.15,
        seed=42,
    )

    # For simplicity: train on train+val, test on test
    trainval_idx = np.concatenate([train_idx, val_idx])

    X_train = X[trainval_idx]
    y_train = [y_q3[i] for i in trainval_idx]
    lengths_train = lengths[trainval_idx]

    X_test = X[test_idx]
    y_test = [y_q3[i] for i in test_idx]
    lengths_test = lengths[test_idx]

    print(f"Train+Val proteins: {len(X_train)}, Test proteins: {len(X_test)}")

    # 3. Build and train the feature-based BiLSTM
    print("\nRunning Feature-based BiLSTM on cb513.npy features...")
    model = FeatureBiLSTM_PSSP(max_len=L, n_features=F)

    history = model.train(
        X_train,
        y_train,
        lengths_train,
        epochs=40,  # EarlyStopping will usually stop earlier
    )

    # 4. Plot loss convergence curve
    os.makedirs("reports", exist_ok=True)
    plot_loss_curve(history, out_path="reports/bilstm_npy_loss_curve.png")

    # 5. Predict on test set
    print("Predicting on test set...")
    y_pred = model.predict(X_test, lengths_test)

    # 6. Evaluate with your existing evaluation helper
    print("\n--- Results for Feature-BiLSTM (cb513.npy) ---")
    evaluate_predictions(y_test, y_pred, "Feature-BiLSTM (npy)")


if __name__ == "__main__":
    import numpy as np  # needed for np.concatenate etc.
    main()