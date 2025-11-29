import os
from src.preprocessing import load_and_process_data, one_hot_encode
from src.chou_fasman import ChouFasman
from src.gor import GOR
from src.bilstm_saved_model import BiLSTM_PSSP
from src.evaluation import evaluate_predictions

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # 1. Load Data
    print("Loading and Preprocessing Data...")
    data_path = 'data/cb513.csv'
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
    X_train_enc, max_len, vocab_size = one_hot_encode(X_train)
    X_test_enc, _, _ = one_hot_encode(X_test, max_len=max_len)
    test_lengths = [len(seq) for seq in X_test]
    
    bilstm = BiLSTM_PSSP(vocab_size, max_len)
    
    # --- CHECK IF MODEL EXISTS ---
    model_path = 'models/bilstm_model.keras'
    plot_path = 'reports/bilstm_loss_curve.png'
    
    if os.path.exists(model_path):
        print(f"Found saved model at {model_path}. Loading...")
        bilstm.load_trained_model(model_path)
    else:
        print("No saved model found. Starting training...")
        bilstm.train(X_train_enc, y_train, epochs=80, plot_path=plot_path)
        bilstm.save_model(model_path)
    
    # Predict
    dl_preds = bilstm.predict(X_test_enc, original_lengths=test_lengths)
    evaluate_predictions(y_test, dl_preds, "BiLSTM")

if __name__ == "__main__":
    main()