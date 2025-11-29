from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(y_true, y_pred, method_name):
    """
    Flattens sequences and calculates metrics.
    """
    # Flatten lists
    y_true_flat = []
    y_pred_flat = []
    
    for t, p in zip(y_true, y_pred):
        # Truncate prediction to match true length (handling padding issues)
        length = min(len(t), len(p))
        y_true_flat.extend(list(t[:length]))
        y_pred_flat.extend(list(p[:length]))

    print(f"--- Results for {method_name} ---")
    acc = accuracy_score(y_true_flat, y_pred_flat)
    print(f"Q3 Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true_flat, y_pred_flat, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=['H', 'E', 'C'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['H','E','C'], yticklabels=['H','E','C'])
    plt.title(f'Confusion Matrix: {method_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'results_{method_name}.png')
    print(f"Confusion matrix saved as results_{method_name}.png\n")