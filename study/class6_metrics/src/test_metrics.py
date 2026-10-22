from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from test_acc import MODEL_PATH, UNDERFITTING_MODEL_PATH, load_model, load_test_data


DIGITS = (1, 2, 3)


def main(weights_path):
    print(f"\n=== Weights: {Path(weights_path).name} ===")

    # Step 1: Load the model and predict test-set probabilities.
    model = load_model(weights_path)
    x, y = load_test_data()
    probabilities = model.predict(x, verbose=0)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Prediction contains non-finite probabilities.")
    predictions = probabilities.argmax(axis=1)

    # Step 2: Calculate precision, recall, and F1 for digits 1, 2, and 3.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=DIGITS, average=None, zero_division=0,
    )
    print("Each digit is positive; all other digits are negative.")
    print("Undefined precision/recall/F1 (zero denominator) is reported as 0.")
    print("Digit | Precision | Recall |     F1 |  AUROC")
    results = {}
    for index, digit in enumerate(DIGITS):
        # Step 3: AUROC uses this digit's probability, not the predicted label.
        actual_positive = y == digit
        if actual_positive.any() and (~actual_positive).any():
            auroc = roc_auc_score(actual_positive, probabilities[:, digit])
        else:
            auroc = np.nan
            print(f"Digit {digit}: AUROC is N/A; both positive and negative samples are required.")
        values = (precision[index], recall[index], f1[index], auroc)
        results[digit] = values
        auc_text = f"{auroc:.4f}" if np.isfinite(auroc) else "N/A"
        print(f"{digit:5d} | {values[0]:9.4f} | {values[1]:6.4f} | {values[2]:6.4f} | {auc_text:>6}")

    # Step 4: Give each of the three digits equal weight in the macro average.
    macro = np.mean(list(results.values()), axis=0)
    auc_text = f"{macro[3]:.4f}" if np.isfinite(macro[3]) else "N/A"
    print(f"Macro | {macro[0]:9.4f} | {macro[1]:6.4f} | {macro[2]:6.4f} | {auc_text:>6}")
    print("Macro averages digits 1, 2, and 3 only; AUROC is N/A if any digit's AUROC is undefined.")
    return results


if __name__ == "__main__":
    main(MODEL_PATH)
    main(UNDERFITTING_MODEL_PATH)
