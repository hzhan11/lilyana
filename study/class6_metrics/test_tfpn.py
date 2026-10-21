from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_acc import MODEL_PATH, UNDERFITTING_MODEL_PATH, load_model, load_test_data


PLOT_DIR = Path(__file__).resolve().parent / "samples"


def main(weights_path):
    print(f"\n=== Weights: {Path(weights_path).name} ===")

    # Step 1: Load the model weights and prepare the test set.
    model = load_model(weights_path)
    x, y = load_test_data()

    # Step 2: Predict one digit for each test image.
    probabilities = model.predict(x, verbose=0)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Prediction contains non-finite probabilities.")
    predictions = probabilities.argmax(axis=1)

    # Step 3: Stack digits 1, 2, and 3 vertically in one figure.
    figure, axes = plt.subplots(3, 1, figsize=(6, 15))
    figure.suptitle(Path(weights_path).name)
    matrices = []
    names = [["TN", "FP"], ["FN", "TP"]]
    for axis, positive_digit in zip(axes, (1, 2, 3)):
        actual_positive = y == positive_digit
        predicted_positive = predictions == positive_digit
        tp = int(np.count_nonzero(actual_positive & predicted_positive))
        fp = int(np.count_nonzero(~actual_positive & predicted_positive))
        fn = int(np.count_nonzero(actual_positive & ~predicted_positive))
        tn = int(np.count_nonzero(~actual_positive & ~predicted_positive))
        matrix = np.array([[tn, fp], [fn, tp]])
        matrices.append(matrix)
        print(f"Digit {positive_digit} | TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        labels = ["Other digits\n(negative)", f"Digit {positive_digit}\n(positive)"]
        axis.set(
            xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
            xlabel="Predicted", ylabel="Actual", title=f"Digit {positive_digit} vs others",
        )
        for row in range(2):
            for column in range(2):
                count = matrix[row, column]
                axis.text(
                    column, row, f"{names[row][column]}\n{count}", ha="center", va="center",
                    fontsize=22, color="white" if count > matrix.max() / 2 else "black",
                )

    # Step 4: Save one combined image for this weights file.
    figure.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"{Path(weights_path).stem}_confusion_matrices.png"
    figure.savefig(plot_path)
    plt.close(figure)
    print(f"Confusion matrix saved to: {plot_path}")
    return np.array(matrices)


if __name__ == "__main__":
    for weights_path in (MODEL_PATH, UNDERFITTING_MODEL_PATH):
        main(weights_path)
