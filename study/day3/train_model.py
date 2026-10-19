from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression


DAY3 = Path(__file__).resolve().parent
DATA_DIR = DAY3.parent / "day2"
MODEL_PATH = DAY3 / "digit_model.joblib"


def load_data(folder):
    files = sorted(folder.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No BMP images found in {folder}")

    images, labels = [], []
    for path in files:
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"Expected a 64x64 image: {path}")
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        images.append(pixels.flatten())
        labels.append(int(path.stem.split("_")[0]))
    return np.array(images), np.array(labels)


def main():
    # Step 1: Read images and labels from the three dataset folders.
    x_train, y_train = load_data(DATA_DIR / "train")
    x_val, y_val = load_data(DATA_DIR / "val")
    x_test, y_test = load_data(DATA_DIR / "test")

    # Step 2: Create a simple model and train it using only the training set.
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Step 3: Evaluate the model on validation and test data.
    print(f"Train accuracy: {model.score(x_train, y_train):.2%}")
    print(f"Val accuracy:   {model.score(x_val, y_val):.2%}")
    print(f"Test accuracy:  {model.score(x_test, y_test):.2%}")

    # Step 4: Save the trained model weights.
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
