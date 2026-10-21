from pathlib import Path
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from PIL import Image


STUDY = Path(__file__).resolve().parent.parent
MODEL_PATH = STUDY / "class3_training_model" / "weights.h5"
UNDERFITTING_MODEL_PATH = STUDY / "class3_training_model" / "weights_underfitting.h5"
TEST_DIR = STUDY / "class2_dataset_split" / "test"


def load_model(weights_path):
    model = keras.Sequential([
        keras.Input(shape=(4096,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.load_weights(weights_path)
    return model


def load_test_data():
    files = sorted(TEST_DIR.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No BMP images found in {TEST_DIR}")
    images, labels = [], []
    for path in files:
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"Expected a 64x64 image: {path}")
            pixels = np.asarray(image.convert("L"), dtype=np.float64).flatten() / 255.0
        label = int(path.stem.split("_")[0])
        if not 0 <= label < 10:
            raise ValueError(f"Expected a digit label from 0 to 9: {path}")

        images.append(pixels)
        labels.append(label)
    return np.array(images), np.array(labels)


def main(weights_path):
    print(f"\n=== Weights: {Path(weights_path).name} ===")
    # Step 1: Create the same network as Class3 and load its trained weights.
    model = load_model(weights_path)
    # Step 2: Prepare all test images and labels.
    x, y = load_test_data()
    # Step 3: Let Keras calculate loss and accuracy without training.
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    results = model.evaluate(x, y, verbose=0, return_dict=True)
    if not np.isfinite(list(results.values())).all():
        raise FloatingPointError("Evaluation returned non-finite metrics.")
    accuracy = results["accuracy"]
    print(f"Test loss: {results['loss']:.4f} | Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    return accuracy


if __name__ == "__main__":
    main(MODEL_PATH)
    main(UNDERFITTING_MODEL_PATH)
