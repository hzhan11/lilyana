from pathlib import Path
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EPOCHS = 50
LEARNING_RATE = 0.001

CLASS3 = Path(__file__).resolve().parent
DATA_DIR = CLASS3.parent / "class2_dataset_split"
MODEL_PATH = CLASS3 / "weights.h5"
PLOT_PATH = CLASS3 / "training_curves.png"


def load_data(folder):
    files = sorted(folder.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No BMP images found in {folder}")
    images, labels = [], []
    for path in files:
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"Expected a 64x64 image: {path}")
            pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        label = int(path.stem.split("_")[0])
        if not 0 <= label < 10:
            raise ValueError(f"Expected a digit label from 0 to 9: {path}")
        images.append(pixels.flatten())
        labels.append(label)
    return np.array(images), np.array(labels)


def main():
    # Step 1: Load training, validation, and test data.
    x_train, y_train = load_data(DATA_DIR / "train")
    x_val, y_val = load_data(DATA_DIR / "val")
    x_test, y_test = load_data(DATA_DIR / "test")

    # Step 2: Create a 4096 -> 32 (ReLU) -> 10 (Softmax) network.
    model = keras.Sequential([
        keras.Input(shape=(4096,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    # Step 3: Configure the optimizer, loss, and accuracy metric.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    # Step 4: Train and record each epoch's training and validation metrics.
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, verbose=2,
    )
    # Step 5: Evaluate on the test set and save the model.
    print(f"Test accuracy: {model.evaluate(x_test, y_test, verbose=0)[1]:.2%}")
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Step 6: Save and display training and validation curves.
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(epochs, history.history[metric], label="Train")
        axis.plot(epochs, history.history[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=metric.title())
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOT_PATH)
    print(f"Training chart saved to: {PLOT_PATH}")
    #plt.show()
    #plt.close(figure)


if __name__ == "__main__":
    main()
