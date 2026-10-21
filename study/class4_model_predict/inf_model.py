from pathlib import Path
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from PIL import Image


STUDY = Path(__file__).resolve().parent.parent
IMAGE_PATH = STUDY / "class2_dataset_split" / "test" / "2_03.bmp"
MODEL_PATH = STUDY / "class3_training_model" / "weights.h5"


def main():
    # Step 1: Load a 64 x 64 grayscale image from the test set.
    with Image.open(IMAGE_PATH) as image:
        if image.size != (64, 64):
            raise ValueError("Expected a 64 x 64 test image.")
        pixels = np.asarray(image.convert("L"), dtype=np.float64)
    print(f"Shape: {pixels.shape}, pixels: {pixels.flatten()[:100]} ... ")

    # Step 2: Normalize pixels from 0-255 to 0-1, just as in training.
    pixels = pixels / 255.0
    print(f"\nStep 2 - Normalized pixels: {pixels.flatten()[:100]} ...")

    # Step 3: Flatten 64 x 64 pixels and add a batch dimension: (1, 4096).
    x = pixels.reshape(1, 4096)
    print(f"\nStep 3 - Model input shape: {x.shape}")

    # Step 4: Load the trained 4096 -> 32 (ReLU) -> 10 (Softmax) model.
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"\nStep 4 - Loaded model: {MODEL_PATH.name}")

    # Step 5: Feed only the pixels, not the label, into the model.
    probabilities = model.predict(x, verbose=0)[0]
    print("\nStep 5 - Output probabilities for digits 0-9:")
    for digit, probability in enumerate(probabilities):
        print(f"{digit}: {probability:.2%}")

    # Step 6: Choose the digit with the highest probability.
    prediction = int(np.argmax(probabilities))
    print(f"\nStep 6 - Prediction: {prediction}")
    print(f"Confidence: {probabilities[prediction]:.2%}")


if __name__ == "__main__":
    main()