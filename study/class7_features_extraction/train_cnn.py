from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


CLASS7 = Path(__file__).resolve().parent
DATA_DIR = CLASS7 / "pic"
MODEL_PATH = CLASS7 / "weights_cnn.pt"
DEMO_PATH = DATA_DIR / "test" / "black_cat_01.png"
CLASS_NAMES = ("cat", "dog")
EPOCHS = 40
LEARNING_RATE = 0.003


def shape_features(path):
    with Image.open(path) as image:
        if image.size != (128, 128):
            raise ValueError(f"Expected a 128x128 image: {path}")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    # Remove color, but keep the foreground's 128x128 spatial layout.
    mask = (pixels < 0.95).any(axis=2).astype(np.float32)
    if not mask.any():
        raise ValueError(f"No foreground pixels found: {path}")
    return mask[np.newaxis, :, :]


def load_data(folder):
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {folder}")
    images, labels = [], []
    for path in paths:
        parts = path.stem.split("_")
        if len(parts) != 3 or parts[1] not in CLASS_NAMES:
            raise ValueError(f"Expected <color>_<cat|dog>_<number>.png: {path}")
        images.append(shape_features(path))
        labels.append(CLASS_NAMES.index(parts[1]))
    if set(labels) != {0, 1}:
        raise ValueError(f"Both cat and dog samples are required in {folder}")
    return torch.from_numpy(np.stack(images)), torch.tensor(labels, dtype=torch.long)


def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Flatten(), nn.Linear(16 * 8 * 8, 2),
    )


def main():
    # Step 1: Load shape masks; the held-out black cat is not used for training.
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    x_train, y_train = load_data(DATA_DIR / "train")
    x_val, y_val = load_data(DATA_DIR / "val")
    print("Input: 1 x 128 x 128 foreground masks. Color is deliberately removed.")
    print(f"Train: {len(y_train)} images | Validation: {len(y_val)} images")

    # Step 2: Learn spatial features with two convolutional layers.
    model = create_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_scores, val_scores = model(x_train), model(x_val)
            train_loss = loss_fn(train_scores, y_train).item()
            val_loss = loss_fn(val_scores, y_val).item()
            train_accuracy = (train_scores.argmax(1) == y_train).float().mean().item()
            val_accuracy = (val_scores.argmax(1) == y_val).float().mean().item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise FloatingPointError("Evaluation loss is not finite.")
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}"
        )

    # Step 3: Save separate CNN weights without overwriting the color model.
    torch.save(model.state_dict(), MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"Weights saved to: {MODEL_PATH}")

    # Step 4: Predict the same black cat using its shape, not its filename.
    image = torch.from_numpy(shape_features(DEMO_PATH)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probabilities = model(image).softmax(dim=1)[0]
    prediction = CLASS_NAMES[probabilities.argmax().item()]
    print(f"\nDemo image: {DEMO_PATH}")
    print(f"Actual: cat | Predicted: {prediction}")
    print(f"Cat probability: {probabilities[0]:.2%} | Dog probability: {probabilities[1]:.2%}")
    print(f"Correct: {prediction == 'cat'}")
    print("This controlled example removes color first; a CNN using RGB can still learn color shortcuts.")
    print("Success on these few cartoon images does not guarantee recognition of real cats and dogs.")


if __name__ == "__main__":
    main()
