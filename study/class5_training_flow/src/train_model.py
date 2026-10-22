from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn


EPOCHS = 50
LEARNING_RATE = 0.001
CLASS5 = Path(__file__).resolve().parent
DATA_DIR = CLASS5.parent.parent / "class2_dataset_split" / "src"
MODEL_PATH = CLASS5 / "weights.pt"
PLOT_PATH = CLASS5 / "training_curves.png"


def load_data(folder):
    files = sorted(folder.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No BMP images found in {folder}")
    images, labels = [], []
    for path in files:
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f"Expected a 64x64 image: {path}")
            images.append(np.asarray(image.convert("L"), dtype=np.float32).flatten() / 255.0)
        label = int(path.stem.split("_")[0])
        if not 0 <= label < 10:
            raise ValueError(f"Expected a digit label from 0 to 9: {path}")
        labels.append(label)
    return torch.from_numpy(np.array(images)), torch.tensor(labels, dtype=torch.long)


def main():
    # Step 1: Load and normalize training and validation images.
    x, y = load_data(DATA_DIR / "train")
    x_val, y_val = load_data(DATA_DIR / "val")

    # Step 2: Create a 4096 -> 32 (ReLU) -> 10 network and an optimizer.
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4096, 32), nn.ReLU(), nn.Linear(32, 10))
    loss_fn = nn.CrossEntropyLoss()  # Includes the log-softmax calculation.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 3: Each epoch uses all training images once (one full-batch update).
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()              # 3.1: Clear gradients from the last epoch.
        scores = model(x)                  # 3.2: Forward pass using current weights.
        loss = loss_fn(scores, y)          # 3.3: Compare predictions with labels.
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss is not finite; check data and learning rate.")
        loss.backward()                    # 3.4: BP computes gradients for both layers.
        optimizer.step()                   # 3.5: Update weights and biases.

        # 3.6: Evaluate both sets after the update, without computing gradients.
        model.eval()
        with torch.no_grad():
            train_scores, val_scores = model(x), model(x_val)
            train_loss = loss_fn(train_scores, y).item()
            val_loss = loss_fn(val_scores, y_val).item()
            train_accuracy = (train_scores.argmax(dim=1) == y).float().mean().item()
            val_accuracy = (val_scores.argmax(dim=1) == y_val).float().mean().item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise FloatingPointError("Evaluation loss is not finite; check data and learning rate.")
        history.append((train_loss, val_loss, train_accuracy, val_accuracy))
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2%}"
        )

    # Step 4: Save the trained weights and biases.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Weights saved to: {MODEL_PATH}")

    # Step 5: Save training and validation loss and accuracy curves.
    epochs = range(1, len(history) + 1)
    train_losses, val_losses, train_accuracies, val_accuracies = zip(*history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, train_values, val_values, title in zip(
        axes, (train_losses, train_accuracies), (val_losses, val_accuracies),
        ("Loss", "Accuracy"),
    ):
        axis.plot(epochs, train_values, label="Train")
        axis.plot(epochs, val_values, label="Validation")
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.legend()
        axis.grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(PLOT_PATH)
    print(f"Training chart saved to: {PLOT_PATH}")
    plt.close(figure)


if __name__ == "__main__":
    main()
