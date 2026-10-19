from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_CLASSES = 10

DAY5 = Path(__file__).resolve().parent
DATA_DIR = DAY5.parent / "day2"
MODEL_PATH = DAY5 / "weights.npz"
PLOT_PATH = DAY5 / "training_curves.png"


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
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"Expected a digit label from 0 to 9: {path}")
        images.append(pixels.flatten())
        labels.append(label)
    return np.array(images), np.array(labels)


class NeuralNetwork:
    def __init__(self, input_size=4096, hidden_size=32, output_size=10):
        rng = np.random.default_rng(42)
        self.weights1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, output_size))
        self.bias2 = np.zeros(output_size)

    def forward(self, x):
        hidden = np.maximum(0, x @ self.weights1 + self.bias1)
        scores = hidden @ self.weights2 + self.bias2
        scores = scores - scores.max(axis=1, keepdims=True)
        log_probs = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
        return hidden, log_probs

    def evaluate(self, x, y):
        _, log_probs = self.forward(x)
        loss = -log_probs[np.arange(len(y)), y].mean()
        accuracy = (log_probs.argmax(axis=1) == y).mean()
        return loss, accuracy

    def gradients(self, x, y):
        hidden, log_probs = self.forward(x)
        output_error = np.exp(log_probs)
        output_error[np.arange(len(y)), y] -= 1
        output_error /= len(y)
        hidden_error = (output_error @ self.weights2.T) * (hidden > 0)
        return (
            x.T @ hidden_error,
            hidden_error.sum(axis=0),
            hidden.T @ output_error,
            output_error.sum(axis=0),
        )

    def train_step(self, x, y, learning_rate):
        weight1_gradient, bias1_gradient, weight2_gradient, bias2_gradient = self.gradients(x, y)
        self.weights1 -= learning_rate * weight1_gradient
        self.bias1 -= learning_rate * bias1_gradient
        self.weights2 -= learning_rate * weight2_gradient
        self.bias2 -= learning_rate * bias2_gradient
        parameters = (self.weights1, self.bias1, self.weights2, self.bias2)
        if not all(np.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("Weights diverged; reduce LEARNING_RATE.")


def plot_history(history):
    epochs, train_loss, valid_loss, train_acc, valid_acc = np.array(history).T
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, valid_loss, label="Valid")
    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, valid_acc, label="Valid")
    axes[1].set_ylim(0, 1.05)
    for axis, title in zip(axes, ("Loss", "Accuracy")):
        axis.set(title=title, xlabel="Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOT_PATH)
    return figure


def main():
    if EPOCHS <= 0 or not np.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        raise ValueError("EPOCHS and LEARNING_RATE must be positive.")
    if not isinstance(HIDDEN_SIZE, int) or HIDDEN_SIZE <= 0:
        raise ValueError("HIDDEN_SIZE must be a positive integer.")

    # Step 1: Load training and validation data.
    x_train, y_train = load_data(DATA_DIR / "train")
    x_valid, y_valid = load_data(DATA_DIR / "val")

    # Step 2: Create a 4096 -> HIDDEN_SIZE -> 10 network with ReLU.
    model = NeuralNetwork(hidden_size=HIDDEN_SIZE, output_size=NUM_CLASSES)
    history = []

    print(f"Network: 4096 -> {HIDDEN_SIZE} (ReLU) -> {NUM_CLASSES}")
    print("Epoch | Train acc | Train loss | Valid acc | Valid loss")
    for epoch in range(EPOCHS + 1):
        # Step 3: Backpropagate training loss and update weights.
        if epoch > 0:
            model.train_step(x_train, y_train, LEARNING_RATE)

        # Step 4: Measure both datasets without updating weights from validation.
        train_loss, train_acc = model.evaluate(x_train, y_train)
        valid_loss, valid_acc = model.evaluate(x_valid, y_valid)
        history.append((epoch, train_loss, valid_loss, train_acc, valid_acc))
        print(
            f"{epoch:5d} | {train_acc:9.2%} | {train_loss:10.4f} | "
            f"{valid_acc:9.2%} | {valid_loss:10.4f}",
            flush=True,
        )

    # Step 5: Save weights and biases for both layers.
    np.savez(
        MODEL_PATH, weights1=model.weights1, bias1=model.bias1,
        weights2=model.weights2, bias2=model.bias2,
    )
    print(f"Weights saved to: {MODEL_PATH}")

    # Step 6: Evaluate test accuracy once, without updating weights.
    x_test, y_test = load_data(DATA_DIR / "test")
    _, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.2%}")

    # Step 7: Save and display training and validation curves.
    figure = plot_history(history)
    print(f"Training plot saved to: {PLOT_PATH}")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
