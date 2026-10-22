import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn


CLASS9 = Path(__file__).resolve().parent
DATA_DIR = CLASS9 / "pic"
CLASS_NAMES = ("cat", "dog")
SCENARIO = "Tiny training set, shared-family validation, independently drawn style-shifted test set."


def load_data(folder):
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {folder}. Run generate_data.py first.")
    images, labels = [], []
    for path in paths:
        label = path.stem.split("_")[0]
        if label not in CLASS_NAMES:
            raise ValueError(f"Expected cat_<number>.png or dog_<number>.png: {path}")
        with Image.open(path) as image:
            if image.size != (128, 128):
                raise ValueError(f"Expected a 128x128 image: {path}")
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        images.append(pixels.transpose(2, 0, 1))
        labels.append(CLASS_NAMES.index(label))
    if set(labels) != {0, 1}:
        raise ValueError(f"Both cats and dogs are required in {folder}")
    return torch.from_numpy(np.stack(images)), torch.tensor(labels, dtype=torch.long), paths


def create_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 2),
    )


def evaluate(model, images, labels):
    model.eval()
    with torch.no_grad():
        scores = model(images)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Model scores contain non-finite values.")
        loss = nn.functional.cross_entropy(scores, labels).item()
        probabilities = scores.softmax(1).numpy()
    if not math.isfinite(loss) or not np.isfinite(probabilities).all():
        raise FloatingPointError("Evaluation loss or probabilities are not finite.")
    predicted = probabilities.argmax(1)
    actual = labels.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], average="macro", zero_division=0,
    )
    return {
        "loss": loss, "accuracy": float((predicted == actual).mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }, probabilities


def save_figures(history, results, paths, probabilities, output):
    epochs = [row["epoch"] for row in history]
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        for split, color in (("train", "#2563eb"), ("val", "#ea580c")):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=split.title(), color=color)
        axis.set(xlabel="Epoch", ylabel=title, title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].set_ylim(0, 1.05)
    figure.suptitle("Small-data fitting: train and validation share base drawing families")
    figure.savefig(output / "training_curves.png", dpi=150)

    figure = Figure(figsize=(11, 5), layout="constrained")
    axis = figure.subplots()
    metrics = ("accuracy", "precision", "recall", "f1")
    positions = np.arange(len(metrics))
    for index, (split, color) in enumerate((("train", "#2563eb"), ("val", "#ea580c"), ("test", "#7c3aed"))):
        bars = axis.bar(
            positions + (index - 1) * 0.24, [results[split][metric] for metric in metrics],
            width=0.24, label=split.title(), color=color,
        )
        axis.bar_label(bars, labels=[f"{bar.get_height():.0%}" for bar in bars], fontsize=9, padding=3)
    axis.set(xticks=positions, xticklabels=("Accuracy", "Macro Precision", "Macro Recall", "Macro F1"), ylim=(0, 1.18))
    axis.set_title("Near-duplicate validation vs. independently drawn test styles")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    axis.set_xlabel(
        "Test evaluated only after training. Undefined P/R/F1 are reported as zero.\n"
        "A style-shift test gap alone is not proof of classical train/validation overfitting.",
        labelpad=12,
    )
    figure.savefig(output / "metrics_comparison.png", dpi=150)

    columns = min(6, len(paths))
    rows = math.ceil(len(paths) / columns)
    figure = Figure(figsize=(columns * 2.1, rows * 2.6), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, path, scores in zip(axes.flat, paths, probabilities):
        actual = path.stem.split("_")[0]
        prediction = CLASS_NAMES[int(scores.argmax())]
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"{path.name}\nTrue: {actual} | Pred: {prediction}\nModel score: {scores.max():.1%}",
            color="#15803d" if actual == prediction else "#b91c1c", fontsize=9,
        )
    figure.suptitle("Fixed test set: all predictions shown, including correct ones")
    figure.savefig(output / "test_predictions.png", dpi=150)


def run_experiment(epochs=120, learning_rate=0.001, seed=42, threads=4, output_dir=None, publish=True):
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("Epochs must be an integer between 1 and 1000.")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("Learning rate must be finite and between 0 (exclusive) and 1.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("CPU threads must be an integer between 1 and 32.")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, y_train, _ = load_data(DATA_DIR / "train")
    x_val, y_val, _ = load_data(DATA_DIR / "val")
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    count = sum(parameter.numel() for parameter in model.parameters())
    output = Path(output_dir) if output_dir is not None else CLASS9 / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = CLASS9 / "dataset_manifest.json"
    config = {
        "epochs": epochs, "learning_rate": learning_rate, "seed": seed,
        "cpu_threads": threads, "parameters": count, "scenario": SCENARIO,
        "regularization": "None: no dropout, no weight decay, no online augmentation.",
        "input": "128x128 RGB / 255; spatial information and color are both retained.",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(SCENARIO)
    print(f"Parameters: {count:,} | Train: {len(y_train)} | Val: {len(y_val)}")
    print("P/R/F1 use macro averages over cat and dog; undefined ratios are reported as zero.")
    print("No early stopping or test-based model selection. Test images are loaded only after training.")
    history = []
    fields = ["epoch"] + [f"{split}_{metric}" for split in ("train", "val") for metric in ("loss", "accuracy", "precision", "recall", "f1")]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_train), y_train)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite; reduce the learning rate.")
            loss.backward()
            optimizer.step()
            train_metrics, _ = evaluate(model, x_train, y_train)
            val_metrics, _ = evaluate(model, x_val, y_val)
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(
                f"Epoch {epoch:3d}/{epochs} | Train loss {train_metrics['loss']:.4f}, "
                f"acc {train_metrics['accuracy']:.1%}, P {train_metrics['precision']:.3f}, "
                f"R {train_metrics['recall']:.3f}, F1 {train_metrics['f1']:.3f} | "
                f"Val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.1%}, "
                f"P {val_metrics['precision']:.3f}, R {val_metrics['recall']:.3f}, F1 {val_metrics['f1']:.3f}"
            )

    # Freeze the final checkpoint before opening the test split.
    torch.save({"model_state_dict": model.state_dict(), "class_names": list(CLASS_NAMES), "config": config}, output / "weights.pt")
    model.load_state_dict(torch.load(output / "weights.pt", weights_only=True)["model_state_dict"])
    x_test, y_test, test_paths = load_data(DATA_DIR / "test")
    test_metrics, probabilities = evaluate(model, x_test, y_test)
    results = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
    report = {
        "scenario": SCENARIO, "counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "metrics": results,
        "interpretation": "Shared-family validation can be optimistic. Style-shift test failure is not by itself classical overfitting.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "actual", "predicted", "cat_probability", "dog_probability"))
        for path, label, scores in zip(test_paths, y_test.tolist(), probabilities):
            writer.writerow((path.name, CLASS_NAMES[label], CLASS_NAMES[int(scores.argmax())], *scores.tolist()))
    save_figures(history, results, test_paths, probabilities, output)
    if publish:
        for name in ("training_curves.png", "metrics_comparison.png", "test_predictions.png"):
            shutil.copyfile(output / name, DATA_DIR / name)
    print("\nFinal metrics (same final checkpoint):")
    for split, metrics in results.items():
        print(f"{split:5s} | ACC {metrics['accuracy']:.1%} | P {metrics['precision']:.3f} | R {metrics['recall']:.3f} | F1 {metrics['f1']:.3f}")
    print(f"Train-test accuracy gap: {train_metrics['accuracy'] - test_metrics['accuracy']:.1%}")
    print("Low style-shift test scores alone do not prove classical train/validation overfitting.")
    print(f"Artifacts saved to: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=SCENARIO)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run_experiment(args.epochs, args.learning_rate, args.seed, args.threads)


if __name__ == "__main__":
    main()
