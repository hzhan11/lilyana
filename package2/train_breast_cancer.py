"""Learn image-level binary classification; this is not a diagnostic tool."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


PACKAGE_DIR = Path(__file__).resolve().parent
DATASET_URL = "https://zenodo.org/records/10519652/files/breastmnist_64.npz?download=1"
DATASET_MD5 = "742edef2a1fd1524b2efff4bd7ba9364"
DATASET_NAME = "BreastMNIST64"
CLASS_NAMES = ("normal_or_benign", "malignant")
LABEL_MAPPING = {"official_0": 1, "official_1": 0}
SPLIT_COUNTS = {"train": 546, "val": 78, "test": 156}
TRAIN_FILES = (
    "breastmnist_training_metrics.csv", "breastmnist_cnn.pt",
    "learning_curves.png", "training_summary.json", "validation_confusion_matrix.png",
)
TEST_FILES = (
    "test_metrics.json", "test_predictions.csv",
    "confusion_matrix.png", "test_examples.png",
)
PREDICT_FILES = ("prediction.png", "prediction.json")
METRIC_FIELDS = (
    "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "val_precision", "val_recall", "val_specificity", "val_f1",
)
NOTICE = "Educational model score, not a diagnosis. No clinical use."


def file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(path):
    if file_md5(path) != DATASET_MD5:
        raise ValueError(f"Dataset checksum mismatch: {path}. Expected official {DATASET_MD5}.")


def ensure_dataset(path, download=False):
    """Download only by explicit request; never replace an invalid existing file."""
    path = Path(path)
    if path.exists():
        verify_dataset(path)
        return path
    if not download:
        raise FileNotFoundError(
            f"Dataset missing: {path}. Run --prepare-data --download first."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique partial file prevents failed downloads from looking complete.
    descriptor, partial_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with urllib.request.urlopen(DATASET_URL, timeout=60) as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
        verify_dataset(partial)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def remap_labels(labels):
    """Official 0=malignant, 1=normal/benign; our positive class is malignant."""
    labels = np.asarray(labels)
    if labels.dtype.kind not in "iu" or not np.isin(labels, [0, 1]).all():
        raise ValueError("Official labels must be integers containing only 0 or 1.")
    return (1 - labels.astype(np.int64)).reshape(-1)


def validate_split(images, labels, split):
    count = SPLIT_COUNTS[split]
    if images.shape != (count, 64, 64):
        raise ValueError(f"{split}_images must have shape {(count, 64, 64)}.")
    if images.dtype != np.uint8:
        raise ValueError(f"{split}_images must contain uint8 pixels (0 through 255).")
    if labels.shape != (count, 1):
        raise ValueError(f"{split}_labels must have shape {(count, 1)}.")
    return images, remap_labels(labels)


def load_splits(path):
    """Verify the official archive and preserve its three existing image splits."""
    verify_dataset(path)
    splits = {}
    with np.load(path, allow_pickle=False) as archive:
        for split in SPLIT_COUNTS:
            keys = (f"{split}_images", f"{split}_labels")
            if any(key not in archive for key in keys):
                raise ValueError(f"Dataset is missing required keys: {keys}.")
            splits[split] = validate_split(archive[keys[0]], archive[keys[1]], split)
    return splits


def normalize_image(image):
    image = np.asarray(image)
    if image.shape != (64, 64) or image.dtype != np.uint8:
        raise ValueError("Input image must be a 64x64 uint8 grayscale array.")
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


class BreastDataset(Dataset):
    """Labels supplied here have already been remapped; do not reverse again."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return normalize_image(self.images[index]), torch.tensor(
            self.labels[index], dtype=torch.float32
        )


class SimpleCNN(nn.Module):
    """Two convolution blocks followed by a small binary-classification head."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs).squeeze(1)


def classify_scores(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be finite and between 0 and 1.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1.")
    return (scores >= threshold).astype(np.int64)


def binary_metrics(labels, scores, threshold=0.5):
    """Pool image counts; undefined denominators return None, never invented 0."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be matching nonempty 1D arrays.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Package labels must be 0 or 1.")
    predicted = classify_scores(scores, threshold)
    tp = int(((labels == 1) & (predicted == 1)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": len(labels),
    }


def guard_output(output_dir, overwrite=False):
    """Protect previous experiments, including artifacts from other modes."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    existing = [
        name for name in TRAIN_FILES + TEST_FILES + PREDICT_FILES
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Existing results in {output_dir}: {', '.join(existing)}. "
            "Choose a new --output-dir or explicitly use --overwrite."
        )


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, allow_nan=False)
        target.write("\n")


def plot_library(show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(split, args, shuffle=False):
    return DataLoader(
        BreastDataset(*split), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )


def run_epoch(model, loader, device, threshold, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    labels_all, scores_all = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss: stop and check the learning rate/data.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
            scores_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    metrics = binary_metrics(labels_all, scores_all, threshold)
    metrics["loss"] = total_loss / metrics["total"]
    return metrics, np.asarray(labels_all), np.asarray(scores_all)


def checkpoint_metadata(args, epoch, val_loss):
    return {
        "architecture": "SimpleCNN", "dataset": DATASET_NAME, "image_size": 64,
        "label_mapping": LABEL_MAPPING, "class_names": list(CLASS_NAMES),
        "threshold": args.threshold, "best_epoch": epoch,
        "best_validation_loss": val_loss, "settings": training_settings(args),
        "dataset_md5": DATASET_MD5, "split_counts": SPLIT_COUNTS,
    }


def training_settings(args):
    return {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "workers": args.workers, "threads": args.threads,
        "device": args.device, "image_size": 64, "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss", "normalization": "uint8 / 255",
    }


def validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a metadata/state_dict dictionary.")
    expected = {
        "architecture": "SimpleCNN", "dataset": DATASET_NAME, "image_size": 64,
        "class_names": list(CLASS_NAMES), "label_mapping": LABEL_MAPPING,
        "dataset_md5": DATASET_MD5, "split_counts": SPLIT_COUNTS,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible checkpoint metadata: {key}.")
    threshold = checkpoint.get("threshold")
    if (
        isinstance(threshold, bool) or not isinstance(threshold, (float, int))
        or not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("Checkpoint threshold must be finite and between 0 and 1.")
    epoch = checkpoint.get("best_epoch")
    loss = checkpoint.get("best_validation_loss")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint best_epoch must be a positive integer.")
    if (
        isinstance(loss, bool) or not isinstance(loss, (float, int))
        or not math.isfinite(loss) or loss < 0
    ):
        raise ValueError("Checkpoint best_validation_loss must be finite and nonnegative.")
    if not isinstance(checkpoint.get("settings"), dict):
        raise ValueError("Checkpoint is missing training settings.")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint is missing state_dict.")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    model = SimpleCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains nonfinite model weights.")
    model.to(device).eval()
    return model, checkpoint


def weights_provenance(path, checkpoint):
    path = Path(path).resolve()
    return {
        "path": str(path), "md5": file_md5(path),
        "architecture": checkpoint["architecture"],
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "training_settings": checkpoint["settings"],
    }


def print_metrics(metrics):
    for key in ("loss", "accuracy", "precision", "recall", "specificity", "f1"):
        value = metrics[key]
        print(f"  {key}: {'N/A' if value is None else f'{value:.4f}'}")
    print("N/A (JSON null / CSV blank) means the metric's denominator is zero.")
    print("All metrics count images, not patients. " + NOTICE)


def save_confusion_matrix(metrics, path, title):
    """Rows are reference image labels; columns are predicted image labels."""
    plt = plot_library()
    figure, axis = plt.subplots(figsize=(6, 5))
    counts = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    axis.imshow(counts, cmap="Blues")
    axis.set(
        xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label", title=title,
    )
    for row in range(2):
        for column in range(2):
            color = "white" if counts[row, column] > counts.max() / 2 else "black"
            axis.text(
                column, row, str(counts[row, column]),
                ha="center", va="center", color=color,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train(args, splits, device):
    train_loader = make_loader(splits["train"], args, shuffle=True)
    val_loader = make_loader(splits["val"], args)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_loss, best_epoch = math.inf, None
    best_val_metrics = None
    csv_path = args.output_dir / TRAIN_FILES[0]
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        target.flush()
        for epoch in range(1, args.epochs + 1):
            train_metrics, _, _ = run_epoch(
                model, train_loader, device, args.threshold, optimizer
            )
            val_metrics, _, _ = run_epoch(model, val_loader, device, args.threshold)
            if val_metrics["loss"] < best_loss:
                best_loss, best_epoch = val_metrics["loss"], epoch
                best_val_metrics = val_metrics.copy()
                checkpoint = checkpoint_metadata(args, epoch, best_loss)
                checkpoint["state_dict"] = model.state_dict()
                torch.save(checkpoint, args.output_dir / TRAIN_FILES[1])
            row = {
                "epoch": epoch, "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                **{f"val_{key}": val_metrics[key] for key in (
                    "loss", "accuracy", "precision", "recall", "specificity", "f1"
                )},
            }
            writer.writerow(row)
            target.flush()
            history.append(row)
            print(
                f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, "
                f"accuracy={train_metrics['accuracy']:.4f}; "
                f"val loss={val_metrics['loss']:.4f}, accuracy={val_metrics['accuracy']:.4f}",
                flush=True,
            )
    plt = plot_library()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot(
                [row["epoch"] for row in history],
                [row[f"{split}_{metric}"] for row in history], label=split,
            )
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=metric.capitalize())
        axis.legend()
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(args.output_dir / TRAIN_FILES[2], dpi=150)
    plt.close(figure)
    if best_val_metrics is None:
        raise RuntimeError("Training did not produce a best-validation checkpoint.")
    save_confusion_matrix(
        best_val_metrics, args.output_dir / TRAIN_FILES[4],
        f"Validation confusion matrix (best epoch {best_epoch})\n"
        f"Threshold: {args.threshold:g}",
    )
    write_json(args.output_dir / TRAIN_FILES[3], {
        "dataset": DATASET_NAME, "dataset_md5": DATASET_MD5,
        "seed": args.seed, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_validation_metrics": best_val_metrics,
        "threshold": args.threshold, "label_names": list(CLASS_NAMES),
        "label_mapping": LABEL_MAPPING, "split_counts": SPLIT_COUNTS,
        "settings": training_settings(args), "notice": NOTICE,
    })
    print(f"Best validation-loss checkpoint: epoch {best_epoch}. Test was NOT evaluated.")
    print("Final-epoch validation image metrics (not necessarily the selected checkpoint):")
    print_metrics(val_metrics)


def test(args, splits, device):
    model, checkpoint = load_checkpoint(args.weights, device)
    threshold = checkpoint["threshold"]
    metrics, labels, scores = run_epoch(
        model, make_loader(splits["test"], args), device, threshold
    )
    predicted = classify_scores(scores, threshold)
    write_json(args.output_dir / TEST_FILES[0], {
        **metrics, "dataset": DATASET_NAME, "dataset_md5": DATASET_MD5,
        "threshold": threshold, "label_names": list(CLASS_NAMES),
        "label_mapping": LABEL_MAPPING,
        "weights_provenance": weights_provenance(args.weights, checkpoint),
        "unit": "image", "notice": NOTICE,
    })
    with (args.output_dir / TEST_FILES[1]).open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("index", "true_label", "predicted_label", "malignant_score"))
        for index, (label, decision, score) in enumerate(zip(labels, predicted, scores)):
            writer.writerow((index, int(label), int(decision), float(score)))
    save_confusion_matrix(
        metrics, args.output_dir / TEST_FILES[2], "Test image confusion matrix",
    )
    plt = plot_library()
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for index, axis in enumerate(axes.flat):
        axis.imshow(splits["test"][0][index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"True: {CLASS_NAMES[labels[index]]}\n"
            f"Pred: {CLASS_NAMES[predicted[index]]}\nMalignant score: {scores[index]:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle("First 8 official test images (fixed order) — not diagnoses")
    figure.tight_layout()
    figure.savefig(args.output_dir / TEST_FILES[3], dpi=150)
    plt.close(figure)
    print("Held-out test image metrics (checkpoint threshold):")
    print_metrics(metrics)


def prepare_prediction_image(path):
    """Apply the same grayscale/64x64 preprocessing for CLI and desktop demos."""
    with Image.open(path) as source:
        image = source.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def predict_image_array(model, array, threshold, device):
    """Return a malignant score and package label without changing weights."""
    model.eval()
    with torch.no_grad():
        inputs = normalize_image(array).unsqueeze(0).to(device)
        score = torch.sigmoid(model(inputs))[0].item()
    decision = int(classify_scores([score], threshold)[0])
    return score, decision


def predict(args, device):
    model, checkpoint = load_checkpoint(args.weights, device)
    array = prepare_prediction_image(args.predict)
    score, decision = predict_image_array(
        model, array, checkpoint["threshold"], device,
    )
    write_json(args.output_dir / PREDICT_FILES[1], {
        "input_image": str(args.predict.resolve()), "predicted_label": decision,
        "predicted_class": CLASS_NAMES[decision], "malignant_score": score,
        "threshold": checkpoint["threshold"], "label_names": list(CLASS_NAMES),
        "label_mapping": LABEL_MAPPING,
        "weights_provenance": weights_provenance(args.weights, checkpoint),
        "notice": NOTICE,
    })
    plt = plot_library(show=not args.no_show)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(array, cmap="gray", vmin=0, vmax=255)
    axis.axis("off")
    axis.set_title(f"{CLASS_NAMES[decision]}\nMalignant model score: {score:.4f}")
    figure.text(0.5, 0.03, NOTICE, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(args.output_dir / PREDICT_FILES[0], dpi=150)
    print(f"Predicted class: {CLASS_NAMES[decision]}; malignant model score: {score:.4f}")
    print(NOTICE)
    if not args.no_show:
        plt.show()
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-data", action="store_true", help="Validate/download only.")
    modes.add_argument("--test", action="store_true", help="Evaluate a saved checkpoint.")
    modes.add_argument("--predict", type=Path, metavar="IMAGE", help="Classify one image.")
    parser.add_argument("--download", action="store_true", help="Allow official dataset download.")
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "BreastMNIST64")
    parser.add_argument("--weights", type=Path, help="Required for test/predict; no resume.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Prediction: save without a window.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=None, help="Training only; default 0.5.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0:
        parser.error("--workers must be nonnegative.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive.")
    if args.threshold is not None:
        if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
            parser.error("--threshold must be finite and between 0 and 1.")
        if args.test or args.predict is not None or args.prepare_data:
            parser.error("--threshold is training-only; test/predict use the checkpoint.")
    else:
        args.threshold = 0.5
    inference = args.test or args.predict is not None
    if inference and args.weights is None:
        parser.error("--weights is required for --test/--predict.")
    if not inference and args.weights is not None:
        parser.error("--weights is only allowed with --test/--predict; training starts fresh.")
    if args.no_show and args.predict is None:
        parser.error("--no-show is only allowed with --predict.")
    if args.predict is not None and args.download:
        parser.error("--predict does not use the dataset; --download is not allowed.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.prepare_data:
        guard_output(args.output_dir, args.overwrite)
    if args.prepare_data:
        path = ensure_dataset(args.data_dir / "breastmnist_64.npz", args.download)
        splits = load_splits(path)
        print(f"Verified {DATASET_NAME}: {path}")
        for name, (_, labels) in splits.items():
            print(
                f"{name}: {len(labels)} images; normal_or_benign={int((labels == 0).sum())}, "
                f"malignant={int((labels == 1).sum())}"
            )
        print("Official label 0 -> package 1; official label 1 -> package 0.")
        return
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable.")
    args.device = device_name
    device = torch.device(device_name)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    print(f"Device: {device}. {NOTICE}")
    splits = None
    if args.predict is None:
        path = ensure_dataset(args.data_dir / "breastmnist_64.npz", args.download)
        splits = load_splits(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.predict is not None:
        predict(args, device)
    elif args.test:
        test(args, splits, device)
    else:
        train(args, splits, device)
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
