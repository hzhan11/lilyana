"""Train animal classification and center localization on data1 or data2."""

import argparse
import math
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import AnimalCNN, CENTER_SCALE, IMAGE_SIZE
from pic.dataset import ANIMALS, load_samples


ROOT = Path(__file__).resolve().parent
DATASET_CLASSES = {1: ANIMALS[:3], 2: ANIMALS[3:], "m": ANIMALS}


def load_dataset(dataset, base_dir=ROOT / "pic"):
    if dataset not in DATASET_CLASSES:
        raise ValueError("Dataset must be 1, 2, or m.")
    classes = DATASET_CLASSES[dataset]
    samples = []
    for number in ((1, 2) if dataset == "m" else (dataset,)):
        subset = load_samples(Path(base_dir) / f"data{number}" / "label.txt")
        for sample in subset:
            if sample.animal not in DATASET_CLASSES[number]:
                raise ValueError(f"{sample.path}: class {sample.animal!r} does not belong to data{number}.")
        samples.extend(subset)
    return samples_to_tensors(samples, classes), classes


def samples_to_tensors(samples, classes):
    """Use identical image and coordinate preprocessing for training and evaluation."""
    images, targets, centers = [], [], []
    for sample in samples:
        if sample.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{sample.path}: training requires 200x200 images, not {sample.size}.")
        with Image.open(sample.path) as image:
            rgba = image.convert("RGBA")
            pixels = torch.frombuffer(bytearray(rgba.tobytes()), dtype=torch.uint8)
            images.append(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 4).float() / 255)
        targets.append(classes.index(sample.animal))
        centers.append(sample.center)
    return TensorDataset(
        torch.stack(images), torch.tensor(targets, dtype=torch.long),
        torch.tensor(centers, dtype=torch.float32),
    )


def split_dataset(data, seed=42):
    train_indices, validation_indices = train_test_split(
        list(range(len(data))), test_size=0.2, random_state=seed,
        stratify=data.tensors[1].tolist(),
    )
    return Subset(data, train_indices), Subset(data, validation_indices)


def multitask_loss(logits, predicted_centers, labels, centers):
    return nn.functional.cross_entropy(logits, labels) + 10 * nn.functional.mse_loss(
        predicted_centers / CENTER_SCALE, centers / CENTER_SCALE,
    )


def measure(model, loader):
    model.eval()
    total_loss, correct, center_error = 0.0, 0, 0.0
    with torch.no_grad():
        for images, labels, centers in loader:
            logits, predicted_centers = model(images)
            loss = multitask_loss(logits, predicted_centers, labels, centers)
            if not torch.isfinite(loss) or not torch.isfinite(predicted_centers).all():
                raise ValueError("Non-finite evaluation loss or predictions.")
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            center_error += (predicted_centers - centers).abs().sum().item()
    size = len(loader.dataset)
    return {"loss": total_loss / size, "accuracy": correct / size,
            "center_mae_pixels": center_error / (size * 2)}


def train(dataset, *, epochs=100, batch_size=5, learning_rate=0.001, seed=42,
          output_dir=ROOT, base_dir=ROOT / "pic"):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    data, classes = load_dataset(dataset, base_dir)
    training, validation = split_dataset(data, seed)
    torch.manual_seed(seed)
    model = AnimalCNN(num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    training_evaluation_loader = DataLoader(training, batch_size=batch_size)
    validation_loader = DataLoader(validation, batch_size=batch_size)
    source = "data1 + data2" if dataset == "m" else f"data{dataset}"
    print(f"Training {source}: {len(data)} images | classes: {', '.join(classes)}", flush=True)
    print(f"80/20 stratified split | train: {len(training)}"
          f" | validation: {len(validation)} | seed: {seed}", flush=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels, centers in loader:
            optimizer.zero_grad()
            logits, predicted_centers = model(images)
            loss = multitask_loss(logits, predicted_centers, labels, centers)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
        training_metrics = measure(model, training_evaluation_loader)
        validation_metrics = measure(model, validation_loader)
        history.append({"epoch": epoch, **training_metrics,
                        **{f"val_{name}": value for name, value in validation_metrics.items()}})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}"
                  f" | train/val loss {training_metrics['loss']:.4f}/{validation_metrics['loss']:.4f}"
                  f" | accuracy {training_metrics['accuracy']:.1%}/{validation_metrics['accuracy']:.1%}"
                  f" | center MAE {training_metrics['center_mae_pixels']:.2f}"
                  f"/{validation_metrics['center_mae_pixels']:.2f} px", flush=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / f"{dataset}.model"
    torch.save({
        "state_dict": model.state_dict(), "classes": list(classes), "dataset": dataset,
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 4], "center_scale": CENTER_SCALE,
        "config": {"epochs": epochs, "batch_size": batch_size,
                   "learning_rate": learning_rate, "seed": seed, "center_loss_weight": 10,
                   "validation_fraction": 0.2},
        "split": {"method": "stratified_random", "train_indices": training.indices,
                  "validation_indices": validation.indices,
                  "train_samples": len(training), "validation_samples": len(validation)},
        "history": history,
    }, checkpoint)
    print(f"Saved checkpoint: {checkpoint}", flush=True)
    return checkpoint


def main(argv=None, *, combined=False):
    description = "Train all six animals on data1 + data2 and save m.model." if combined else __doc__
    parser = argparse.ArgumentParser(description=description)
    if combined:
        parser.set_defaults(dataset="m")
    else:
        parser.add_argument("dataset", type=int, choices=(1, 2), help="1: cat/dog/pig; 2: rabbit/bear/fox.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (1-4).")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    options = parser.parse_args(argv)
    if not 1 <= options.threads <= 4:
        parser.error("threads must be between 1 and 4.")
    torch.set_num_threads(options.threads)
    try:
        train(options.dataset, epochs=options.epochs, batch_size=options.batch_size,
              learning_rate=options.learning_rate, seed=options.seed, output_dir=options.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
