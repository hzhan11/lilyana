from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import tarfile
import tempfile
import threading
import time
from typing import Callable
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CLASS8 = Path(__file__).resolve().parent
DATA_DIR = CLASS8 / "data"
RUNS_DIR = CLASS8 / "runs"
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


@dataclass(frozen=True)
class Config:
    blocks: int = 3
    convs_per_block: int = 2
    base_channels: int = 32
    dropout: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "AdamW"
    weight_decay: float = 0.0001
    train_limit: int = 5000
    val_limit: int = 1000
    augment: bool = True
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 4


def validate_config(config):
    bounds = {
        "blocks": (1, 4), "convs_per_block": (1, 3),
        "epochs": (1, 500), "batch_size": (8, 512),
        "train_limit": (0, 45000), "val_limit": (0, 5000),
        "seed": (0, 2**32 - 1), "cpu_threads": (1, 32),
    }
    for name, (low, high) in bounds.items():
        value = getattr(config, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
    if type(config.base_channels) is not int or config.base_channels not in (8, 16, 32, 64, 128):
        raise ValueError("Base channels must be 8, 16, 32, 64, or 128.")
    for name, low, high in (
        ("dropout", 0, 0.8), ("learning_rate", 0, 1), ("weight_decay", 0, 1),
    ):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be a finite number between {low} and {high}.")
    if config.learning_rate == 0:
        raise ValueError("Learning rate must be greater than zero.")
    if config.optimizer not in ("AdamW", "SGD"):
        raise ValueError("Optimizer must be AdamW or SGD.")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    if type(config.augment) is not bool:
        raise ValueError("Augmentation must be enabled or disabled.")


def build_model(config):
    validate_config(config)
    layers = []
    in_channels = 3
    for block in range(config.blocks):
        out_channels = min(config.base_channels * 2**block, 512)
        for _ in range(config.convs_per_block):
            layers.extend((
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(2))
    layers.extend((
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(config.dropout), nn.Linear(in_channels, 10),
    ))
    return nn.Sequential(*layers)


def parameter_count(config):
    # Do not allocate large preview models or advance the training RNG.
    with torch.device("meta"):
        return sum(parameter.numel() for parameter in build_model(config).parameters())


def split_indices(targets, seed, train_limit=0, val_limit=0):
    labels = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_groups, val_groups = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) != 5000:
            raise ValueError("Expected the official CIFAR-10 training set: 5,000 images per class.")
        val_groups.append(indices[:500])
        train_groups.append(indices[500:])

    def select(groups, limit):
        total = sum(len(group) for group in groups)
        count = limit or total
        if not 1 <= count <= total:
            raise ValueError(f"Sample limit must be zero (all) or between 1 and {total}.")
        quotient, remainder = divmod(count, 10)
        selected = np.concatenate([
            group[:quotient + (label < remainder)] for label, group in enumerate(groups)
        ])
        return rng.permutation(selected).tolist()

    return select(train_groups, train_limit), select(val_groups, val_limit)


class CancelledTraining(Exception):
    pass


def check_stop(stop_event):
    if stop_event.is_set():
        raise CancelledTraining()


def ensure_dataset(data_dir, emit, stop_event):
    check_stop(stop_event)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        return datasets.CIFAR10(str(data_dir), train=True, download=False)
    except RuntimeError as error:
        if "Dataset not found or corrupted" not in str(error):
            raise
        emit({"type": "status", "message": f"CIFAR-10 not cached: {error}"})

    archive = data_dir / datasets.CIFAR10.filename
    expected_md5 = datasets.CIFAR10.tgz_md5

    def valid_archive(path):
        if not path.is_file():
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "md5").hexdigest() == expected_md5

    if not valid_archive(archive):
        emit({"type": "status", "message": "Downloading official CIFAR-10 (~170 MB); Stop cancels the download."})
        request = urllib.request.Request(
            datasets.CIFAR10.url, headers={"User-Agent": "Class8-CNN-Playground"}
        )
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    received, last_report = 0, 0.0
                    while True:
                        check_stop(stop_event)
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report >= 1:
                            size = f" / {total / 1e6:.1f} MB" if total else " MB"
                            emit({"type": "status", "message": f"Downloading CIFAR-10: {received / 1e6:.1f}{size}"})
                            last_report = time.monotonic()
                stream.close()
                check_stop(stop_event)
                if not valid_archive(temporary):
                    raise ValueError("CIFAR-10 archive checksum failed; download was not accepted.")
                temporary.replace(archive)
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)
    check_stop(stop_event)
    emit({"type": "status", "message": "Verifying and extracting CIFAR-10. The official test set will not be used."})
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            check_stop(stop_event)
            source.extract(member, data_dir, filter="data")
    check_stop(stop_event)
    return datasets.CIFAR10(str(data_dir), train=True, download=False)


class ImageSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


def make_loaders(dataset, config, device):
    train_indices, val_indices = split_indices(
        dataset.targets, config.seed, config.train_limit, config.val_limit
    )
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    augmentation = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if config.augment else []
    )
    train_transform = transforms.Compose(augmentation + [transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    common = {"batch_size": config.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train = DataLoader(
        ImageSubset(dataset, train_indices, train_transform),
        shuffle=True, generator=torch.Generator().manual_seed(config.seed), **common,
    )
    val = DataLoader(ImageSubset(dataset, val_indices, val_transform), shuffle=False, **common)
    return train, val, train_indices, val_indices


def run_training(
    config: Config,
    emit: Callable[[dict], None],
    stop_event: threading.Event,
    *,
    data_dir=DATA_DIR,
    runs_dir=RUNS_DIR,
):
    validate_config(config)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected, but no CUDA-capable PyTorch device is available.")
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device
    )
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    model, optimizer = None, None
    completed, best_accuracy, partial_epoch = 0, None, False
    started = time.monotonic()

    def checkpoint(name, status):
        if model is None:
            return
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config), "class_names": list(CLASS_NAMES),
            "epoch": completed, "partial_epoch": partial_epoch,
            "best_accuracy": best_accuracy, "status": status,
        }
        temporary = output / f"{name}.tmp"
        torch.save(state, temporary)
        temporary.replace(output / name)

    try:
        emit({"type": "status", "message": "Preparing CIFAR-10: 45,000 train / 5,000 validation; 10,000 test images reserved."})
        dataset = ensure_dataset(Path(data_dir), emit, stop_event)
        check_stop(stop_event)
        train, val, train_indices, val_indices = make_loaders(dataset, config, device)
        (output / "split_indices.json").write_text(
            json.dumps({"train": train_indices, "val": val_indices}), encoding="utf-8"
        )
        model = build_model(config).to(device)
        optimizer = (
            torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            if config.optimizer == "AdamW" else
            torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        )
        loss_fn = nn.CrossEntropyLoss()
        emit({
            "type": "started", "output_dir": str(output), "device": str(device),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_size": len(train.dataset), "val_size": len(val.dataset),
            "total_batches": len(train) + len(val),
        })
        fields = ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "elapsed_seconds")
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, config.epochs + 1):
                check_stop(stop_event)
                partial_epoch = True
                summary = {"epoch": epoch}
                for phase, loader in (("train", train), ("val", val)):
                    model.train(phase == "train")
                    total_loss, correct, seen, last_report = 0.0, 0, 0, 0.0
                    with torch.set_grad_enabled(phase == "train"):
                        for batch, (images, labels) in enumerate(loader, 1):
                            check_stop(stop_event)
                            images, labels = images.to(device), labels.to(device)
                            if phase == "train":
                                optimizer.zero_grad(set_to_none=True)
                            scores = model(images)
                            loss = loss_fn(scores, labels)
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f"Non-finite {phase} loss; try a lower learning rate.")
                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                            total_loss += loss.item() * len(labels)
                            correct += (scores.argmax(1) == labels).sum().item()
                            seen += len(labels)
                            if batch in (1, len(loader)) or time.monotonic() - last_report >= 0.2:
                                emit({
                                    "type": "metrics", "phase": phase,
                                    "epoch": epoch - 1 + batch / len(loader),
                                    "loss": total_loss / seen, "accuracy": correct / seen,
                                    "batch": batch, "batches": len(loader),
                                    "epoch_number": epoch, "epochs": config.epochs,
                                })
                                last_report = time.monotonic()
                    summary[f"{phase}_loss"] = total_loss / seen
                    summary[f"{phase}_accuracy"] = correct / seen
                completed, partial_epoch = epoch, False
                summary["elapsed_seconds"] = time.monotonic() - started
                writer.writerow(summary)
                stream.flush()
                if best_accuracy is None or summary["val_accuracy"] > best_accuracy:
                    best_accuracy = summary["val_accuracy"]
                    checkpoint("best.pt", "running")
                checkpoint("last.pt", "completed" if completed == config.epochs else "running")
                emit({"type": "epoch", **summary})
        check_stop(stop_event)
    except CancelledTraining:
        checkpoint("last.pt", "stopped")
        emit({
            "type": "stopped", "output_dir": str(output),
            "best_accuracy": best_accuracy, "epochs_completed": completed,
        })
        return output
    emit({
        "type": "done", "output_dir": str(output),
        "best_accuracy": best_accuracy, "epochs_completed": completed,
    })
    return output
