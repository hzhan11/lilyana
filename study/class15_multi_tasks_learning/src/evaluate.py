"""Load 1.model / 2.model / m.model and evaluate classification and centers on data3."""

import argparse
import json
from pathlib import Path
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from model import AnimalCNN, CENTER_SCALE, IMAGE_SIZE
from pic.dataset import ANIMALS, load_samples
from train import DATASET_CLASSES, ROOT, samples_to_tensors


def load_model(path, dataset):
    if dataset not in DATASET_CLASSES:
        raise ValueError("Model must be 1, 2, or m.")
    path = Path(path)
    if not path.is_file():
        action = "mtrain" if dataset == "m" else f"training {dataset}"
        raise ValueError(
            f"Model not found: {path}. Run start.bat {action} "
            f"or sh start.sh {action} first."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: expected a checkpoint dictionary.")
    classes = checkpoint.get("classes")
    if (
        not isinstance(classes, list) or len(classes) != len(DATASET_CLASSES[dataset])
        or any(not isinstance(name, str) for name in classes)
        or set(classes) != set(DATASET_CLASSES[dataset])
        or checkpoint.get("dataset") != dataset
        or checkpoint.get("input_shape") != [IMAGE_SIZE, IMAGE_SIZE, 4]
        or checkpoint.get("center_scale") != CENTER_SCALE
    ):
        raise ValueError(f"{path}: checkpoint metadata does not match model {dataset}.")
    model = AnimalCNN(num_classes=len(classes))
    try:
        model.load_state_dict(checkpoint.get("state_dict", {}))
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"{path}: incompatible model weights: {error}") from error
    model.eval()
    return model, classes


def calculate_metrics(actual, predicted, target_centers, predicted_centers, classes):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics require equal, nonempty actual and predicted labels.")
    target = torch.as_tensor(target_centers, dtype=torch.float64)
    estimated = torch.as_tensor(predicted_centers, dtype=torch.float64)
    if target.shape != (len(actual), 2) or estimated.shape != target.shape:
        raise ValueError("Center metrics require one [x,y] pair per label.")
    if not torch.isfinite(target).all() or not torch.isfinite(estimated).all():
        raise ValueError("Center metrics require finite coordinates.")
    if any(name not in classes for name in (*actual, *predicted)):
        raise ValueError("Metric labels must belong to the supplied class list.")
    report = classification_report(
        actual, predicted, labels=list(classes), output_dict=True, zero_division=0,
    )
    error = estimated - target
    return {
        "samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro": {key: float(report["macro avg"][key]) for key in ("precision", "recall", "f1-score")},
        "weighted": {key: float(report["weighted avg"][key]) for key in ("precision", "recall", "f1-score")},
        "per_class": {name: report[name] for name in classes},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(classes)).tolist(),
        "class_order": list(classes),
        "center": {
            "mae_pixels": error.abs().mean().item(),
            "mae_x_pixels": error[:, 0].abs().mean().item(),
            "mae_y_pixels": error[:, 1].abs().mean().item(),
            "rmse_pixels": error.square().mean().sqrt().item(),
            "mean_distance_pixels": error.square().sum(dim=1).sqrt().mean().item(),
        },
    }


def print_metrics(title, metrics):
    print(f"\n{title} ({metrics['samples']} images)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Average       Precision    Recall        F1")
    for average in ("macro", "weighted"):
        values = metrics[average]
        print(f"{average:<13} {values['precision']:.4f}       {values['recall']:.4f}"
              f"       {values['f1-score']:.4f}")
    print("Class         Precision    Recall        F1  Support")
    for name, values in metrics["per_class"].items():
        print(f"{name:<13} {values['precision']:.4f}       {values['recall']:.4f}"
              f"       {values['f1-score']:.4f}  {int(values['support']):7d}")
    print("Confusion matrix: rows=actual, columns=predicted")
    print("Class order: " + ", ".join(metrics["class_order"]))
    for row in metrics["confusion_matrix"]:
        print(" ".join(f"{value:6d}" for value in row))
    center = metrics["center"]
    print(f"Center MAE: {center['mae_pixels']:.2f} px"
          f" | RMSE: {center['rmse_pixels']:.2f} px"
          f" | mean distance: {center['mean_distance_pixels']:.2f} px")


def evaluate(dataset, *, model_dir=ROOT, data_dir=ROOT / "pic" / "data3",
             output_dir=ROOT / "runs" / "evaluation"):
    if dataset not in DATASET_CLASSES:
        raise ValueError("Model must be 1, 2, or m.")
    path = Path(model_dir) / f"{dataset}.model"
    model, classes = load_model(path, dataset)
    samples = load_samples(Path(data_dir) / "label.txt")
    evaluation_classes = (*ANIMALS, *sorted({sample.animal for sample in samples} - set(ANIMALS)))
    data = samples_to_tensors(samples, evaluation_classes)
    with torch.inference_mode():
        logits, centers = model(data.tensors[0])
        if not torch.isfinite(logits).all() or not torch.isfinite(centers).all():
            raise ValueError(f"{path}: model produced non-finite predictions.")
        probabilities, indices = logits.softmax(dim=1).max(dim=1)
    actual = [sample.animal for sample in samples]
    predicted = [classes[index] for index in indices.tolist()]
    target_centers = data.tensors[2].tolist()
    predicted_centers = centers.tolist()
    known = [index for index, name in enumerate(actual) if name in classes]
    full = calculate_metrics(actual, predicted, target_centers, predicted_centers, evaluation_classes)
    known_metrics = None
    if known:
        known_metrics = calculate_metrics(
            [actual[index] for index in known], [predicted[index] for index in known],
            [target_centers[index] for index in known], [predicted_centers[index] for index in known],
            classes,
        )
    report = {
        "model": str(path.resolve()), "dataset": str(Path(data_dir).resolve()),
        "known_classes": classes, "unseen_class_samples": len(samples) - len(known),
        "zero_division_policy": "Undefined precision, recall, and F1 are reported as 0.",
        "full_dataset": full, "known_classes_only": known_metrics,
        "predictions": [
            {"filename": sample.path.name, "actual": sample.animal, "predicted": predicted[index],
             "confidence": probabilities[index].item(), "actual_center": target_centers[index],
             "predicted_center": predicted_centers[index], "known_class": sample.animal in classes}
            for index, sample in enumerate(samples)
        ],
    }
    print(f"\nModel {dataset}: {path}")
    print("Known classes: " + ", ".join(classes))
    print(f"Unseen-class images: {report['unseen_class_samples']} (counted as incorrect in full scores).")
    print(report["zero_division_policy"])
    print_metrics(f"All test images ({len(evaluation_classes)}-class scores)", full)
    if known_metrics is not None:
        print_metrics(f"Known classes only ({len(classes)}-class scores)", known_metrics)
    else:
        print("\nKnown-class scores: N/A (no known-class images; none were silently substituted).")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{dataset}.metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved report and per-image predictions: {output}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=("1", "2", "m", "compare"),
                        help="1, 2, m, or compare (all three with charts); omit for models 1 and 2.")
    parser.add_argument("--model-dir", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "pic" / "data3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "evaluation")
    parser.add_argument("--threads", type=int, default=2)
    options = parser.parse_args(argv)
    if not 1 <= options.threads <= 4:
        parser.error("threads must be between 1 and 4.")
    torch.set_num_threads(options.threads)
    datasets = (1, 2)
    if options.dataset == "compare":
        datasets = (1, 2, "m")
    elif options.dataset is not None:
        datasets = ("m" if options.dataset == "m" else int(options.dataset),)
    try:
        reports = {}
        for dataset in datasets:
            reports[str(dataset)] = evaluate(
                dataset, model_dir=options.model_dir, data_dir=options.data_dir,
                output_dir=options.output_dir,
            )
        if options.dataset == "compare":
            from charts import write_comparison

            for path in write_comparison(reports, options.output_dir):
                print(f"Saved comparison: {path}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
