import argparse
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn

if __package__:
    from .generate_data import DATA_DIR, expected_price
else:
    from generate_data import DATA_DIR, expected_price


CLASS13 = Path(__file__).resolve().parent


def load_data(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No house records found: {path}")
    features = np.array([[float(row["area_sqm"]), float(row["rooms"])] for row in rows], dtype=np.float32)
    targets = np.array([float(row["price_kusd"]) for row in rows], dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"Non-finite values in {path}")
    if (features <= 0).any() or (targets <= 0).any() or not np.equal(features[:, 1], np.floor(features[:, 1])).all():
        raise ValueError(f"Area/price must be positive and rooms must be positive integers: {path}")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return features, targets, ids


def create_model():
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Expected matching nonempty one-dimensional targets and predictions.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise FloatingPointError("Targets and predictions must be finite.")
    error = actual.astype(np.float64) - predicted.astype(np.float64)
    return {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean())}


def predict(model, features, scaling):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2 or not np.isfinite(features).all():
        raise ValueError("Expected finite rows of [area_sqm, rooms].")
    normalized = (features - np.asarray(scaling["x_mean"])) / np.asarray(scaling["x_scale"])
    model.eval()
    with torch.no_grad():
        result = model(torch.tensor(normalized, dtype=torch.float32)).squeeze(1).numpy()
    result = result * scaling["y_scale"] + scaling["y_mean"]
    if not np.isfinite(result).all():
        raise FloatingPointError("Predictions are not finite.")
    return result


def save_figures(data, model, scaling, history, predictions, metrics, output):
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axis = figure.subplots()
    points = axis.scatter(x_train[:, 0], y_train, c=x_train[:, 1], s=12, alpha=0.6, cmap="viridis", vmin=1, vmax=7)
    figure.colorbar(points, ax=axis, label="Rooms", ticks=range(1, 8))
    axis.set(xlabel="Area (square meters)", ylabel="Price ($1,000)", title="Synthetic house-price data: two inputs, one continuous target")
    axis.set_xlabel("Area (square meters)\n1,400 training / 300 validation / 300 test rows; no row overlap.")
    figure.savefig(output / "dataset.png", dpi=150)

    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, metric, label in zip(axes, ("mse", "mae"), ("MSE (($1,000)^2)", "MAE ($1,000)")):
        for split in ("train", "val"):
            axis.plot([row["epoch"] for row in history], [row[f"{split}_{metric}"] for row in history], label=split.title())
        axis.set(xlabel="Epoch", ylabel=label, title=metric.upper())
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Training curves: errors converted back to original price units")
    figure.savefig(output / "training_curves.png", dpi=150)

    figure = Figure(figsize=(12, 5), layout="constrained")
    curve, scatter = figure.subplots(1, 2)
    for rooms, color in ((2, "#2563eb"), (4, "#16a34a"), (6, "#9333ea")):
        area = np.linspace(max(40, rooms * 38 - 45), min(240, rooms * 38 + 45), 100)
        features = np.column_stack((area, np.full_like(area, rooms)))
        curve.plot(area, predict(model, features, scaling), color=color, label=f"NN: {rooms} rooms")
        curve.plot(area, expected_price(area, rooms), color=color, linestyle=":", alpha=0.7)
    curve.set(xlabel="Area (square meters)", ylabel="Price ($1,000)", title="Predicted price vs. area, with rooms held fixed")
    curve.legend()
    curve.text(0.03, 0.97, "Dotted: noise-free simulation formula\nCurves restricted to typical area/room combinations.",
               transform=curve.transAxes, va="top", fontsize=8)
    scatter.scatter(y_test, predictions["neural_network"], alpha=0.6, s=16)
    low = min(y_test.min(), predictions["neural_network"].min())
    high = max(y_test.max(), predictions["neural_network"].max())
    scatter.plot([low, high], [low, high], "--", color="#b91c1c", label="Perfect prediction")
    scatter.set(xlabel="Actual price ($1,000)", ylabel="Predicted price ($1,000)", title="Held-out test predictions")
    scatter.legend()
    figure.suptitle("Regression predicts a number, not a cat/dog class")
    figure.savefig(output / "prediction_curves.png", dpi=150)

    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axes = figure.subplots(1, 2)
    names = ("mean_baseline", "linear_regression", "neural_network")
    for axis, metric, units in zip(axes, ("mae", "mse"), ("$1,000", "($1,000)^2")):
        bars = axis.bar(("Training mean", "Linear", "Neural network"), [metrics[name][metric] for name in names],
                        color=("#94a3b8", "#ea580c", "#2563eb"))
        axis.bar_label(bars, fmt="%.1f", padding=3)
        axis.set(ylabel=f"{metric.upper()} ({units})", title=f"Test {metric.upper()}: lower is better")
        axis.margins(y=0.2)
    figure.suptitle("Same held-out houses; baselines fitted using training data only")
    figure.savefig(output / "metrics_comparison.png", dpi=150)


def run_training(epochs=400, learning_rate=0.01, seed=42, data_dir=DATA_DIR, output_dir=None, publish=True):
    if type(epochs) is not int or not 1 <= epochs <= 2000:
        raise ValueError("Epochs must be an integer between 1 and 2000.")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("Learning rate must be finite and greater than zero, up to 1.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    data_dir = Path(data_dir)
    x_train, y_train, train_ids = load_data(data_dir / "train.csv")
    x_val, y_val, val_ids = load_data(data_dir / "val.csv")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation IDs overlap.")
    scaling = {
        "x_mean": x_train.mean(0).tolist(), "x_scale": x_train.std(0).tolist(),
        "y_mean": float(y_train.mean()), "y_scale": float(y_train.std()),
    }
    if min(scaling["x_scale"]) <= 0 or scaling["y_scale"] <= 0:
        raise ValueError("Training features and targets must have nonzero variance.")
    normalized_x = torch.tensor((x_train - scaling["x_mean"]) / scaling["x_scale"], dtype=torch.float32)
    normalized_y = torch.tensor((y_train - scaling["y_mean"]) / scaling["y_scale"], dtype=torch.float32).unsqueeze(1)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_mse, best_state, best_epoch = [], float("inf"), None, None
    output = Path(output_dir) if output_dir is not None else CLASS13 / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print("2 inputs (area, rooms) -> 32 ReLU -> 32 ReLU -> 1 price. No Softmax.")
    print("Synthetic prices in $1,000. Scaling uses training data only; test stays held out.")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(normalized_x), normalized_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        loss.backward()
        optimizer.step()
        train_metrics = regression_metrics(y_train, predict(model, x_train, scaling))
        val_metrics = regression_metrics(y_val, predict(model, x_val, scaling))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["mse"] < best_mse:
            best_mse, best_epoch = val_metrics["mse"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | Train MAE {train_metrics['mae']:.2f}, MSE {train_metrics['mse']:.2f}"
                  f" | Val MAE {val_metrics['mae']:.2f}, MSE {val_metrics['mse']:.2f}")
    model.load_state_dict(best_state)
    config = {
        "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "best_epoch": best_epoch,
        "features": ["area_sqm", "rooms"], "units": "thousand USD (synthetic)", "scaling": scaling,
        "feature_min": x_train.min(0).tolist(), "feature_max": x_train.max(0).tolist(),
    }
    torch.save({"model_state_dict": model.state_dict(), **config}, output / "weights.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Only the validation split selects the checkpoint; open test targets afterwards.
    x_test, y_test, test_ids = load_data(data_dir / "test.csv")
    if set(test_ids) & (set(train_ids) | set(val_ids)):
        raise ValueError("Test sample IDs overlap training or validation.")
    design = np.column_stack((np.ones(len(x_train)), x_train))
    coefficients = np.linalg.lstsq(design, y_train, rcond=None)[0]
    predictions = {
        "mean_baseline": np.full(len(y_test), y_train.mean()),
        "linear_regression": np.column_stack((np.ones(len(x_test)), x_test)) @ coefficients,
        "neural_network": predict(model, x_test, scaling),
    }
    metrics = {name: regression_metrics(y_test, values) for name, values in predictions.items()}
    report = {
        "synthetic": True, "units": {"mae": "thousand USD", "mse": "(thousand USD)^2"},
        "counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "best_epoch": best_epoch, "test": metrics, "linear_coefficients": coefficients.tolist(),
        "train_target_mean": float(y_train.mean()),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "area_sqm", "rooms", "actual_price_kusd", *predictions))
        for index, sample_id in enumerate(test_ids):
            writer.writerow((sample_id, *x_test[index], y_test[index], *(values[index] for values in predictions.values())))
    data = {"train": (x_train, y_train), "test": (x_test, y_test)}
    save_figures(data, model, scaling, history, predictions, metrics, output)
    if publish:
        (CLASS13 / "pic").mkdir(exist_ok=True)
        for name in ("dataset.png", "training_curves.png", "prediction_curves.png", "metrics_comparison.png"):
            shutil.copyfile(output / name, CLASS13 / "pic" / name)
    print(f"\nBest validation checkpoint: epoch {best_epoch}")
    for name, values in metrics.items():
        print(f"{name:18s} | Test MAE {values['mae']:.2f} ($1,000) | MSE {values['mse']:.2f} (($1,000)^2)")
    print(f"Artifacts: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic house-price regression: area + rooms -> price.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(args.epochs, args.learning_rate, args.seed)
