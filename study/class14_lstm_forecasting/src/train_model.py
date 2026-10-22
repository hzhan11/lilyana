"""CPU LSTM lesson. Run: python train_model.py (uses the frozen CSV, never regenerates it)."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import DEFAULT_DATA, ROOT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def valid_prices(prices):
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Prices must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must all be finite and strictly positive")
    return values


def load_data(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True:
        raise ValueError("This lesson requires a labeled synthetic dataset")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Dataset checksum differs from its frozen manifest")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["day_index", "date", "synthetic_price"]:
            raise ValueError("Expected CSV columns: day_index,date,synthetic_price")
        rows = list(reader)
    if len(rows) != manifest["rows"]:
        raise ValueError("CSV row count differs from manifest")
    indices = [int(row["day_index"]) for row in rows]
    dates = [row["date"] for row in rows]
    parsed = [date.fromisoformat(value) for value in dates]
    if indices != list(range(len(rows))) or any(a >= b for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Day indices must be contiguous and dates strictly chronological")
    if any(day.weekday() >= 5 for day in parsed):
        raise ValueError("Synthetic trading dates must be weekdays")
    return valid_prices([float(row["synthetic_price"]) for row in rows]), dates, manifest


def fit_scaler(training_prices):
    values = valid_prices(training_prices)
    scale = float(values.std())
    if scale <= 1e-12:
        raise ValueError("Training prices need nonzero variance")
    return {"mean": float(values.mean()), "scale": scale}


def make_windows(prices, lookback):
    values = valid_prices(prices)
    if not isinstance(lookback, int) or lookback < 1 or len(values) <= lookback:
        raise ValueError("Lookback must be positive and smaller than the series length")
    targets = np.arange(lookback, len(values))
    x = np.stack([values[t - lookback:t] for t in targets])[:, :, None]
    return x, values[targets, None], targets


def prepare(prices, lookback=25):
    values = valid_prices(prices)
    if len(values) < max(200, 4 * lookback):
        raise ValueError("Need at least 200 prices and four lookback windows")
    train_end, val_end = len(values) * 70 // 100, len(values) * 85 // 100
    scaler = fit_scaler(values[:train_end])
    raw_x, raw_y, targets = make_windows(values, lookback)
    x = ((raw_x - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    y = ((raw_y - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    masks = {"train": targets < train_end,
             "validation": (targets >= train_end) & (targets < val_end),
             "test": targets >= val_end}
    splits = {name: (x[mask], y[mask], targets[mask]) for name, mask in masks.items()}
    return splits, scaler, (train_end, val_end)


class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != 1:
            raise ValueError("LSTM input must have shape (batch, lookback, 1)")
        if not torch.isfinite(x).all():
            raise ValueError("LSTM input must contain only finite values")
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


def fit_model(train, validation, scaler, config, run):
    model = PriceLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train[0]), torch.from_numpy(train[1])),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    vx, vy = torch.from_numpy(validation[0]), torch.from_numpy(validation[1])
    best, history = float("inf"), []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vx), vy).item()
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_mse": total / len(train[0]),
                        "validation_mse": val_loss})
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "scaler": scaler, "config": config,
                        "architecture": {"input_size": 1, "hidden_size": 32, "num_layers": 1,
                                         "head": "Linear(32,1)"},
                        "epoch": epoch, "validation_mse": val_loss}, run / "best_model.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
            print(f"Epoch {epoch:3d}/{config['epochs']} | train MSE {history[-1]['train_mse']:.6f}"
                  f" | validation MSE {val_loss:.6f}", flush=True)
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return history


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PriceLSTM()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_next(checkpoint_path, observed_prices):
    """One next-price estimate from exactly lookback already observed actual prices."""
    model, checkpoint = load_checkpoint(checkpoint_path)
    values = valid_prices(observed_prices)
    if len(values) != checkpoint["config"]["lookback"]:
        raise ValueError("Provide exactly the checkpoint lookback count of observed prices")
    scaler = checkpoint["scaler"]
    x = ((values - scaler["mean"]) / scaler["scale"]).astype(np.float32)[None, :, None]
    with torch.no_grad():
        result = model(torch.from_numpy(x)).item() * scaler["scale"] + scaler["mean"]
    if not np.isfinite(result):
        raise ValueError("Model produced a non-finite prediction")
    return result


def persistence(prices, targets):
    values = valid_prices(prices)
    targets = np.asarray(targets)
    if (targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer)
            or len(targets) == 0 or (targets < 1).any() or (targets >= len(values)).any()):
        raise ValueError("Persistence requires valid integer target indices >= 1")
    return values[targets - 1]


def scores(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError("Metrics require equal nonempty one-dimensional shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = predicted - actual
    return {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.threads <= 4:
        parser.error("epochs must be positive; threads must be between 1 and 4")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    prices, dates, manifest = load_data(args.data)
    splits, scaler, boundaries = prepare(prices)
    run = ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    config = {"seed": args.seed, "epochs": args.epochs, "batch_size": 64,
              "learning_rate": 0.001, "lookback": 25, "threads": args.threads,
              "device": "cpu", "data_path": str(args.data.resolve()),
              "data_sha256": manifest["sha256"], "split_fractions": [0.70, 0.15, 0.15],
              "raw_rows": len(prices), "train_end_exclusive": boundaries[0],
              "validation_end_exclusive": boundaries[1],
              "numpy_version": str(np.__version__), "torch_version": str(torch.__version__),
              "selection": "Lowest validation scaled MSE over fixed epochs; test never selects.",
              "evaluation": "Rolling one-step; prior actual test prices are observed, not recursive forecasts."}
    write_json(run / "config.json", config)
    write_json(run / "scaler.json", scaler)
    write_json(run / "data_manifest.json", manifest)
    print("SYNTHETIC educational prices: no real ticker, investment advice, or profit claims.")
    print("Target-index chronological split; scaler fitted ONLY to raw training prices.")
    for name, (_, _, targets) in splits.items():
        print(f"{name}: {len(targets)} targets, indices {targets[0]}..{targets[-1]}, "
              f"{dates[targets[0]]} .. {dates[targets[-1]]}")
    print("Validation/test use only already observed actual history: rolling ONE-step, not multi-day.")
    print(f"Run artifacts: {run}", flush=True)
    history = fit_model(splits["train"], splits["validation"], scaler, config, run)

    # The test inputs are first evaluated only after validation-based selection is finished.
    model, checkpoint = load_checkpoint(run / "best_model.pt")
    test_x, _, targets = splits["test"]
    with torch.no_grad():
        predicted = model(torch.from_numpy(test_x)).numpy().ravel() * scaler["scale"] + scaler["mean"]
    actual, baseline = prices[targets], persistence(prices, targets)
    metrics = {"synthetic": True, "best_epoch": checkpoint["epoch"],
               "best_validation_scaled_mse": checkpoint["validation_mse"],
               "test_count": len(targets), "test_start": dates[targets[0]],
               "test_end": dates[targets[-1]], "price_unit": "fictional price units",
               "mse_unit": "fictional price units squared", "lstm": scores(actual, predicted),
               "persistence": scores(actual, baseline),
               "caveat": "One frozen synthetic path; no evidence of real stock predictability.",
               "protocol": config["evaluation"]}
    metrics["outcome"] = ("LSTM has lower test MAE on this synthetic run."
                          if metrics["lstm"]["mae"] < metrics["persistence"]["mae"]
                          else "Persistence has equal or lower test MAE; keep this honest result.")
    write_json(run / "metrics.json", metrics)
    with (run / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_index", "date", "actual", "lstm", "persistence"])
        writer.writerows(zip(targets.tolist(), [dates[t] for t in targets],
                            actual.tolist(), predicted.tolist(), baseline.tolist()))
    from lesson_figures import draw_lesson
    draw_lesson(run, prices, dates, splits, boundaries, history, predicted, baseline, metrics)
    for image in (run / "pic").glob("*.png"):
        (ROOT / "pic").mkdir(exist_ok=True)
        shutil.copy2(image, ROOT / "pic" / image.name)
    write_json(ROOT / "pic" / "latest_run.json",
               {"run": str(run.relative_to(ROOT)), "data_sha256": manifest["sha256"],
                "metrics": metrics, "config": config})
    print(json.dumps(metrics, indent=2))
    print("Reload: predict_next(run / 'best_model.pt', last_25_observed_prices)")
    print("Do not tune on this test set; a strong persistence baseline can win.")


if __name__ == "__main__":
    main()
