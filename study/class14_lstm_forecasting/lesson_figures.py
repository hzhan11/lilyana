"""English, headless lesson pages. No GUI backend or pyplot state."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

COLORS = {"train": "#2563eb", "validation": "#d97706", "test": "#15803d"}


def page(title, subtitle):
    figure = Figure(figsize=(14, 9), dpi=140, facecolor="white")
    figure.suptitle(title, fontsize=22, fontweight="bold", x=0.07, y=0.97, ha="left")
    figure.text(0.07, 0.91, subtitle, fontsize=11, color="#475569")
    figure.text(0.07, 0.025,
                "SYNTHETIC DATA ONLY  |  Educational experiment, not investment advice or a real-stock forecast.",
                fontsize=10, color="#475569")
    return figure


def save(figure, folder, name):
    figure.savefig(folder / name, dpi=140, facecolor="white")
    figure.clear()


def draw_lesson(run, prices, dates, splits, boundaries, history, predicted, baseline, metrics):
    folder = Path(run) / "pic"
    folder.mkdir()
    train_end, val_end = boundaries
    lookback = splits["train"][0].shape[1]
    test_count = len(splits["test"][2])
    figure = page("Class 14 | A first LSTM forecasting experiment",
                  "Goal: use the last 25 observed synthetic prices to estimate exactly one next price.")
    figure.text(0.08, 0.81, "1  Prepare a frozen time series", fontsize=17, fontweight="bold")
    figure.text(0.10, 0.75, f"{len(prices):,} weekday rows | seeded noisy trend + cycles + autocorrelated random returns\n"
                "CSV stays readable. Its generation formula, seed and checksum are saved in data/.", fontsize=13)
    figure.text(0.08, 0.64, "2  Learn a small sequence model", fontsize=17, fontweight="bold")
    figure.text(0.10, 0.57, "[64, 25, 1] batch of windows  ->  LSTM(1, 32)  ->  last hidden output\n"
                "->  Linear(32, 1)  ->  [64, 1] next-price estimates (last batch can be smaller)", fontsize=13)
    figure.text(0.08, 0.46, "3  Run locally from this lesson folder", fontsize=17, fontweight="bold")
    figure.text(0.10, 0.36,
                "python -m pip install -r requirements.txt\n"
                "python train_model.py\n"
                "python -m unittest discover -s . -p \"test_*.py\" -v",
                fontsize=13, family="monospace", linespacing=1.7)
    figure.text(0.10, 0.23, "The supplied CSV is already frozen. To create a fresh copy elsewhere:\n"
                "python generate_data.py --output data\\new_prices.csv\n"
                "Existing files are refused unless --overwrite is explicitly supplied.", fontsize=12)
    figure.text(0.08, 0.10, "No downloads. No pretrained weights. CPU only. No promises of profit.\n"
                "Do not change data or hyperparameters after inspecting this test result.", fontsize=13,
                color="#9a3412")
    save(figure, folder, "01_hands_on.png")

    figure = page("Data first | Respect the arrow of time",
                  "70% / 15% / 15% of raw rows; each window belongs to the split containing its TARGET day.")
    ax = figure.add_axes([0.08, 0.47, 0.86, 0.38])
    ax.plot(np.arange(len(prices)), prices, color="#334155", linewidth=1)
    ends = [0, *boundaries, len(prices)]
    for i, name in enumerate(("train", "validation", "test")):
        ax.axvspan(ends[i], ends[i + 1] - 1, alpha=0.12, color=COLORS[name],
                   label=f"{name}: raw indices {ends[i]}-{ends[i+1]-1}")
    ax.set(xlabel="Chronological weekday index", ylabel="Synthetic price (fictional units)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.15)
    figure.text(0.08, 0.36, f"Window at the first test target (index {val_end}):",
                fontsize=15, fontweight="bold")
    figure.text(0.10, 0.29, f"Observed indices {val_end-lookback} ... {val_end-1}"
                f"  ->  predict index {val_end}\n"
                f"Then observe actual {val_end}; indices {val_end-lookback+1} ... {val_end}"
                f"  ->  predict {val_end+1}.",
                fontsize=13, family="monospace")
    figure.text(0.08, 0.18, f"Train targets: {lookback}-{train_end-1} ({train_end-lookback})"
                f"  |  Validation: {train_end}-{val_end-1} ({val_end-train_end})"
                f"  |  Test: {val_end}-{len(prices)-1} ({test_count})\n"
                f"Mean and standard deviation are fitted on raw training indices 0-{train_end-1} ONLY.\n"
                "Past split context is allowed. Future targets never appear in their own input windows.",
                fontsize=13)
    figure.text(0.08, 0.085, f"Dates: {dates[0]} to {dates[-1]} | Weekends skipped; exchange holidays not modeled.",
                fontsize=11, color="#475569")
    save(figure, folder, "02_data_and_windows.png")

    figure = page("Training | Select with validation, not test",
                  "Adam, learning rate 0.001, batch size 64 | 1 layer, 32 hidden units | CPU")
    ax = figure.add_axes([0.09, 0.25, 0.84, 0.59])
    for key, label, color in (("train_mse", "Training", COLORS["train"]),
                              ("validation_mse", "Validation", COLORS["validation"])):
        ax.plot([row["epoch"] for row in history], [row[key] for row in history],
                label=label, color=color)
    ax.axvline(metrics["best_epoch"], color="#64748b", linestyle="--",
               label=f"Selected epoch {metrics['best_epoch']}")
    ax.set(xlabel="Epoch", ylabel="MSE (standardized price units squared)", yscale="log")
    ax.grid(alpha=0.2)
    ax.legend()
    figure.text(0.09, 0.12, "Save the checkpoint with the lowest validation MSE across the fixed training schedule.\n"
                "Only after selection ends: reload that checkpoint and evaluate the untouched test targets.\n"
                "The scaler is frozen; shuffling training windows never shuffles chronological split membership.",
                fontsize=12)
    save(figure, folder, "03_training_loss.png")

    figure = page("Test | Rolling one-step estimates",
                  "Every prediction uses only previously observed actual prices. This is NOT a recursive multi-day forecast.")
    targets = splits["test"][2]
    for rect, zoom in (([0.09, 0.49, 0.84, 0.35], False), ([0.09, 0.16, 0.84, 0.24], True)):
        ax = figure.add_axes(rect)
        selection = slice(0, 50) if zoom else slice(None)
        ax.plot(targets[selection], prices[targets][selection], color="#111827", label="Actual", linewidth=1.7)
        ax.plot(targets[selection], predicted[selection], color="#2563eb", label="LSTM", linewidth=1.2)
        ax.plot(targets[selection], baseline[selection], color="#d97706", linestyle="--",
                label="Persistence: tomorrow = today", linewidth=1.2)
        ax.set(xlabel="Target weekday index", ylabel="Price (fictional units)")
        ax.set_title("Zoom: first 50 test targets" if zoom else
                     f"All {len(targets)} identical test dates: {metrics['test_start']} to {metrics['test_end']}",
                     fontsize=12)
        ax.grid(alpha=0.2)
        if not zoom:
            ax.legend(fontsize=10)
    figure.text(0.09, 0.07, "The dashed baseline shifts actual prices by ONE day; it is not today's target value.",
                fontsize=12)
    save(figure, folder, "04_test_predictions.png")

    figure = page("Evaluation | A strong simple baseline can win",
                  f"Lower is better. Both models use exactly the same {test_count} test target indices and dates.")
    for left, key, unit in ((0.09, "mae", "fictional price units"),
                             (0.55, "mse", "fictional price units squared")):
        ax = figure.add_axes([left, 0.33, 0.36, 0.48])
        values = [metrics["lstm"][key], metrics["persistence"][key]]
        bars = ax.bar(["LSTM", "Persistence"], values, color=["#2563eb", "#d97706"], width=0.58)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=12)
        ax.set(title=key.upper(), ylabel=unit, ylim=(0, max(values) * 1.25))
        ax.grid(axis="y", alpha=0.15)
    figure.text(0.09, 0.22, metrics["outcome"], fontsize=15, fontweight="bold")
    figure.text(0.09, 0.12, "MAE = average absolute error. MSE = average squared error (penalizes large misses).\n"
                "A complicated model is not automatically better. Keep the measured outcome, even when it loses.\n"
                "One synthetic path cannot establish real-market predictability. No test-based retuning.",
                fontsize=12)
    save(figure, folder, "05_metric_comparison.png")
