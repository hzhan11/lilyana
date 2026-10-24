"""Render headless comparison charts from fresh model evaluation reports."""

import json
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


MODELS = ("1", "2", "m")
COLORS = ("#2563eb", "#ea580c", "#16a34a")


def grouped_bars(axis, labels, rows, legends, *, percentage):
    width = 0.24
    for index, (values, legend, color) in enumerate(zip(rows, legends, COLORS)):
        positions = [position + (index - 1) * width for position in range(len(labels))]
        bars = axis.bar(positions, values, width, label=legend, color=color)
        annotations = [
            "" if not math.isfinite(value) else
            f"{value:.0%}" if percentage else f"{value:.1f}"
            for value in values
        ]
        axis.bar_label(bars, labels=annotations, padding=3, fontsize=8)
        for position, value in zip(positions, values):
            if not math.isfinite(value):
                axis.text(position, 0.02, "N/A", ha="center", fontsize=8)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    if percentage:
        axis.set_ylim(0, 1.17)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        axis.set_ylim(bottom=0)
        axis.margins(y=0.22)
        axis.set_ylabel("Pixels")


def classification_values(metrics):
    if metrics is None:
        return [float("nan")] * 4
    return [metrics["accuracy"], metrics["macro"]["precision"],
            metrics["macro"]["recall"], metrics["macro"]["f1-score"]]


def write_comparison(reports, output_dir):
    if set(reports) != set(MODELS):
        raise ValueError("Comparison requires reports for models 1, 2, and m.")
    actual = [
        [(row["filename"], row["actual"], row["actual_center"]) for row in reports[name]["predictions"]]
        for name in MODELS
    ]
    if not actual[0] or any(rows != actual[0] for rows in actual[1:]):
        raise ValueError("Comparison reports must evaluate exactly the same images and labels.")
    ordered = [reports[name] for name in MODELS]
    classes = ordered[0]["full_dataset"]["class_order"]
    if any(report["full_dataset"]["class_order"] != classes for report in ordered):
        raise ValueError("Comparison reports must use the same class order.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(15, 11), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    legends = [f"Model {name} (n={report['full_dataset']['samples']})"
               for name, report in zip(MODELS, ordered)]
    score_labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
    grouped_bars(
        axes[0, 0], score_labels,
        [classification_values(report["full_dataset"]) for report in ordered],
        legends, percentage=True,
    )
    axes[0, 0].set_title("All images, six-class scores - higher is better")
    known_legends = [
        f"Model {name} (n={report['known_classes_only']['samples'] if report['known_classes_only'] else 0})"
        for name, report in zip(MODELS, ordered)
    ]
    grouped_bars(
        axes[0, 1], score_labels,
        [classification_values(report["known_classes_only"]) for report in ordered],
        known_legends, percentage=True,
    )
    axes[0, 1].set_title("Known classes only - DIFFERENT subsets and class counts")
    center_keys = ("mae_pixels", "rmse_pixels", "mean_distance_pixels")
    grouped_bars(
        axes[1, 0], ["Center MAE", "Center RMSE", "Mean distance"],
        [[report["full_dataset"]["center"][key] for key in center_keys] for report in ordered],
        legends, percentage=False,
    )
    axes[1, 0].set_title("Center errors on all images - lower is better")
    grouped_bars(
        axes[1, 1], classes,
        [[report["full_dataset"]["per_class"][animal]["f1-score"] for animal in classes]
         for report in ordered],
        legends, percentage=True,
    )
    axes[1, 1].set_title("Per-class F1 on all images - higher is better")
    figure.suptitle(
        "Models 1 vs 2 vs m: test-set comparison\n"
        "Models 1/2 know three classes each; m knows all six. Unseen classes count as incorrect.",
        fontsize=14,
    )
    dashboard = destination / "comparison.png"
    figure.savefig(dashboard, dpi=150)
    figure.clear()

    figure = Figure(figsize=(16, 5.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    matrices = [report["full_dataset"]["confusion_matrix"] for report in ordered]
    maximum = max(max(row) for matrix in matrices for row in matrix)
    for axis, name, matrix in zip(axes, MODELS, matrices):
        heatmap = axis.imshow(matrix, vmin=0, vmax=max(1, maximum), cmap="Blues")
        axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Predicted animal")
        axis.set_ylabel("Actual animal")
        axis.set_title(f"Model {name}")
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axis.text(column, row, str(value), ha="center", va="center",
                          color="white" if value > maximum / 2 else "black")
    figure.colorbar(heatmap, ax=list(axes), label="Image count", shrink=0.8)
    figure.suptitle("Confusion matrices: same test images, same class order and color scale")
    confusion = destination / "comparison_confusion.png"
    figure.savefig(confusion, dpi=150)
    figure.clear()
    snapshot = destination / "comparison.metrics.json"
    snapshot.write_text(
        json.dumps({"models": reports}, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return dashboard, confusion, snapshot
