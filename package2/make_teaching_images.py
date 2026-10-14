"""Export labeled training examples for the English lesson, not clinical use."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_breast_cancer import load_splits


def main():
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file", type=Path,
        default=package_dir / "data" / "breastmnist_64.npz",
    )
    args = parser.parse_args()
    images, labels = load_splits(args.data_file)["train"]
    labels = labels.reshape(-1)
    selected = [
        int(index)
        for label in (0, 1)
        for index in np.flatnonzero(labels == label)[:4]
    ]
    if len(selected) != 8:
        raise ValueError("The training split needs at least four examples per class.")

    output_dir = package_dir / "teaching_assets"
    output_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 21)
    small_font = ImageFont.truetype("arial.ttf", 17)
    canvas = Image.new("RGB", (1280, 820), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (22, 16), "Real BreastMNIST64 TRAIN images with reference labels",
        fill="#0f172a", font=font,
    )
    records = []
    for position, index in enumerate(selected):
        label = int(labels[index])
        left = 20 + (position % 4) * 320
        top = 65 + (position // 4) * 350
        example = Image.fromarray(images[index])
        displayed = example.resize((280, 280), Image.Resampling.NEAREST)
        canvas.paste(displayed, (left, top))
        title = "0: normal / benign" if label == 0 else "1: malignant"
        draw.text((left, top + 287), title, fill="#0f172a", font=font)
        draw.text(
            (left, top + 315), f"Official train index {index}",
            fill="#475569", font=small_font,
        )
        records.append(
            {"split": "train", "index": index, "label": label, "official_label": 1 - label}
        )

    draw.text(
        (22, 788),
        "Source: MedMNIST / BUSI, CC BY 4.0. Enlarged for display; "
        "labels are references, NOT predictions.",
        fill="#475569", font=small_font,
    )
    canvas.save(output_dir / "dataset_examples.png")
    example_index = next(index for index in selected if labels[index] == 1)
    Image.fromarray(images[example_index]).save(package_dir / "example_ultrasound.png")
    manifest = {
        "dataset": "BreastMNIST64",
        "mapping": {"0": "normal_or_benign", "1": "malignant"},
        "label_transform": "package_label = 1 - official_label",
        "illustration_examples": records,
        "example_ultrasound": {
            "split": "train", "index": example_index, "label": 1,
            "independent_test": False,
        },
        "changes": "Nearest-neighbor enlargement for the contact sheet and text headings.",
    }
    (output_dir / "image_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Teaching images and provenance saved in {output_dir}")
    print(f"example_ultrasound.png is TRAIN index {example_index}; not an independent test.")


if __name__ == "__main__":
    main()
