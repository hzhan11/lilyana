import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
FIELDS = ("sample_id", "area_sqm", "rooms", "price_kusd", "split")


def expected_price(area_sqm, rooms):
    return 80 + 1.9 * np.asarray(area_sqm) + 25 * np.asarray(rooms) + 0.004 * (np.asarray(area_sqm) - 120) ** 2


def generate_data(output_dir=DATA_DIR, overwrite=False):
    output = Path(output_dir)
    names = ("houses.csv", "train.csv", "val.csv", "test.csv", "manifest.json")
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError("Dataset already exists. Use --overwrite only to replace generated data.")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    areas = rng.uniform(40, 240, 2000)
    rooms = np.clip(np.rint(areas / 38 + rng.normal(0, 0.9, len(areas))), 1, 7).astype(int)
    prices = expected_price(areas, rooms) + rng.normal(0, 15, len(areas))
    order = np.random.default_rng(43).permutation(len(areas))
    split = np.empty(len(areas), dtype=object)
    split[order[:1400]], split[order[1400:1700]], split[order[1700:]] = "train", "val", "test"
    records = [
        {
            "sample_id": f"H{index + 1:04d}", "area_sqm": f"{area:.3f}",
            "rooms": int(room), "price_kusd": f"{price:.3f}", "split": group,
        }
        for index, (area, room, price, group) in enumerate(zip(areas, rooms, prices, split))
    ]
    for name in ("houses", "train", "val", "test"):
        rows = records if name == "houses" else [row for row in records if row["split"] == name]
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "synthetic": True, "seed": 42, "split_seed": 43,
        "features": ["area_sqm", "rooms"], "target": "price_kusd",
        "target_units": "thousand USD (simulated; not real property valuations)",
        "formula": "80 + 1.9*area_sqm + 25*rooms + 0.004*(area_sqm-120)^2 + Normal(0,15)",
        "counts": {"train": 1400, "val": 300, "test": 300},
        "split": "Random, disjoint rows. Scaling is fitted only to the training split.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated 2,000 synthetic houses: 1,400 train / 300 val / 300 test at {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic area/rooms -> house price data.")
    parser.add_argument("--overwrite", action="store_true")
    generate_data(overwrite=parser.parse_args().overwrite)
