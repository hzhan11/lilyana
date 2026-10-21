"""Freeze a reproducible, entirely synthetic stock-like teaching dataset."""

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "synthetic_prices.csv"


def synthetic_prices(days=1400, seed=42):
    if not isinstance(days, int) or days < 200:
        raise ValueError("days must be an integer of at least 200")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 0.006, days)
    returns = np.zeros(days)
    for t in range(1, days):
        returns[t] = 0.55 * returns[t - 1] + innovations[t]
    t = np.arange(days)
    log_price = (np.log(100.0) + 0.0003 * t
                 + 0.08 * np.sin(t / 95.0) + 0.04 * np.sin(t / 23.0)
                 + np.cumsum(returns))
    prices = np.exp(log_price)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Generation produced invalid prices")
    return prices


def generate(output=DEFAULT_DATA, days=1400, seed=42, overwrite=False):
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Frozen dataset exists; use --overwrite explicitly to regenerate.")
    prices = synthetic_prices(days, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    current = date(2019, 1, 2)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["day_index", "date", "synthetic_price"])
        for index, price in enumerate(prices):
            while current.weekday() >= 5:
                current += timedelta(days=1)
            writer.writerow([index, current.isoformat(), f"{price:.6f}"])
            current += timedelta(days=1)
    manifest = {
        "synthetic": True, "seed": seed, "rows": days,
        "calendar": "Weekdays only; exchange holidays are not modeled.",
        "start_date": "2019-01-02", "price_unit": "fictional price units",
        "parameters": {"start_price": 100, "daily_log_drift": 0.0003,
                       "ar_coefficient": 0.55, "innovation_std": 0.006,
                       "cycle_amplitudes": [0.08, 0.04],
                       "cycle_divisors_in_days": [95, 23]},
        "formula": ("r[0]=0; r[t]=0.55*r[t-1]+Normal(0,0.006); "
                    "price[t]=exp(log(100)+0.0003*t+0.08*sin(t/95)"
                    "+0.04*sin(t/23)+cumsum(r)[t])"),
        "rng": "numpy.random.default_rng(seed), PCG64; draw days innovations, discard index 0",
        "storage": "CSV prices rounded to six decimals; this CSV is the training source.",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy": "Frozen before training; do not regenerate or tune after seeing test results.",
        "warning": "Synthetic educational data, not a real ticker or investment forecast.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {days} SYNTHETIC weekday prices: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--days", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace data; never use to improve test results.")
    args = parser.parse_args()
    generate(args.output, args.days, args.seed, args.overwrite)
