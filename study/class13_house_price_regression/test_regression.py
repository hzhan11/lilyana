from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generate_data import generate_data
from predict import predict_house
import train_model as lesson


class RegressionTests(unittest.TestCase):
    def test_dataset_splits_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            first, second = root / "first", root / "second"
            generate_data(first)
            generate_data(second)
            all_ids = set()
            for split, count in (("train", 1400), ("val", 300), ("test", 300)):
                x, y, ids = lesson.load_data(first / f"{split}.csv")
                self.assertEqual(x.shape, (count, 2))
                self.assertEqual(y.shape, (count,))
                self.assertFalse(all_ids & set(ids))
                all_ids.update(ids)
                self.assertTrue((first / f"{split}.csv").read_bytes() == (second / f"{split}.csv").read_bytes())
            self.assertEqual(len(all_ids), 2000)
            with self.assertRaises(FileExistsError):
                generate_data(first)

    def test_regression_metrics_and_output(self):
        metrics = lesson.regression_metrics([1, 2, 3], [2, 2, 1])
        self.assertEqual(metrics["mae"], 1)
        self.assertAlmostEqual(metrics["mse"], 5 / 3)
        self.assertEqual(tuple(lesson.create_model()(torch.zeros(4, 2)).shape), (4, 1))
        with self.assertRaises(ValueError):
            lesson.regression_metrics([1], [[1]])
        with self.assertRaises(FloatingPointError):
            lesson.regression_metrics([float("nan")], [1])

    def test_invalid_parameters(self):
        for kwargs in ({"epochs": 0}, {"learning_rate": 0}, {"learning_rate": True}, {"learning_rate": float("nan")}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lesson.run_training(**kwargs)
        with self.assertRaises(ValueError):
            predict_house(-5, 3, "not-read.pt")

    def test_train_only_scaling_test_holdout_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = generate_data(root / "data")
            output = root / "run"
            original = lesson.load_data
            loaded = []

            def load(path):
                loaded.append(Path(path).name)
                if Path(path).name == "test.csv":
                    self.assertTrue((output / "weights.pt").is_file())
                return original(path)

            with patch.object(lesson, "load_data", side_effect=load):
                lesson.run_training(epochs=3, data_dir=data, output_dir=output, publish=False)
            self.assertEqual(loaded, ["train.csv", "val.csv", "test.csv"])
            saved = torch.load(output / "weights.pt", weights_only=True)
            x, y, _ = original(data / "train.csv")
            np.testing.assert_allclose(saved["scaling"]["x_mean"], x.mean(0))
            np.testing.assert_allclose(saved["scaling"]["x_scale"], x.std(0))
            self.assertAlmostEqual(saved["scaling"]["y_mean"], float(y.mean()))
            with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            best_epoch = min(history, key=lambda row: float(row["val_mse"]))["epoch"]
            self.assertEqual(saved["best_epoch"], int(best_epoch))
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"train": 1400, "val": 300, "test": 300})
            price, outside = predict_house(120, 3, output / "weights.pt")
            self.assertTrue(np.isfinite(price))
            self.assertFalse(outside)
            for name in ("dataset.png", "training_curves.png", "prediction_curves.png", "metrics_comparison.png"):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
