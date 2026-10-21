"""Run from this folder: python -m unittest discover -s . -p "test_*.py" -v"""

import csv
import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from generate_data import DEFAULT_DATA, ROOT, generate, synthetic_prices
from train_model import (PriceLSTM, fit_model, fit_scaler, load_checkpoint, load_data,
                         make_windows, persistence, predict_next, prepare, scores, valid_prices)


class LessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.scratch = ROOT / "runs" / ("test_" + uuid.uuid4().hex)
        cls.scratch.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch)

    def test_official_frozen_data_integrity(self):
        prices, dates, manifest = load_data(DEFAULT_DATA)
        self.assertEqual(len(prices), 1400)
        self.assertEqual(len(set(dates)), 1400)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(dates[0], "2019-01-02")
        np.testing.assert_array_equal(prices, np.round(synthetic_prices(1400, 42), 6))
        self.assertTrue((prices > 0).all())
        self.assertGreater(float(np.std(np.diff(prices))), 0.1)
        splits, _, boundaries = prepare(prices)
        self.assertEqual(boundaries, (980, 1190))
        self.assertEqual([len(splits[key][2]) for key in ("train", "validation", "test")],
                         [955, 210, 210])

    def test_generator_is_deterministic_and_refuses_overwrite(self):
        first = generate(self.scratch / "first.csv", days=220)
        second = generate(self.scratch / "second.csv", days=220)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        frozen = first.read_bytes()
        with self.assertRaises(FileExistsError):
            generate(first, days=230)
        self.assertEqual(first.read_bytes(), frozen)
        generate(first, days=230, overwrite=True)
        self.assertEqual(len(load_data(first)[0]), 230)
        first.write_bytes(first.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_data(first)

    def test_windows_are_strictly_past_only(self):
        prices = np.arange(1.0, 301.0)
        x, y, targets = make_windows(prices, 25)
        self.assertEqual(x.shape, (275, 25, 1))
        self.assertEqual(y.shape, (275, 1))
        for window, target_value, t in zip(x, y, targets):
            np.testing.assert_array_equal(window[:, 0], prices[t - 25:t])
            self.assertEqual(target_value[0], prices[t])
            self.assertEqual(window[-1, 0], prices[t - 1])

    def test_chronological_target_splits_and_train_only_scaler(self):
        prices = np.arange(1.0, 1001.0)
        splits, scaler, boundaries = prepare(prices)
        self.assertEqual(boundaries, (700, 850))
        for name, start, end in (("train", 25, 700), ("validation", 700, 850), ("test", 850, 1000)):
            np.testing.assert_array_equal(splits[name][2], np.arange(start, end))
        self.assertEqual(scaler, fit_scaler(prices[:700]))
        changed = prices.copy()
        changed[700:] *= 100
        altered, new_scaler, _ = prepare(changed)
        self.assertEqual(new_scaler, scaler)
        np.testing.assert_array_equal(splits["train"][0], altered["train"][0])
        np.testing.assert_array_equal(splits["train"][1], altered["train"][1])
        first_test = splits["test"][0][0, :, 0] * scaler["scale"] + scaler["mean"]
        np.testing.assert_allclose(first_test, prices[825:850], atol=0.0001)
        next_test = splits["test"][0][1, :, 0] * scaler["scale"] + scaler["mean"]
        self.assertAlmostEqual(float(next_test[-1]), float(prices[850]), places=3)

    def test_lstm_shapes_and_finite_output(self):
        model = PriceLSTM()
        result = model(torch.zeros(4, 25, 1))
        self.assertEqual(tuple(result.shape), (4, 1))
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.zeros(4, 25))
        with self.assertRaisesRegex(ValueError, "finite"):
            model(torch.full((1, 25, 1), float("nan")))

    def test_persistence_alignment_and_metrics(self):
        prices = np.array([10., 13., 12., 17., 15.])
        targets = np.array([2, 3, 4])
        result = persistence(prices, targets)
        np.testing.assert_array_equal(result, [13., 12., 17.])
        measured = scores(prices[targets], result)
        self.assertEqual(measured["mae"], 8 / 3)
        self.assertEqual(measured["mse"], 10)
        with self.assertRaises(ValueError):
            persistence(prices, np.array([0]))
        with self.assertRaises(ValueError):
            scores(prices, prices[:, None])

    def test_invalid_prices_windows_and_scalers_fail_explicitly(self):
        for values in ([], [[1, 2]], [1, float("nan")], [1, float("inf")], [1, 0], [1, -2]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    valid_prices(values)
        with self.assertRaises(ValueError):
            fit_scaler([2, 2, 2])
        for lookback in (0, -1, 5, 2.5):
            with self.assertRaises(ValueError):
                make_windows([1, 2, 3], lookback)
        with self.assertRaises(ValueError):
            prepare(np.arange(1., 101.))

    def test_small_training_loop_checkpoint_reload_and_next_price(self):
        torch.manual_seed(42)
        splits, scaler, _ = prepare(synthetic_prices(240))
        run = self.scratch / "small_training"
        run.mkdir()
        config = {"seed": 42, "epochs": 2, "batch_size": 64, "learning_rate": 0.001,
                  "lookback": 25}
        history = fit_model(splits["train"], splits["validation"], scaler, config, run)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite([row["train_mse"] for row in history]).all())
        model, checkpoint = load_checkpoint(run / "best_model.pt")
        expected_best = min(history, key=lambda row: row["validation_mse"])
        self.assertEqual(checkpoint["epoch"], expected_best["epoch"])
        self.assertEqual(checkpoint["validation_mse"], expected_best["validation_mse"])
        observed = synthetic_prices(240)[-25:]
        result = predict_next(run / "best_model.pt", observed)
        x = ((observed - scaler["mean"]) / scaler["scale"]).astype(np.float32)[None, :, None]
        with torch.no_grad():
            reloaded_result = model(torch.from_numpy(x)).item() * scaler["scale"] + scaler["mean"]
        self.assertAlmostEqual(result, reloaded_result, places=6)
        self.assertTrue(np.isfinite(result))
        with self.assertRaises(ValueError):
            predict_next(run / "best_model.pt", observed[:-1])

    def test_published_run_reproduces_saved_metrics(self):
        latest = ROOT / "pic" / "latest_run.json"
        if not latest.exists():
            self.skipTest("Run python train_model.py first to verify published results")
        published = json.loads(latest.read_text(encoding="utf-8"))
        run = ROOT / published["run"]
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        prices, dates, manifest = load_data(Path(config["data_path"]))
        self.assertEqual(config["data_sha256"], manifest["sha256"])
        splits, scaler, _ = prepare(prices, config["lookback"])
        model, checkpoint = load_checkpoint(run / "best_model.pt")
        self.assertEqual(checkpoint["scaler"], scaler)
        with torch.no_grad():
            predicted = model(torch.from_numpy(splits["test"][0])).numpy().ravel()
        predicted = predicted * scaler["scale"] + scaler["mean"]
        with (run / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        targets = splits["test"][2]
        np.testing.assert_array_equal([int(row["target_index"]) for row in rows], targets)
        self.assertEqual([row["date"] for row in rows], [dates[t] for t in targets])
        np.testing.assert_allclose([float(row["lstm"]) for row in rows], predicted, rtol=0, atol=1e-6)
        np.testing.assert_array_equal([float(row["actual"]) for row in rows], prices[targets])
        baseline = persistence(prices, targets)
        np.testing.assert_array_equal([float(row["persistence"]) for row in rows], baseline)
        self.assertEqual(scores(prices[targets], predicted), metrics["lstm"])
        self.assertEqual(scores(prices[targets], baseline), metrics["persistence"])
        with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda row: float(row["validation_mse"]))
        self.assertEqual(checkpoint["epoch"], int(best["epoch"]))
        self.assertEqual(metrics["best_epoch"], checkpoint["epoch"])
        self.assertEqual(metrics, published["metrics"])
        for name in ("01_hands_on.png", "02_data_and_windows.png", "03_training_loss.png",
                     "04_test_predictions.png", "05_metric_comparison.png"):
            self.assertEqual(hashlib.sha256((ROOT / "pic" / name).read_bytes()).hexdigest(),
                             hashlib.sha256((run / "pic" / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
