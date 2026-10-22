from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from training import (
    CancelledTraining, Config, build_model, check_stop, parameter_count,
    ensure_dataset, run_training, split_indices, validate_config,
)


class TinyCIFAR:
    targets = np.repeat(np.arange(10), 5000).tolist()

    def __getitem__(self, index):
        label = self.targets[index]
        pixels = np.full((32, 32, 3), 20, dtype=np.uint8)
        pixels[2 + label:12 + label, :, label % 3] = 200
        return Image.fromarray(pixels), label


class TrainingTests(unittest.TestCase):
    def test_config_validation(self):
        validate_config(Config())
        for changes in (
            {"blocks": 0}, {"blocks": 5}, {"convs_per_block": 4},
            {"base_channels": 7}, {"base_channels": 32.0},
            {"learning_rate": 0}, {"learning_rate": True},
            {"learning_rate": float("nan")}, {"dropout": 1},
            {"batch_size": 0}, {"train_limit": 45001},
            {"val_limit": 5001}, {"device": "missing"},
            {"optimizer": "missing"}, {"seed": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(replace(Config(), **changes))

    def test_architecture_dimensions_and_growth(self):
        torch.set_num_threads(2)
        counts = []
        for blocks, width in ((1, 8), (2, 8), (2, 16), (4, 16)):
            config = Config(blocks=blocks, base_channels=width, convs_per_block=1)
            model = build_model(config).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 10))
            count = sum(parameter.numel() for parameter in model.parameters())
            self.assertEqual(count, parameter_count(config))
            counts.append(count)
        self.assertEqual(sorted(counts), counts)
        largest = Config(blocks=4, convs_per_block=3, base_channels=128)
        self.assertGreater(parameter_count(largest), 5_000_000)

    def test_preview_preserves_rng(self):
        before = torch.random.get_rng_state()
        parameter_count(Config())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_balanced_reproducible_disjoint_splits(self):
        targets = TinyCIFAR.targets
        train, val = split_indices(targets, 42, 1000, 200)
        self.assertEqual((len(train), len(val)), (1000, 200))
        self.assertFalse(set(train) & set(val))
        self.assertEqual((train, val), split_indices(targets, 42, 1000, 200))
        self.assertTrue(np.all(np.bincount(np.asarray(targets)[train]) == 100))
        self.assertTrue(np.all(np.bincount(np.asarray(targets)[val]) == 20))
        full_train, full_val = split_indices(targets, 42)
        self.assertEqual((len(full_train), len(full_val)), (45000, 5000))
        self.assertEqual(len(set(full_train) | set(full_val)), 50000)
        self.assertTrue(set(train) <= set(full_train))
        self.assertTrue(set(val) <= set(full_val))

    def test_training_events_artifacts_and_reload(self):
        config = Config(
            blocks=1, convs_per_block=1, base_channels=8, epochs=2,
            train_limit=40, val_limit=20, batch_size=8, cpu_threads=2,
        )
        for optimizer in ("AdamW", "SGD"):
            with self.subTest(optimizer=optimizer), tempfile.TemporaryDirectory() as directory:
                events = []
                with patch("training.ensure_dataset", return_value=TinyCIFAR()):
                    output = run_training(
                        replace(config, optimizer=optimizer), events.append,
                        threading.Event(), runs_dir=directory,
                    )
                self.assertEqual(events[-1]["type"], "done")
                self.assertEqual(events[-1]["epochs_completed"], 2)
                for phase in ("train", "val"):
                    metrics = [e for e in events if e["type"] == "metrics" and e["phase"] == phase]
                    self.assertGreaterEqual(len(metrics), 4)
                    self.assertEqual(metrics[-1]["epoch"], 2)
                    self.assertTrue(all(0 <= e["accuracy"] <= 1 for e in metrics))
                for filename in ("config.json", "history.csv", "split_indices.json", "best.pt", "last.pt"):
                    self.assertTrue((output / filename).is_file())
                self.assertEqual(len((output / "history.csv").read_text().splitlines()), 3)
                split = json.loads((output / "split_indices.json").read_text())
                self.assertFalse(set(split["train"]) & set(split["val"]))
                saved = torch.load(output / "last.pt", weights_only=True)
                self.assertEqual(saved["status"], "completed")
                self.assertFalse(saved["partial_epoch"])
                model = build_model(Config(**saved["config"]))
                model.load_state_dict(saved["model_state_dict"])
                self.assertEqual(tuple(model.eval()(torch.zeros(2, 3, 32, 32)).shape), (2, 10))

    def test_stop_before_loading_data(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(CancelledTraining):
            check_stop(stop)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            run_training(Config(), events.append, stop, runs_dir=directory, data_dir=Path(directory) / "data")
            self.assertEqual(events[-1]["type"], "stopped")
            self.assertEqual(events[-1]["epochs_completed"], 0)
            self.assertIsNone(events[-1]["best_accuracy"])

    def test_stop_mid_epoch(self):
        stop = threading.Event()
        events = []

        def emit(event):
            events.append(event)
            if event["type"] == "metrics" and event["phase"] == "train":
                stop.set()

        config = Config(blocks=1, base_channels=8, epochs=3, train_limit=40, val_limit=20, batch_size=8)
        with tempfile.TemporaryDirectory() as directory, patch("training.ensure_dataset", return_value=TinyCIFAR()):
            output = run_training(config, emit, stop, runs_dir=directory)
            self.assertEqual(events[-1]["type"], "stopped")
            self.assertNotIn("done", [event["type"] for event in events])
            state = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["epoch"], 0)
            self.assertTrue(state["partial_epoch"])

    def test_download_can_be_cancelled_and_cleans_partial_file(self):
        stop = threading.Event()
        response = MagicMock()
        response.headers = {"Content-Length": "500000"}

        def read(_):
            stop.set()
            return b"partial download"

        response.read.side_effect = read
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            with patch("training.datasets.CIFAR10") as dataset:
                dataset.side_effect = RuntimeError("Dataset not found or corrupted.")
                dataset.filename = "cifar-10-python.tar.gz"
                dataset.tgz_md5 = "irrelevant"
                dataset.url = "https://example.invalid/cifar.tar.gz"
                with patch("training.urllib.request.urlopen", return_value=response):
                    with self.assertRaises(CancelledTraining):
                        ensure_dataset(Path(directory), lambda event: None, stop)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_nonfinite_loss_is_not_reported_as_success(self):
        bad = TinyCIFAR()
        config = Config(blocks=1, base_channels=8, train_limit=20, val_limit=10, epochs=1)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("training.ensure_dataset", return_value=bad):
                with patch("training.nn.CrossEntropyLoss.forward", return_value=torch.tensor(float("nan"))):
                    with self.assertRaises(FloatingPointError):
                        run_training(config, events.append, threading.Event(), runs_dir=directory)
        self.assertNotIn("done", [event["type"] for event in events])


if __name__ == "__main__":
    unittest.main()
