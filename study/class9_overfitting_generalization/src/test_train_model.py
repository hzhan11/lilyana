from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

import train_model as lesson


class ExperimentTests(unittest.TestCase):
    def test_model_capacity_and_shapes(self):
        torch.set_num_threads(2)
        model = lesson.create_model()
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(count, 1_000_000)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 128, 128)).shape), (2, 2))

    def test_evaluation_metrics_are_measured(self):
        class ConstantDog(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([[0.0, 2.0]]).repeat(len(images), 1)

        metrics, probabilities = lesson.evaluate(
            ConstantDog(), torch.zeros(4, 3, 128, 128), torch.tensor([0, 0, 1, 1]),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.25)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 1 / 3)
        np.testing.assert_allclose(probabilities.sum(1), 1)

    def test_invalid_inputs_are_rejected(self):
        for kwargs in ({"epochs": 0}, {"learning_rate": 0}, {"learning_rate": float("nan")}, {"seed": -1}, {"threads": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                lesson.run_experiment(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(FileNotFoundError):
                lesson.load_data(folder)
            Image.new("RGB", (32, 32)).save(folder / "cat_01.png")
            with self.assertRaises(ValueError):
                lesson.load_data(folder)

    def test_test_split_is_loaded_only_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "pic"
            output = root / "run"
            for split in ("train", "val", "test"):
                (data / split).mkdir(parents=True)
                for label, color in (("cat", "#cc6688"), ("dog", "#555566")):
                    image = Image.new("RGB", (128, 128), "white")
                    ImageDraw.Draw(image).rectangle((20, 20, 90, 110), fill=color)
                    image.save(data / split / f"{label}_01.png")
            (root / "dataset_manifest.json").write_text("{}", encoding="utf-8")
            original_loader = lesson.load_data
            loaded = []

            def load(folder):
                loaded.append(folder.name)
                if folder.name == "test":
                    self.assertTrue((output / "weights.pt").is_file())
                    with (output / "history.csv").open(encoding="utf-8") as stream:
                        self.assertEqual(len(stream.read().splitlines()), 3)
                return original_loader(folder)

            with patch.object(lesson, "CLASS9", root), patch.object(lesson, "DATA_DIR", data):
                with patch.object(lesson, "load_data", side_effect=load), redirect_stdout(io.StringIO()):
                    lesson.run_experiment(epochs=2, threads=2, output_dir=output, publish=False)
            self.assertEqual(loaded, ["train", "val", "test"])
            for filename in (
                "config.json", "history.csv", "weights.pt", "metrics.json",
                "test_predictions.csv", "training_curves.png", "metrics_comparison.png", "test_predictions.png",
            ):
                self.assertTrue((output / filename).is_file())
            saved = torch.load(output / "weights.pt", weights_only=True)
            model = lesson.create_model()
            model.load_state_dict(saved["model_state_dict"])
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["metrics"]), {"train", "val", "test"})
            self.assertEqual(report["counts"], {"train": 2, "val": 2, "test": 2})


if __name__ == "__main__":
    unittest.main()
