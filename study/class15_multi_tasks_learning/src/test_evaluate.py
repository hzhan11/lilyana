import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from evaluate import calculate_metrics, evaluate, load_model, main
from model import AnimalCNN
from pic.dataset import ANIMALS
from train import DATASET_CLASSES, ROOT


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for dataset in (1, 2, "m"):
            self.save_model(dataset)

    def save_model(self, dataset, classes=None):
        model = AnimalCNN(len(DATASET_CLASSES[dataset]))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        checkpoint = {
            "state_dict": model.state_dict(),
            "classes": list(DATASET_CLASSES[dataset]) if classes is None else classes,
            "dataset": dataset, "input_shape": [200, 200, 4], "center_scale": 199,
        }
        path = self.folder / f"{dataset}.model"
        torch.save(checkpoint, path)
        return path

    def test_exact_classification_and_center_metrics(self):
        metrics = calculate_metrics(
            ["cat", "cat", "dog", "pig"], ["cat", "dog", "dog", "dog"],
            [[0, 0]] * 4, [[3, 4], [0, 0], [0, 0], [0, 0]], ANIMALS[:3],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro"]["precision"], 4 / 9)
        self.assertAlmostEqual(metrics["macro"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["macro"]["f1-score"], 7 / 18)
        self.assertAlmostEqual(metrics["weighted"]["f1-score"], 11 / 24)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(metrics["per_class"]["pig"]["precision"], 0)
        self.assertEqual(metrics["center"]["mae_pixels"], 7 / 8)
        self.assertAlmostEqual(metrics["center"]["rmse_pixels"], math.sqrt(25 / 8))
        self.assertEqual(metrics["center"]["mean_distance_pixels"], 5 / 4)
        self.assertEqual(metrics["center"]["mae_x_pixels"], 3 / 4)
        self.assertEqual(metrics["center"]["mae_y_pixels"], 1)

    def test_data3_evaluation_counts_unseen_classes_and_saves_predictions(self):
        data_dir = ROOT / "pic" / "data3"
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in data_dir.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, unseen, accuracy, known_accuracy in (
                (1, 5, 0.1, 0.2), (2, 5, 0.2, 0.4), ("m", 0, 0.1, 0.1),
            ):
                report = evaluate(dataset, model_dir=self.folder, output_dir=self.folder)
                self.assertEqual(report["full_dataset"]["samples"], 10)
                self.assertEqual(report["known_classes_only"]["samples"], 10 - unseen)
                self.assertEqual(report["unseen_class_samples"], unseen)
                self.assertEqual(report["full_dataset"]["class_order"], list(ANIMALS))
                self.assertEqual(len(report["full_dataset"]["confusion_matrix"]), 6)
                self.assertEqual(sum(map(sum, report["full_dataset"]["confusion_matrix"])), 10)
                self.assertEqual(len(report["predictions"]), 10)
                self.assertEqual(report["full_dataset"]["accuracy"], accuracy)
                self.assertEqual(report["known_classes_only"]["accuracy"], known_accuracy)
                if dataset == "m":
                    self.assertEqual(report["full_dataset"], report["known_classes_only"])
                    self.assertTrue(all(row["known_class"] for row in report["predictions"]))
                for row in report["predictions"]:
                    self.assertEqual(row["predicted_center"], [99.5, 99.5])
                    self.assertEqual(row["predicted"], DATASET_CLASSES[dataset][0])
                    if not row["known_class"]:
                        self.assertNotEqual(row["actual"], row["predicted"])
                saved = json.loads((self.folder / f"{dataset}.metrics.json").read_text())
                self.assertEqual(saved, report)
        self.assertEqual(
            before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before},
        )

    def test_saved_class_order_is_used_for_decoding(self):
        classes = list(reversed(DATASET_CLASSES[1]))
        self.save_model(1, classes)
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluate(1, model_dir=self.folder, output_dir=self.folder)
        self.assertTrue(all(row["predicted"] == "pig" for row in report["predictions"]))

    def test_no_known_class_images_is_explicit_and_not_a_fake_zero_score(self):
        folder = self.folder / "unseen"
        folder.mkdir()
        Image.new("RGB", (200, 200), "white").save(folder / "fox.png")
        (folder / "label.txt").write_text("fox.png,fox,[80,90]\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report = evaluate(1, model_dir=self.folder, data_dir=folder, output_dir=self.folder)
        self.assertIsNone(report["known_classes_only"])
        self.assertEqual(report["full_dataset"]["accuracy"], 0)
        self.assertEqual(report["unseen_class_samples"], 1)
        self.assertIn("N/A", output.getvalue())

    def test_missing_corrupt_wrong_metadata_and_wrong_weights_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "training 1"):
            load_model(self.folder / "missing.model", 1)
        with self.assertRaisesRegex(ValueError, "mtrain"):
            load_model(self.folder / "missing.model", "m")
        path = self.folder / "bad.model"
        path.write_bytes(b"corrupt checkpoint")
        with self.assertRaisesRegex(ValueError, "Cannot load checkpoint"):
            load_model(path, 1)
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_model(self.folder / "2.model", 1)
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_model(self.folder / "1.model", "m")
        checkpoint = torch.load(self.folder / "1.model", weights_only=True)
        checkpoint["state_dict"] = AnimalCNN(6).state_dict()
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "incompatible model weights"):
            load_model(path, 1)

    def test_metrics_reject_invalid_inputs(self):
        for actual, predicted, centers in (
            ([], [], []), (["cat"], [], [[0, 0]]),
            (["cat"], ["cat"], [[float("nan"), 0]]),
            (["bird"], ["cat"], [[0, 0]]),
        ):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                calculate_metrics(actual, predicted, centers, centers, ANIMALS)

    def test_cli_defaults_to_both_models_and_supports_selection(self):
        for args, expected in (([], [1, 2]), (["1"], [1]), (["2"], [2]), (["m"], ["m"])):
            with self.subTest(args=args), patch("evaluate.evaluate") as run:
                self.assertEqual(main(args), 0)
                self.assertEqual([call.args[0] for call in run.call_args_list], expected)
        for args in (["3"], ["1", "--threads", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
