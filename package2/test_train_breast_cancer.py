"""Offline unit tests: no downloads, dataset files, or training experiments."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import train_breast_cancer as lesson


class DataTests(unittest.TestCase):
    def test_official_labels_reversed_once(self):
        official = np.array([[0], [1], [1], [0]], dtype=np.uint8)
        mapped = lesson.remap_labels(official)
        np.testing.assert_array_equal(mapped, [1, 0, 0, 1])
        self.assertEqual(mapped.dtype, np.int64)
        dataset = lesson.BreastDataset(np.zeros((4, 64, 64), np.uint8), mapped)
        self.assertEqual(dataset[0][1].item(), 1)
        self.assertEqual(dataset[1][1].item(), 0)
        np.testing.assert_array_equal(official[:, 0], [0, 1, 1, 0])

    def test_invalid_official_labels(self):
        for labels in (
            np.array([[2]]), np.array([[-1]]), np.array([[0.5]]),
            np.array([[np.nan]]), np.array([[np.inf]]), np.array([[True]]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                lesson.remap_labels(labels)

    def test_valid_shapes_and_all_splits(self):
        for split, count in lesson.SPLIT_COUNTS.items():
            images = np.zeros((count, 64, 64), np.uint8)
            labels = np.zeros((count, 1), np.uint8)
            actual_images, actual_labels = lesson.validate_split(images, labels, split)
            self.assertIs(actual_images, images)
            np.testing.assert_array_equal(actual_labels, np.ones(count, np.int64))

    def test_invalid_shapes_dtypes_values(self):
        images = np.zeros((78, 64, 64), np.uint8)
        labels = np.zeros((78, 1), np.uint8)
        for invalid_images, invalid_labels in (
            (images[:77], labels), (images[:, :32], labels),
            (images[..., np.newaxis], labels), (images.astype(np.float32), labels),
            (images.astype(np.int16) - 1, labels),
            (np.full(images.shape, np.nan), labels),
            (images, labels[:, 0]), (images, np.full(labels.shape, 2)),
            (images, labels.astype(np.float32)),
        ):
            with self.subTest(shape=invalid_images.shape), self.assertRaises(ValueError):
                lesson.validate_split(invalid_images, invalid_labels, "val")

    def test_load_splits_remaps_without_resplitting(self):
        archive = {}
        for split, count in lesson.SPLIT_COUNTS.items():
            archive[f"{split}_images"] = np.zeros((count, 64, 64), np.uint8)
            archive[f"{split}_labels"] = np.ones((count, 1), np.uint8)
        with patch.object(lesson, "verify_dataset") as verify:
            with patch.object(lesson.np, "load") as load:
                load.return_value.__enter__.return_value = archive
                splits = lesson.load_splits(Path("unused.npz"))
        verify.assert_called_once_with(Path("unused.npz"))
        load.assert_called_once_with(Path("unused.npz"), allow_pickle=False)
        for split, count in lesson.SPLIT_COUNTS.items():
            self.assertEqual(len(splits[split][0]), count)
            self.assertTrue((splits[split][1] == 0).all())

    def test_load_splits_requires_all_keys(self):
        with patch.object(lesson, "verify_dataset"):
            with patch.object(lesson.np, "load") as load:
                load.return_value.__enter__.return_value = {}
                with self.assertRaises(ValueError):
                    lesson.load_splits(Path("unused.npz"))

    def test_checksum_rejection(self):
        with patch.object(lesson, "file_md5", return_value="incorrect"):
            with self.assertRaisesRegex(ValueError, "checksum"):
                lesson.verify_dataset("unused.npz")

    def test_missing_dataset_requires_explicit_download(self):
        with patch.object(Path, "exists", return_value=False):
            with patch.object(lesson.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(FileNotFoundError, "--download"):
                    lesson.ensure_dataset(Path("missing.npz"))
                network.assert_not_called()

    def test_download_verifies_before_atomic_replace_and_cleans_failure(self):
        destination = lesson.PACKAGE_DIR / "data" / "breastmnist_64.npz"
        partial = destination.with_name("breastmnist_64.npz.unit.partial")
        for failure in (None, ValueError("checksum mismatch")):
            with self.subTest(failure=failure), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(Path, "exists", side_effect=[False, True]))
                stack.enter_context(patch.object(Path, "mkdir"))
                unlink = stack.enter_context(patch.object(Path, "unlink"))
                make_partial = stack.enter_context(
                    patch.object(lesson.tempfile, "mkstemp", return_value=(123, str(partial)))
                )
                stack.enter_context(patch.object(lesson.os, "fdopen"))
                source = stack.enter_context(patch.object(lesson.urllib.request, "urlopen"))
                source.return_value.__enter__.return_value = io.BytesIO(b"offline sample")
                verify = stack.enter_context(
                    patch.object(lesson, "verify_dataset", side_effect=failure)
                )
                replace = stack.enter_context(patch.object(lesson.os, "replace"))
                if failure:
                    with self.assertRaises(ValueError):
                        lesson.ensure_dataset(destination, download=True)
                    replace.assert_not_called()
                else:
                    replace.side_effect = lambda *args: verify.assert_called_once_with(partial)
                    self.assertEqual(lesson.ensure_dataset(destination, download=True),
                                     destination)
                    replace.assert_called_once_with(partial, destination)
                self.assertEqual(make_partial.call_args.kwargs["dir"], destination.parent)
                verify.assert_called_once_with(partial)
                unlink.assert_called_once()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_uint8_divided_by_255_and_channel(self):
        image = np.zeros((64, 64), np.uint8)
        image[0, 0], image[0, 1] = 255, 128
        normalized = lesson.normalize_image(image)
        self.assertEqual(tuple(normalized.shape), (1, 64, 64))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(normalized[0, 0, 0].item(), 1)
        self.assertAlmostEqual(normalized[0, 0, 1].item(), 128 / 255, places=6)
        self.assertEqual(normalized[0, 1, 0].item(), 0)
        with self.assertRaises(ValueError):
            lesson.normalize_image(image.astype(np.float32))
        with self.assertRaises(ValueError):
            lesson.normalize_image(image[:32])

    def test_logits_shape_including_single_image(self):
        model = lesson.SimpleCNN()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 9505)
        with torch.no_grad():
            for batch_size in (1, 3):
                logits = model(torch.zeros(batch_size, 1, 64, 64))
                self.assertEqual(tuple(logits.shape), (batch_size,))
                self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(any(isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.Dropout))
                             for layer in model.modules()))

    def test_evaluation_pools_images_including_short_last_batch(self):
        class ConstantLogit(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(images.shape[0], device=images.device)

        dataset = lesson.BreastDataset(
            np.zeros((3, 64, 64), np.uint8), np.array([1, 0, 1])
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        metrics, labels, scores = lesson.run_epoch(
            ConstantLogit(), loader, torch.device("cpu"), 0.5
        )
        self.assertAlmostEqual(metrics["loss"], np.log(2), places=6)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        np.testing.assert_array_equal(labels, [1, 0, 1])
        np.testing.assert_array_equal(scores, [0.5, 0.5, 0.5])


class MetricTests(unittest.TestCase):
    def test_known_counts_and_metrics(self):
        result = lesson.binary_metrics(
            [1, 1, 1, 0, 0, 0], [0.9, 0.8, 0.1, 0.7, 0.2, 0.3]
        )
        self.assertEqual({key: result[key] for key in ("tp", "fp", "fn", "tn", "total")},
                         {"tp": 2, "fp": 1, "fn": 1, "tn": 2, "total": 6})
        for name in ("accuracy", "precision", "recall", "specificity", "f1"):
            self.assertAlmostEqual(result[name], 2 / 3)

    def test_undefined_is_none_not_invented_zero(self):
        result = lesson.binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])
        self.assertEqual(result["specificity"], 1)
        result = lesson.binary_metrics([1, 1], [0.1, 0.2])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["specificity"])
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)
        result = lesson.binary_metrics([0, 0], [0.8, 0.9])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["f1"], 0)

    def test_threshold_equality_and_endpoints(self):
        np.testing.assert_array_equal(
            lesson.classify_scores([0, 0.499, 0.5, 0.501, 1], 0.5), [0, 0, 1, 1, 1]
        )
        np.testing.assert_array_equal(lesson.classify_scores([0, 1], 0), [1, 1])
        np.testing.assert_array_equal(lesson.classify_scores([0, 1], 1), [0, 1])

    def test_invalid_metric_inputs(self):
        for labels, scores in (
            ([], []), ([0], [0, 1]), ([[0]], [[0.1]]),
            ([2], [0.1]), ([0], [np.nan]), ([0], [np.inf]), ([0], [1.1]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                lesson.binary_metrics(labels, scores)
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                lesson.classify_scores([0.5], threshold)

    def test_na_printed_explicitly(self):
        metrics = lesson.binary_metrics([0], [0.1])
        metrics["loss"] = 0.1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            lesson.print_metrics(metrics)
        self.assertIn("precision: N/A", output.getvalue())
        self.assertIn("denominator is zero", output.getvalue())


class CheckpointAndCLITests(unittest.TestCase):
    def make_checkpoint(self):
        args = lesson.parse_args([])
        checkpoint = lesson.checkpoint_metadata(args, 1, 0.6)
        checkpoint["state_dict"] = lesson.SimpleCNN().state_dict()
        return checkpoint

    def test_valid_checkpoint_and_safe_load(self):
        checkpoint = self.make_checkpoint()
        with patch.object(lesson.torch, "load", return_value=checkpoint) as load:
            model, loaded = lesson.load_checkpoint("unused.pt", torch.device("cpu"))
        load.assert_called_once_with("unused.pt", map_location="cpu", weights_only=True)
        self.assertIs(loaded, checkpoint)
        self.assertFalse(model.training)

    def test_incompatible_checkpoint_metadata(self):
        for key, value in (
            ("label_mapping", {"official_0": 0, "official_1": 1}),
            ("class_names", ["malignant", "normal_or_benign"]),
            ("architecture", "UNet"), ("image_size", 28), ("dataset", "BUSI"),
            ("threshold", float("nan")), ("threshold", 2),
            ("best_epoch", 0), ("best_validation_loss", float("inf")),
            ("state_dict", None), ("settings", None),
        ):
            checkpoint = self.make_checkpoint()
            checkpoint[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                lesson.validate_checkpoint(checkpoint)

    def test_incompatible_state_dict(self):
        checkpoint = self.make_checkpoint()
        checkpoint["state_dict"] = {}
        with patch.object(lesson.torch, "load", return_value=checkpoint):
            with self.assertRaises(RuntimeError):
                lesson.load_checkpoint("unused.pt", torch.device("cpu"))

    def test_overwrite_guard_and_intentional_override(self):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_dir", return_value=True):
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    lesson.guard_output(Path("unused"))
                lesson.guard_output(Path("unused"), overwrite=True)
        with patch.object(Path, "exists", return_value=False):
            lesson.guard_output(Path("unused"))
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_dir", return_value=False):
                with self.assertRaises(ValueError):
                    lesson.guard_output(Path("unused"), overwrite=True)

    def test_defaults_and_relative_paths(self):
        args = lesson.parse_args([])
        self.assertEqual(
            (args.epochs, args.batch_size, args.lr, args.seed, args.workers,
             args.threads, args.threshold), (10, 32, 0.001, 42, 0, 2, 0.5)
        )
        self.assertEqual(args.data_dir, lesson.PACKAGE_DIR / "data")
        self.assertEqual(args.output_dir, lesson.PACKAGE_DIR / "outputs" / "BreastMNIST64")
        self.assertEqual(lesson.parse_args(["--data-dir", "custom"]).data_dir, Path("custom"))

    def test_cli_invalid_ranges_and_modes(self):
        cases = [
            ["--epochs", "0"], ["--batch-size", "-1"], ["--workers", "-1"],
            ["--threads", "0"], ["--lr", "0"], ["--lr", "nan"], ["--lr", "inf"],
            ["--seed", "-1"], ["--seed", "4294967296"],
            ["--threshold", "nan"], ["--threshold", "-0.1"], ["--threshold", "1.1"],
            ["--test"], ["--predict", "image.png"],
            ["--weights", "model.pt"], ["--no-show"],
            ["--prepare-data", "--test"],
            ["--test", "--weights", "model.pt", "--threshold", "0.5"],
            ["--predict", "image.png", "--weights", "model.pt", "--download"],
            ["--prepare-data", "--threshold", "0.5"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    lesson.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_valid_modes(self):
        lesson.parse_args(["--prepare-data", "--download"])
        lesson.parse_args(["--test", "--weights", "model.pt"])
        lesson.parse_args(["--predict", "image.png", "--weights", "model.pt", "--no-show"])
        for threshold in ("0", "0.5", "1"):
            self.assertEqual(lesson.parse_args(["--threshold", threshold]).threshold,
                             float(threshold))

    def test_predict_main_never_reads_dataset(self):
        with patch.object(lesson, "guard_output"):
            with patch.object(Path, "mkdir"):
                with patch.object(lesson, "predict") as prediction:
                    with patch.object(lesson, "ensure_dataset") as data:
                        with contextlib.redirect_stdout(io.StringIO()):
                            lesson.main(["--predict", "image.png", "--weights", "model.pt",
                                         "--device", "cpu", "--no-show"])
        prediction.assert_called_once()
        data.assert_not_called()

    def test_training_saves_best_loss_immediately_without_testing(self):
        args = lesson.parse_args(["--epochs", "3", "--device", "cpu"])
        splits = {"train": object(), "val": object(), "test": object()}
        calls = []
        output = io.StringIO()
        results = []
        for loss in (0.8, 0.6, 0.7, 0.4, 0.5, 0.7):
            metrics = lesson.binary_metrics([0, 1], [0.1, 0.8])
            metrics["loss"] = loss
            results.append((metrics, None, None))
        results[-1][0].update(lesson.binary_metrics([0, 1], [0.8, 0.1]))

        def fake_epoch(*unused, **kwargs):
            calls.append("epoch")
            return results.pop(0)

        def save_checkpoint(checkpoint, path):
            calls.append(f"save{checkpoint['best_epoch']}")
            self.assertEqual(path.name, "breastmnist_cnn.pt")
            lesson.validate_checkpoint(checkpoint)

        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            loader = stack.enter_context(patch.object(lesson, "make_loader"))
            stack.enter_context(patch.object(lesson, "run_epoch", side_effect=fake_epoch))
            stack.enter_context(patch.object(lesson.torch.optim, "Adam"))
            stack.enter_context(patch.object(lesson.torch, "save", side_effect=save_checkpoint))
            open_file = stack.enter_context(patch.object(Path, "open"))
            open_file.return_value.__enter__.return_value = output
            pyplot = stack.enter_context(patch.object(lesson, "plot_library"))
            pyplot.return_value.subplots.return_value = (
                MagicMock(), (MagicMock(), MagicMock())
            )
            summary = stack.enter_context(patch.object(lesson, "write_json"))
            matrix = stack.enter_context(patch.object(lesson, "save_confusion_matrix"))
            lesson.train(args, splits, torch.device("cpu"))
        self.assertEqual(calls, ["epoch", "epoch", "save1", "epoch", "epoch",
                                 "save2", "epoch", "epoch"])
        self.assertEqual([call.args[0] for call in loader.call_args_list],
                         [splits["train"], splits["val"]])
        rows = list(lesson.csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(tuple(rows[0]), lesson.METRIC_FIELDS)
        self.assertEqual(len(rows), 3)
        payload = summary.call_args.args[1]
        self.assertEqual(payload["best_epoch"], 2)
        self.assertEqual(payload["best_validation_loss"], 0.4)
        self.assertEqual(payload["best_validation_metrics"]["accuracy"], 1)
        matrix.assert_called_once()
        plotted_metrics, plotted_path, title = matrix.call_args.args
        self.assertEqual(plotted_path.name, "validation_confusion_matrix.png")
        self.assertEqual(plotted_metrics, payload["best_validation_metrics"])
        self.assertEqual(
            [plotted_metrics[key] for key in ("tn", "fp", "fn", "tp")], [1, 0, 0, 1],
        )
        self.assertIn("best epoch 2", title)
        self.assertFalse(any("test" in key for key in payload))


class PlotTests(unittest.TestCase):
    def test_confusion_matrix_orientation_and_output(self):
        metrics = {"tn": 72, "fp": 8, "fn": 4, "tp": 16}
        path = Path("validation_confusion_matrix.png")
        figure, axis = MagicMock(), MagicMock()
        with patch.object(lesson, "plot_library") as pyplot:
            pyplot.return_value.subplots.return_value = figure, axis
            lesson.save_confusion_matrix(metrics, path, "Validation matrix")
        np.testing.assert_array_equal(axis.imshow.call_args.args[0], [[72, 8], [4, 16]])
        self.assertEqual(axis.set.call_args.kwargs["xlabel"], "Predicted label")
        self.assertEqual(axis.set.call_args.kwargs["ylabel"], "True label")
        self.assertEqual(axis.text.call_args_list[0].kwargs["color"], "white")
        figure.savefig.assert_called_once_with(path, dpi=150)
        pyplot.return_value.close.assert_called_once_with(figure)

    def test_validation_matrix_is_protected_from_overwrite(self):
        def exists(path):
            return path.name in ("existing_run", "validation_confusion_matrix.png")

        with patch.object(Path, "exists", exists):
            with patch.object(Path, "is_dir", return_value=True):
                with self.assertRaisesRegex(FileExistsError, "validation_confusion_matrix"):
                    lesson.guard_output(Path("existing_run"))


if __name__ == "__main__":
    unittest.main()
