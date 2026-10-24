import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from model import AnimalCNN
import mtrain
from pic.dataset import ANIMALS, load_samples
from train import load_dataset, main, measure, split_dataset, train


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shapes_pixel_coordinates_and_both_heads_receive_gradients(self):
        model = AnimalCNN()
        logits, centers = model(torch.rand(2, 200, 200, 4))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(centers.shape, (2, 2))
        self.assertTrue(((centers >= 0) & (centers <= 199)).all())
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 5]))
        loss += nn.MSELoss()(centers / 199, torch.zeros_like(centers))
        loss.backward()
        for weights in (model.features[0].weight, model.classifier.weight, model.center_head[0].weight):
            self.assertTrue(torch.isfinite(weights.grad).all())
            self.assertGreater(weights.grad.abs().sum().item(), 0)

    def test_invalid_inputs_fail_explicitly(self):
        model = AnimalCNN()
        for images in (
            torch.zeros(200, 200, 4), torch.zeros(1, 4, 200, 200),
            torch.zeros(1, 200, 200, 3), torch.zeros(1, 200, 200, 4, dtype=torch.uint8),
            torch.full((1, 200, 200, 4), float("nan")),
            torch.full((1, 200, 200, 4), 255.0),
        ):
            with self.subTest(shape=images.shape, dtype=images.dtype):
                with self.assertRaises(ValueError):
                    model(images)

    def test_real_datasets_have_rgba_input_and_correct_class_mapping(self):
        for number, expected in ((1, ("cat", "dog", "pig")), (2, ("rabbit", "bear", "fox"))):
            with self.subTest(dataset=number):
                data, classes = load_dataset(number)
                images, labels, centers = data.tensors
                self.assertEqual(classes, expected)
                self.assertEqual(images.shape, (100, 200, 200, 4))
                self.assertEqual(images.dtype, torch.float32)
                self.assertTrue(((images >= 0) & (images <= 1)).all())
                self.assertTrue((images[:, :, :, 3] == 1).all())
                self.assertEqual(set(labels.tolist()), {0, 1, 2})
                self.assertEqual(centers.shape, (100, 2))

    def test_rgba_alpha_is_preserved_and_wrong_size_or_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            folder = base / "data1"
            folder.mkdir()
            labels = folder / "label.txt"
            labels.write_text("image.png,cat,[10,20]\n", encoding="utf-8")
            Image.new("RGBA", (200, 200), (10, 20, 30, 128)).save(folder / "image.png")
            data, _ = load_dataset(1, base)
            self.assertAlmostEqual(data.tensors[0][0, 0, 0, 3].item(), 128 / 255)
            Image.new("RGB", (100, 100)).save(folder / "image.png")
            with self.assertRaisesRegex(ValueError, "200x200"):
                load_dataset(1, base)
            Image.new("RGB", (200, 200)).save(folder / "image.png")
            labels.write_text("image.png,fox,[10,20]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not belong"):
                load_dataset(1, base)

    def test_combined_dataset_keeps_two_hundred_images_and_remaps_all_six_classes(self):
        with patch("train.load_samples", wraps=load_samples) as load:
            combined, classes = load_dataset("m")
        self.assertEqual([call.args[0].parent.name for call in load.call_args_list], ["data1", "data2"])
        first, _ = load_dataset(1)
        second, _ = load_dataset(2)
        self.assertEqual(classes, ANIMALS)
        self.assertEqual(combined.tensors[0].shape, (200, 200, 200, 4))
        self.assertTrue(torch.equal(
            combined.tensors[0], torch.cat((first.tensors[0], second.tensors[0])),
        ))
        self.assertTrue(torch.equal(
            combined.tensors[1], torch.cat((first.tensors[1], second.tensors[1] + 3)),
        ))
        self.assertTrue(torch.equal(
            combined.tensors[2], torch.cat((first.tensors[2], second.tensors[2])),
        ))
        self.assertEqual(set(combined.tensors[1].tolist()), set(range(6)))

    def test_training_all_modes_saves_reloadable_separate_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            for dataset in (1, 2, "m"):
                with self.subTest(dataset=dataset):
                    path = train(dataset, epochs=2, output_dir=temporary)
                    self.assertEqual(path, Path(temporary) / f"{dataset}.model")
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    data, classes = load_dataset(dataset)
                    self.assertEqual(checkpoint["classes"], list(classes))
                    self.assertEqual(checkpoint["input_shape"], [200, 200, 4])
                    self.assertEqual(len(checkpoint["history"]), 2)
                    split = checkpoint["split"]
                    self.assertEqual(split["train_samples"], 80)
                    self.assertEqual(split["validation_samples"], 20)
                    self.assertTrue(set(split["train_indices"]).isdisjoint(split["validation_indices"]))
                    self.assertEqual(
                        set(split["train_indices"]) | set(split["validation_indices"])
                        | set(split["unused_indices"]),
                        set(range(len(data))),
                    )
                    self.assertEqual(len(split["unused_indices"]), 100 if dataset == "m" else 0)
                    training, validation = split_dataset(
                        data, checkpoint["config"]["seed"], combined=dataset == "m",
                    )
                    self.assertEqual(split["train_indices"], training.indices)
                    self.assertEqual(split["validation_indices"], validation.indices)
                    model = AnimalCNN(len(classes))
                    model.load_state_dict(checkpoint["state_dict"])
                    model.eval()
                    with torch.no_grad():
                        logits, centers = model(data.tensors[0][:5])
                    self.assertEqual(logits.shape, (5, len(classes)))
                    self.assertTrue(torch.isfinite(centers).all())
                    self.assertEqual(checkpoint["dataset"], dataset)
                    validation_loader = torch.utils.data.DataLoader(validation, batch_size=5)
                    metrics = measure(model, validation_loader)
                    for name, value in metrics.items():
                        self.assertAlmostEqual(checkpoint["history"][-1][f"val_{name}"], value)
            self.assertTrue((Path(temporary) / "1.model").is_file())
            self.assertTrue((Path(temporary) / "2.model").is_file())
            self.assertTrue((Path(temporary) / "m.model").is_file())

    def test_split_is_exact_disjoint_stratified_and_reproducible_in_all_modes(self):
        for size, classes, combined in ((100, 3, False), (200, 6, True)):
            labels = torch.tensor([index % classes for index in range(size)])
            data = torch.utils.data.TensorDataset(torch.zeros(size, 1), labels)
            training, validation = split_dataset(data, 42, combined=combined)
            repeated_train, repeated_validation = split_dataset(data, 42, combined=combined)
            changed_train, _ = split_dataset(data, 43, combined=combined)
            self.assertEqual(len(training), 80)
            self.assertEqual(len(validation), 20)
            self.assertTrue(set(training.indices).isdisjoint(validation.indices))
            self.assertEqual(len(set(training.indices) | set(validation.indices)), 100)
            self.assertEqual(training.indices, repeated_train.indices)
            self.assertEqual(validation.indices, repeated_validation.indices)
            self.assertNotEqual(training.indices, changed_train.indices)
            self.assertIs(training.dataset, data)
            for subset in (training, validation):
                self.assertEqual(set(labels[subset.indices].tolist()), set(range(classes)))

    def test_combined_split_takes_40_train_10_val_per_source_without_crossing_baseline_splits(self):
        for seed in (0, 42, 123):
            labels = torch.tensor(([0] * 34 + [1] * 33 + [2] * 33)
                                  + ([3] * 34 + [4] * 33 + [5] * 33))
            data = torch.utils.data.TensorDataset(torch.zeros(200, 1), labels)
            training, validation = split_dataset(data, seed, combined=True)
            for offset in (0, 100):
                source = torch.utils.data.TensorDataset(
                    torch.zeros(100, 1), labels[offset:offset + 100],
                )
                baseline_train, baseline_val = split_dataset(source, seed)
                selected_train = {index - offset for index in training.indices if offset <= index < offset + 100}
                selected_val = {index - offset for index in validation.indices if offset <= index < offset + 100}
                self.assertEqual(len(selected_train), 40)
                self.assertEqual(len(selected_val), 10)
                self.assertTrue(selected_train <= set(baseline_train.indices))
                self.assertTrue(selected_val <= set(baseline_val.indices))
                self.assertTrue(selected_train.isdisjoint(selected_val))

    def test_combined_split_uses_all_samples_at_minimum_and_rejects_insufficient_sources(self):
        labels = torch.empty(100, dtype=torch.long)
        data = torch.utils.data.TensorDataset(torch.zeros(100, 1), labels)
        labels[:50] = torch.arange(50) % 3
        labels[50:] = 3 + torch.arange(50) % 3
        training, validation = split_dataset(data, combined=True)
        self.assertEqual(len(training), 80)
        self.assertEqual(len(validation), 20)
        self.assertEqual(set(training.indices) | set(validation.indices), set(range(100)))
        small = torch.utils.data.TensorDataset(torch.zeros(99, 1), labels[:99])
        with self.assertRaisesRegex(ValueError, "at least 50 images from data2"):
            split_dataset(small, combined=True)

    def test_validation_samples_never_receive_gradient_updates(self):
        size = 20
        images = torch.arange(size).float().view(size, 1, 1, 1).expand(size, 200, 200, 4) / size
        data = torch.utils.data.TensorDataset(
            images, torch.arange(size) % 3, torch.full((size, 2), 100.0),
        )
        model = AnimalCNN(3)
        observations = []

        def record(module, inputs):
            indices = (inputs[0][:, 0, 0, 0] * size).round().int().tolist()
            observations.append((module.training, torch.is_grad_enabled(), indices))

        handle = model.register_forward_pre_hook(record)
        self.addCleanup(handle.remove)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with patch("train.load_dataset", return_value=(data, ANIMALS[:3])):
                with patch("train.AnimalCNN", return_value=model):
                    train(1, epochs=1, batch_size=4, output_dir=temporary)
        training, validation = split_dataset(data, 42)
        updated = [index for mode, gradients, indices in observations if mode and gradients for index in indices]
        measured = [index for mode, gradients, indices in observations if not mode and not gradients for index in indices]
        self.assertEqual(sorted(updated), sorted(training.indices))
        self.assertTrue(set(updated).isdisjoint(validation.indices))
        self.assertEqual(sorted(measured), list(range(size)))

    def test_mtrain_cli_uses_shared_training_with_no_dataset_argument(self):
        with patch("train.train") as run:
            self.assertEqual(mtrain.main(["--epochs", "2", "--batch-size", "4"]), 0)
        self.assertEqual(run.call_args.args, ("m",))
        self.assertEqual(run.call_args.kwargs["epochs"], 2)
        self.assertEqual(run.call_args.kwargs["batch_size"], 4)
        for args in (["1"], ["--epochs", "0"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    mtrain.main(args)
                self.assertEqual(error.exception.code, 2)

    def test_training_reduces_combined_loss(self):
        data, classes = load_dataset(1)
        small_data = torch.utils.data.TensorDataset(*(tensor[:20] for tensor in data.tensors))
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with patch("train.load_dataset", return_value=(small_data, classes)):
                path = train(1, epochs=12, output_dir=temporary)
            history = torch.load(path, weights_only=True)["history"]
            self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_invalid_cli_requests_do_not_save_checkpoints(self):
        for args in (
            [], ["3"], ["1", "--epochs", "0"], ["1", "--batch-size", "0"],
            ["1", "--learning-rate", "nan"], ["1", "--threads", "0"],
            ["1", "--seed", "-1"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
