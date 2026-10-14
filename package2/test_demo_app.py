"""Offline tests for shared inference and desktop-demo behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

import demo_app
import train_breast_cancer as lesson


class PredictionTests(unittest.TestCase):
    def test_rgb_image_uses_shared_grayscale_resize(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.png"
            Image.new("RGB", (100, 80), (255, 255, 255)).save(path)
            result = lesson.prepare_prediction_image(path)
            self.assertEqual(result.shape, (64, 64))
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue((result == 255).all())

    def test_missing_image_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                lesson.prepare_prediction_image(Path(folder) / "missing.png")

    def test_threshold_equality_and_inference_without_gradients(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, images):
                self.grad_enabled = torch.is_grad_enabled()
                self.input_shape = tuple(images.shape)
                return torch.zeros(len(images))

        model = ConstantModel()
        score, decision = lesson.predict_image_array(
            model, np.zeros((64, 64), np.uint8), 0.5, torch.device("cpu"),
        )
        self.assertEqual((score, decision), (0.5, 1))
        self.assertFalse(model.training)
        self.assertFalse(model.grad_enabled)
        self.assertEqual(model.input_shape, (1, 1, 64, 64))

    def test_invalid_array_does_not_produce_a_prediction(self):
        with self.assertRaises(ValueError):
            lesson.predict_image_array(
                lesson.SimpleCNN(), np.zeros((32, 32), np.uint8),
                0.5, torch.device("cpu"),
            )


class DesktopTests(unittest.TestCase):
    def make_app_without_window(self):
        app = demo_app.DemoApp.__new__(demo_app.DemoApp)
        for name in (
            "root", "result_text", "result_label", "score_text", "status_text",
            "image_label", "image_button", "weights_button", "predict_button",
        ):
            setattr(app, name, MagicMock())
        app.image_path = Path("example.png")
        app.weights_path = Path("model.pt")
        app.preview_photo = "old preview"
        app.device = torch.device("cpu")
        return app

    def test_default_paths_are_package_relative(self):
        args = demo_app.parse_args([])
        self.assertEqual(args.image, lesson.PACKAGE_DIR / "example_ultrasound.png")
        self.assertEqual(
            args.weights, lesson.PACKAGE_DIR / "outputs" / "first_try" / "breastmnist_cnn.pt",
        )

    def test_failure_clears_previous_result_and_notifies(self):
        app = self.make_app_without_window()
        with patch.object(demo_app, "load_checkpoint", side_effect=ValueError("Wrong labels")):
            with patch.object(demo_app.messagebox, "showerror") as dialog:
                with self.assertLogs(demo_app.LOGGER, level="ERROR"):
                    app.predict()
        app.result_text.set.assert_called_with("Prediction failed")
        app.score_text.set.assert_called_with("")
        self.assertIsNone(app.preview_photo)
        dialog.assert_called_once()
        for button in (app.image_button, app.weights_button, app.predict_button):
            button.configure.assert_called_with(state="normal")

    def test_success_uses_saved_threshold_and_shared_inference(self):
        app = self.make_app_without_window()
        model = MagicMock()
        array = np.zeros((64, 64), np.uint8)
        with patch.object(demo_app, "load_checkpoint", return_value=(model, {"threshold": 0.7})):
            with patch.object(demo_app, "prepare_prediction_image", return_value=array):
                with patch.object(demo_app, "predict_image_array", return_value=(0.6, 0)) as predict:
                    with patch.object(demo_app.ImageTk, "PhotoImage", return_value="new preview"):
                        app.predict()
        predict.assert_called_once_with(model, array, 0.7, app.device)
        app.result_text.set.assert_called_with("Prediction: Normal / benign (label 0)")
        self.assertIn("Threshold: 0.70", app.score_text.set.call_args.args[0])
        self.assertEqual(app.preview_photo, "new preview")

    def test_cancelled_image_picker_keeps_current_result(self):
        app = self.make_app_without_window()
        with patch.object(demo_app.filedialog, "askopenfilename", return_value=""):
            with patch.object(app, "predict") as predict:
                app.choose_image()
        self.assertEqual(app.image_path, Path("example.png"))
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
