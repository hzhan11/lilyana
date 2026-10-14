"""Desktop ultrasound classification demo. Educational use only, not a diagnosis."""

import argparse
import logging
import pickle
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk

from train_breast_cancer import (
    PACKAGE_DIR,
    load_checkpoint,
    predict_image_array,
    prepare_prediction_image,
)


DEFAULT_IMAGE = PACKAGE_DIR / "example_ultrasound.png"
DEFAULT_WEIGHTS = PACKAGE_DIR / "outputs" / "first_try" / "breastmnist_cnn.pt"
CLASS_TITLES = ("Normal / benign (label 0)", "Malignant (label 1)")
LOGGER = logging.getLogger(__name__)


class DemoApp:
    def __init__(self, root, image_path=DEFAULT_IMAGE, weights_path=DEFAULT_WEIGHTS,
                 auto_predict=True):
        self.root = root
        self.image_path = Path(image_path)
        self.weights_path = Path(weights_path)
        self.preview_photo = None
        self.device = torch.device("cpu")
        torch.set_num_threads(2)

        root.title("BreastMNIST64 - Image Classification Demo")
        root.minsize(760, 650)
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame, text="One ultrasound image, one class",
            font=("Arial", 20, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="0 = normal or benign    |    1 = malignant",
        ).pack(anchor="w", pady=(4, 12))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self.image_button = ttk.Button(
            controls, text="Choose image...", command=self.choose_image,
        )
        self.image_button.pack(side="left")
        self.weights_button = ttk.Button(
            controls, text="Load weights...", command=self.choose_weights,
        )
        self.weights_button.pack(side="left", padx=8)
        self.predict_button = ttk.Button(
            controls, text="Predict", command=self.predict,
        )
        self.predict_button.pack(side="left")

        self.image_text = tk.StringVar()
        self.weights_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.image_text, wraplength=700).pack(
            anchor="w", pady=(12, 4),
        )
        ttk.Label(frame, textvariable=self.weights_text, wraplength=700).pack(
            anchor="w",
        )
        self.refresh_paths()

        self.image_label = ttk.Label(frame, anchor="center")
        self.image_label.pack(fill="x", pady=12)
        self.result_text = tk.StringVar(value="Ready")
        self.result_label = ttk.Label(
            frame, textvariable=self.result_text, font=("Arial", 17, "bold"),
        )
        self.result_label.pack(anchor="w")
        self.score_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.score_text).pack(anchor="w", pady=6)
        self.status_text = tk.StringVar(value="Only trusted model checkpoints should be loaded.")
        ttk.Label(frame, textvariable=self.status_text, wraplength=700).pack(
            anchor="w", pady=(0, 12),
        )
        ttk.Separator(frame).pack(fill="x")
        ttk.Label(
            frame,
            text=(
                "Educational demo only. The included short-run model misses many "
                "malignant images. Do not use predictions for medical decisions.\n"
                "This predicts normal/benign versus malignant, NOT lesion presence. "
                "Benign images may also contain lesions. The default bundled image "
                "is a reference-malignant training example; the included model "
                "misclassifies it."
            ),
            wraplength=700, foreground="#9a3412",
        ).pack(anchor="w", pady=12)
        if auto_predict:
            root.after_idle(self.predict)

    def refresh_paths(self):
        self.image_text.set(f"Image: {self.image_path}")
        self.weights_text.set(f"Weights: {self.weights_path}")

    def choose_image(self):
        selected = filedialog.askopenfilename(
            parent=self.root, title="Choose an ultrasound image",
            initialdir=str(self.image_path.parent),
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.image_path = Path(selected)
            self.refresh_paths()
            self.predict()

    def choose_weights(self):
        selected = filedialog.askopenfilename(
            parent=self.root, title="Choose a trusted package2 checkpoint",
            initialdir=str(self.weights_path.parent),
            filetypes=[("PyTorch checkpoint", "*.pt")],
        )
        if selected:
            self.weights_path = Path(selected)
            self.refresh_paths()
            self.predict()

    def predict(self):
        self.result_text.set("Predicting...")
        self.result_label.configure(foreground="#334155")
        self.score_text.set("")
        self.status_text.set("Loading the checkpoint and running CPU inference...")
        self.image_label.configure(image="")
        self.preview_photo = None
        buttons = (self.image_button, self.weights_button, self.predict_button)
        for button in buttons:
            button.configure(state="disabled")
        self.root.update_idletasks()
        try:
            model, checkpoint = load_checkpoint(self.weights_path, self.device)
            array = prepare_prediction_image(self.image_path)
            score, decision = predict_image_array(
                model, array, checkpoint["threshold"], self.device,
            )
            preview = Image.fromarray(array).resize(
                (320, 320), Image.Resampling.NEAREST,
            )
            self.preview_photo = ImageTk.PhotoImage(preview, master=self.root)
            self.image_label.configure(image=self.preview_photo)
            self.result_text.set(f"Prediction: {CLASS_TITLES[decision]}")
            self.result_label.configure(
                foreground="#b91c1c" if decision else "#1d4ed8",
            )
            self.score_text.set(
                f"Malignant model score: {score:.4f}    |    "
                f"Threshold: {checkpoint['threshold']:.2f}"
            )
            self.status_text.set(
                "Showing the 64 x 64 grayscale input enlarged for display. "
                "The score is not calibrated clinical risk."
            )
        except (
            OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
            Image.DecompressionBombError,
        ) as error:
            LOGGER.exception("Prediction failed for %s using %s", self.image_path, self.weights_path)
            self.result_text.set("Prediction failed")
            self.status_text.set(str(error))
            messagebox.showerror(
                "Prediction failed",
                f"{error}\n\nChoose a valid image and a trusted package2 checkpoint.",
                parent=self.root,
            )
        finally:
            for button in buttons:
                button.configure(state="normal")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    root = tk.Tk()
    DemoApp(root, args.image, args.weights)
    root.mainloop()


if __name__ == "__main__":
    main()
