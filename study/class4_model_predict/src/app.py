from pathlib import Path
import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageTk
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


CLASS3 = Path(__file__).resolve().parents[2] / "class3_training_model" / "src"
DRAW_SIZE = 256


def load_model(path):
    if Path(path).suffix.lower() in (".h5", ".hdf5", ".keras"):
        model = keras.models.load_model(path, compile=False)
        if model.input_shape != (None, 4096) or model.output_shape != (None, 10):
            raise ValueError("Expected a neural network with 4096 inputs and 10 outputs.")
        return model
    if Path(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as weights:
            weights1 = weights["weights1"]
            bias1 = weights["bias1"]
            weights2 = weights["weights2"]
            bias2 = weights["bias2"]
        if weights1.ndim != 2 or weights1.shape[0] != 64 * 64:
            raise ValueError("The model must accept 64 x 64 grayscale images.")
        hidden_size = weights1.shape[1]
        if (
            hidden_size <= 0 or bias1.shape != (hidden_size,)
            or weights2.shape != (hidden_size, 10) or bias2.shape != (10,)
        ):
            raise ValueError("Expected weights for a 4096 -> hidden -> 10 neural network.")
        parameters = (weights1, bias1, weights2, bias2)
        if not all(
            parameter.dtype.kind in "fiu" and np.isfinite(parameter).all()
            for parameter in parameters
        ):
            raise ValueError("Model weights must contain finite real numbers.")
        model = keras.Sequential([
            keras.Input(shape=(4096,)),
            keras.layers.Dense(hidden_size, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        model.set_weights(parameters)
        return model
    model = joblib.load(path)
    if not isinstance(model, LogisticRegression):
        raise ValueError("Please select a LogisticRegression model saved by class3_training_model.")
    check_is_fitted(model)
    if model.n_features_in_ != 64 * 64:
        raise ValueError("The model must accept 64 x 64 grayscale images.")
    return model


def prepare_image(image):
    image = image.convert("L")
    if ImageChops.invert(image).getbbox() is None:
        raise ValueError("Please draw a digit first.")
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float64) / 255.0
    return pixels.reshape(1, -1)


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        root.title("Handwritten Digit Prediction")
        root.resizable(False, False)
        panel = ttk.Frame(root, padding=16)
        panel.pack()

        # Step 1: Select the model saved by class3_training_model.
        ttk.Button(panel, text="Load model", command=self.choose_model).pack(fill=tk.X)
        self.model_info = tk.StringVar(value="No model loaded.")
        ttk.Label(
            panel, textvariable=self.model_info, wraplength=320
        ).pack(pady=8)
        ttk.Label(panel, text="Only load trusted model files.").pack()

        # Step 2: Draw on a white canvas using the mouse.
        self.canvas = tk.Canvas(
            panel, width=DRAW_SIZE, height=DRAW_SIZE, background="white",
            highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(pady=12)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.bind("<Button-1>", self.begin_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        # Step 3: Predict the digit or clear the canvas.
        self.predict_button = ttk.Button(
            panel, text="Predict", command=self.predict, state=tk.DISABLED
        )
        self.predict_button.pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Clear", command=self.clear).pack(fill=tk.X, pady=4)
        self.result = tk.StringVar()
        ttk.Label(
            panel, textvariable=self.result, font=("Arial", 18), wraplength=320
        ).pack(pady=12)
        self.clear()

    def choose_model(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Select a trusted class3_training_model model",
            initialdir=CLASS3, initialfile="weights.h5",
            filetypes=[
                ("Class3 trained model", "*.h5 *.hdf5"), ("Keras model", "*.keras"),
                ("Neural network weights", "*.npz"),
                ("Joblib model", "*.joblib"),
            ],
        )
        if not path:
            return
        self.model = None
        self.predict_button.configure(state=tk.DISABLED)
        self.result.set("Prediction: -")
        self.model_info.set("No model loaded.")
        try:
            model = load_model(path)
        except (
            OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError,
            KeyError, IndexError, AttributeError, ImportError,
        ) as error:
            messagebox.showerror("Load failed", str(error), parent=self.root)
            return
        self.model = model
        classes = np.arange(10) if isinstance(model, keras.Model) else model.classes_
        labels = ", ".join(str(label) for label in classes)
        self.model_info.set(f"Loaded: {Path(path).name}\nLabels: {labels}")
        self.predict_button.configure(state=tk.NORMAL)

    def clear(self):
        self.image = Image.new("L", (DRAW_SIZE, DRAW_SIZE), 255)
        self.pen = ImageDraw.Draw(self.image)
        self.last_point = None
        self.result.set("Prediction: -")
        self.render()

    def render(self):
        self.photo = ImageTk.PhotoImage(self.image, master=self.root)
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)

    def begin_stroke(self, event):
        self.last_point = None
        self.draw(event)

    def draw(self, event):
        point = (
            max(0, min(DRAW_SIZE - 1, event.x)),
            max(0, min(DRAW_SIZE - 1, event.y)),
        )
        if self.last_point is not None:
            self.pen.line([self.last_point, point], fill=0, width=14)
        x, y = point
        self.pen.ellipse((x - 7, y - 7, x + 7, y + 7), fill=0)
        self.last_point = point
        self.result.set("Prediction: -")
        self.render()

    def end_stroke(self, event):
        if self.last_point is not None:
            self.draw(event)
        self.last_point = None

    def predict(self):
        if self.model is None:
            messagebox.showerror(
                "No model", "Please load a model first.", parent=self.root
            )
            return
        try:
            pixels = prepare_image(self.image)
            if isinstance(self.model, keras.Model):
                prediction = self.model.predict(pixels, verbose=0).argmax(axis=1)[0]
            else:
                prediction = self.model.predict(pixels)[0]
        except ValueError as error:
            self.result.set("Prediction: -")
            messagebox.showerror("Prediction failed", str(error), parent=self.root)
            return
        self.result.set(f"Prediction: {prediction}")


def main():
    root = tk.Tk()
    PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
