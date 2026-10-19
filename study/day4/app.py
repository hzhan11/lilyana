from pathlib import Path
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageTk
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


DAY3 = Path(__file__).resolve().parent.parent / "day3"
DRAW_SIZE = 256


def load_model(path):
    model = joblib.load(path)
    if not isinstance(model, LogisticRegression):
        raise ValueError("Please select a LogisticRegression model saved by day3.")
    check_is_fitted(model)
    if model.n_features_in_ != 64 * 64:
        raise ValueError("The model must accept 64 x 64 grayscale images.")
    return model


def prepare_image(image):
    image = image.convert("L")
    if ImageChops.invert(image).getbbox() is None:
        raise ValueError("Please draw a digit first.")
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, -1)


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        root.title("Handwritten Digit Prediction")
        root.resizable(False, False)
        panel = ttk.Frame(root, padding=16)
        panel.pack()

        # Step 1: Select the model saved by day3.
        ttk.Button(panel, text="Load model", command=self.choose_model).pack(fill=tk.X)
        self.model_info = tk.StringVar(value="No model loaded.")
        ttk.Label(
            panel, textvariable=self.model_info, wraplength=320
        ).pack(pady=8)
        ttk.Label(panel, text="Only load trusted .joblib files.").pack()

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
            parent=self.root, title="Select a trusted day3 model",
            initialdir=DAY3, initialfile="digit_model.joblib",
            filetypes=[("Joblib model", "*.joblib")],
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
        labels = ", ".join(str(label) for label in model.classes_)
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
