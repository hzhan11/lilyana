"""Browse and extend the handwritten digit dataset with a Tkinter window."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageChops, ImageDraw, ImageTk


DATA_DIR = Path(__file__).resolve().parent / "png"
IMAGE_SIZE = 64
DRAW_SIZE = 256


def image_label(path):
    label, separator, index = path.stem.partition("_")
    if label not in tuple(str(number) for number in range(10)):
        raise ValueError(f"Invalid digit label in filename: {path.name}")
    if not separator or not index.isdecimal():
        raise ValueError(f"Expected filename <label>_<index>.bmp: {path.name}")
    return int(label)


def load_samples(directory):
    samples = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".bmp":
            label = image_label(path)
            with Image.open(path) as image:
                thumbnail = image.convert("L").resize((80, 80))
            samples.append((path, label, thumbnail))
    return sorted(samples, key=lambda sample: (
        sample[1], int(sample[0].stem.partition("_")[2]), sample[0].name
    ))


def save_sample(directory, image, label):
    if label not in tuple(str(number) for number in range(10)):
        raise ValueError("Choose a label from 0 to 9.")
    grayscale = image.convert("L")
    if ImageChops.invert(grayscale).getbbox() is None:
        raise ValueError("Draw a digit before saving.")
    output = grayscale.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = directory / f"{label}_{index:02d}.bmp"
        try:
            stream = path.open("xb")
        except FileExistsError:
            index += 1
            continue
        with stream:
            output.save(stream, format="BMP")
        return path


class DatasetApp:
    def __init__(self, root, directory=DATA_DIR):
        self.root = root
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.photos = []
        self.last_point = None
        self.root.title("Handwritten Digit Dataset")
        self.root.geometry("1000x650")
        self.root.minsize(830, 450)

        gallery_panel = ttk.Frame(root, padding=12)
        gallery_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.count = tk.StringVar()
        ttk.Label(gallery_panel, textvariable=self.count).pack(anchor=tk.W)
        gallery_body = ttk.Frame(gallery_panel)
        gallery_body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.gallery = tk.Canvas(gallery_body, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            gallery_body, orient=tk.VERTICAL, command=self.gallery.yview
        )
        self.gallery.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cards = ttk.Frame(self.gallery)
        self.gallery.create_window((0, 0), window=self.cards, anchor=tk.NW)
        self.cards.bind("<Configure>", self.update_scroll_region)
        self.gallery.bind("<Enter>", self.enable_wheel)
        self.gallery.bind("<Leave>", self.disable_wheel)

        editor = ttk.Frame(root, padding=16)
        editor.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(editor, text="Draw a digit with the mouse").pack(anchor=tk.W)
        self.drawing = tk.Canvas(
            editor, width=DRAW_SIZE, height=DRAW_SIZE,
            background="white", highlightthickness=1,
            highlightbackground="#999999", cursor="crosshair",
        )
        self.drawing.pack(pady=10)
        self.drawing.bind("<Button-1>", self.begin_stroke)
        self.drawing.bind("<B1-Motion>", self.continue_stroke)
        self.drawing.bind("<ButtonRelease-1>", self.end_stroke)

        labels = ttk.Frame(editor)
        labels.pack(fill=tk.X, pady=8)
        ttk.Label(labels, text="label:").pack(side=tk.LEFT)
        self.label = tk.StringVar(value="1")
        ttk.Combobox(
            labels, textvariable=self.label, values=tuple(range(10)),
            state="readonly", width=5,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(editor, text="Save and draw next", command=self.save).pack(
            fill=tk.X, pady=4
        )
        ttk.Button(editor, text="Clear", command=self.clear).pack(
            fill=tk.X, pady=4
        )
        ttk.Button(editor, text="Refresh images", command=self.refresh).pack(
            fill=tk.X, pady=4
        )
        self.status = tk.StringVar(value="Saved as 64 x 64 grayscale BMP.")
        ttk.Label(
            editor, textvariable=self.status, wraplength=DRAW_SIZE
        ).pack(anchor=tk.W, pady=12)
        self.clear()
        self.refresh()

    def update_scroll_region(self, event=None):
        self.gallery.configure(scrollregion=self.gallery.bbox("all"))

    def enable_wheel(self, event):
        self.root.bind("<MouseWheel>", self.scroll)

    def disable_wheel(self, event):
        self.root.unbind("<MouseWheel>")

    def scroll(self, event):
        self.gallery.yview_scroll(-int(event.delta / 120), "units")

    def refresh(self):
        try:
            samples = load_samples(self.directory)
        except (OSError, ValueError) as error:
            self.status.set(f"Could not load images: {error}")
            messagebox.showerror("Load failed", str(error), parent=self.root)
            return
        for child in self.cards.winfo_children():
            child.destroy()
        self.photos.clear()
        for index, (path, label, image) in enumerate(samples):
            card = ttk.Frame(self.cards, padding=8, relief=tk.RIDGE)
            card.grid(row=index // 4, column=index % 4, padx=4, pady=4)
            photo = ImageTk.PhotoImage(image, master=self.root)
            self.photos.append(photo)
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=f"label: {label}").pack()
            ttk.Label(card, text=path.name).pack()
        self.count.set(f"Dataset: {len(samples)} images")
        self.cards.update_idletasks()
        self.update_scroll_region()

    def clear(self):
        self.drawing.delete("all")
        self.image = Image.new("L", (DRAW_SIZE, DRAW_SIZE), 255)
        self.pen = ImageDraw.Draw(self.image)
        self.last_point = None

    def position(self, event):
        return (
            max(0, min(DRAW_SIZE - 1, event.x)),
            max(0, min(DRAW_SIZE - 1, event.y)),
        )

    def begin_stroke(self, event):
        self.last_point = self.position(event)
        self.paint(self.last_point, self.last_point)

    def continue_stroke(self, event):
        point = self.position(event)
        if self.last_point is not None:
            self.paint(self.last_point, point)
            self.last_point = point

    def end_stroke(self, event):
        self.continue_stroke(event)
        self.last_point = None

    def paint(self, start, end):
        width = 14
        self.drawing.create_line(*start, *end, fill="black", width=width)
        self.pen.line([start, end], fill=0, width=width)
        radius = width / 2
        for x, y in (start, end):
            bounds = (x - radius, y - radius, x + radius, y + radius)
            self.drawing.create_oval(*bounds, fill="black", outline="black")
            self.pen.ellipse(bounds, fill=0)

    def save(self):
        try:
            path = save_sample(self.directory, self.image, self.label.get())
        except (OSError, ValueError) as error:
            self.status.set(f"Could not save image: {error}")
            messagebox.showerror("Save failed", str(error), parent=self.root)
            return
        self.clear()
        self.status.set(f"Saved {path.name}. Ready for the next digit.")
        self.refresh()
        self.gallery.yview_moveto(0)


def main():
    root = tk.Tk()
    DatasetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
