"""Read-only viewer for animal classes and image-center annotations."""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

if __package__:
    from .dataset import ANIMALS, Sample, load_samples
else:
    from dataset import ANIMALS, Sample, load_samples


PIC_DIR = Path(__file__).resolve().parent
MARKER_COLOR = "#e11d48"


def display_geometry(image_size, canvas_size):
    width, height = image_size
    canvas_width, canvas_height = canvas_size
    scale = min(max(1, canvas_width - 32) / width, max(1, canvas_height - 32) / height)
    shown_width, shown_height = max(1, round(width * scale)), max(1, round(height * scale))
    return (
        shown_width, shown_height,
        (canvas_width - shown_width) / 2, (canvas_height - shown_height) / 2,
    )


class DatasetViewer:
    def __init__(self, root, base_dir=PIC_DIR):
        self.root = root
        self.base_dir = Path(base_dir)
        self.samples = []
        self.index = -1
        self.image = None
        self.photo = None
        self.dataset = tk.StringVar(root)
        self.label_path = tk.StringVar(root)
        self.info = tk.StringVar(root, value="Choose a dataset to view its labeled images.")
        self.show_center = tk.BooleanVar(root, value=True)

        root.title("Class 15 | Animal Dataset and Center Viewer")
        root.geometry("1000x720")
        root.minsize(760, 560)
        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Dataset:").pack(side=tk.LEFT)
        self.folder_picker = ttk.Combobox(
            toolbar, textvariable=self.dataset, state="readonly", width=22,
        )
        self.folder_picker.pack(side=tk.LEFT, padx=8)
        self.folder_picker.bind("<<ComboboxSelected>>", self.choose_dataset)
        ttk.Button(toolbar, text="Refresh folders", command=self.refresh).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Open label file...", command=self.open_label).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            toolbar, text="Show center", variable=self.show_center, command=self.render,
        ).pack(side=tk.RIGHT)
        ttk.Label(root, textvariable=self.label_path, wraplength=900, padding=(12, 0)).pack(fill=tk.X)

        body = ttk.Frame(root, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        sidebar = ttk.Frame(body)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        ttk.Label(sidebar, text="Labeled images").pack(anchor=tk.W, pady=(0, 6))
        self.image_list = tk.Listbox(sidebar, width=29, exportselection=False)
        scrollbar = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=scrollbar.set)
        self.image_list.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_list.bind("<<ListboxSelect>>", self.choose_image)
        self.canvas = tk.Canvas(body, background="#e8edf3", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.render)

        ttk.Label(root, textvariable=self.info, padding=(12, 6), wraplength=900).pack(fill=tk.X)
        navigation = ttk.Frame(root, padding=(12, 4, 12, 8))
        navigation.pack(fill=tk.X)
        self.previous = ttk.Button(navigation, text="Previous", command=lambda: self.select(self.index - 1))
        self.previous.pack(side=tk.LEFT)
        self.next = ttk.Button(navigation, text="Next", command=lambda: self.select(self.index + 1))
        self.next.pack(side=tk.LEFT, padx=8)
        ttk.Label(
            root, padding=(12, 0, 12, 12),
            text="Origin: top-left | x increases rightward, y downward | Read-only: source files are not changed.",
        ).pack(fill=tk.X)
        self.refresh()

    def clear(self, message):
        self.samples = []
        self.index = -1
        self.image = self.photo = None
        self.image_list.delete(0, tk.END)
        self.canvas.delete("all")
        self.label_path.set("")
        self.info.set(message)
        self.update_navigation()

    def update_navigation(self):
        self.previous.configure(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next.configure(state=tk.NORMAL if 0 <= self.index < len(self.samples) - 1 else tk.DISABLED)

    def refresh(self):
        try:
            folders = sorted(
                path.name for path in self.base_dir.iterdir()
                if path.is_dir() and path.name.startswith("data")
            )
        except OSError as error:
            self.clear(f"Cannot list dataset folders: {error}")
            messagebox.showerror("Folder loading failed", str(error), parent=self.root)
            return
        self.folder_picker.configure(values=folders)
        if not folders:
            self.dataset.set("")
            self.clear("No data folders found. Create data1/label.txt or use Open label file.")
            return
        selected = self.dataset.get()
        if selected not in folders:
            selected = next(
                (name for name in folders if (self.base_dir / name / "label.txt").is_file()),
                folders[0],
            )
        self.dataset.set(selected)
        self.choose_dataset()

    def choose_dataset(self, event=None):
        self.load_labels(self.base_dir / self.dataset.get() / "label.txt")

    def open_label(self):
        filename = filedialog.askopenfilename(
            parent=self.root, title="Open animal labels", initialdir=self.base_dir,
            filetypes=(("Label files", "*.txt"), ("All files", "*.*")),
        )
        if filename:
            self.load_labels(Path(filename))

    def load_labels(self, filename):
        self.clear("Loading labels...")
        try:
            samples = load_samples(filename)
        except (OSError, UnicodeError, ValueError) as error:
            self.info.set(f"Cannot load labels: {error}")
            messagebox.showerror("Label loading failed", str(error), parent=self.root)
            return
        self.samples = samples
        self.dataset.set(Path(filename).parent.name)
        self.label_path.set(str(Path(filename).resolve()))
        for sample in samples:
            self.image_list.insert(tk.END, f"{sample.path.name}  |  {sample.animal}")
        self.select(0)

    def choose_image(self, event=None):
        selected = self.image_list.curselection()
        if selected:
            self.select(selected[0])

    def select(self, index):
        if not 0 <= index < len(self.samples):
            return
        self.index = index
        self.image = self.photo = None
        self.canvas.delete("all")
        self.image_list.selection_clear(0, tk.END)
        self.image_list.selection_set(index)
        self.image_list.see(index)
        self.update_navigation()
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                self.image = image.convert("RGBA")
            if self.image.size != sample.size:
                raise ValueError("Image dimensions changed. Reload the label file before continuing.")
        except (OSError, ValueError) as error:
            self.image = None
            self.info.set(f"Cannot display {sample.path.name}: {error}")
            messagebox.showerror("Image loading failed", str(error), parent=self.root)
            return
        x, y = sample.center
        width, height = sample.size
        self.info.set(
            f"{index + 1}/{len(self.samples)} | {sample.path.name} | {sample.animal} | "
            f"Center [{x:g},{y:g}] pixels | {width}x{height}"
        )
        self.render()

    def render(self, event=None):
        self.canvas.delete("all")
        if self.image is None:
            return
        canvas_size = (
            (event.width, event.height) if event is not None
            else (self.canvas.winfo_width(), self.canvas.winfo_height())
        )
        width, height, left, top = display_geometry(self.image.size, canvas_size)
        shown = self.image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(shown, master=self.canvas)
        self.canvas.create_image(left, top, image=self.photo, anchor=tk.NW, tags="sample_image")
        if self.show_center.get():
            sample = self.samples[self.index]
            x = left + sample.center[0] * width / sample.size[0]
            y = top + sample.center[1] * height / sample.size[1]
            self.canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6, outline=MARKER_COLOR, width=2, tags="center",
            )
            self.canvas.create_line(
                x - 11, y, x + 11, y, fill=MARKER_COLOR, width=2,
                tags=("center", "center_horizontal"),
            )
            self.canvas.create_line(
                x, y - 11, x, y + 11, fill=MARKER_COLOR, width=2, tags="center",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Open label.txt from this data folder.")
    args = parser.parse_args()
    root = tk.Tk()
    viewer = DatasetViewer(root)
    if args.data_dir is not None:
        viewer.load_labels(args.data_dir / "label.txt")
    root.mainloop()


if __name__ == "__main__":
    main()
