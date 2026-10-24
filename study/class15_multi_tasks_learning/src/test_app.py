import hashlib
import json
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops

from pic.app import DatasetViewer, display_geometry, load_samples
from pic.dataset import UNSEEN_ANIMALS


def write_dataset(folder, lines):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "label.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder / "label.txt"


class LabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        Image.new("RGB", (200, 200), "white").save(self.folder / "image.png")

    def test_loads_filename_class_coordinates_and_actual_size(self):
        labels = write_dataset(self.folder, ["image.png,cat,[82,100]"])
        sample, = load_samples(labels)
        self.assertEqual(sample.path, self.folder / "image.png")
        self.assertEqual(sample.animal, "cat")
        self.assertEqual(sample.center, (82, 100))
        self.assertEqual(sample.size, (200, 200))

    def test_accepts_bom_blank_lines_whitespace_and_fractional_centers(self):
        labels = self.folder / "label.txt"
        labels.write_text("\ufeff\n image.png , pig , [0, 199.5] \n", encoding="utf-8")
        self.assertEqual(load_samples(labels)[0].center, (0, 199.5))

    def test_accepts_the_three_new_animal_classes(self):
        for animal in ("rabbit", "bear", "fox", *UNSEEN_ANIMALS):
            with self.subTest(animal=animal):
                labels = write_dataset(self.folder, [f"image.png,{animal},[82,100]"])
                self.assertEqual(load_samples(labels)[0].animal, animal)

    def test_rejects_malformed_and_empty_labels(self):
        for line in ("", "image.png,cat", "image.png,cat,not-json", "image.png,dragon,[1,2]"):
            with self.subTest(line=line):
                labels = write_dataset(self.folder, [line])
                with self.assertRaises(ValueError):
                    load_samples(labels)

    def test_rejects_invalid_centers(self):
        for center in (
            [-1, 50], [200, 50], [50, 200], [1], [1, 2, 3], {"x": 1, "y": 2},
            [True, 1], ["5", 1], [float("nan"), 1], [float("inf"), 1], [10**400, 1],
        ):
            with self.subTest(center=center):
                labels = write_dataset(self.folder, [f"image.png,cat,{json.dumps(center)}"])
                with self.assertRaises(ValueError):
                    load_samples(labels)

    def test_rejects_paths_and_duplicate_names(self):
        for filename in ("../image.png", "..\\image.png", "C:\\image.png", "image.jpg"):
            with self.subTest(filename=filename):
                labels = write_dataset(self.folder, [f"{filename},cat,[1,2]"])
                with self.assertRaises(ValueError):
                    load_samples(labels)
        labels = write_dataset(self.folder, ["image.png,cat,[1,2]", "image.png,dog,[3,4]"])
        with self.assertRaisesRegex(ValueError, "line 2.*duplicate"):
            load_samples(labels)

    def test_missing_or_corrupt_images_identify_the_label_line(self):
        labels = write_dataset(self.folder, ["missing.png,cat,[1,2]"])
        with self.assertRaisesRegex(ValueError, "line 1.*missing.png"):
            load_samples(labels)
        (self.folder / "image.png").write_bytes(b"not an image")
        labels = write_dataset(self.folder, ["image.png,cat,[1,2]"])
        with self.assertRaisesRegex(ValueError, "line 1.*image.png"):
            load_samples(labels)

    def test_a_different_image_format_cannot_disguise_itself_as_png(self):
        Image.new("RGB", (200, 200)).save(self.folder / "image.png", format="JPEG")
        labels = write_dataset(self.folder, ["image.png,cat,[1,2]"])
        with self.assertRaisesRegex(ValueError, "not a PNG"):
            load_samples(labels)

    def test_supplied_dataset_loads_without_modifying_files(self):
        for dataset, animals in (
            ("data1", {"cat", "dog", "pig"}),
            ("data2", {"rabbit", "bear", "fox"}),
            ("data3", set(UNSEEN_ANIMALS)),
        ):
            with self.subTest(dataset=dataset):
                folder = Path(__file__).resolve().parent / "pic" / dataset
                paths = list(folder.iterdir())
                before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                samples = load_samples(folder / "label.txt")
                self.assertEqual(len(samples), 10 if dataset == "data3" else 100)
                self.assertEqual({sample.animal for sample in samples}, animals)
                self.assertTrue(all(sample.size == (200, 200) for sample in samples))
                self.assertEqual(
                    {sample.path.name for sample in samples},
                    {path.name for path in folder.glob("*.png")},
                )
                for sample in samples:
                    with Image.open(sample.path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.mode, "RGB")
                        bounds = ImageChops.difference(
                            image, Image.new("RGB", image.size, "white"),
                        ).getbbox()
                        self.assertIsNotNone(bounds)
                        left, top, right, bottom = bounds
                        self.assertEqual(
                            sample.center, ((left + right) // 2, (top + bottom) // 2),
                        )
                        self.assertTrue(0 < left < right < image.width)
                        self.assertTrue(0 < top < bottom < image.height)
                self.assertEqual(
                    before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                )

    def test_geometry_preserves_aspect_ratio_and_accounts_for_padding(self):
        self.assertEqual(display_geometry((200, 200), (640, 480)), (448, 448, 96, 16))
        self.assertEqual(display_geometry((200, 100), (640, 480)), (608, 304, 16, 88))
        width, height, _, _ = display_geometry((200, 200), (1, 1))
        self.assertGreaterEqual(width, 1)
        self.assertGreaterEqual(height, 1)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for name, color in (("data1", "white"), ("data2", "green")):
            dataset = self.folder / name
            dataset.mkdir()
            Image.new("RGB", (200, 200), color).save(dataset / "first.png")
            Image.new("RGB", (200, 200), color).save(dataset / "second.png")
            animals = ("cat", "pig") if name == "data1" else ("rabbit", "fox")
            write_dataset(dataset, [
                f"first.png,{animals[0]},[82,100]", f"second.png,{animals[1]},[40,80]",
            ])
        (self.folder / "data3").mkdir()
        error_patch = patch("pic.app.messagebox.showerror")
        self.error = error_patch.start()
        self.addCleanup(error_patch.stop)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = DatasetViewer(self.root, self.folder)

    def test_startup_selects_a_labeled_dataset_and_shows_details(self):
        self.assertEqual(self.app.dataset.get(), "data1")
        self.assertEqual(self.app.index, 0)
        self.assertEqual(self.app.image_list.size(), 2)
        self.assertIn("cat", self.app.info.get())
        self.assertIn("[82,100]", self.app.info.get())
        self.assertEqual(str(self.app.previous["state"]), "disabled")
        self.assertEqual(str(self.app.next["state"]), "normal")
        self.error.assert_not_called()

    def test_center_is_scaled_with_image_and_can_be_hidden(self):
        event = SimpleNamespace(width=640, height=480)
        self.app.render(event)
        x, y = 96 + 82 * 448 / 200, 16 + 100 * 448 / 200
        self.assertEqual(self.app.canvas.coords("sample_image"), [96, 16])
        self.assertEqual(self.app.canvas.coords("center_horizontal"), [x - 11, y, x + 11, y])
        self.assertEqual(len(self.app.canvas.find_withtag("center")), 3)
        self.app.show_center.set(False)
        self.app.render(event)
        self.assertEqual(self.app.canvas.find_withtag("center"), ())
        self.assertIn("[82,100]", self.app.info.get())

    def test_navigation_and_dataset_switching(self):
        self.app.next.invoke()
        self.assertEqual(self.app.index, 1)
        self.assertIn("pig", self.app.info.get())
        self.assertEqual(str(self.app.next["state"]), "disabled")
        self.app.previous.invoke()
        self.assertEqual(self.app.index, 0)
        self.app.dataset.set("data2")
        self.app.choose_dataset()
        self.assertEqual(self.app.samples[0].path.parent.name, "data2")
        self.assertIn("data2", self.app.label_path.get())
        self.assertIn("rabbit", self.app.info.get())

    def test_empty_dataset_reports_error_and_clears_stale_image(self):
        self.app.dataset.set("data3")
        self.app.choose_dataset()
        self.assertEqual(self.app.samples, [])
        self.assertIsNone(self.app.image)
        self.assertEqual(self.app.canvas.find_all(), ())
        self.assertEqual(self.app.image_list.size(), 0)
        self.assertEqual(self.app.label_path.get(), "")
        self.assertIn("Cannot load labels", self.app.info.get())
        self.error.assert_called_once()

    def test_removed_image_reports_error_without_previous_overlay(self):
        self.app.samples[1].path.unlink()
        self.app.select(1)
        self.assertIsNone(self.app.image)
        self.assertEqual(self.app.canvas.find_all(), ())
        self.assertIn("Cannot display second.png", self.app.info.get())
        self.error.assert_called_once()
        self.app.select(0)
        self.assertIsNotNone(self.app.image)

    def test_cancelled_file_picker_keeps_current_view(self):
        original = self.app.samples
        with patch("pic.app.filedialog.askopenfilename", return_value=""):
            self.app.open_label()
        self.assertIs(self.app.samples, original)
        self.error.assert_not_called()

    def test_refresh_discovers_new_dataset_and_handles_no_folders(self):
        (self.folder / "data4").mkdir()
        self.app.refresh()
        self.assertIn("data4", self.app.folder_picker["values"])
        empty = self.folder / "empty"
        empty.mkdir()
        self.app.base_dir = empty
        self.app.refresh()
        self.assertEqual(self.app.samples, [])
        self.assertIn("No data folders found", self.app.info.get())
        self.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
