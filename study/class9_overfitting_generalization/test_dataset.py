from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parent


class DatasetTests(unittest.TestCase):
    def test_fixed_dataset_is_balanced_and_matches_manifest(self):
        manifest = json.loads((ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
        samples = manifest["samples"]
        self.assertEqual(manifest["image_size"], [128, 128])
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(len(samples), 48)
        self.assertEqual(Counter(sample["split"] for sample in samples), {"train": 24, "val": 12, "test": 12})
        hashes = set()
        families = {}
        for split, count in (("train", 12), ("val", 6), ("test", 6)):
            group = [sample for sample in samples if sample["split"] == split]
            self.assertEqual(Counter(sample["label"] for sample in group), {"cat": count, "dog": count})
            self.assertEqual(
                {str((ROOT / sample["path"]).resolve()) for sample in group},
                {str(path.resolve()) for path in (ROOT / "pic" / split).glob("*.png")},
            )
            families[split] = {sample["family_id"] for sample in group}
            for sample in group:
                path = ROOT / sample["path"]
                self.assertEqual(path.stem.split("_")[0], sample["label"])
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(checksum, sample["sha256"])
                hashes.add(checksum)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (128, 128))
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.format, "PNG")
                    image.verify()
                if split in ("train", "val"):
                    self.assertLessEqual(abs(sample["angle"]), 10)
        self.assertEqual(len(hashes), 48)
        self.assertEqual(families["train"], families["val"])
        self.assertEqual(len(families["train"]), 6)
        self.assertEqual(len(families["test"]), 12)
        self.assertFalse(families["test"] & families["train"])


if __name__ == "__main__":
    unittest.main()
