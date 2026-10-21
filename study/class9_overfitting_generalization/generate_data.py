"""Generate a fixed, hand-drawn cat/dog dataset using Pillow only."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SEED = 42
SIZE = 128
SCALE = 4
ROOT = Path(__file__).resolve().parent
BACKGROUND = "#fffdf9"
INK = "#302e36"
PALETTE = ("#dba2b4", "#4a4951", "#c89a70", "#9aa5af")
NOTE = (
    "Training and validation intentionally share three base drawing families per "
    "class, with small rotations, translations and overlapping coat palettes. "
    "Validation is near-duplicate validation, not a classic independent validation "
    "set. Test uses six separately drawn families per class and a shifted "
    "illustration style. All samples are fixed before training; no labels, images "
    "or test selections depend on model predictions. Performance is not guaranteed."
)


class Canvas:
    """Draw in logical 128-pixel coordinates at four times the resolution."""

    def __init__(self):
        self.image = Image.new("RGB", (SIZE * SCALE, SIZE * SCALE), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

    @staticmethod
    def box(values):
        return tuple(round(value * SCALE) for value in values)

    def ellipse(self, box, fill, width=1.7, outline=INK):
        self.draw.ellipse(
            self.box(box), fill=fill, outline=outline, width=round(width * SCALE)
        )

    def polygon(self, points, fill, width=1.7):
        points = [(round(x * SCALE), round(y * SCALE)) for x, y in points]
        self.draw.polygon(points, fill=fill)
        self.draw.line(points + [points[0]], fill=INK, width=round(width * SCALE),
                       joint="curve")

    def line(self, points, width=1.6, fill=INK):
        self.draw.line(
            [(round(x * SCALE), round(y * SCALE)) for x, y in points],
            fill=fill, width=round(width * SCALE), joint="curve"
        )

    def arc(self, box, start, end, width=2, fill=INK):
        self.draw.arc(self.box(box), start, end, fill=fill, width=round(width * SCALE))


def eyes(c, left, right, y, coat, sleepy=False):
    if sleepy:
        c.arc((left - 3, y - 3, left + 3, y + 3), 185, 350, 1.6)
        c.arc((right - 3, y - 3, right + 3, y + 3), 185, 350, 1.6)
    else:
        for x in (left, right):
            c.ellipse((x - 2.5, y - 3, x + 2.5, y + 3), INK, 0, None)
            c.ellipse((x - 0.8, y - 2, x + 0.3, y - 0.9), "white", 0, None)


def cat_face(c, cx, cy, radius, coat, sleepy=False, stripes=True):
    c.polygon([(cx - radius + 2, cy - 6), (cx - radius + 1, cy - radius - 13),
               (cx - 5, cy - radius + 3)], coat)
    c.polygon([(cx + 5, cy - radius + 3), (cx + radius - 1, cy - radius - 13),
               (cx + radius - 2, cy - 6)], coat)
    c.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), coat)
    for sign in (-1, 1):
        x = cx + sign * (radius - 7)
        c.line([(x, cy - radius + 1), (x + sign * 1, cy - radius - 7)],
               2.2, "#d991a3")
    eyes(c, cx - radius * .42, cx + radius * .42, cy - 2, coat, sleepy)
    c.polygon([(cx - 2.5, cy + 5), (cx + 2.5, cy + 5), (cx, cy + 8)],
              "#db859c", 1)
    c.line([(cx, cy + 8), (cx - 4, cy + 11), (cx - 7, cy + 9)], 1.2)
    c.line([(cx, cy + 8), (cx + 4, cy + 11), (cx + 7, cy + 9)], 1.2)
    for sign in (-1, 1):
        for dy in (-3, 3):
            c.line([(cx + sign * 10, cy + 7 + dy / 3),
                    (cx + sign * (radius + 10), cy + 7 + dy)], 1.1)
    if stripes:
        for dx in (-6, 0, 6):
            c.line([(cx + dx, cy - radius + 3),
                    (cx + dx * .75, cy - radius + 9)], 1.5)


def dog_face(c, cx, cy, radius, coat, pointed=False, sleepy=False):
    if pointed:
        c.polygon([(cx - radius, cy), (cx - radius - 4, cy - radius - 17),
                   (cx - 5, cy - radius + 2)], coat)
        c.polygon([(cx + 5, cy - radius + 2),
                   (cx + radius + 4, cy - radius - 17), (cx + radius, cy)], coat)
    else:
        c.ellipse((cx - radius - 12, cy - radius + 1,
                   cx - radius + 10, cy + radius + 4), coat)
        c.ellipse((cx + radius - 10, cy - radius + 1,
                   cx + radius + 12, cy + radius + 4), coat)
    c.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), coat)
    eyes(c, cx - radius * .43, cx + radius * .43, cy - 4, coat, sleepy)
    c.ellipse((cx - radius * .64, cy + 1, cx + radius * .64, cy + radius - 1),
              "#e8d3be", 1.3)
    c.ellipse((cx - 5, cy + 4, cx + 5, cy + 10), INK, 0, None)
    c.line([(cx, cy + 9), (cx, cy + 14), (cx - 4, cy + 16)], 1.2)
    c.line([(cx, cy + 14), (cx + 4, cy + 16)], 1.2)


def familiar_drawing(label, family, coat):
    c = Canvas()
    if family == 0:
        c.arc((70, 67, 106, 110), 270, 445, 9, coat)
        c.arc((70, 67, 106, 110), 270, 445, 1.4)
        c.ellipse((40, 62, 87, 115), coat)
        for x in (48, 68):
            c.ellipse((x, 96, x + 13, 116), coat)
        cx, cy, radius = 63, 48, 24
    elif family == 1:
        c.ellipse((30, 73, 98, 140), coat)
        cx, cy, radius = 64, 60, 32
    else:
        c.arc((76, 69, 113, 110), 260, 445, 8, coat)
        c.arc((76, 69, 113, 110), 260, 445, 1.5)
        c.ellipse((34, 72, 92, 111), coat)
        c.ellipse((38, 98, 60, 113), coat)
        c.ellipse((67, 98, 89, 113), coat)
        cx, cy, radius = 60, 54, 25
    if label == "cat":
        cat_face(c, cx, cy, radius, coat)
    else:
        dog_face(c, cx, cy, radius, coat)
    return c.image


def profile(c, label, coat, standing=False):
    """Independent side-view construction, not a transformed familiar drawing."""
    if label == "cat":
        c.arc((14, 35, 51, 88), 90, 300, 7, coat)
        c.arc((14, 35, 51, 88), 90, 300, 1.5)
    else:
        c.line([(36, 73), (23, 56), (18, 42)], 8, coat)
        c.line([(36, 73), (23, 56), (18, 42)], 1.4)
    c.ellipse((28, 57, 92, 96), coat)
    if standing:
        for x in (37, 51, 76, 88):
            c.polygon([(x - 4, 84), (x + 4, 84), (x + 3, 112),
                       (x - 6, 112)], coat, 1.3)
    else:
        c.ellipse((31, 84, 55, 100), coat)
        c.ellipse((69, 85, 98, 100), coat)
    if label == "cat":
        c.polygon([(77, 61), (74, 29), (88, 42), (102, 30), (105, 65)], coat)
        c.ellipse((73, 40, 108, 75), coat)
        c.line([(87, 58), (110, 54)], 1)
        c.line([(88, 61), (113, 64)], 1)
        c.polygon([(104, 55), (109, 57), (104, 60)], "#cf8299", 1)
    else:
        c.ellipse((73, 33, 103, 75), coat)
        c.ellipse((91, 51, 119, 70), "#e8d3be")
        c.ellipse((111, 51, 120, 59), INK, 0, None)
        c.ellipse((72, 36, 86, 67), coat)
    c.ellipse((94, 46, 98, 51), INK, 0, None)


TEST_STYLES = (
    "fine-line portrait", "side-view illustration", "sleepy reclining sketch",
    "angular geometric portrait", "standing profile", "round storybook portrait"
)


def independent_drawing(label, index, coat):
    c = Canvas()
    if index in (1, 4):
        profile(c, label, coat, standing=index == 4)
    elif index == 0:
        # Unfilled contours and long necks differ from the colored training art.
        c.line([(41, 112), (44, 86), (50, 77)], 1.4)
        c.line([(82, 77), (88, 89), (90, 112)], 1.4)
        if label == "cat":
            cat_face(c, 64, 57, 27, BACKGROUND, stripes=False)
        else:
            dog_face(c, 64, 56, 26, BACKGROUND)
        c.line([(49, 101), (79, 101)], 1.2)
    elif index == 2:
        c.ellipse((17, 63, 110, 103), coat)
        c.arc((20, 65, 74, 104), 15, 275, 5, coat)
        c.arc((20, 65, 74, 104), 15, 275, 1.3)
        c.ellipse((77, 87, 113, 103), coat)
        if label == "cat":
            cat_face(c, 87, 65, 21, coat, sleepy=True, stripes=False)
        else:
            dog_face(c, 88, 65, 20, coat, sleepy=True)
    elif index == 3:
        c.polygon([(33, 104), (45, 83), (83, 83), (96, 104)], coat)
        if label == "cat":
            points = [(28, 65), (27, 20), (48, 34), (79, 34), (102, 19),
                      (99, 69), (83, 87), (46, 87)]
        else:
            points = [(29, 41), (42, 28), (84, 28), (100, 42), (111, 74),
                      (95, 81), (85, 63), (83, 89), (43, 89), (41, 63),
                      (31, 82), (18, 75)]
        c.polygon(points, coat)
        eyes(c, 48, 80, 53, coat)
        if label == "cat":
            c.polygon([(60, 65), (68, 65), (64, 70)], "#ce8299", 1)
            for side in (-1, 1):
                c.line([(64 + side * 12, 69), (64 + side * 39, 65)], 1)
                c.line([(64 + side * 12, 73), (64 + side * 39, 78)], 1)
        else:
            c.polygon([(48, 64), (80, 64), (85, 78), (64, 88), (43, 78)],
                      "#e8d3be")
            c.polygon([(59, 67), (69, 67), (64, 73)], INK, 1)
        c.line([(64, 70), (64, 78)], 1)
    else:
        c.ellipse((38, 74, 90, 117), coat)
        c.ellipse((32, 104, 58, 118), coat)
        c.ellipse((72, 104, 98, 118), coat)
        if label == "cat":
            cat_face(c, 64, 56, 33, coat, stripes=False)
        else:
            dog_face(c, 64, 52, 28, coat, pointed=True)
        c.ellipse((43, 78, 85, 102), "#e8d3be", 1.2)
    return c.image


def transform(image, angle, shift):
    image = image.rotate(angle, Image.Resampling.BICUBIC, fillcolor=BACKGROUND)
    image = image.transform(
        image.size, Image.Transform.AFFINE,
        (1, 0, -shift[0] * SCALE, 0, 1, -shift[1] * SCALE),
        resample=Image.Resampling.BICUBIC, fillcolor=BACKGROUND
    )
    return image.resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert("RGB")


def font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(samples):
    sheet = Image.new("RGB", (1120, 1500), "#eef0f4")
    d = ImageDraw.Draw(sheet)
    d.text((24, 16), "Cats & dogs: fixed small-data generalization study",
           fill=INK, font=font(28))
    d.text((24, 59), "Validation uses the SAME base families as training (near-duplicates).",
           fill=INK, font=font(19))
    d.text((24, 87), "Test: independently drawn styles, frozen before training. No performance promise.",
           fill=INK, font=font(19))
    y = 130
    for split, title in (
        ("train", "TRAIN  |  24 drawings  |  3 base families per class"),
        ("val", "VALIDATION  |  12 drawings  |  same base families, new variants"),
        ("test", "TEST  |  12 drawings  |  distinct families and illustration styles"),
    ):
        group = [sample for sample in samples if sample["split"] == split]
        rows = math.ceil(len(group) / 8)
        d.rounded_rectangle((16, y, 1104, y + 44 + rows * 158),
                            radius=12, fill="white")
        d.text((30, y + 10), title, fill=INK, font=font(21))
        for i, sample in enumerate(group):
            x = 32 + (i % 8) * 135
            top = y + 45 + (i // 8) * 158
            with Image.open(ROOT / sample["path"]) as image:
                sheet.paste(image, (x, top))
            d.text((x + 8, top + 130), Path(sample["path"]).name,
                   fill=INK, font=font(14))
        y += 59 + rows * 158
    sheet.crop((0, 0, 1120, y + 10)).save(ROOT / "pic" / "dataset_overview.png")


def generate(overwrite=False):
    targets = [
        ROOT / "pic" / split / f"{label}_{index:02d}.png"
        for split, count in (("train", 12), ("val", 6), ("test", 6))
        for label in ("cat", "dog") for index in range(1, count + 1)
    ]
    targets += [ROOT / "pic" / "dataset_overview.png", ROOT / "dataset_manifest.json"]
    existing = [path for path in targets if path.exists()]
    split_files = [
        path for split in ("train", "val", "test")
        for path in (ROOT / "pic" / split).glob("*") if path.is_file()
    ]
    if (existing or split_files) and not overwrite:
        raise FileExistsError(
            "Dataset output already exists. Use --overwrite to replace only "
            "the generator's explicitly named output files."
        )
    for split in ("train", "val", "test"):
        (ROOT / "pic" / split).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    samples = []
    # Both classes use identical augmentation schedules and overlapping colors.
    schedules = {}
    for split, count in (("train", 4), ("val", 2)):
        schedules[split] = [
            [(round(rng.uniform(-8, 8), 3),
              (rng.randint(-2, 2), rng.randint(-2, 2)),
              PALETTE[(family + variant + (2 if split == "val" else 0)) % 4])
             for variant in range(count)]
            for family in range(3)
        ]
    for split in ("train", "val", "test"):
        for label in ("cat", "dog"):
            count = 12 if split == "train" else 6
            per_family = 4 if split == "train" else 2
            for index in range(count):
                if split == "test":
                    family = index
                    angle, shift = 0.0, (0, 0)
                    coat = PALETTE[index % len(PALETTE)]
                    style = TEST_STYLES[index]
                    family_id = f"{label}_independent_{family + 1}"
                    image = independent_drawing(label, index, coat)
                else:
                    family, variant = divmod(index, per_family)
                    angle, shift, coat = schedules[split][family][variant]
                    style = ("seated mascot", "large-head portrait",
                             "compact seated mascot")[family]
                    family_id = f"{label}_familiar_{family + 1}"
                    image = familiar_drawing(label, family, coat)
                image = transform(image, angle, shift)
                path = ROOT / "pic" / split / f"{label}_{index + 1:02d}.png"
                image.save(path)
                samples.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "split": split, "label": label, "family_id": family_id,
                    "style": style, "angle": angle, "color": coat,
                    "shift": list(shift),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    if len({sample["sha256"] for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate image hashes detected; dataset is not valid.")
    manifest = {"image_size": [SIZE, SIZE], "seed": SEED, "note": NOTE,
                "samples": samples}
    (ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    contact_sheet(samples)
    print("Generated 48 fixed RGB drawings: 24 train, 12 validation, 12 test.")
    print(NOTE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only explicitly named generator outputs.")
    generate(parser.parse_args().overwrite)
