"""Shared label validation for the viewer and training program."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image


ANIMALS = ("cat", "dog", "pig", "rabbit", "bear", "fox")
UNSEEN_ANIMALS = ("elephant", "lion", "turtle", "bird", "fish", "cow", "horse", "sheep", "zebra", "giraffe")


@dataclass(frozen=True)
class Sample:
    path: Path
    animal: str
    center: tuple[float, float]
    size: tuple[int, int]


def load_samples(label_file):
    label_file = Path(label_file)
    samples, filenames = [], set()
    for number, line in enumerate(label_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        context = f"{label_file.name}, line {number}"
        parts = line.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"{context}: expected filename,animal,[x,y].")
        filename, animal, coordinates = (part.strip() for part in parts)
        if (
            not filename or "/" in filename or "\\" in filename
            or Path(filename).name != filename or Path(filename).suffix.lower() != ".png"
        ):
            raise ValueError(f"{context}: expected a PNG filename without a directory.")
        if filename.casefold() in filenames:
            raise ValueError(f"{context}: duplicate image filename {filename}.")
        if animal not in (*ANIMALS, *UNSEEN_ANIMALS):
            raise ValueError(
                f"{context}: unknown animal {animal!r}; expected {', '.join((*ANIMALS, *UNSEEN_ANIMALS))}."
            )
        try:
            center = json.loads(coordinates)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid center; expected [x,y].") from error
        if (
            not isinstance(center, list) or len(center) != 2
            or any(
                not (type(value) is int or (type(value) is float and math.isfinite(value)))
                for value in center
            )
        ):
            raise ValueError(f"{context}: center must contain two finite numbers.")
        path = label_file.parent / filename
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"{context}: {filename} is not a PNG image.")
                image.load()
                width, height = image.size
        except OSError as error:
            raise ValueError(f"{context}: cannot read {filename}: {error}") from error
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            raise ValueError(f"{context}: center {center} is outside the {width}x{height} image.")
        samples.append(Sample(path, animal, (center[0], center[1]), (width, height)))
        filenames.add(filename.casefold())
    if not samples:
        raise ValueError(f"{label_file}: no image labels found.")
    return samples
