from math import isclose
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DAY2 = Path(__file__).resolve().parent
SOURCE = DAY2.parent / "day1" / "png"


def main():

    # Step 1: Read images.
    images = sorted(SOURCE.glob("*.bmp"))

    # Step 2: Split into train, validation, and test sets.
    train, remaining = train_test_split(
        images, train_size=TRAIN_RATIO, random_state=42
    )
    val, test = train_test_split(
        remaining, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42
    )

    # Step 3: Delete old output folders, recreate them, and copy images from day1 png folder.
    for name, files in (("train", train), ("val", val), ("test", test)):
        folder = DAY2 / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        for image in files:
            shutil.copy2(image, folder / image.name)
        print(f"{name}: {len(files)} images")


if __name__ == "__main__":
    main()
