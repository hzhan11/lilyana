from math import isclose
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

CLASS2 = Path(__file__).resolve().parent
SOURCE = CLASS2.parent / "class1_dataset_prepare" / "png"


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

    # Step 3: Recreate the output folders and copy images from class1_dataset_prepare.
    for name, files in (("train", train), ("val", val), ("test", test)):
        folder = CLASS2 / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        for image in files:
            shutil.copy2(image, folder / image.name)
        print(f"{name}: {len(files)} images")


if __name__ == "__main__":
    main()
