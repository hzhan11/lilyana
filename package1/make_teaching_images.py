"""Create attributed teaching illustrations from the local BUSI image-mask pairs."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    package_dir = Path(__file__).resolve().parent
    data_dir = package_dir / "data" / "BUSI" / "BUSI"
    output_dir = package_dir / "teaching_assets"
    output_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 22)
    small_font = ImageFont.truetype("arial.ttf", 17)

    for name in ("benign_148", "malignant_1"):
        with Image.open(data_dir / "images" / f"{name}.png") as source:
            image = source.convert("RGB")
        with Image.open(data_dir / "labels" / f"{name}.png") as source:
            mask_values = np.asarray(source.convert("L"))
        if mask_values.shape != (image.height, image.width):
            raise ValueError(f"Image and mask dimensions differ for {name}.")

        lesion = mask_values > 0
        mask = Image.fromarray(lesion.astype(np.uint8) * 255).convert("RGB")
        overlay_values = np.asarray(image).copy()
        overlay_values[lesion] = (
            0.55 * overlay_values[lesion] + 0.45 * np.array([255, 90, 40])
        ).astype(np.uint8)
        overlay = Image.fromarray(overlay_values)

        canvas = Image.new("RGB", (1260, 680), "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        panels = (
            (image, "Ultrasound input"),
            (mask, "Reference mask (NOT a prediction)"),
            (overlay, "Reference region overlaid"),
        )
        for index, (panel, title) in enumerate(panels):
            left = 20 + index * 420
            draw.text((left, 20), title, fill="#0f172a", font=font)
            fitted = ImageOps.contain(panel, (400, 560))
            canvas.paste(
                fitted,
                (left + (400 - fitted.width) // 2, 65 + (560 - fitted.height) // 2),
            )
        draw.text(
            (20, 637),
            f"BUSI subset: {name}. White = nonzero reference-mask pixels; "
            "orange = reference annotation.",
            fill="#334155",
            font=small_font,
        )
        canvas.save(output_dir / f"{name}_reference.png")


if __name__ == "__main__":
    main()
