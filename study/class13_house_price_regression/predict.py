import argparse
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .train_model import CLASS13, create_model, predict
else:
    from train_model import CLASS13, create_model, predict


def predict_house(area_sqm, rooms, checkpoint_path):
    if not np.isfinite(area_sqm) or area_sqm <= 0 or type(rooms) is not int or rooms <= 0:
        raise ValueError("Area must be finite and positive; rooms must be a positive integer.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_model()
    model.load_state_dict(saved["model_state_dict"])
    features = np.array([[area_sqm, rooms]], dtype=np.float32)
    result = float(predict(model, features, saved["scaling"])[0])
    outside = bool(((features[0] < saved["feature_min"]) | (features[0] > saved["feature_max"])).any())
    return result, outside


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a synthetic house price; not a real property valuation.")
    parser.add_argument("--area", type=float, default=120)
    parser.add_argument("--rooms", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoints = sorted((CLASS13 / "runs").glob("*/weights.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint found. Run train_model.py first.")
        checkpoint = checkpoints[-1]
    price, outside = predict_house(args.area, args.rooms, checkpoint)
    print(f"Area: {args.area:g} square meters | Rooms: {args.rooms}")
    print(f"Predicted price: ${price * 1000:,.0f} (synthetic teaching example)")
    if outside:
        print("WARNING: Inputs fall outside the training ranges; prediction may be unreliable.")
