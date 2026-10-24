"""A shared CNN with classification and pixel-coordinate regression heads."""

import torch
from torch import nn


IMAGE_SIZE = 200
CENTER_SCALE = IMAGE_SIZE - 1


class AnimalCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2.")
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # Keep spatial information so the center head can learn position.
            nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)
        self.center_head = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())

    def forward(self, images):
        """Accept float RGBA (batch, 200, 200, 4) in [0,1]; return logits, pixel [x,y]."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 4):
            raise ValueError("Input must have shape (batch, 200, 200, 4).")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Input must contain finite floating-point RGBA values.")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("RGBA values must be normalized to [0,1].")
        features = self.features(images.permute(0, 3, 1, 2).contiguous())
        return self.classifier(features), self.center_head(features) * CENTER_SCALE
