"""Small nn.Module models used throughout the guide."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Two hidden layers with ReLU and dropout; returns raw logits."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyCNN(nn.Module):
    """A two-block CNN for 1x28x28 inputs (MNIST-sized)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # -> 8 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 8 x 14 x 14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> 16 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> 16 x 7 x 7
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
