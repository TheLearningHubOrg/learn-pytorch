"""Synthetic, CPU-friendly datasets and DataLoader helpers."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BlobDataset(Dataset):
    """Gaussian clusters placed evenly on a circle, one cluster per class.

    Placing the centres on a circle (instead of sampling them) keeps the
    classes separable for every seed, so tests are deterministic.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        n_classes: int = 3,
        n_features: int = 2,
        radius: float = 3.0,
        spread: float = 1.0,
        seed: int = 0,
    ) -> None:
        if n_features < 2:
            raise ValueError("n_features must be at least 2")
        g = torch.Generator().manual_seed(seed)

        # 1. One centre per class on a circle in the first two dimensions.
        angles = torch.arange(n_classes, dtype=torch.float32) * (2 * math.pi / n_classes)
        centres = torch.zeros(n_classes, n_features)
        centres[:, 0] = radius * torch.cos(angles)
        centres[:, 1] = radius * torch.sin(angles)

        # 2. Sample labels, then jitter each point around its class centre.
        self.labels = torch.randint(0, n_classes, (n_samples,), generator=g)
        noise = spread * torch.randn(n_samples, n_features, generator=g)
        self.features = centres[self.labels] + noise
        self.n_classes = n_classes
        self.n_features = n_features

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def make_loaders(
    dataset: Dataset,
    batch_size: int = 64,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split a dataset into train/validation loaders with a fixed seed."""
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader
