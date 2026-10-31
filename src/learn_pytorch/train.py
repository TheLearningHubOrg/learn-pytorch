"""A plain, explicit PyTorch training loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from learn_pytorch.data import BlobDataset, make_loaders
from learn_pytorch.model import MLPClassifier


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-2
    weight_decay: float = 1e-4
    n_samples: int = 1024
    n_classes: int = 3
    seed: int = 0
    device: str = "auto"


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def pick_device(preference: str = "auto") -> torch.device:
    """Return CUDA, then Apple MPS, then CPU, unless a device is named."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        # 1. Forward pass and loss.
        logits = model(features)
        loss = loss_fn(logits, labels)

        # 2. Clear old gradients, backpropagate, update weights.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item() * labels.shape[0]
        count += labels.shape[0]
    return total / count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total, correct, count = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        total += loss_fn(logits, labels).item() * labels.shape[0]
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.shape[0]
    return total / count, correct / count


def fit(cfg: TrainConfig | None = None) -> tuple[nn.Module, History]:
    """Train an MLP on synthetic blobs and return the model and its history."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    dataset = BlobDataset(n_samples=cfg.n_samples, n_classes=cfg.n_classes, seed=cfg.seed)
    train_loader, val_loader = make_loaders(dataset, batch_size=cfg.batch_size, seed=cfg.seed)

    model = MLPClassifier(dataset.n_features, dataset.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = History()
    for _ in range(cfg.epochs):
        history.train_loss.append(train_one_epoch(model, train_loader, loss_fn, optimizer, device))
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
    return model, history
