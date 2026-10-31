"""Companion code for the Learn PyTorch guide."""

from learn_pytorch.data import BlobDataset, make_loaders
from learn_pytorch.model import MLPClassifier, TinyCNN
from learn_pytorch.train import TrainConfig, evaluate, fit, pick_device, train_one_epoch

__all__ = [
    "BlobDataset",
    "MLPClassifier",
    "TinyCNN",
    "TrainConfig",
    "evaluate",
    "fit",
    "make_loaders",
    "pick_device",
    "train_one_epoch",
]

__version__ = "1.0.0"
