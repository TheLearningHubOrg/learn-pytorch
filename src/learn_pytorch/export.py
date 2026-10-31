"""Checkpointing and torch.export helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save only the state_dict (tensors), never the pickled module."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path) -> nn.Module:
    """Load weights into an already-constructed model.

    weights_only=True refuses to unpickle arbitrary Python objects.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def export_program(model: nn.Module, example: torch.Tensor, path: str | Path) -> None:
    """Capture a graph with torch.export and save it as a .pt2 archive.

    The batch dimension is marked dynamic so the exported program accepts
    any batch size, not just the one used for tracing.
    """
    model = model.eval()
    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.export.export(model, (example,), dynamic_shapes={"x": {0: batch}})
    torch.export.save(program, path)


def load_program(path: str | Path) -> nn.Module:
    """Load a .pt2 archive and return a callable module."""
    return torch.export.load(path).module()
