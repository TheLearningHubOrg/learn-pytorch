"""Command line entry point: python -m learn_pytorch --epochs 20 --out artifacts"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from learn_pytorch.export import export_program, save_checkpoint
from learn_pytorch.train import TrainConfig, fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Learn PyTorch sample model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, device=args.device)
    model, history = fit(cfg)
    rows = zip(history.train_loss, history.val_loss, history.val_acc)
    for epoch, (tl, vl, va) in enumerate(rows, start=1):
        print(f"epoch {epoch:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    ckpt, prog = args.out / "model.pt", args.out / "model.pt2"
    save_checkpoint(model, ckpt)
    export_program(model, torch.randn(4, 2), prog)
    print(f"saved {ckpt} and {prog}")


if __name__ == "__main__":
    main()
