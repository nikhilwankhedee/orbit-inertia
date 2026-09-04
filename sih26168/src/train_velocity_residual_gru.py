#!/usr/bin/env python3
"""
Velocity Residual GRU — Training Script (SIH26168)
===================================================

Trains a 1-layer GRU (hidden=32) to predict 2D EN velocity residuals
relative to classical A0 DR. Model learns Δv = v_ref_EN − v_dr_EN.

Usage (Kaggle):
  python train_velocity_residual_gru.py --data-root /kaggle/input/sih26168 \
      --context-len 20 --stride 5 --epochs 200 --batch-size 256

Outputs (in outputs/ml/):
  - best_model.pt          PyTorch checkpoint
  - normalization.npz      Z-score stats (from ml_common)
  - training_log.csv       Per-epoch train/val loss
  - training_curves.png    Loss curves plot
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import ml_common as mc

OUT_DIR = mc.OUT_DIR


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
class VelocityResidualGRU(nn.Module):
    """1-layer GRU predicting 2D EN velocity residual.

    Input:  (batch, context_len, n_features)
    Output: (batch, 2)  — [delta_ve, delta_vn] in normalized space
    """

    def __init__(
        self,
        n_features: int = 16,
        hidden_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, context_len, n_features)."""
        _, h_n = self.gru(x)           # h_n: (n_layers, batch, hidden)
        last_h = h_n[-1]               # (batch, hidden) — last layer
        out = self.fc(last_h)          # (batch, 2)
        return out


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for feats, tgts in loader:
        feats, tgts = feats.to(device), tgts.to(device)
        optimizer.zero_grad()
        pred = model(feats)
        loss = criterion(pred, tgts)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        n_samples += feats.size(0)

    return total_loss / n_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for feats, tgts in loader:
            feats, tgts = feats.to(device), tgts.to(device)
            pred = model(feats)
            loss = criterion(pred, tgts)
            total_loss += loss.item() * feats.size(0)
            n_samples += feats.size(0)

    return total_loss / n_samples


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    patience=20,
    min_delta=1e-5,
):
    """Full training loop with early stopping. Returns training history dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    print(f"\n  Training config:")
    print(f"    Epochs: {epochs}, LR: {lr}, Weight decay: {weight_decay}")
    print(f"    Patience: {patience}, Device: {device}")
    print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"    Model params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "n_features": model.n_features,
                "hidden_size": model.hidden_size,
                "n_layers": model.n_layers,
                "config": {
                    "n_features": model.n_features,
                    "hidden_size": model.hidden_size,
                    "n_layers": model.n_layers,
                },
            }, OUT_DIR / "best_model.pt")
        else:
            wait += 1

        scheduler.step(val_loss)

        # Logging (every 10 epochs or first/last)
        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | "
                  f"lr={current_lr:.1e} | best@{best_epoch} ({best_val_loss:.6f}) | "
                  f"{elapsed:.0f}s")

        if wait >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - t0
    print(f"\n  Training complete: {epoch} epochs, {total_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    return history


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
def plot_training_curves(history: dict):
    """Plot train/val loss and learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], "b-", alpha=0.7, label="Train")
    ax.plot(history["epoch"], history["val_loss"], "r-", alpha=0.7, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1]
    ax.plot(history["epoch"], history["lr"], "k-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Velocity Residual GRU — Training Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'training_curves.png'}")


def save_training_log(history: dict):
    """Save training history to CSV."""
    path = OUT_DIR / "training_log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({
                "epoch": history["epoch"][i],
                "train_loss": history["train_loss"][i],
                "val_loss": history["val_loss"][i],
                "lr": history["lr"][i],
            })
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train velocity residual GRU")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root (e.g. /kaggle/input/sih26168)")
    parser.add_argument("--context-len", type=int, default=20,
                        help="Context window length in samples (default: 20 = 2s)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding window stride (default: 5)")
    parser.add_argument("--hidden-size", type=int, default=32,
                        help="GRU hidden size (default: 32)")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="GRU layers (default: 1)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override data path if specified
    if args.data_root:
        mc.DATA_FILE = Path(args.data_root) / "processed" / "S4_synced.csv"
        if not mc.DATA_FILE.exists():
            # Try alternative layout
            mc.DATA_FILE = Path(args.data_root) / "S4_synced.csv"

    print("=" * 70)
    print("VELOCITY RESIDUAL GRU — TRAINING")
    print("=" * 70)
    print(f"  Data: {mc.DATA_FILE}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Prepare data
    result = mc.prepare_ml_data(
        context_len=args.context_len,
        stride=args.stride,
    )

    train_dataset = result["train_dataset"]
    val_dataset = result["val_dataset"]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Build model
    model = VelocityResidualGRU(
        n_features=mc.N_FEATURES,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {mc.N_FEATURES} features → GRU(hidden={args.hidden_size}, "
          f"layers={args.n_layers}) → 2 outputs")
    print(f"  Parameters: {param_count:,}")

    # Train
    history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # Save artifacts
    save_training_log(history)
    plot_training_curves(history)

    # Save config
    config = {
        "n_features": mc.N_FEATURES,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "context_len": args.context_len,
        "stride": args.stride,
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": min(history["val_loss"]),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seed": args.seed,
    }
    config_path = OUT_DIR / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best model: {OUT_DIR / 'best_model.pt'}")
    print(f"  Normalization: {OUT_DIR / 'normalization.npz'}")
    print(f"  Next: python evaluate_velocity_residual_gru.py")


if __name__ == "__main__":
    main()
