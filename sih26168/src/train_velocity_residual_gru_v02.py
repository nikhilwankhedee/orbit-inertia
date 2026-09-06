#!/usr/bin/env python3
"""
V0.2 Deployment-Valid Velocity Residual GRU — Training Script (SIH26168)
========================================================================

Trains a small 1-layer GRU (hidden=32, ~4.1k params) to predict the 2D EN
velocity residual relative to the classical A0 DR, using ONLY
deployment-available inputs: 7 measured phone-IMU channels + 2 A0
internal-nav-state channels (nav_speed, nav_heading).

This is the deployment-valid retrain demanded by the V0.1 verdict (CASE B):
the V0 16-feature checkpoint read vehicle-CAN/GNSS reference channels that
do not exist in a GNSS-denied phone deployment and its recursive rollout
collapsed to +8.6% avg.

Usage (Kaggle GPU):
  python train_velocity_residual_gru_v02.py --data-root /kaggle/input/sih26168 \
      --context-len 20 --stride 5 --epochs 200 --batch-size 256 --seed 42

Outputs (outputs/ml_v02/):
  best_model.pt       PyTorch checkpoint
  normalization.npz   Z-score stats (fit on training data only)
  train_config.json   Reproducible config
  training_log.csv    Per-epoch train/val loss + lr
  training_curves.png Loss curves plot
  training_report.txt Training summary report

NOTE: this script writes to a configurable output dir (--out-dir) so smoke
tests can run isolated without polluting outputs/ml_v02/.
"""

import argparse
import csv
import json
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

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import ml_common as mc
import ml_v02_common as v2c

OUT_DIR = v2c.OUT_DIR


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────
class VelocityResidualGRU(nn.Module):
    """1-layer GRU predicting 2D EN velocity residual (V0.2, 9 features).

    Input:  (batch, context_len, 9)
    Output: (batch, 2)  — [delta_ve, delta_vn] in normalized space
    """

    def __init__(
        self,
        n_features: int = 9,
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
        _, h_n = self.gru(x)   # h_n: (n_layers, batch, hidden)
        last_h = h_n[-1]       # (batch, hidden)
        return self.fc(last_h)  # (batch, 2)


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
    loss_name="mse",
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
    criterion = nn.MSELoss() if loss_name == "mse" else nn.SmoothL1Loss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    print(f"\n  Training config:")
    print(f"    Epochs: {epochs}, LR: {lr}, Weight decay: {weight_decay}")
    print(f"    Loss: {loss_name}, Patience: {patience}, Device: {device}")
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

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
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
# PLOTS / LOGS
# ──────────────────────────────────────────────────────────────
def plot_training_curves(history: dict):
    """Plot train/val loss and learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], "b-", alpha=0.7, label="Train")
    ax.plot(history["epoch"], history["val_loss"], "r-", alpha=0.7, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss (V0.2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["lr"], "k-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle("V0.2 Velocity Residual GRU — Training Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'training_curves.png'}")


def save_training_log(history: dict):
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


def generate_training_report(
    history: dict,
    config: dict,
    model,
    train_count: int,
    val_count: int,
):
    lines = []
    add = lines.append

    add("=" * 78)
    add("V0.2 DEPLOYMENT-VALID TRAINING REPORT")
    add("=" * 78)
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Model: 1-layer GRU, hidden={model.hidden_size}, " 
        f"params={sum(p.numel() for p in model.parameters()):,}")
    add(f"Features: {model.n_features} (7 phone-IMU + nav_speed + nav_heading)")
    add(f"Target: dv = v_reference_EN - v_classical_A0_EN (m/s)")
    add(f"Context window: {config['context_len']} samples (~2 s at 10 Hz), "
        f"stride {config['stride']}")
    add(f"Split: Train Seg0+Seg1 ({train_count}), Val Seg2 ({val_count}), "
        f"Test Seg3 (held out)")
    add(f"Normalization: z-score fit on training windows only")
    add(f"Loss: {config['loss']}")
    add(f"Optimizer: Adam lr={config['lr']}, weight_decay={config['weight_decay']}")
    add(f"Early stopping: patience={config['patience']}")
    add(f"Seed: {config['seed']}")
    add(f"Epochs trained: {config['epochs_trained']} "
        f"(best val loss {config['best_val_loss']:.6f} @ epoch {config['best_epoch']})")
    add("")
    add("Reproducibility note: full GPU training is the official V0.2 run;")
    add("this report is regenerated on every training execution.")
    add("=" * 78)

    report = "\n".join(lines)
    path = OUT_DIR / "training_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved: {path}")
    return report


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train V0.2 deployment-valid GRU")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root (e.g. /kaggle/input/sih26168)")
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"])
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir override (smoke tests only)")
    return parser.parse_args()


def main():
    args = parse_args()
    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        v2c.OUT_DIR = OUT_DIR
        mc.OUT_DIR = OUT_DIR

    if args.data_root:
        mc.DATA_FILE = Path(args.data_root) / "processed" / "S4_synced.csv"
        if not mc.DATA_FILE.exists():
            mc.DATA_FILE = Path(args.data_root) / "S4_synced.csv"

    print("=" * 70)
    print("V0.2 DEPLOYMENT-VALID VELOCITY RESIDUAL GRU — TRAINING")
    print("=" * 70)
    print(f"  Data: {mc.DATA_FILE}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    result = v2c.prepare_v02_data(
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

    model = VelocityResidualGRU(
        n_features=v2c.V02_N_FEATURES,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {v2c.V02_N_FEATURES} features -> GRU(hidden={args.hidden_size}, "
          f"layers={args.n_layers}) -> 2 outputs")
    print(f"  Parameters: {param_count:,}")

    history = train(
        model, train_loader, val_loader, device,
        loss_name=args.loss,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    save_training_log(history)
    plot_training_curves(history)

    config = {
        "n_features": v2c.V02_N_FEATURES,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "context_len": args.context_len,
        "stride": args.stride,
        "epochs_trained": len(history["epoch"]),
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history["val_loss"])),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "loss": args.loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
    }
    config_path = OUT_DIR / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    generate_training_report(history, config, model,
                             len(train_dataset), len(val_dataset))

    print("\n" + "=" * 70)
    print("V0.2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best model: {OUT_DIR / 'best_model.pt'}")
    print(f"  Normalization: {OUT_DIR / 'normalization.npz'}")
    print(f"  Next: python evaluate_velocity_residual_gru_v02.py")


if __name__ == "__main__":
    main()