#!/usr/bin/env python3
"""
Kaggle one-shot runner: train + evaluate velocity residual GRU (SIH26168)
=========================================================================

Clones (already done by notebook) the orbit-inertia repo to the working dir,
then trains and evaluates the GRU in one command.

Usage (Kaggle notebook cell):
  !git clone --depth 1 https://github.com/nikhilwankhedee/orbit-inertia.git
  !python orbit-inertia/sih26168/src/run_all.py \
      --data-root /kaggle/input/<your-dataset>

Expects the S4 dataset mounted at /kaggle/input/<your-dataset>/S4_synced.csv
(or <your-dataset>/processed/S4_synced.csv).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate in one shot")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Kaggle dataset mount path, e.g. /kaggle/input/sih26168")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    repo_dir = src_dir.parent  # .../sih26168 (outputs live here)

    print("=" * 70)
    print("ORBIT-INERTIA KAGGLE RUNNER")
    print("=" * 70)
    print(f"  Data root: {args.data_root}")
    print(f"  Source:    {src_dir}")

    # Locate the CSV. Accept either the dataset dir OR the CSV file itself
    # as --data-root (handles pasting the Kaggle mount path to the .csv).
    raw = Path(args.data_root)
    if raw.is_file() and raw.name.lower() == "s4_synced.csv":
        csv_path = raw
    else:
        candidates = [
            raw / "S4_synced.csv",
            raw / "processed" / "S4_synced.csv",
        ]
        csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        print("\n*** Could not find S4_synced.csv under --data-root")
        print("    Tried:")
        for c in candidates:
            print(f"      {c}")
        print("\n    Tip: pass the DATASET DIRECTORY (the mount), not the CSV file path.")
        print("    e.g. --data-root /kaggle/input/datasets/nikhilwankhedee/v0dataset")
        sys.exit(1)
    print(f"  Dataset:   {csv_path}")

    # Resolve the dataset directory to pass to child scripts
    data_dir = str(csv_path.parent)

    # ── Step 1: Train ──
    print("\n" + "=" * 70)
    print("STEP 1: TRAIN")
    print("=" * 70)
    train_cmd = [
        sys.executable, str(src_dir / "train_velocity_residual_gru.py"),
        "--data-root", data_dir,
        "--context-len", str(args.context_len),
        "--stride", str(args.stride),
        "--hidden-size", str(args.hidden_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    print(f"  Running: {' '.join(train_cmd)}")
    r = subprocess.run(train_cmd, cwd=src_dir.parents[1])
    if r.returncode != 0:
        print("\n*** Training failed. Aborting.")
        sys.exit(1)

    # ── Step 2: Evaluate ──
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATE")
    print("=" * 70)
    eval_cmd = [
        sys.executable, str(src_dir / "evaluate_velocity_residual_gru.py"),
        "--data-root", data_dir,
        "--model-path", str(repo_dir / "outputs" / "ml" / "best_model.pt"),
        "--norm-path", str(repo_dir / "outputs" / "ml" / "normalization.npz"),
        "--context-len", str(args.context_len),
        "--seed", str(args.seed),
    ]
    r = subprocess.run(eval_cmd, cwd=src_dir.parents[1])
    if r.returncode != 0:
        print("\n*** Evaluation failed.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("  Outputs in outputs/ml/")


if __name__ == "__main__":
    main()
