#!/usr/bin/env python3
"""
Kaggle one-shot runner: train + evaluate residual GRUs (SIH26168)
=================================================================

Clones (already done by notebook) the orbit-inertia repo to the working dir,
then trains and evaluates GRU variants in one command.

Usage (Kaggle notebook cell):
  !git clone --depth 1 https://github.com/nikhilwankhedee/orbit-inertia.git
  !python orbit-inertia/sih26168/src/run_all.py \
      --data-root /kaggle/input/<your-dataset>

Variants:
  --variant v0   V0 (16-feature, vehicle-CAN) train + teacher + recursive eval
  --variant v02  V0.2 (9-feature, phone-only) train + recursive + oracle eval
  --variant all  both (default: v0 for backward compatibility)

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
    parser.add_argument("--variant", type=str, default="v0",
                        choices=["v0", "v02", "all"],
                        help="Pipeline variant to run (default: v0)")
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
    print(f"  Variant:   {args.variant}")
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

    def run(step_name, script, extra):
        print("\n" + "=" * 70)
        print(f"STEP {step_name}")
        print("=" * 70)
        cmd = [sys.executable, str(src_dir / script),
               "--data-root", data_dir, *extra,
               "--context-len", str(args.context_len),
               "--seed", str(args.seed)]
        print(f"  Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=repo_dir.parent)
        if r.returncode != 0:
            print(f"\n*** {step_name} failed. Aborting.")
            sys.exit(1)

    common = ["--hidden-size", str(args.hidden_size),
              "--epochs", str(args.epochs),
              "--batch-size", str(args.batch_size)]
    train_extra = [*common, "--stride", str(args.stride)]

    if args.variant in ("v0", "all"):
        run("1: TRAIN (V0 16-feature)",
            "train_velocity_residual_gru.py", train_extra)
        run("2: EVALUATE (V0 teacher)",
            "evaluate_velocity_residual_gru.py",
            ["--model-path", str(repo_dir / "outputs" / "ml" / "best_model.pt"),
             "--norm-path", str(repo_dir / "outputs" / "ml" / "normalization.npz")])
        run("3: RECURSIVE EVALUATION (V0.1 audit)",
            "evaluate_recursive_gru.py",
            ["--model-path", str(repo_dir / "outputs" / "ml" / "best_model.pt"),
             "--norm-path", str(repo_dir / "outputs" / "ml" / "normalization.npz")])

    if args.variant in ("v02", "all"):
        v02_out = str(repo_dir / "outputs" / "ml_v02")
        run("4: TRAIN (V0.2 phone-only)",
            "train_velocity_residual_gru_v02.py",
            [*train_extra, "--out-dir", v02_out])
        run("5: EVALUATE (V0.2 recursive + oracle)",
            "evaluate_velocity_residual_gru_v02.py",
            ["--model-path", str(Path(v02_out) / "best_model.pt"),
             "--norm-path", str(Path(v02_out) / "normalization.npz"),
             "--out-dir", v02_out])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("  V0  outputs in outputs/ml/ and outputs/ml/recursive/")
    print("  V0.2 outputs in outputs/ml_v02/")


if __name__ == "__main__":
    main()
