#!/usr/bin/env python3
"""
Velocity Residual GRU — Evaluation Script (SIH26168)
=====================================================

Evaluates the trained GRU model on simulated GNSS blackouts using
recursive rollout. Compares against classical A0 baseline.

Usage (Kaggle):
  python evaluate_velocity_residual_gru.py \
      --model-path outputs/ml/best_model.pt \
      --norm-path outputs/ml/normalization.npz

Requires: outputs/ml/best_model.pt, outputs/ml/normalization.npz
           (produced by train_velocity_residual_gru.py)
           plus data infrastructure from ml_common.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import ml_common as mc
from ml_common import (
    N_FEATURES, FEATURE_KEYS, FEATURE_SPECS, META_SPECS,
    ll2enu, propagate_classical_dr_velocity, compute_reference_en_velocity,
)

OUT_DIR = mc.OUT_DIR


# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
def load_model(model_path: Path, device: torch.device):
    """Load trained GRU model from checkpoint."""
    from train_velocity_residual_gru import VelocityResidualGRU

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = VelocityResidualGRU(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Model loaded: {model_path}")
    print(f"    Config: {config}")
    print(f"    Checkpoint epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.6f}")
    print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, config


# ──────────────────────────────────────────────────────────────
# ML-ASSISTED DR PROPAGATION
# ──────────────────────────────────────────────────────────────
def run_ml_dr(
    model: torch.nn.Module,
    data: dict,
    features: np.ndarray,
    norm: dict,
    window: dict,
    context_len: int,
    device: torch.device,
) -> dict:
    """ML-corrected DR: A0 kinematic + GRU velocity residual correction.

    At each step:
      1. Classical A0 propagates heading (gyro) + constant speed
      2. GRU predicts velocity residual from context window of sensor features
      3. Corrected velocity = classical + denorm(residual_pred) applied in EN

    Causal: uses only sensor features available at each timestep.

    Returns dict with headings, positions_east/north, velocities_east/north.
    """
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]
    n_eval = i2 - i0

    # ── INITIALIZATION ──
    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]
    east0, north0, _ = ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0,
    )
    init_heading = np.radians(data["veh_heading"][i0])
    init_speed = data["veh_velocity"][i0] / 3.6

    # ── CLASSICAL DR HEADINGS (same as A0) ──
    headings, cls_ve, cls_vn = propagate_classical_dr_velocity(
        data["gyro_pitch"][i0:i2],
        data["sync_time"][i0:i2],
        init_heading,
        init_speed,
        n_eval,
    )

    # ── ML CORRECTION (batched inference) ──
    feat_mean = norm["feat_mean"]
    feat_std = norm["feat_std"]
    tgt_mean = norm["tgt_mean"]
    tgt_std = norm["tgt_std"]

    # Build all context windows at once (vectorized, no Python loop)
    all_ctx = np.zeros((n_eval, context_len, N_FEATURES), dtype=np.float32)
    for k in range(n_eval):
        ctx_start = max(0, k - context_len + 1)
        ctx_end = k + 1
        feat_ctx = features[i0 + ctx_start: i0 + ctx_end]
        pad_len = context_len - len(feat_ctx)
        if pad_len > 0:
            feat_ctx = np.concatenate([np.zeros((pad_len, N_FEATURES)), feat_ctx], axis=0)
        all_ctx[k] = (feat_ctx - feat_mean) / feat_std

    # Single batched forward pass
    with torch.no_grad():
        x = torch.tensor(all_ctx, dtype=torch.float32).to(device)
        preds = model(x).cpu().numpy()  # (n_eval, 2) normalized

    # Denormalize
    delta_v = preds * tgt_std + tgt_mean  # (n_eval, 2) m/s

    ve = cls_ve + delta_v[:, 0]
    vn = cls_vn + delta_v[:, 1]

    # Propagate position (vectorized)
    dt_arr = np.diff(data["sync_time"][i0:i2], prepend=data["sync_time"][i0] - mc.DT_NOMINAL)
    dt_arr = np.clip(dt_arr, 0, mc.DT_MAX_GAP)
    e_pos = np.cumsum(ve * dt_arr) + east0
    n_pos = np.cumsum(vn * dt_arr) + north0

    return {
        "headings": headings,
        "positions_east": e_pos,
        "positions_north": n_pos,
        "velocities_east": ve,
        "velocities_north": vn,
        "init_heading_deg": np.degrees(init_heading),
        "init_speed_ms": init_speed,
        "init_east": east0,
        "init_north": north0,
        "ref_lat0": ref_lat0,
        "ref_lon0": ref_lon0,
    }


# ──────────────────────────────────────────────────────────────
# CLASSICAL A0 DR (for comparison)
# ──────────────────────────────────────────────────────────────
def run_classical_a0(data, window):
    """Reproduce A0 baseline DR from classical_dr_baseline.py."""
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]
    n_eval = i2 - i0

    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]
    east0, north0, _ = ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0,
    )
    heading0 = np.radians(data["veh_heading"][i0])
    speed0 = data["veh_velocity"][i0] / 3.6

    headings, ve, vn = propagate_classical_dr_velocity(
        data["gyro_pitch"][i0:i2],
        data["sync_time"][i0:i2],
        heading0,
        speed0,
        n_eval,
    )

    e_pos = np.zeros(n_eval)
    n_pos = np.zeros(n_eval)
    e_pos[0] = east0
    n_pos[0] = north0

    for k in range(1, n_eval):
        dt = data["sync_time"][i0 + k] - data["sync_time"][i0 + k - 1]
        if dt <= 0 or dt > mc.DT_MAX_GAP:
            e_pos[k] = e_pos[k - 1]
            n_pos[k] = n_pos[k - 1]
            continue
        e_pos[k] = e_pos[k - 1] + ve[k] * dt
        n_pos[k] = n_pos[k - 1] + vn[k] * dt

    return {
        "headings": headings,
        "positions_east": e_pos,
        "positions_north": n_pos,
        "velocities_east": ve,
        "velocities_north": vn,
        "init_heading_deg": np.degrees(heading0),
        "init_speed_ms": speed0,
        "init_east": east0,
        "init_north": north0,
        "ref_lat0": ref_lat0,
        "ref_lon0": ref_lon0,
    }


# ──────────────────────────────────────────────────────────────
# REFERENCE TRAJECTORY
# ──────────────────────────────────────────────────────────────
def get_reference_enu(data, i0, i2, ref_lat0, ref_lon0):
    n = i2 - i0
    ref_east = np.zeros(n)
    ref_north = np.zeros(n)
    ref_heading = np.zeros(n)

    for k in range(n):
        idx = i0 + k
        e, n_, _ = ll2enu(
            data["veh_lat"][idx], data["veh_lon"][idx], 0.0,
            ref_lat0, ref_lon0, 0.0,
        )
        ref_east[k] = e
        ref_north[k] = n_
        ref_heading[k] = data["veh_heading"][idx]

    return ref_east, ref_north, ref_heading


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_metrics(dr_east, dr_north, dr_heading,
                    ref_east, ref_north, ref_heading,
                    sync_time, blackout_end_rel):
    n = len(dr_east)
    t_since = sync_time[:n] - sync_time[0]

    de = dr_east - ref_east
    dn = dr_north - ref_north
    pos_error = np.sqrt(de ** 2 + dn ** 2)

    dh = np.degrees(dr_heading) - ref_heading
    dh = (dh + 180) % 360 - 180
    heading_error = np.abs(dh)

    results = {}
    for dur in [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]:
        mask = t_since >= dur
        if mask.any():
            idx = np.argmax(mask)
            results[dur] = {
                "t_s": t_since[idx],
                "pos_error_m": pos_error[idx],
                "heading_error_deg": heading_error[idx],
            }

    blackout_mask = t_since <= (t_since[blackout_end_rel] if blackout_end_rel < n else t_since[-1])
    if blackout_mask.sum() > 0:
        bm = blackout_mask
        results["blackout"] = {
            "mae_m": float(np.mean(pos_error[bm])),
            "rmse_m": float(np.sqrt(np.mean(pos_error[bm] ** 2))),
            "max_m": float(np.max(pos_error[bm])),
            "final_m": float(pos_error[blackout_end_rel]) if blackout_end_rel < n else float(pos_error[-1]),
            "heading_mae_deg": float(np.mean(heading_error[bm])),
            "heading_final_deg": float(heading_error[blackout_end_rel]) if blackout_end_rel < n else float(heading_error[-1]),
        }

    return {
        "t_since": t_since,
        "pos_error": pos_error,
        "heading_error": heading_error,
        "at_durations": results,
    }


def summarize(all_met):
    summary = {}
    for dur, met_list in all_met.items():
        blackout_metrics = []
        for met in met_list:
            if "blackout" in met["at_durations"]:
                blackout_metrics.append(met["at_durations"]["blackout"])
        if not blackout_metrics:
            continue
        keys = list(blackout_metrics[0].keys())
        avg = {}
        for k in keys:
            vals = [bm[k] for bm in blackout_metrics]
            avg[f"{k}_mean"] = float(np.mean(vals))
            avg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            avg[f"{k}_median"] = float(np.median(vals))
        avg["n_windows"] = len(blackout_metrics)
        summary[dur] = avg
    return summary


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
COLORS = {"A0": "#1f77b4", "ML": "#d62728"}


def plot_position_error_vs_time(all_met_a0, all_met_ml, duration):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, met in enumerate(all_met_a0):
        t = met["t_since"]
        err = met["pos_error"]
        ax.plot(t, err, alpha=0.5, color=COLORS["A0"], linewidth=0.8,
                label="A0" if i == 0 else None)

    for i, met in enumerate(all_met_ml):
        t = met["t_since"]
        err = met["pos_error"]
        ax.plot(t, err, alpha=0.5, color=COLORS["ML"], linewidth=0.8,
                label="ML-GRU" if i == 0 else None)

    ax.set_xlabel("Time since blackout start (s)", fontsize=12)
    ax.set_ylabel("Position error (m)", fontsize=12)
    ax.set_title(f"Position Error — {duration}s Blackout (A0 vs ML-GRU)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)

    outpath = OUT_DIR / f"ml_position_error_{duration}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_trajectory_example(ml_res, a0_res, ref_enu, window):
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    n_bo = i1 - i0
    n_total = window["eval_end_idx"] - i0

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Reference (full)
    ax.plot(ref_enu[0], ref_enu[1], "gray", alpha=0.3, linewidth=0.5,
            label="Reference (full)")
    # Reference blackout
    ax.plot(ref_enu[0][:n_bo], ref_enu[1][:n_bo], "b-", linewidth=2.5,
            alpha=0.7, label="Reference (blackout)")
    # A0
    ax.plot(a0_res["positions_east"][:n_total],
            a0_res["positions_north"][:n_total],
            color=COLORS["A0"], linewidth=1.5, alpha=0.8,
            label="A0 (classical)")
    # ML
    ax.plot(ml_res["positions_east"][:n_total],
            ml_res["positions_north"][:n_total],
            color=COLORS["ML"], linewidth=1.5, alpha=0.8,
            label="ML-GRU (corrected)")

    ax.plot(ref_enu[0][0], ref_enu[1][0], "go", markersize=10, label="Blackout start")
    if n_bo < len(ref_enu[0]):
        ax.plot(ref_enu[0][n_bo], ref_enu[1][n_bo], "r^", markersize=10,
                label="Blackout end")

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title(f"Trajectory: {window['duration_s']}s Blackout — A0 vs ML-GRU", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    outpath = OUT_DIR / f"ml_trajectory_{window['duration_s']}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_error_vs_blackout_duration(summary_a0, summary_ml):
    durations = sorted(set(summary_a0.keys()) & set(summary_ml.keys()))
    if not durations:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(durations))
    width = 0.35

    for ax, metric, title in zip(
        axes,
        ["mae_m_mean", "rmse_m_mean", "max_m_mean"],
        ["Mean Absolute Error", "Root Mean Square Error", "Maximum Error"],
    ):
        vals_a0 = [summary_a0[d].get(metric, 0) for d in durations]
        vals_ml = [summary_ml[d].get(metric, 0) for d in durations]

        ax.bar(x - width / 2, vals_a0, width, color=COLORS["A0"], alpha=0.8, label="A0")
        ax.bar(x + width / 2, vals_ml, width, color=COLORS["ML"], alpha=0.8, label="ML-GRU")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}s" for d in durations])
        ax.set_ylabel("Error (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("DR Error vs Blackout Duration — A0 vs ML-GRU", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = OUT_DIR / "ml_error_vs_duration.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_improvement_bar(summary_a0, summary_ml):
    """Bar chart showing absolute and percentage improvement."""
    durations = sorted(set(summary_a0.keys()) & set(summary_ml.keys()))
    if not durations:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, unit in zip(
        axes,
        ["final_m_mean", "rmse_m_mean"],
        ["m", "m"],
    ):
        vals_a0 = [summary_a0[d].get(metric, 0) for d in durations]
        vals_ml = [summary_ml[d].get(metric, 0) for d in durations]
        improvements = [(a - m) / a * 100 if a > 0 else 0 for a, m in zip(vals_a0, vals_ml)]

        colors = ["green" if imp > 0 else "red" for imp in improvements]
        ax.bar(range(len(durations)), improvements, color=colors, alpha=0.8)
        ax.set_xticks(range(len(durations)))
        ax.set_xticklabels([f"{d}s" for d in durations])
        ax.set_ylabel("Improvement (%)")
        ax.set_xlabel("Blackout Duration")
        label = "Final Position Error" if "final" in metric else "RMSE"
        ax.set_title(f"{label} Improvement (ML over A0)")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        for i, (imp, v0, v1) in enumerate(zip(improvements, vals_a0, vals_ml)):
            ax.text(i, imp + 1, f"{imp:.1f}%", ha="center", fontsize=9)

    fig.suptitle("ML-GRU Improvement over A0 Baseline", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = OUT_DIR / "ml_improvement.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_velocity_residuals(ml_res, a0_res, ref_enu, window):
    """Plot velocity residual (ML correction) over the blackout window."""
    n = min(len(ml_res["velocities_east"]),
            len(a0_res["velocities_east"]),
            len(ref_enu[0]))
    n_bo = window["blackout_end_idx"] - window["start_idx"]

    # Reference velocity (from ENU differences)
    ref_ve = np.zeros(n)
    ref_vn = np.zeros(n)
    for k in range(1, n):
        dt = 0.1  # nominal
        ref_ve[k] = (ref_enu[0][k] - ref_enu[0][k - 1]) / dt
        ref_vn[k] = (ref_enu[1][k] - ref_enu[1][k - 1]) / dt

    # Residuals
    delta_ve = ml_res["velocities_east"][:n] - a0_res["velocities_east"][:n]
    delta_vn = ml_res["velocities_north"][:n] - a0_res["velocities_north"][:n]
    residual_mag = np.sqrt(delta_ve ** 2 + delta_vn ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = np.arange(n) * 0.1

    ax = axes[0]
    ax.plot(t, residual_mag, color=COLORS["ML"], linewidth=1)
    ax.axvline(t[n_bo], color="r", linestyle="--", alpha=0.5, label="Blackout end")
    ax.set_ylabel("Velocity correction magnitude (m/s)")
    ax.set_title("ML Velocity Residual Correction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, delta_ve, linewidth=0.8, alpha=0.7, label="East")
    ax.plot(t, delta_vn, linewidth=0.8, alpha=0.7, label="North")
    ax.axvline(t[n_bo], color="r", linestyle="--", alpha=0.5, label="Blackout end")
    ax.set_xlabel("Time since blackout start (s)")
    ax.set_ylabel("Velocity residual (m/s)")
    ax.set_title("East / North Velocity Residual Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    outpath = OUT_DIR / f"ml_velocity_residuals_{window['duration_s']}s.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────
def generate_report(summary_a0, summary_ml, windows, model_config):
    lines = []
    add = lines.append

    add("=" * 80)
    add("ML RESIDUAL GRU — EVALUATION REPORT")
    add("=" * 80)
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Model: GRU (1-layer, hidden={model_config.get('hidden_size', '?')})")
    add(f"Features: {model_config.get('n_features', '?')} dimensions")
    add(f"Context window: {model_config.get('context_len', '?')} samples")
    add(f"Training epochs: {model_config.get('epochs_trained', '?')}")
    add(f"Best val loss (MSE): {model_config.get('best_val_loss', '?'):.6f}" if isinstance(model_config.get('best_val_loss'), float) else "")
    add("")

    add("-" * 80)
    add("COMPARISON: A0 (classical) vs ML-GRU (corrected)")
    add("-" * 80)
    add(f"{'Duration':>10s} | {'A0 MAE':>10s} | {'ML MAE':>10s} | {'Improvement':>12s} | {'A0 RMSE':>10s} | {'ML RMSE':>10s}")
    add("-" * 80)

    durations = sorted(set(summary_a0.keys()) & set(summary_ml.keys()))
    for d in durations:
        a0_mae = summary_a0[d].get("mae_m_mean", 0)
        ml_mae = summary_ml[d].get("mae_m_mean", 0)
        a0_rmse = summary_a0[d].get("rmse_m_mean", 0)
        ml_rmse = summary_ml[d].get("rmse_m_mean", 0)
        imp = (a0_mae - ml_mae) / a0_mae * 100 if a0_mae > 0 else 0

        add(f"{d:>9d}s | {a0_mae:>9.1f}m | {ml_mae:>9.1f}m | {imp:>+10.1f}%   | {a0_rmse:>9.1f}m | {ml_rmse:>9.1f}m")

    add("-" * 80)
    add("")
    add("NOTES:")
    add("  - MAE = Mean Absolute Error over blackout period")
    add("  - RMSE = Root Mean Square Error over blackout period")
    add("  - Improvement = (A0_MAE - ML_MAE) / A0_MAE * 100%")
    add("  - Positive improvement = ML is better than A0")
    add("  - Model was trained on Seg0+Seg1, validated on Seg2")
    add("  - Evaluation windows match classical_dr_baseline.py criteria")
    add("")
    add("=" * 80)

    report = "\n".join(lines)
    report_path = OUT_DIR / "ml_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")
    return report


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
BLACKOUT_DURATIONS = [10, 30, 60, 120]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate velocity residual GRU")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root")
    parser.add_argument("--model-path", type=str,
                        default=str(OUT_DIR / "best_model.pt"),
                        help="Path to trained model checkpoint")
    parser.add_argument("--norm-path", type=str,
                        default=str(OUT_DIR / "normalization.npz"),
                        help="Path to normalization stats")
    parser.add_argument("--context-len", type=int, default=20,
                        help="Context window length")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_root:
        mc.DATA_FILE = Path(args.data_root) / "processed" / "S4_synced.csv"
        if not mc.DATA_FILE.exists():
            mc.DATA_FILE = Path(args.data_root) / "S4_synced.csv"

    print("=" * 70)
    print("VELOCITY RESIDUAL GRU — EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Load model ──
    print("\nLoading model...")
    model, model_config = load_model(Path(args.model_path), device)

    # ── Load normalization ──
    print("\nLoading normalization...")
    norm = mc.load_normalization(Path(args.norm_path))
    print(f"  Train samples: {norm['n_train']}")

    # ── Load data ──
    data, cols, df = mc.load_data()
    segments = mc.detect_segments(data["sync_time"])

    print("\nExtracting features...")
    features = mc.extract_features(data)

    # ── Select blackout windows ──
    windows = mc.select_blackout_windows(data, segments, BLACKOUT_DURATIONS)
    if not windows:
        print("\n*** No valid blackout windows. Cannot evaluate. ***")
        sys.exit(1)

    # ── Evaluate each window ──
    all_met_a0 = {}
    all_met_ml = {}

    for w in windows:
        dur = w["duration_s"]
        print(f"\n  Evaluating {dur}s blackout at T={w['t_start']:.1f}...")

        # Reference
        ref_e, ref_n, ref_h = get_reference_enu(
            data, w["start_idx"], w["eval_end_idx"],
            w["segment"]["t_start"], 0.0,
        )
        # Use actual ref origin
        ref_lat0 = data["veh_lat"][w["start_idx"]]
        ref_lon0 = data["veh_lon"][w["start_idx"]]
        ref_e, ref_n, ref_h = get_reference_enu(
            data, w["start_idx"], w["eval_end_idx"],
            ref_lat0, ref_lon0,
        )

        n_bo = w["blackout_end_idx"] - w["start_idx"]
        t_eval = data["sync_time"][w["start_idx"]:w["eval_end_idx"]]

        # A0 baseline
        print("    A0 baseline...")
        res_a0 = run_classical_a0(data, w)
        met_a0 = compute_metrics(
            res_a0["positions_east"], res_a0["positions_north"], res_a0["headings"],
            ref_e, ref_n, ref_h,
            t_eval, n_bo,
        )
        all_met_a0.setdefault(dur, []).append(met_a0)

        # ML-GRU
        print("    ML-GRU...")
        model.to(device)  # ensure params on active device (defensive)
        res_ml = run_ml_dr(model, data, features, norm, w, args.context_len, device)
        met_ml = compute_metrics(
            res_ml["positions_east"], res_ml["positions_north"], res_ml["headings"],
            ref_e, ref_n, ref_h,
            t_eval, n_bo,
        )
        all_met_ml.setdefault(dur, []).append(met_ml)

        # Plots
        plot_position_error_vs_time([met_a0], [met_ml], dur)
        plot_trajectory_example(res_ml, res_a0, (ref_e, ref_n), w)
        if dur == 30:
            plot_velocity_residuals(res_ml, res_a0, (ref_e, ref_n), w)

    # ── Summaries ──
    summary_a0 = summarize(all_met_a0)
    summary_ml = summarize(all_met_ml)

    # ── Aggregate plots ──
    plot_error_vs_blackout_duration(summary_a0, summary_ml)
    plot_improvement_bar(summary_a0, summary_ml)

    # ── Report ──
    generate_report(summary_a0, summary_ml, windows, model_config)

    # ── Print final comparison ──
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Dur':>5s} | {'A0 MAE':>10s} | {'ML MAE':>10s} | {'Δ MAE':>10s} | {'Improve':>8s}")
    print("-" * 70)
    for d in sorted(summary_a0.keys()):
        if d not in summary_ml:
            continue
        a0 = summary_a0[d]["mae_m_mean"]
        ml = summary_ml[d]["mae_m_mean"]
        delta = a0 - ml
        pct = delta / a0 * 100 if a0 > 0 else 0
        print(f"  {d:>4d}s | {a0:>9.1f}m | {ml:>9.1f}m | {delta:>+9.1f}m | {pct:>+7.1f}%")

    print(f"\n  Outputs: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
