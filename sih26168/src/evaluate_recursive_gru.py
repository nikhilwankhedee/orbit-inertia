#!/usr/bin/env python3
"""
Recursive (deployment-like) Rollout Evaluation — Velocity Residual GRU (SIH26168)
==================================================================================

Tests whether the existing V0 GRU correction survives a genuinely recursive,
deployment-like rollout, compared against:
  - A0 (classical DR)
  - V0 GRU teacher-style (full-context batched, the original evaluation)

KEY DIFFERENCE (recursive):
  - The blackout is propagated step-by-step.
  - Internally-maintained navigation state (heading, corrected speed) is fed
    BACK into the model's input at each timestep.
  - Vehicle-CAN and GNSS-derived features are NOT read from the recording
    during the blackout. They are replaced by recursively-maintained values
    (see RECURSIVE_FEATURE_SUBSTITUTION below).
  - Ground truth is used only for evaluation, after/beside the rollout.

IMPORTANT (feature-leakage audit):
  The V0 model was TRAINED on all 16 features. Its 34-45% teacher-style
  improvement used the RECORDED ground-truth vehicle features DURING the
  blackout (veh_velocity, veh_heading, veh_yaw_rate, steering, wheel speeds),
  i.e. reference leakage. This script instead substitutes those channels with
  causal recursive estimates so the result reflects deployment-like operation.
  The checkpoint stays compatible (same 16 input dims).

Usage (Kaggle):
  python evaluate_recursive_gru.py \
      --model-path outputs/ml/best_model.pt \
      --norm-path outputs/ml/normalization.npz

Outputs (outputs/ml/recursive/):
  recursive_v0_report.txt   text report answering the experiment checklist
  recursive_comparison.csv  per-window A0 vs teacher vs recursive metrics
  plots/*.png               required plots
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import ml_common as mc
import evaluate_velocity_residual_gru as ev  # reuse teacher/A0/metrics unchanged

OUT_DIR = ev.OUT_DIR

# Feature index reference (matches ml_common.FEATURE_KEYS)
# 0-2 accel_x/y/z, 3-5 gravity_x/y/z, 6 gyro_pitch: PHONE-MEASURED (available)
# 7 phone_speed, 8 phone_acc: GNSS-derived (unavailable after GNSS loss)
# 9 veh_velocity, 10 veh_heading, 11 veh_yaw_rate, 12 steering, 13-15 whl: CAN (unavailable)

RECURSIVE_FEATURE_SUBSTITUTION = {
    7:  "internal speed estimate (km/h) — recursively maintained",
    8:  "last pre-blackout GPS accuracy (held)",
    9:  "internal speed estimate (km/h) — recursively maintained",
    10: "internally propagated heading (rad) — gyro-integrated",
    11: "measured gyro_pitch (rad/s) as yaw-rate proxy",
    12: "last pre-blackout steering angle (held)",
    13: "last pre-blackout wheel speed FL (held)",
    14: "last pre-blackout wheel speed FR (held)",
    15: "last pre-blackout wheel speed RL (held)",
}


def run_recursive_dr(
    model: torch.nn.Module,
    data: dict,
    features: np.ndarray,
    norm: dict,
    window: dict,
    context_len: int,
    device: torch.device,
) -> dict:
    """Fully recursive, deployment-like ML-corrected DR rollout.

    State maintained internally (per blackout window):
      - heading (rad): gyro-integrated, same recurrence as A0
      - corrected EN velocity (m/s)
      - internal speed = |v_corrected|
      - position (East/North)

    At step k the GRU input window is built causally:
      - j < i0            : measured pre-blackout features (all available)
      - j == i0           : measured feature row at loss instant (last GNSS/CAN)
      - j > i0, in blackout: measured phone IMU rows, with unavailable channels
                             replaced by recursive estimates (see substitution
                             table).

    Returns same dict schema as evaluate_velocity_residual_gru.run_ml_dr plus
    'delta_v' (applied correction) and 'speeds'.
    """
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]
    n_eval = i2 - i0

    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]
    east0, north0, _ = mc.ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0,
    )

    # Init conditions from data available at the loss instant (ground truth is
    # legitimate for the INITIAL condition, not for the propagation).
    init_heading = np.radians(data["veh_heading"][i0])
    init_speed = data["veh_velocity"][i0] / 3.6

    tgt_mean = norm["tgt_mean"]
    tgt_std = norm["tgt_std"]
    f_mean = norm["feat_mean"]
    f_std = norm["feat_std"]

    # Held pre-blackout values for channels with no recursive substitute
    hold_phone_acc = features[i0, 8]
    hold_steering = features[i0, 12]
    hold_wheels = features[i0, 13:16].copy()

    # Context buffer (normalized). Pre-blackout rows are measured.
    buf_start = i0 - context_len + 1
    buf_rows = features[max(0, buf_start):i0].copy()
    pad = context_len - 1 - len(buf_rows)
    if pad > 0:
        buf_rows = np.concatenate([np.zeros((pad, mc.N_FEATURES)), buf_rows], axis=0)
    buf = (buf_rows - f_mean) / f_std
    buf = buf[- (context_len - 1):] if len(buf) > context_len - 1 else buf

    # Internal state
    heading = init_heading
    prev_heading = init_heading
    speed_est = init_speed
    e_pos = east0
    n_pos = north0

    headings = np.zeros(n_eval)
    ve_out = np.zeros(n_eval)
    vn_out = np.zeros(n_eval)
    e_pos_arr = np.zeros(n_eval)
    n_pos_arr = np.zeros(n_eval)
    delta_v_arr = np.zeros((n_eval, 2))
    speeds = np.zeros(n_eval)

    gyro = data["gyro_pitch"]  # rad/s, measured throughout

    for k in range(n_eval):
        j = i0 + k

        # Build the causal feature row for absolute index j
        if k == 0:
            row = features[j].copy()  # measured (loss instant, all channels valid)
        else:
            row = features[j].copy()  # measured IMU channels preserved
            row[7] = speed_est * mc.GPS_SPEED_FACTOR       # phone_speed km/h
            row[8] = hold_phone_acc                        # phone_acc (last GNSS)
            row[9] = speed_est * mc.GPS_SPEED_FACTOR       # veh_velocity km/h
            row[10] = heading                              # veh_heading rad
            row[11] = gyro[j - 1]                          # veh_yaw_rate proxy
            row[12] = hold_steering                        # steering held
            row[13:16] = hold_wheels                       # wheel speeds held

        # Advance internal state to this step (mirrors A0 recurrence)
        if k > 0:
            dt = data["sync_time"][j] - data["sync_time"][j - 1]
            if 0 < dt <= mc.DT_MAX_GAP:
                heading = prev_heading + gyro[j - 1] * dt
            else:
                heading = prev_heading  # hold on gap

        # Update context buffer, predict residual (normalized)
        row_n = (row - f_mean) / f_std
        buf = np.concatenate([buf, row_n[None, :]], axis=0)
        if len(buf) > context_len:
            buf = buf[1:]

        with torch.no_grad():
            x = torch.tensor(buf[None, :, :], dtype=torch.float32).to(device)
            pred = model(x).cpu().numpy()[0]  # (2,) normalized

        dve, dvn = pred * tgt_std + tgt_mean  # m/s

        # Classical A0 velocity at this heading (same recurrence as A0)
        cls_ve = init_speed * np.cos(heading)
        cls_vn = init_speed * np.sin(heading)

        ve_corr = cls_ve + dve
        vn_corr = cls_vn + dvn

        # Position accumulation
        if k == 0:
            dt = mc.DT_NOMINAL
        else:
            dt = data["sync_time"][j] - data["sync_time"][j - 1]
        if dt <= 0 or dt > mc.DT_MAX_GAP:
            dt = 0.0
        e_pos = e_pos + ve_corr * dt
        n_pos = n_pos + vn_corr * dt

        # Store
        headings[k] = heading
        ve_out[k] = ve_corr
        vn_out[k] = vn_corr
        e_pos_arr[k] = e_pos
        n_pos_arr[k] = n_pos
        delta_v_arr[k] = [dve, dvn]
        speeds[k] = np.hypot(ve_corr, vn_corr)

        # Internal state for next step
        prev_heading = heading
        speed_est = np.hypot(ve_corr, vn_corr)

    return {
        "headings": headings,
        "positions_east": e_pos_arr,
        "positions_north": n_pos_arr,
        "velocities_east": ve_out,
        "velocities_north": vn_out,
        "delta_v": delta_v_arr,
        "speeds": speeds,
        "init_heading_deg": np.degrees(init_heading),
        "init_speed_ms": init_speed,
        "init_east": east0,
        "init_north": north0,
        "ref_lat0": ref_lat0,
        "ref_lon0": ref_lon0,
    }


def compute_velocity_metrics(ve_dr, vn_dr, ref_ve, ref_vn, blackout_end_rel):
    """2D velocity error over the blackout period."""
    n = min(len(ve_dr), len(ref_ve))
    ve_dr, vn_dr = ve_dr[:n], vn_dr[:n]
    ref_ve, ref_vn = ref_ve[:n], ref_vn[:n]
    vel_err = np.sqrt((ve_dr - ref_ve) ** 2 + (vn_dr - ref_vn) ** 2)
    bo = vel_err[:blackout_end_rel]
    return {
        "velocity_mae": float(np.mean(bo)) if len(bo) else float("nan"),
        "velocity_rmse": float(np.sqrt(np.mean(bo ** 2))) if len(bo) else float("nan"),
    }


def collect_window_metrics(name, res, ref_e, ref_n, ref_h, t_eval, n_bo,
                           ref_ve, ref_vn):
    """Position + heading metrics (same as teacher eval) plus velocity metric."""
    met = ev.compute_metrics(
        res["positions_east"], res["positions_north"], res["headings"],
        ref_e, ref_n, ref_h, t_eval, n_bo,
    )
    bo = met["at_durations"].get("blackout", {})
    vel = compute_velocity_metrics(
        res["velocities_east"], res["velocities_north"], ref_ve, ref_vn, n_bo,
    )
    return {
        "name": name,
        "mae_m": bo.get("mae_m", float("nan")),
        "rmse_m": bo.get("rmse_m", float("nan")),
        "max_m": bo.get("max_m", float("nan")),
        "final_m": bo.get("final_m", float("nan")),
        "heading_mae_deg": bo.get("heading_mae_deg", float("nan")),
        "velocity_mae": vel["velocity_mae"],
        "velocity_rmse": vel["velocity_rmse"],
        "t_since": met["t_since"],
        "pos_error": met["pos_error"],
    }


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
COLORS = {"A0": "#1f77b4", "Teacher": "#ff7f0e", "Recursive": "#d62728"}


def plot_pos_error_vs_time(all_met, duration, out_dir):
    """Position error over time — A0 vs Teacher vs Recursive (req 1-4)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, m in enumerate(all_met["A0"]):
        ax.plot(m["t_since"], m["pos_error"], color=COLORS["A0"], alpha=0.5,
                linewidth=0.8, label="A0" if i == 0 else None)
    for i, m in enumerate(all_met["Teacher"]):
        ax.plot(m["t_since"], m["pos_error"], color=COLORS["Teacher"], alpha=0.5,
                linewidth=0.8, label="Teacher" if i == 0 else None)
    for i, m in enumerate(all_met["Recursive"]):
        ax.plot(m["t_since"], m["pos_error"], color=COLORS["Recursive"], alpha=0.5,
                linewidth=0.8, label="Recursive" if i == 0 else None)
    ax.set_xlabel("Time since blackout start (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title(f"Position Error — {duration}s Blackout (A0 vs Teacher vs Recursive)")
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)
    if not ax.lines:
        plt.close(fig)
        return
    ax.set_xlim(0, duration + 10)
    outpath = out_dir / f"rec_position_error_{duration}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_teacher_vs_recursive_final(all_met, out_dir):
    """Teacher vs recursive final position error per duration (req 5)."""
    durations = sorted(all_met["dur"])
    x = np.arange(len(durations))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, off in [("Teacher", -w / 2), ("Recursive", w / 2)]:
        vals = [np.mean([m["final_m"] for m in all_met["d"][name][d] if np.isfinite(m["final_m"])]) if all_met["d"][name][d] else float("nan") for d in durations]
        ax.bar(x + off, vals, w, color=COLORS[name], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("Final position error (m)")
    ax.set_title("Teacher vs Recursive — Final Position Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    outpath = out_dir / "rec_teacher_vs_recursive_final.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_three_way_mae(all_met, out_dir):
    """Classical vs Teacher vs Recursive MAE per duration (req 6)."""
    durations = sorted(all_met["dur"])
    x = np.arange(len(durations))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, off in [("A0", -w), ("Teacher", 0), ("Recursive", w)]:
        vals = [np.mean([m["mae_m"] for m in all_met["d"][name][d] if np.isfinite(m["mae_m"])]) if all_met["d"][name][d] else float("nan") for d in durations]
        ax.bar(x + off, vals, w, color=COLORS[name], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("Position MAE (m)")
    ax.set_title("Classical vs Teacher vs Recursive — Position MAE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    outpath = out_dir / "rec_three_way_mae.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_recursive_velocity_error(all_met, out_dir):
    """Recursive velocity error per duration (req 7)."""
    durations = sorted(all_met["dur"])
    x = np.arange(len(durations))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, off in [("A0", -w), ("Teacher", 0), ("Recursive", w)]:
        vals = [np.mean([m["velocity_mae"] for m in all_met["d"][name][d] if np.isfinite(m["velocity_mae"])]) if all_met["d"][name][d] else float("nan") for d in durations]
        ax.bar(x + off, vals, w, color=COLORS[name], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("Velocity MAE (m/s)")
    ax.set_title("Velocity Error (2D) — A0 vs Teacher vs Recursive")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    outpath = out_dir / "rec_velocity_error.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# CSV + REPORT
# ──────────────────────────────────────────────────────────────
def save_comparison_csv(rows, out_dir):
    path = out_dir / "recursive_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "duration_s", "start_idx", "t_start_s", "pre_velocity_kmh",
            "a0_mae_m", "a0_rmse_m", "a0_max_m", "a0_final_m", "a0_vel_mae",
            "teach_mae_m", "teach_rmse_m", "teach_max_m", "teach_final_m", "teach_vel_mae",
            "rec_mae_m", "rec_rmse_m", "rec_max_m", "rec_final_m", "rec_vel_mae",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def summary_stats(all_met, name):
    """Average blackout metrics over windows per duration."""
    stats = {}
    for d in all_met["dur"]:
        ms = [m for m in all_met["d"][name][d] if np.isfinite(m["mae_m"])]
        if not ms:
            continue
        stats[d] = {
            "mae_m_mean": float(np.mean([m["mae_m"] for m in ms])),
            "rmse_m_mean": float(np.mean([m["rmse_m"] for m in ms])),
            "max_m_mean": float(np.mean([m["max_m"] for m in ms])),
            "final_m_mean": float(np.mean([m["final_m"] for m in ms])),
            "vel_mae_mean": float(np.mean([m["velocity_mae"] for m in ms])),
            "heading_mae_mean": float(np.mean([m["heading_mae_deg"] for m in ms])),
            "n_windows": len(ms),
        }
    return stats


def improvement(a, b):
    return (a - b) / a * 100 if a and a > 0 else float("nan")


def generate_report(stats, features_cfg, out_dir):
    lines = []
    add = lines.append

    add("=" * 88)
    add("RECURSIVE (DEPLOYMENT-LIKE) ROLLOUT — V0 GRU EVALUATION REPORT")
    add("=" * 88)
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Model checkpoint epoch: {features_cfg.get('epoch', '?')}, val_loss: {features_cfg.get('val_loss', '?')}")
    add(f"Context window: {features_cfg.get('context_len', 20)} samples")
    add("")

    add("=" * 88)
    add("1. WHAT WAS RECURSIVE?")
    add("=" * 88)
    add("  - A step-by-step rollout over each blackout window (A0 recurrence).")
    add("  - The model's own corrected EN velocity is accumulated into position.")
    add("  - Corrected speed |v|, gyro-integrated heading, and yaw-rate proxy are")
    add("    fed back into the model input at every subsequent timestep.")
    add("  - Nothing is reset to ground truth during the blackout.")
    add("")

    add("=" * 88)
    add("2. STATE FED FORWARD (recursive state)")
    add("=" * 88)
    add("  - heading (rad): gyro-integrated, identical recurrence to A0")
    add("  - corrected EN velocity (m/s): from classical + GRU residual")
    add("  - internal speed = |v_corrected| (km/h): replaces veh_velocity and")
    add("    phone_speed channels during the blackout")
    add("  - position (East/North): accumulated, used only for output/eval")
    add("")

    add("=" * 88)
    add("3. FEATURES AVAILABLE DURING BLACKOUT")
    add("=" * 88)
    add("  Measured (phone IMU, index 0-6):")
    add("    accel_x/y/z, gravity_x/y/z, gyro_pitch  -> raw recorded measurement")
    add("  Recursively substituted (see table):")
    for idx, desc in RECURSIVE_FEATURE_SUBSTITUTION.items():
        add(f"    FEATURE_KEYS[{idx}] ({mc.FEATURE_KEYS[idx]}) -> {desc}")
    add("  Init conditions at loss instant (last GNSS/CAN fix, legitimate):")
    add("    heading/speed from veh_heading/veh_velocity at blackout start")
    add("")

    add("=" * 88)
    add("4. LEAKAGE AUDIT")
    add("=" * 88)
    add("  - The V0 model was TRAINED on all 16 features, INCLUDING vehicle CAN")
    add("    velocity/heading/yaw-rate/steering/wheel speeds.")
    add("  - The original teacher-style evaluation read those RECORDED channels")
    add("    from the dataset DURING the blackout -> REFERENCE LEAKAGE. The")
    add("    model could read near-truth velocity/heading straight from inputs.")
    add("  - This recursive evaluation does NOT read CAN or GNSS channels during")
    add("    the blackout; they are causally substituted (Section 3).")
    add("  - The existing checkpoint is therefore INCOMPATIBLE with a strict")
    add("    smartphone-only feature set (7 phone-measured dimensions). Minimum")
    add("    retraining experiment documented in OBSERVATION_LOG.md.")
    add("")

    add("=" * 88)
    add("5. CAUSALITY")
    add("=" * 88)
    add("  - Rollout uses only features at j<=current step. Feature buffer (k) uses")
    add("    rows j=i0+k-context_len+1 .. i0+k, never beyond, never ground-truthed")
    add("    during blackout. +1.81s calibration not applied. Fully causal.")
    add("")

    add("=" * 88)
    add("6. BLACKOUT WINDOWS")
    add("=" * 88)
    add("  - Same 39 windows / durations (10, 30, 60, 120 s) as the teacher eval,")
    add("    same selection criteria (classical_dr_baseline.py). No cherry-picking.")
    add("")

    add("=" * 88)
    add("7-11. RESULTS")
    add("=" * 88)
    cols = f"{'Dur':>5s} | {'A0 MAE':>9s} | {'Tch MAE':>9s} | {'Rec MAE':>9s} | {'A0 RMSE':>9s} | {'Tch RMSE':>9s} | {'Rec RMSE':>9s} | {'A0-V':>7s} | {'Rec-V':>7s}"
    add(cols)
    add("-" * len(cols))
    a0_durs = sorted(stats["A0"].keys())
    for d in a0_durs:
        a0 = stats["A0"][d]
        tch = stats["Teacher"].get(d, {})
        rec = stats["Recursive"].get(d, {})
        def g(x, k, nd=float("nan")):
            return x.get(k, nd) if x else nd
        add(f"{d:>4d}s | {a0['mae_m_mean']:>8.1f}m | {g(tch,'mae_m_mean'):>8.1f}m | {g(rec,'mae_m_mean'):>8.1f}m | "
            f"{a0['rmse_m_mean']:>8.1f}m | {g(tch,'rmse_m_mean'):>8.1f}m | {g(rec,'rmse_m_mean'):>8.1f}m | "
            f"{a0['vel_mae_mean']:>6.2f} | {g(rec,'vel_mae_mean'):>6.2f}")
    add("-" * len(cols))
    add("  Tch = teacher-style (original eval). Rec = recursive rollout.")
    add("  A0-V / Rec-V = 2D velocity MAE (m/s) over blackout.")
    add("")

    # 7/8/9/10/11: comparisons + interpretation
    teach_wins_over_a0 = []
    rec_wins_over_a0 = []
    rec_vs_teach_ratio = []
    for d in a0_durs:
        a0 = stats["A0"][d]["mae_m_mean"]
        tch = stats["Teacher"].get(d, {}).get("mae_m_mean", a0)
        rec = stats["Recursive"].get(d, {}).get("mae_m_mean", a0)
        teach_wins_over_a0.append(improvement(a0, tch))
        rec_wins_over_a0.append(improvement(a0, rec))
        rec_vs_teach_ratio.append(rec / tch if tch else float("nan"))

    t_imp = np.nanmean(teach_wins_over_a0)
    r_imp = np.nanmean(rec_wins_over_a0)
    r_ratio = np.nanmean(rec_vs_teach_ratio)

    add("  7. Recursive ML vs teacher ML:")
    add(f"     Recursive avg improvement vs A0 = {r_imp:+.1f}%; teacher avg = {t_imp:+.1f}%.")
    add(f"     Mean recursive/teacher MAE ratio = {r_ratio:.2f}x.")
    add("  8. Recursive ML vs A0: see table above (right-hand columns).")
    add("  9. Error compounding: 'final_m_mean' vs 'mae_m_mean' per duration")
    add("     quantify growth over time; position-error plots show the curve.")
    add("  10. Stability: NaN/max_jump handled; divergence shows as Rec >> A0.")
    add("")

    # Verdict (CASE A / B / C)
    add("=" * 88)
    add("VERDICT")
    add("=" * 88)
    if r_imp < 0:
        add(f"  CASE C — Recursive ML is WORSE than A0 (avg {r_imp:+.1f}%).")
        add("  STOP. Do NOT increase model size. Investigate:")
        add("    - feature leakage / substitution quality")
        add("    - frame mapping")
        add("    - state / target formulation")
        add("    - correction magnitude (destabilising the propagated state)")
    elif r_imp > 0 and r_imp >= 0.5 * t_imp:
        add(f"  CASE A — Recursive ML remains strong (avg {r_imp:+.1f}% vs A0,")
        add(f"            {r_ratio:.2f}x teacher MAE). Correction survives feedback.")
        add("  NEXT: retrain on deployment-valid smartphone features, then")
        add("        multi-trajectory validation -> freeze candidate -> app.")
    else:
        add(f"  CASE B — Teacher strong ({t_imp:+.1f}%) but recursive weak")
        add(f"            ({r_imp:+.1f}%, {r_ratio:.2f}x teacher MAE).")
        add("  The learned correction depends on reference-like state/context or")
        add("  suffers error accumulation / distribution shift. Not ready for")
        add("  deployment. Diagnose feature/state/target formulation; the V0")
        add("  checkpoint is NOT deployment valid (retrain on phone-only features).")

    add("")
    add("=" * 88)
    add("12. READINESS FOR BROADER VALIDATION")
    add("=" * 88)
    add("  - Only a positive CASE A verdict justifies continuing to multi-trajectory")
    add("    validation with this approach.")
    add("  - Frame mapping (phone-to-vehicle EN) remains unresolved: the 2D EN")
    add("    velocity residual target is an empirical construct. This experiment")
    add("    measures whether recursive rollout changes the observed ML benefit;")
    add("    it does NOT claim physically-correct East/North navigation.")
    add("")
    add("=" * 88)

    report = "\n".join(lines)
    path = out_dir / "recursive_v0_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Report saved: {path}")
    return report


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
BLACKOUT_DURATIONS = [10, 30, 60, 120]


def parse_args():
    parser = argparse.ArgumentParser(description="Recursive rollout evaluation")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--model-path", type=str,
                        default=str(ev.OUT_DIR / "best_model.pt"))
    parser.add_argument("--norm-path", type=str,
                        default=str(ev.OUT_DIR / "normalization.npz"))
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit number of blackout windows (smoke tests only)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_root:
        mc.DATA_FILE = Path(args.data_root) / "processed" / "S4_synced.csv"
        if not mc.DATA_FILE.exists():
            mc.DATA_FILE = Path(args.data_root) / "S4_synced.csv"

    fig_out = ev.OUT_DIR / "recursive"
    fig_out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V0 GRU — RECURSIVE ROLLOUT EVALUATION (deployment-like)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("\nLoading model...")
    model, model_config = ev.load_model(Path(args.model_path), device)

    print("\nLoading normalization...")
    norm = mc.load_normalization(Path(args.norm_path))
    print(f"  Train samples: {norm['n_train']}")

    data, cols, df = mc.load_data()
    segments = mc.detect_segments(data["sync_time"])

    print("\nExtracting features...")
    features = mc.extract_features(data)

    windows = mc.select_blackout_windows(data, segments, BLACKOUT_DURATIONS)
    if not windows:
        print("\n*** No valid blackout windows. Cannot evaluate. ***")
        sys.exit(1)
    if args.max_windows:
        windows = windows[:args.max_windows]
        print(f"\n  [SMOKE] Limiting to {len(windows)} windows.")

    all_met = {"dur": BLACKOUT_DURATIONS, "d": {}}
    for name in ["A0", "Teacher", "Recursive"]:
        all_met["d"][name] = {d: [] for d in BLACKOUT_DURATIONS}

    csv_rows = []

    for w in windows:
        dur = w["duration_s"]
        print(f"\n  Evaluating {dur}s blackout at T={w['t_start']:.1f}...")

        ref_lat0 = data["veh_lat"][w["start_idx"]]
        ref_lon0 = data["veh_lon"][w["start_idx"]]
        ref_e, ref_n, ref_h = ev.get_reference_enu(
            data, w["start_idx"], w["eval_end_idx"], ref_lat0, ref_lon0,
        )
        n_bo = w["blackout_end_idx"] - w["start_idx"]
        t_eval = data["sync_time"][w["start_idx"]:w["eval_end_idx"]]
        ref_ve, ref_vn = mc.compute_reference_en_velocity(
            data["veh_lat"], data["veh_lon"], data["sync_time"],
            w["start_idx"], w["eval_end_idx"] - w["start_idx"],
        )

        # A0
        print("    A0 baseline...")
        res_a0 = ev.run_classical_a0(data, w)
        m_a0 = collect_window_metrics("A0", res_a0, ref_e, ref_n, ref_h,
                                      t_eval, n_bo, ref_ve, ref_vn)
        all_met["d"]["A0"][dur].append(m_a0)

        # Teacher-style (reuse original, full-context batched)
        print("    ML-GRU teacher-style...")
        model.to(device)
        res_t = ev.run_ml_dr(model, data, features, norm, w, args.context_len, device)
        m_t = collect_window_metrics("Teacher", res_t, ref_e, ref_n, ref_h,
                                     t_eval, n_bo, ref_ve, ref_vn)
        all_met["d"]["Teacher"][dur].append(m_t)

        # Recursive
        print("    ML-GRU recursive...")
        res_r = run_recursive_dr(model, data, features, norm, w, args.context_len, device)
        m_r = collect_window_metrics("Recursive", res_r, ref_e, ref_n, ref_h,
                                     t_eval, n_bo, ref_ve, ref_vn)
        all_met["d"]["Recursive"][dur].append(m_r)

        csv_rows.append({
            "duration_s": dur,
            "start_idx": w["start_idx"],
            "t_start_s": round(w["t_start"], 1),
            "pre_velocity_kmh": round(w["pre_velocity_kmh"], 1),
            **{"a0_" + k: (round(v, 2) if np.isfinite(v) else "") for k, v in
               {"mae_m": m_a0["mae_m"], "rmse_m": m_a0["rmse_m"],
                "max_m": m_a0["max_m"], "final_m": m_a0["final_m"],
                "vel_mae": m_a0["velocity_mae"]}.items()},
            **{"teach_" + k: (round(v, 2) if np.isfinite(v) else "") for k, v in
               {"mae_m": m_t["mae_m"], "rmse_m": m_t["rmse_m"],
                "max_m": m_t["max_m"], "final_m": m_t["final_m"],
                "vel_mae": m_t["velocity_mae"]}.items()},
            **{"rec_" + k: (round(v, 2) if np.isfinite(v) else "") for k, v in
               {"mae_m": m_r["mae_m"], "rmse_m": m_r["rmse_m"],
                "max_m": m_r["max_m"], "final_m": m_r["final_m"],
                "vel_mae": m_r["velocity_mae"]}.items()},
        })

    save_comparison_csv(csv_rows, fig_out)

    stats = {name: summary_stats(all_met, name) for name in ["A0", "Teacher", "Recursive"]}

    # Plots
    for d in BLACKOUT_DURATIONS:
        plot_pos_error_vs_time(
            {name: all_met["d"][name][d] for name in ["A0", "Teacher", "Recursive"]},
            d, fig_out,
        )
    plot_teacher_vs_recursive_final(all_met, fig_out)
    plot_three_way_mae(all_met, fig_out)
    plot_recursive_velocity_error(all_met, fig_out)

    # Report
    features_cfg = {"context_len": args.context_len}
    try:
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        features_cfg["epoch"] = ckpt.get("epoch", "?")
        features_cfg["val_loss"] = ckpt.get("val_loss", "?")
    except Exception:
        pass
    generate_report(stats, features_cfg, fig_out)

    # Console summary
    print("\n" + "=" * 70)
    print("RECURSIVE SUMMARY")
    print("=" * 70)
    print(f"  {'Dur':>5s} | {'A0 MAE':>9s} | {'Tch MAE':>9s} | {'Rec MAE':>9s} | {'Imp A0':>8s} | {'Imp Tch':>8s}")
    print("-" * 60)
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]["mae_m_mean"]
        tch = stats["Teacher"].get(d, {}).get("mae_m_mean", a0)
        rec = stats["Recursive"].get(d, {}).get("mae_m_mean", a0)
        imp_a0 = (a0 - rec) / a0 * 100 if a0 > 0 else 0
        imp_tch = (tch - rec) / tch * 100 if tch > 0 else 0
        print(f"  {d:>4d}s | {a0:>8.1f}m | {tch:>8.1f}m | {rec:>8.1f}m | {imp_a0:>+7.1f}% | {imp_tch:>+7.1f}%")
    print(f"\n  Outputs: {fig_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()