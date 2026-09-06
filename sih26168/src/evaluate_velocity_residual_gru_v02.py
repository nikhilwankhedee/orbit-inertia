#!/usr/bin/env python3
"""
V0.2 Deployment-Valid GRU — Recursive Rollout Evaluation (SIH26168)
====================================================================

Evaluates the phone-only V0.2 checkpoint under a FULLY RECURSIVE,
deployment-like rollout against the A0 classical baseline, plus an
oracle-state ("teacher-style") diagnostic.

PRIMARY RESULT = RECURSIVE. The oracle-state row (recorded reference
speed/heading fed into the nav channels during the blackout) is labelled
DIAGNOSTIC ONLY — it is the V0 analog of the leaked teacher-style eval and
must NOT be headlined.

Available during the blackout (recursive):
  - accel_x/y/z, gravity_x/y/z, gyro_pitch      (measured phone IMU)
  - nav_speed, nav_heading                        (A0 internal nav state,
    identical to the training-time distribution)
Forbidden: vehicle CAN, steering, wheel speeds, future/phone GPS,
ground-truth position/velocity.

Usage (Kaggle):
  python evaluate_velocity_residual_gru_v02.py \
      --model-path outputs/ml_v02/best_model.pt \
      --norm-path outputs/ml_v02/normalization.npz

Outputs (outputs/ml_v02/):
  recursive_evaluation_report.txt   per-duration metrics + verdict
  v02_report.txt                    full 15-item experiment report
  recursive_comparison.csv          per-window A0 vs V0.2 vs Oracle metrics
  plots/*.png                       required plots
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
import ml_v02_common as v2c
import evaluate_velocity_residual_gru as ev  # reuse A0 / reference / metrics

OUT_DIR = v2c.OUT_DIR

BLACKOUT_DURATIONS = [10, 30, 60, 120]
COLORS = {"A0": "#1f77b4", "V0.2": "#2ca02c", "Oracle": "#ff7f0e"}


# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
def load_model(model_path: Path, device: torch.device):
    from train_velocity_residual_gru_v02 import VelocityResidualGRU

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

    vl = ckpt.get("val_loss")
    vl_str = f"{vl:.6f}" if isinstance(vl, (int, float)) else str(vl)
    print(f"  Model loaded: {model_path}")
    print(f"    Config: {config}")
    print(f"    Checkpoint epoch: {ckpt.get('epoch', '?')}, val_loss: {vl_str}")
    print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, config


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
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
        "correction_mean": float(np.mean(np.sqrt(
            res["delta_v"][:, 0] ** 2 + res["delta_v"][:, 1] ** 2
        ))) if res.get("delta_v") is not None else float("nan"),
    }


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
def plot_pos_error_vs_time(all_met, duration, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, color in COLORS.items():
        for i, m in enumerate(all_met[name]):
            ax.plot(m["t_since"], m["pos_error"], color=color, alpha=0.5,
                    linewidth=0.8, label=name if i == 0 else None)
    ax.set_xlabel("Time since blackout start (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title(f"Position Error — {duration}s Blackout (A0 vs V0.2 vs Oracle)")
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)
    outpath = out_dir / f"v02_position_error_{duration}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_final_error(all_met, out_dir):
    durations = sorted(all_met["A0"].keys())
    x = np.arange(len(durations))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, off in [("A0", -w), ("V0.2", 0), ("Oracle", w)]:
        vals = []
        for d in durations:
            ms = [m["final_m"] for m in all_met[name][d] if np.isfinite(m["final_m"])]
            vals.append(float(np.mean(ms)) if ms else float("nan"))
        ax.bar(x + off, vals, w, color=COLORS[name], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("Final position error (m)")
    ax.set_title("Final Position Error — A0 vs V0.2 (Recursive) vs Oracle")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    outpath = out_dir / "v02_final_error.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_three_way_mae(all_met, out_dir):
    durations = sorted(all_met["A0"].keys())
    x = np.arange(len(durations))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for name, off in [("A0", -w), ("V0.2", 0), ("Oracle", w)]:
        vals = []
        for d in durations:
            ms = [m["mae_m"] for m in all_met[name][d] if np.isfinite(m["mae_m"])]
            vals.append(float(np.mean(ms)) if ms else float("nan"))
        ax.bar(x + off, vals, w, color=COLORS[name], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.set_ylabel("Position MAE (m)")
    ax.set_title("Position MAE — A0 vs V0.2 (Recursive) vs Oracle")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    outpath = out_dir / "v02_three_way_mae.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_correction_magnitude(all_met, out_dir):
    durations = sorted(all_met["A0"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, d in zip(axes.ravel(), durations):
        for name, color in [("V0.2", COLORS["V0.2"]), ("Oracle", COLORS["Oracle"])]:
            for i, m in enumerate(all_met[name][d]):
                curve = m.get("correction_curve")
                if curve is None:
                    continue
                ax.plot(m["t_since"][:len(curve)], curve,
                        color=color, alpha=0.5, linewidth=0.8,
                        label=f"{name} (correction)" if i == 0 else None)
        ax.set_title(f"{d}s blackout")
        ax.set_xlabel("Time since blackout start (s)")
        ax.set_ylabel("Correction magnitude (m/s)")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("Recursive Correction Magnitude |Δv| over Time", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = out_dir / "v02_correction_magnitude.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_pred_vs_target(model, feats, targets, test_windows, norm, context_len,
                        device, out_dir):
    """Scatter of GRU prediction vs target residual on TEST (Seg3) windows.

    Uses the training-convention deployment features and the recorded target.
    Diagnostic only — shows how well the learned correction fits the residual.
    """
    preds = []
    targs = []
    for w in test_windows:
        targs.append(w["target"])
        i = w["abs_idx"]
        ctx_start = max(0, i - context_len + 1)
        ctx = feats[ctx_start:i + 1]
        pad = context_len - len(ctx)
        if pad > 0:
            ctx = np.concatenate([np.zeros((pad, v2c.V02_N_FEATURES)), ctx], axis=0)
        ctx = (ctx - norm["feat_mean"]) / norm["feat_std"]
        with torch.no_grad():
            x = torch.tensor(ctx[None, :, :], dtype=torch.float32).to(device)
            p = model(x).cpu().numpy()[0]
        p = p * norm["tgt_std"] + norm["tgt_mean"]
        preds.append(p)

    preds = np.array(preds)
    targs = np.array(targs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ci, clabel in zip(axes, range(2), ["ΔEast (m/s)", "ΔNorth (m/s)"]):
        ax.scatter(targs[:, ci], preds[:, ci], s=6, alpha=0.6, color="#2ca02c")
        lim = np.nanpercentile(np.abs(np.concatenate([targs[:, ci], preds[:, ci]])), 99)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Target {clabel}")
        ax.set_ylabel(f"Predicted {clabel}")
        ax.set_title(f"Predicted vs Target Residual — {clabel}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("V0.2 — Predicted vs Target Residual (Test windows, diagnostic)", fontsize=13)
    fig.tight_layout()
    outpath = out_dir / "v02_pred_vs_target.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# SUMMARY + REPORT
# ──────────────────────────────────────────────────────────────
def summary_stats(all_met, name):
    stats = {}
    for d in sorted(all_met[name]):
        ms = [m for m in all_met[name][d] if np.isfinite(m["mae_m"])]
        if not ms:
            continue
        stats[d] = {
            "mae_m_mean": float(np.mean([m["mae_m"] for m in ms])),
            "rmse_m_mean": float(np.mean([m["rmse_m"] for m in ms])),
            "max_m_mean": float(np.mean([m["max_m"] for m in ms])),
            "final_m_mean": float(np.mean([m["final_m"] for m in ms])),
            "vel_mae_mean": float(np.mean([m["velocity_mae"] for m in ms])),
            "heading_mae_mean": float(np.mean([m["heading_mae_deg"] for m in ms])),
            "corr_mean": float(np.mean([m["correction_mean"] for m in ms])),
            "n_windows": len(ms),
        }
    return stats


def improvement(a, b):
    return (a - b) / a * 100 if a and a > 0 else float("nan")


def save_comparison_csv(rows, out_dir):
    path = out_dir / "recursive_comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "duration_s", "start_idx", "t_start_s", "pre_velocity_kmh",
            "a0_mae_m", "a0_rmse_m", "a0_max_m", "a0_final_m", "a0_vel_mae",
            "v02_mae_m", "v02_rmse_m", "v02_max_m", "v02_final_m", "v02_vel_mae",
            "oracle_mae_m", "oracle_final_m",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def generate_recursive_report(stats, config, features_cfg, out_dir):
    lines = []
    add = lines.append

    add("=" * 88)
    add("V0.2 DEPLOYMENT-VALID GRU — RECURSIVE ROLLOUT REPORT")
    add("=" * 88)
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add(f"Checkpoint: epoch {features_cfg.get('epoch', '?')}, "
        f"val_loss {features_cfg.get('val_loss', '?')}")
    add(f"Context: {features_cfg.get('context_len', 20)} samples "
        f"(causal, ~2 s @ 10 Hz)")
    add(f"Features: {config.get('n_features', 9)} "
        f"(7 phone-IMU + nav_speed + nav_heading)")
    add("")
    add("PRIMARY RESULT = RECURSIVE. Oracle-state rows are DIAGNOSTIC ONLY")

    add("")
    add("=" * 88)
    add("1-3. WHAT WAS EVALUATED")
    add("=" * 88)
    add("  - A0: classical DR (gyro-integrated heading, constant speed).")
    add("  - V0.2 (RECURSIVE, PRIMARY): 9-dim phone-only features; the model's")
    add("    corrected velocity is accumulated into position step-by-step; no")
    add("    reference/CAN/GPS channel is read during the blackout.")
    add("  - Oracle (DIAGNOSTIC ONLY): same rollout but nav_speed/nav_heading")
    add("    channels are replaced by RECORDED reference state during the")
    add("    blackout (the V0-style 'teacher' analog — LEAKED by design).")
    add("")

    add("=" * 88)
    add("4. AVAILABLE FEATURES DURING BLACKOUT (recursive)")
    add("=" * 88)
    for i, key in enumerate(v2c.DEPLOYMENT_FEATURE_KEYS):
        src = ("measured phone IMU" if i < 7
               else "A0 internal nav state (train/deploy identical)")
        add(f"  {i}. {key:12s} <- {src}")
    add("")

    add("=" * 88)
    add("5. RECURSIVE STATE")
    add("=" * 88)
    add("  - heading: A0 gyro-integrated (same recurrence as training/A0)")
    add("  - corrected EN velocity: v_classical + GRU residual")
    add("  - position: accumulated from v_corrected (never reset to truth)")
    add("  - nav_speed/nav_heading channels: A0 classical state — the EXACT")
    add("    training distribution (no state feedback loop into inputs).")
    add("")

    add("=" * 88)
    add("6. RESULTS")
    add("=" * 88)
    cols = (f"{'Dur':>5s} | {'A0 MAE':>9s} | {'V02 MAE':>9s} | {'Orc MAE':>9s} | "
            f"{'A0 RMSE':>9s} | {'V02 RMSE':>9s} | {'A0-F':>8s} | {'V02-F':>8s} | "
            f"{'A0-V':>7s} | {'V02-V':>7s}")
    add(cols)
    add("-" * len(cols))
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]
        v02 = stats["V0.2"].get(d, {})
        orc = stats["Oracle"].get(d, {})
        def g(x, k, nd=float("nan")):
            return x.get(k, nd) if x else nd
        add(f"{d:>4d}s | {a0['mae_m_mean']:>8.1f}m | {g(v02,'mae_m_mean'):>8.1f}m | "
            f"{g(orc,'mae_m_mean'):>8.1f}m | {a0['rmse_m_mean']:>8.1f}m | "
            f"{g(v02,'rmse_m_mean'):>8.1f}m | {a0['final_m_mean']:>7.1f}m | "
            f"{g(v02,'final_m_mean'):>7.1f}m | {a0['vel_mae_mean']:>6.2f} | "
            f"{g(v02,'vel_mae_mean'):>6.2f}")
    add("-" * len(cols))
    add("  A0-F / V02-F = final position error (m). A0-V / V02-V = velocity MAE (m/s).")
    add("")

    # improvement + verdict
    rec_imps = []
    orc_imps = []
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]["mae_m_mean"]
        rec = stats["V0.2"].get(d, {}).get("mae_m_mean", a0)
        orc = stats["Oracle"].get(d, {}).get("mae_m_mean", a0)
        rec_imps.append(improvement(a0, rec))
        orc_imps.append(improvement(a0, orc))

    r_imp = np.nanmean(rec_imps)
    o_imp = np.nanmean(orc_imps)

    add("  7. Recursive V0.2 vs A0:")
    for d, im in zip(sorted(stats["A0"].keys()), rec_imps):
        add(f"     {d}s: {im:+.1f}%")
    add(f"     Average recursive improvement = {r_imp:+.1f}%")
    add(f"  8. Oracle-state (DIAGNOSTIC) avg improvement = {o_imp:+.1f}%")
    add("  9. Error growth: final vs MAE per duration and position-error plots")
    add("     show whether error compounds unstably over the blackout.")
    add("  10. Correction magnitude per duration:")
    for d in sorted(stats["A0"].keys()):
        v02 = stats["V0.2"].get(d, {})
        if v02:
            add(f"     {d}s: mean |Δv| = {v02['corr_mean']:.2f} m/s")
    add("")

    add("=" * 88)
    add("VERDICT")
    add("=" * 88)
    if not np.isfinite(r_imp):
        add("  CASE UNKNOWN — recursive metrics unavailable.")
    elif r_imp < 0:
        add(f"  CASE C — Recursive V0.2 is WORSE than A0 (avg {r_imp:+.1f}%).")
        add("  STOP. Do not increase model size. Diagnose formulation "
            "(state / target / frame mapping).")
    elif o_imp > 0 and r_imp >= 0.5 * o_imp:
        add(f"  CASE A — Recursive V0.2 remains strong (avg {r_imp:+.1f}% vs A0).")
        add("  V0.2 is promising -> proceed to broader multi-trajectory validation.")
    else:
        add(f"  CASE B — Recursive V0.2 improvement exists but is weak/inconsistent")
        add(f"            (avg {r_imp:+.1f}%; oracle-state {o_imp:+.1f}%).")
        add("  Diagnose state distribution / frame / target before increasing "
            "complexity. Do not headlined oracle/teacher numbers.")
    add("")
    add("=" * 88)

    report = "\n".join(lines)
    path = out_dir / "recursive_evaluation_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Report saved: {path}")
    return report


def generate_v02_report(stats, config, features_cfg, verdict, out_dir):
    """Full 15-item experiment report per the V0.2 spec."""
    lines = []
    add = lines.append

    add("=" * 88)
    add("V0.2 DEPLOYMENT-VALID PHONE-ONLY GRU — EXPERIMENT REPORT")
    add("=" * 88)
    add(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    add("")
    add("1. WHY V0 WAS REJECTED")
    add("   V0.1 recursive audit (CASE B): the V0 checkpoint's teacher-style")
    add("   +34-45% collapsed to +8.6% avg under a deployment-like rollout; at 30s")
    add("   the model was WORSE than classical A0 (-16.6%).")
    add("")
    add("2. WHAT LEAKAGE V0 CONTAINED")
    add("   V0 trained on 16 features including vehicle-CAN velocity/heading/")
    add("   yaw-rate/steering/wheel-speeds; its teacher-style eval READ those")
    add("   recorded near-truth channels during the simulated blackout ->")
    add("   reference information was available in the model input.")
    add("")
    add("3. HOW V0.2 REMOVES THE LEAKAGE")
    add("   Model retrained from scratch on 9 strictly phone-only dims; no")
    add("   vehicle-CAN, steering, wheel-speed, or GPS-derived channel exists in")
    add("   the input space. Recursive eval reads only phone IMU + internal A0")
    add("   state during the blackout. Oracle-state eval (reference state in the")
    add("   nav channels) is retained ONLY as a labelled diagnostic.")
    add("")
    add("4. EXACT DEPLOYMENT FEATURE LIST (9)")
    for i, key in enumerate(v2c.DEPLOYMENT_FEATURE_KEYS):
        unit = "m/s^2" if i < 6 else ("rad/s" if key == "gyro_pitch"
                                      else ("km/h" if key == "nav_speed" else "rad"))
        src = "measured phone IMU" if i < 7 else "A0 internal nav state"
        add(f"   {i}. {key:12s} ({unit:6s}) <- {src}")
    add("")
    add("5. EXACT TARGET")
    add("   dv = v_reference_EN - v_classical_A0_EN (m/s), 2 dims [dEast, dNorth].")
    add("   Reference velocity from recorded vehicle GPS; used ONLY for target")
    add("   construction and evaluation, NEVER as a model input.")
    add("")
    add("6. EXACT SPLIT")
    add("   Segment-level (no overlapping windows across splits):")
    add("   Train Seg0+Seg1 | Val Seg2 | Test Seg3. Windows never cross segments.")
    add("")
    add("7. EXACT MODEL ARCHITECTURE")
    add(f"   1-layer GRU(hidden={config.get('hidden_size', 32)}, "
        f"input={config.get('n_features', 9)}) -> Linear(2). "
        f"~{config.get('params', '?')} parameters.")
    add("   Same deliberately small V0 architecture - no transformers/attention/")
    add("   LSTM stacks/ensembles.")
    add("")
    add("8. EXACT CAUSAL WINDOW")
    add(f"   {features_cfg.get('context_len', 20)} samples (~2 s @ 10 Hz), "
        f"stride {config.get('stride', 5)}. Causal history only; no future/"
        f"centered windows.")
    add("")
    add("9. HOW RECURSIVE STATE IS MAINTAINED")
    add("   Per blackout: gyro-integrated heading + v_corrected = v_classical + dv;")
    add("   position accumulated from v_corrected. nav_speed/nav_heading channels")
    add("   are the A0 classical state (same definition as training). Nothing is")
    add("   reset to ground truth during the blackout.")
    add("")
    add("10. TRAIN/DEPLOYMENT DISTRIBUTION MISMATCH")
    add("    V0.1 root-cause: recursive substitution fed a DISTRIBUTION-SHIFTED")
    add("    state (corrected speed |v| + held CAN channels) into inputs trained")
    add("    on reference-like CAN features. V0.2 defines nav state once, as A0")
    add("    classical state, and uses the SAME definition for training features,")
    add("    targets, and recursive rollout inputs -> no state-channel feedback")
    add("    loop and no substitution. (Optional state-noise perturbation is not")
    add("    used in the clean crop; documented as a further hardening step.)")
    add("")
    add("11-13. RECURSIVE RESULTS vs A0")
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]["mae_m_mean"]
        rec = stats["V0.2"].get(d, {}).get("mae_m_mean", float("nan"))
        imp = improvement(a0, rec)
        add(f"   {d:>4d}s | A0 {a0:>7.1f}m | V0.2 {rec:>7.1f}m | {imp:+.1f}%")
    add("")
    add("14. REMAINING LIMITATIONS")
    add("    - Single trajectory (S4); Test=Seg3 small -> no strong generalization")
    add("      claim from S4 alone.")
    add("    - Phone<->vehicle EN frame mapping unresolved; the 2D EN residual is")
    add("      an empirical construct (V0 limitation retained).")
    add("    - nav_speed is constant per segment (A0 speed) - coarse state signal.")
    add("    - Oracle-state eval is DIAGNOSTIC ONLY and must not be headlined.")
    add("")
    add("15. RECOMMENDATION FOR NEXT PHASE")
    add("    - CASE results above; full details in recursive_evaluation_report.txt")
    add("    - Only a positive, consistent CASE A from recursive V0.2 justifies")
    add("      broader multi-trajectory validation. Otherwise diagnose")
    add("      state/frame/target formulation before any complexity increase.")
    add("")
    add("=" * 88)

    report = "\n".join(lines)
    path = out_dir / "v02_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Report saved: {path}")
    return report


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V0.2 recursive rollout")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--model-path", type=str,
                        default=str(v2c.OUT_DIR / "best_model.pt"))
    parser.add_argument("--norm-path", type=str,
                        default=str(v2c.OUT_DIR / "normalization.npz"))
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit number of blackout windows (smoke tests only)")
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

    fig_out = OUT_DIR / "plots"
    fig_out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V0.2 DEPLOYMENT-VALID GRU — RECURSIVE ROLLOUT EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("\nLoading model...")
    model, model_config = load_model(Path(args.model_path), device)

    print("\nLoading normalization...")
    norm = v2c.load_normalization(Path(args.norm_path))

    data, cols, df = mc.load_data()
    segments = mc.detect_segments(data["sync_time"])

    print("\nComputing A0 classical state + deployment features...")
    ve_cls, vn_cls, heading_cls, speed_kmh = v2c.compute_a0_state(data, segments)
    features = v2c.build_deployment_features(data, speed_kmh, heading_cls)

    try:
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
        features_cfg = {
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": ckpt.get("val_loss", "?"),
            "context_len": args.context_len,
        }
    except Exception:
        features_cfg = {"context_len": args.context_len}

    windows = mc.select_blackout_windows(data, segments, BLACKOUT_DURATIONS)
    if not windows:
        print("\n*** No valid blackout windows. Cannot evaluate. ***")
        sys.exit(1)
    if args.max_windows:
        windows = windows[:args.max_windows]
        print(f"\n  [SMOKE] Limiting to {len(windows)} windows.")

    all_met = {"dur": BLACKOUT_DURATIONS, "d": {}}
    for name in ["A0", "V0.2", "Oracle"]:
        all_met["d"][name] = {d: [] for d in BLACKOUT_DURATIONS}
        # also store the correction magnitude curve
    corrections = {"V0.2": {d: [] for d in BLACKOUT_DURATIONS},
                   "Oracle": {d: [] for d in BLACKOUT_DURATIONS}}

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

        # V0.2 RECURSIVE (primary)
        print("    V0.2 recursive...")
        model.to(device)
        row_fn = v2c.recursive_row_fn(features)
        res_r = v2c.run_v02_rollout(model, row_fn, data, norm, w,
                                    args.context_len, device)
        m_r = collect_window_metrics("V0.2", res_r, ref_e, ref_n, ref_h,
                                     t_eval, n_bo, ref_ve, ref_vn)
        all_met["d"]["V0.2"][dur].append(m_r)
        corrections["V0.2"][dur].append(
            np.sqrt((res_r["delta_v"][:, 0] ** 2 + res_r["delta_v"][:, 1] ** 2)))

        # Oracle-state (DIAGNOSTIC ONLY)
        print("    Oracle-state (DIAGNOSTIC)...")
        row_fn_o = v2c.oracle_row_fn(data, features, w)
        res_o = v2c.run_v02_rollout(model, row_fn_o, data, norm, w,
                                    args.context_len, device)
        m_o = collect_window_metrics("Oracle", res_o, ref_e, ref_n, ref_h,
                                     t_eval, n_bo, ref_ve, ref_vn)
        all_met["d"]["Oracle"][dur].append(m_o)
        corrections["Oracle"][dur].append(
            np.sqrt((res_o["delta_v"][:, 0] ** 2 + res_o["delta_v"][:, 1] ** 2)))

        csv_rows.append({
            "duration_s": dur,
            "start_idx": w["start_idx"],
            "t_start_s": round(w["t_start"], 1),
            "pre_velocity_kmh": round(w["pre_velocity_kmh"], 1),
            **{"a0_" + k: (round(v, 2) if np.isfinite(v) else "") for k, v in
               {"mae_m": m_a0["mae_m"], "rmse_m": m_a0["rmse_m"],
                "max_m": m_a0["max_m"], "final_m": m_a0["final_m"],
                "vel_mae": m_a0["velocity_mae"]}.items()},
            **{"v02_" + k: (round(v, 2) if np.isfinite(v) else "") for k, v in
               {"mae_m": m_r["mae_m"], "rmse_m": m_r["rmse_m"],
                "max_m": m_r["max_m"], "final_m": m_r["final_m"],
                "vel_mae": m_r["velocity_mae"]}.items()},
            **{"oracle_mae_m": (round(m_o["mae_m"], 2)
                                if np.isfinite(m_o["mae_m"]) else ""),
               "oracle_final_m": (round(m_o["final_m"], 2)
                                  if np.isfinite(m_o["final_m"]) else "")},
        })

    # Attach correction curves onto the metric dicts for plotting
    for name in ["V0.2", "Oracle"]:
        for d in BLACKOUT_DURATIONS:
            for m, curve in zip(all_met["d"][name][d], corrections[name][d]):
                m["correction_curve"] = curve

    save_comparison_csv(csv_rows, OUT_DIR)

    stats = {name: summary_stats(all_met["d"], name) for name in ["A0", "V0.2", "Oracle"]}

    for d in BLACKOUT_DURATIONS:
        plot_pos_error_vs_time(
            {name: all_met["d"][name][d] for name in ["A0", "V0.2", "Oracle"]},
            d, fig_out,
        )
    plot_final_error(all_met["d"], fig_out)
    plot_three_way_mae(all_met["d"], fig_out)
    plot_correction_magnitude(all_met["d"], fig_out)

    # Predicted vs target residual scatter on TEST (Seg3) windows
    print("\n  Pred vs target residual diagnostic on Test (Seg3) windows...")
    targets_all = v2c.compute_v02_targets(data, segments, ve_cls, vn_cls)
    windows_all = mc.build_windows(features, targets_all, data["sync_time"],
                                   segments, context_len=args.context_len, stride=5)
    test_windows = [w for w in windows_all if w["segment_idx"] == 3]
    if test_windows:
        plot_pred_vs_target(model, features, targets_all, test_windows, norm,
                            args.context_len, device, fig_out)
    else:
        print("  [skip] no Seg3 test windows found for pred-vs-target diagnostic.")

    config = {"n_features": v2c.V02_N_FEATURES, "stride": 5}
    generate_recursive_report(stats, config, features_cfg, OUT_DIR)

    # Verdict
    rec_imps = []
    orc_imps = []
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]["mae_m_mean"]
        rec = stats["V0.2"].get(d, {}).get("mae_m_mean", a0)
        orc = stats["Oracle"].get(d, {}).get("mae_m_mean", a0)
        rec_imps.append(improvement(a0, rec))
        orc_imps.append(improvement(a0, orc))
    r_imp = np.nanmean(rec_imps)
    o_imp = np.nanmean(orc_imps)
    if not np.isfinite(r_imp):
        verdict = "CASE UNKNOWN"
    elif r_imp < 0:
        verdict = "CASE C"
    elif o_imp > 0 and r_imp >= 0.5 * o_imp:
        verdict = "CASE A"
    else:
        verdict = "CASE B"

    generate_v02_report(stats, {"params": "4194",
                                "n_features": v2c.V02_N_FEATURES,
                                "stride": 5,
                                "hidden_size": 32},
                        features_cfg, verdict, OUT_DIR)

    # Console summary
    print("\n" + "=" * 70)
    print("V0.2 RECURSIVE SUMMARY")
    print("=" * 70)
    print(f"  {'Dur':>5s} | {'A0 MAE':>9s} | {'V02 MAE':>9s} | {'Orc MAE':>9s} | {'Imp A0':>8s} | {'Imp Orc':>8s}")
    print("-" * 70)
    for d in sorted(stats["A0"].keys()):
        a0 = stats["A0"][d]["mae_m_mean"]
        rec = stats["V0.2"].get(d, {}).get("mae_m_mean", a0)
        orc = stats["Oracle"].get(d, {}).get("mae_m_mean", a0)
        imp_a0 = (a0 - rec) / a0 * 100 if a0 > 0 else 0
        imp_orc = (orc - rec) / orc * 100 if orc > 0 else 0
        print(f"  {d:>4d}s | {a0:>8.1f}m | {rec:>8.1f}m | {orc:>8.1f}m | {imp_a0:>+7.1f}% | {imp_orc:>+7.1f}%")
    print(f"\n  Verdict: {verdict}")
    print(f"  Outputs: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()