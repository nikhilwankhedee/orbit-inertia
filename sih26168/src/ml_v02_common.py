#!/usr/bin/env python3
"""
V0.2 Deployment-Valid ML Pipeline — Shared Infrastructure (SIH26168)
====================================================================

Deployment-valid (phone-only) feature pipeline for the V0.2 GRU experiment.

The V0.1 recursive audit showed the V0 16-feature checkpoint is NOT
deployment-valid: it was trained on vehicle-CAN/GNSS channels that do not
exist on a GNSS-denied smartphone deployment, and its teacher-style benefit
(+34-45%) collapsed under a genuinely recursive rollout (+8.6%).

V0.2 retrains from scratch using strictly deployment-available inputs:
  7 measured phone-IMU channels + 2 internal navigation-state channels
  (nav_speed, nav_heading). The internal nav state is the A0 classical
  dead-reckoning state propagated from segment start — the SAME state the
  classical baseline and training targets are built from — so the model's
  input distribution during training and during recursive deployment is
  IDENTICAL. This removes the train/deployment distribution shift on the
  state channels that V0.1 identified as a root-cause of the collapse.

Feature set (9):
  accel_x/y/z, gravity_x/y/z, gyro_pitch      (measured phone IMU)
  nav_speed   (km/h, A0 classical speed, const per segment)
  nav_heading (rad, gyro-integrated from segment start)

Target (2D EN velocity residual, m/s):
  dv = v_reference_EN - v_classical_A0_EN
  Reference velocity comes from recorded vehicle GPS and is used ONLY for
  target construction / evaluation, NEVER as a model input.

Split (segment-level per V0/V0.1):
  Train Seg0+Seg1, Val Seg2, Test Seg3.

Usage:
  import ml_v02_common as v2c
  result = v2c.prepare_v02_data()
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import ml_common as mc

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
OUT_DIR = SRC_DIR.parent / "outputs" / "ml_v02"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# DEPLOYMENT FEATURE SET (9)
# ──────────────────────────────────────────────────────────────
# Every feature must be available on a smartphone during a GNSS blackout.
#   measured phone IMU (7): accel_x/y/z, gravity_x/y/z, gyro_pitch
#   internal nav state (2): nav_speed (km/h), nav_heading (rad)
# Forbidden during blackout: vehicle CAN, steering, wheel speeds, future GPS,
# phone/vehicle GPS-derived speed/accuracy, ground-truth position/velocity.
DEPLOYMENT_FEATURE_KEYS = [
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_pitch",
    "nav_speed", "nav_heading",
]
V02_N_FEATURES = len(DEPLOYMENT_FEATURE_KEYS)  # 9

V02_TARGET_KEYS = ["delta_ve", "delta_vn"]     # 2D EN velocity residual (m/s)

# Column index of the internal-state channels inside the 9-dim feature vector
NAV_SPEED_IDX = DEPLOYMENT_FEATURE_KEYS.index("nav_speed")      # 7
NAV_HEADING_IDX = DEPLOYMENT_FEATURE_KEYS.index("nav_heading")  # 8

# The 7 measured phone-IMU channel indices
IMU_FEATURE_IDX = list(range(7))


# ──────────────────────────────────────────────────────────────
# A0 CLASSICAL STATE (identical formulation for train + deploy)
# ──────────────────────────────────────────────────────────────
def compute_a0_state(
    data: dict,
    segments: List[dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A0 classical DR state propagated from segment start across all rows.

    Returns (ve_cls, vn_cls, heading_cls, speed_kmh), each length N:
      - ve_cls/vn_cls: classical EN velocity (m/s)
      - heading_cls:   gyro-integrated heading (rad)
      - speed_kmh:     classical speed (km/h), constant per segment

    Init at segment start from veh_heading / veh_velocity (available before
    any blackout). This is the same state used to build training targets and
    the nav_speed/nav_heading channels, so training and recursive deployment
    share a single consistent state definition.
    """
    n = len(data["sync_time"])
    ve_cls = np.zeros(n, dtype=np.float64)
    vn_cls = np.zeros(n, dtype=np.float64)
    heading_cls = np.zeros(n, dtype=np.float64)
    speed_kmh = np.zeros(n, dtype=np.float64)

    for seg in segments:
        s, e = seg["start"], seg["end"]
        seg_len = e - s
        init_heading = np.radians(data["veh_heading"][s])
        init_speed = data["veh_velocity"][s] / mc.GPS_SPEED_FACTOR  # km/h -> m/s

        h, ve, vn = mc.propagate_classical_dr_velocity(
            data["gyro_pitch"][s:e],
            data["sync_time"][s:e],
            init_heading,
            init_speed,
            seg_len,
        )
        ve_cls[s:e] = ve
        vn_cls[s:e] = vn
        heading_cls[s:e] = h
        speed_kmh[s:e] = init_speed * mc.GPS_SPEED_FACTOR

    return ve_cls, vn_cls, heading_cls, speed_kmh


# ──────────────────────────────────────────────────────────────
# DEPLOYMENT FEATURES (9-dim)
# ──────────────────────────────────────────────────────────────
def build_deployment_features(
    data: dict,
    nav_speed_kmh: np.ndarray,
    nav_heading_rad: np.ndarray,
) -> np.ndarray:
    """Build the (N, 9) deployment-valid feature array.

    Ordering follows DEPLOYMENT_FEATURE_KEYS:
      0-2 accel_x/y/z, 3-5 gravity_x/y/z, 6 gyro_pitch  (measured)
      7 nav_speed (km/h), 8 nav_heading (rad)           (internal A0 state)
    """
    n = len(data["sync_time"])
    features = np.zeros((n, V02_N_FEATURES), dtype=np.float64)

    for i, key in enumerate(DEPLOYMENT_FEATURE_KEYS):
        if key == "nav_speed":
            features[:, i] = nav_speed_kmh
        elif key == "nav_heading":
            features[:, i] = nav_heading_rad
        elif key == "gyro_pitch":
            features[:, i] = data["gyro_pitch"]  # rad/s (measured)
        else:
            features[:, i] = data[key]           # accel/gravity m/s^2 (measured)

    return features


# ──────────────────────────────────────────────────────────────
# TARGETS (2D EN velocity residual)
# ──────────────────────────────────────────────────────────────
def compute_v02_targets(
    data: dict,
    segments: List[dict],
    ve_cls: np.ndarray,
    vn_cls: np.ndarray,
) -> np.ndarray:
    """(N, 2) target dv = v_reference_EN - v_classical_A0_EN (m/s).

    Reference EN velocity from recorded vehicle GPS (target construction /
    evaluation only, never a model input).
    """
    n = len(data["sync_time"])
    targets = np.zeros((n, 2), dtype=np.float64)

    for seg in segments:
        s, e = seg["start"], seg["end"]
        seg_len = e - s
        ref_ve, ref_vn = mc.compute_reference_en_velocity(
            data["veh_lat"], data["veh_lon"], data["sync_time"],
            s, seg_len,
        )
        targets[s:e, 0] = ref_ve - ve_cls[s:e]
        targets[s:e, 1] = ref_vn - vn_cls[s:e]

    return targets


# ──────────────────────────────────────────────────────────────
# NORMALIZATION (fit on TRAINING windows only)
# ──────────────────────────────────────────────────────────────
def compute_normalization(windows: List[dict]) -> dict:
    """Z-score normalization from training windows (9 features, 2 targets)."""
    feat_list = [w["feature_window"] for w in windows]
    tgt_list = [w["target"] for w in windows]

    nfeat = feat_list[0].shape[1]
    all_feats = np.concatenate([f.reshape(-1, nfeat) for f in feat_list], axis=0)
    all_tgts = np.stack(tgt_list, axis=0)

    feat_mean = np.mean(all_feats, axis=0)
    feat_std = np.std(all_feats, axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    tgt_mean = np.mean(all_tgts, axis=0)
    tgt_std = np.std(all_tgts, axis=0)
    tgt_std[tgt_std < 1e-8] = 1.0

    norm = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "n_train": len(windows),
    }

    print(f"\n  Normalization stats from {len(windows)} training windows:")
    print(f"    Feature mean range: [{feat_mean.min():.4f}, {feat_mean.max():.4f}]")
    print(f"    Feature std  range: [{feat_std.min():.4f}, {feat_std.max():.4f}]")
    print(f"    Target mean: {tgt_mean}")
    print(f"    Target std:  {tgt_std}")

    return norm


def save_normalization(norm: dict, path: Optional[Path] = None):
    if path is None:
        path = OUT_DIR / "normalization.npz"
    np.savez(
        path,
        feat_mean=norm["feat_mean"],
        feat_std=norm["feat_std"],
        tgt_mean=norm["tgt_mean"],
        tgt_std=norm["tgt_std"],
        n_train=norm["n_train"],
    )
    print(f"  Saved normalization: {path}")


def load_normalization(path: Optional[Path] = None) -> dict:
    if path is None:
        path = OUT_DIR / "normalization.npz"
    npz = np.load(path)
    return {
        "feat_mean": npz["feat_mean"],
        "feat_std": npz["feat_std"],
        "tgt_mean": npz["tgt_mean"],
        "tgt_std": npz["tgt_std"],
        "n_train": int(npz["n_train"]),
    }


# ──────────────────────────────────────────────────────────────
# FULL DATA PREPARATION
# ──────────────────────────────────────────────────────────────
def prepare_v02_data(
    context_len: int = 20,
    stride: int = 5,
    train_segs: Optional[List[int]] = None,
    val_segs: Optional[List[int]] = None,
    test_segs: Optional[List[int]] = None,
) -> dict:
    """Full V0.2 data preparation. Returns dict with everything needed."""
    print("\n" + "=" * 70)
    print("V0.2 DEPLOYMENT-VALID DATA PREPARATION")
    print("=" * 70)

    data, cols, df = mc.load_data()
    segments = mc.detect_segments(data["sync_time"])

    print("\nComputing A0 classical state (shared train/deploy definition)...")
    ve_cls, vn_cls, heading_cls, speed_kmh = compute_a0_state(data, segments)

    print("\nBuilding deployment features (9-dim: 7 IMU + nav_speed + nav_heading)...")
    features = build_deployment_features(data, speed_kmh, heading_cls)
    print(f"  Feature matrix: {features.shape}")

    print("\nComputing velocity residual targets...")
    targets = compute_v02_targets(data, segments, ve_cls, vn_cls)
    print(f"  Target matrix: {targets.shape}")
    print(f"  Target mean: [{targets[:, 0].mean():.4f}, {targets[:, 1].mean():.4f}] m/s")
    print(f"  Target std:  [{targets[:, 0].std():.4f}, {targets[:, 1].std():.4f}] m/s")

    print("\nBuilding windows...")
    windows_all = mc.build_windows(features, targets, data["sync_time"],
                                   segments, context_len=context_len, stride=stride)

    train_raw, val_raw, test_raw = mc.split_windows(windows_all, train_segs, val_segs, test_segs)

    print("\nComputing normalization from training data only...")
    norm = compute_normalization(train_raw)
    save_normalization(norm)

    train_norm = mc.apply_normalization(train_raw, norm)
    val_norm = mc.apply_normalization(val_raw, norm)
    test_norm = mc.apply_normalization(test_raw, norm)

    print("\nCreating PyTorch datasets...")
    train_dataset = mc.make_dataset(train_norm)
    val_dataset = mc.make_dataset(val_norm)
    test_dataset = mc.make_dataset(test_norm)

    return {
        "data": data,
        "cols": cols,
        "df": df,
        "segments": segments,
        "features": features,
        "targets": targets,
        "ve_cls": ve_cls,
        "vn_cls": vn_cls,
        "nav_heading_rad": heading_cls,
        "nav_speed_kmh": speed_kmh,
        "windows_train_raw": train_raw,
        "windows_val_raw": val_raw,
        "windows_test_raw": test_raw,
        "norm": norm,
        "windows_train_norm": train_norm,
        "windows_val_norm": val_norm,
        "windows_test_norm": test_norm,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


# ──────────────────────────────────────────────────────────────
# ROLLOUT FEATURE ROW PROVIDERS
# ──────────────────────────────────────────────────────────────
def recursive_row_fn(features: np.ndarray) -> Callable[[int], np.ndarray]:
    """Row provider for the RECURSIVE (primary) rollout.

    Returns features[j] for any absolute index j: measured IMU + A0 internal
    nav state — the exact training distribution throughout the blackout.
    No reference channel is read during the blackout.
    """
    return lambda j: features[j]


def oracle_row_fn(
    data: dict,
    features: np.ndarray,
    window: dict,
) -> Callable[[int], np.ndarray]:
    """Row provider for the ORACLE-STATE (teacher-style, DIAGNOSTIC ONLY) rollout.

    For indices >= blackout start, nav_speed/nav_heading channels are replaced
    by the RECORDED REFERENCE state (|v_ref| and veh_heading). This measures
    how much of the residual the model could explain with near-perfect state
    inputs — the V0 analog of leaked teacher-style evaluation — and is NOT a
    deployment-valid result.
    """
    i0 = window["start_idx"]
    i2 = window["eval_end_idx"]
    oracle = features[i0:i2].copy()

    ref_ve, ref_vn = mc.compute_reference_en_velocity(
        data["veh_lat"], data["veh_lon"], data["sync_time"],
        i0, i2 - i0,
    )
    oracle[:, NAV_SPEED_IDX] = np.hypot(ref_ve, ref_vn) * mc.GPS_SPEED_FACTOR  # km/h
    oracle[:, NAV_HEADING_IDX] = np.radians(data["veh_heading"][i0:i2])       # rad

    return lambda j: oracle[j - i0] if j >= i0 else features[j]


def run_v02_rollout(
    model,
    row_fn: Callable[[int], np.ndarray],
    data: dict,
    norm: dict,
    window: dict,
    context_len: int,
    device,
) -> dict:
    """Causal, recursive ML-corrected DR rollout on 9-dim deployment features.

    row_fn(j) returns the (9,) feature row for absolute index j (recursive or
    oracle-state provider). For each step k:
      1. Causal context buffer of normalized rows (auto-regressive, no future).
      2. GRU predicts dv = (dve, dvn) in m/s.
      3. v_classical comes from the nav-state channels at row k (A0 speed and
         gyro-integrated heading); v_corrected = v_classical + dv.
      4. Position accumulates v_corrected. Nothing is reset to ground truth,
         no reference velocity/GPS/CAN is injected.

    Returns the standard result dict used by the metric helpers.
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

    f_mean = norm["feat_mean"]
    f_std = norm["feat_std"]
    tgt_mean = norm["tgt_mean"]
    tgt_std = norm["tgt_std"]

    # Pre-blackout context buffer (causal, includes measured rows). Rows are
    # pulled through row_fn so oracle diagnostics get base features pre-loss.
    buf_start = i0 - context_len + 1
    pre = [row_fn(j) for j in range(max(0, buf_start), i0)]
    pad = context_len - 1 - len(pre)
    if pad > 0:
        pre = [np.zeros(V02_N_FEATURES) for _ in range(pad)] + pre
    buf = np.stack(pre).astype(np.float64)
    buf = (buf - f_mean) / f_std
    if len(buf) > context_len - 1:
        buf = buf[-(context_len - 1):]

    headings = np.zeros(n_eval)
    ve_out = np.zeros(n_eval)
    vn_out = np.zeros(n_eval)
    e_pos = east0
    n_pos = north0
    e_pos_arr = np.zeros(n_eval)
    n_pos_arr = np.zeros(n_eval)
    delta_v_arr = np.zeros((n_eval, 2))
    speeds = np.zeros(n_eval)

    import torch

    for k in range(n_eval):
        j = i0 + k
        row = row_fn(j).astype(np.float64)

        # Classical A0 velocity from the nav-state channels of this row
        speed_ms = row[NAV_SPEED_IDX] / mc.GPS_SPEED_FACTOR
        heading = row[NAV_HEADING_IDX]
        cls_ve = speed_ms * np.cos(heading)
        cls_vn = speed_ms * np.sin(heading)

        # Predict residual from normalized causal buffer
        row_n = (row - f_mean) / f_std
        buf = np.concatenate([buf, row_n[None, :]], axis=0)
        if len(buf) > context_len:
            buf = buf[1:]

        with torch.no_grad():
            x = torch.tensor(buf[None, :, :], dtype=torch.float32).to(device)
            pred = model(x).cpu().numpy()[0]  # (2,) normalized

        dve, dvn = pred * tgt_std + tgt_mean  # m/s

        ve_corr = cls_ve + dve
        vn_corr = cls_vn + dvn

        if k == 0:
            dt = mc.DT_NOMINAL
        else:
            dt = data["sync_time"][j] - data["sync_time"][j - 1]
        if dt <= 0 or dt > mc.DT_MAX_GAP:
            dt = 0.0
        e_pos = e_pos + ve_corr * dt
        n_pos = n_pos + vn_corr * dt

        headings[k] = heading
        ve_out[k] = ve_corr
        vn_out[k] = vn_corr
        e_pos_arr[k] = e_pos
        n_pos_arr[k] = n_pos
        delta_v_arr[k] = [dve, dvn]
        speeds[k] = np.hypot(ve_corr, vn_corr)

    return {
        "headings": headings,
        "positions_east": e_pos_arr,
        "positions_north": n_pos_arr,
        "velocities_east": ve_out,
        "velocities_north": vn_out,
        "delta_v": delta_v_arr,
        "speeds": speeds,
        "init_heading_deg": np.degrees(headings[0]),
        "init_speed_ms": (row_fn(i0)[NAV_SPEED_IDX] / mc.GPS_SPEED_FACTOR),
        "init_east": east0,
        "init_north": north0,
        "ref_lat0": ref_lat0,
        "ref_lon0": ref_lon0,
    }


# ──────────────────────────────────────────────────────────────
# MAIN (smoke test)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ML V0.2 COMMON — SMOKE TEST")
    print("=" * 70)
    result = prepare_v02_data()
    print("\nSmoke test passed.")
    print(f"  Train: {len(result['train_dataset'])} samples")
    print(f"  Val:   {len(result['val_dataset'])} samples")
    print(f"  Test:  {len(result['test_dataset'])} samples")
    print(f"  Feature dims: {result['features'].shape[1]}")