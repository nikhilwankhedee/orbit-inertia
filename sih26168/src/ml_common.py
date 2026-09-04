#!/usr/bin/env python3
"""
ML Pipeline — Shared Data Infrastructure (SIH26168)
====================================================

Reusable utilities for ML residual-learning experiments:
  - Data loading + column matching (semantic/mojibake-robust)
  - Segment detection from timestamp gaps
  - ENU coordinate system (local tangent plane)
  - Classical DR velocity propagation (A0 yaw-only kinematic)
  - Feature extraction (16-dim phone + vehicle)
  - Target computation (2D EN velocity residual)
  - Sliding-window dataset construction
  - Segment-level train/val/test split
  - Z-score normalization (fit on train only)
  - PyTorch Dataset + DataLoader helpers

Usage:
  import ml_common as mc
  data, cols, df = mc.load_data()
  segments = mc.detect_segments(data["sync_time"])
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT / "processed" / "S4_synced.csv"
OUT_DIR = SRC_DIR.parent / "outputs" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
DT_NOMINAL = 0.1
DT_MAX_GAP = 0.2
GPS_SPEED_FACTOR = 3.6
R_EARTH = 6371000.0

# Feature columns (canonical key → semantic keywords for matching)
FEATURE_SPECS = {
    "accel_x":      ["accelerometer x"],
    "accel_y":      ["accelerometer y"],
    "accel_z":      ["accelerometer z"],
    "gravity_x":    ["gravity x"],
    "gravity_y":    ["gravity y"],
    "gravity_z":    ["gravity z"],
    "gyro_pitch":   ["gyroscope pitch"],
    "phone_speed":  ["gps speed"],
    "phone_acc":    ["gps accuracy"],
    "veh_velocity": ["velocity (km/hr)"],
    "veh_heading":  ["heading (degrees)"],
    "veh_yaw_rate": ["yaw rate (deg/sec)"],
    "steering":     ["steering angle"],
    "whl_fl":       ["wheel speed front left"],
    "whl_fr":       ["wheel speed front right"],
    "whl_rl":       ["wheel speed rear left"],
}

# Additional non-feature columns we load
META_SPECS = {
    "sync_time":    ["sync_time_s"],
    "phone_date":   ["date (yyyy"],
    "phone_lat":    ["gps latitude"],
    "phone_lon":    ["gps longitude"],
    "veh_lat":      ["latitude (degrees)"],
    "veh_lon":      ["longitude (degrees)"],
}

# Indices into FEATURE_SPECS in order (deterministic feature ordering)
FEATURE_KEYS = [
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_pitch",
    "phone_speed", "phone_acc",
    "veh_velocity", "veh_heading", "veh_yaw_rate",
    "steering",
    "whl_fl", "whl_fr", "whl_rl",
]

N_FEATURES = len(FEATURE_KEYS)  # 16

# Feature groups for normalization (accel-like vs angular vs rate-like)
FEATURE_GROUPS = {
    "accel":   ["accel_x", "accel_y", "accel_z"],
    "gravity": ["gravity_x", "gravity_y", "gravity_z"],
    "angular": ["gyro_pitch", "veh_heading", "veh_yaw_rate"],
    "speed":   ["phone_speed", "phone_acc", "veh_velocity"],
    "steering":["steering"],
    "wheel":   ["whl_fl", "whl_fr", "whl_rl"],
}


# ──────────────────────────────────────────────────────────────
# COLUMN MATCHING
# ──────────────────────────────────────────────────────────────
def match_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Match required columns using semantic keywords.

    Priority: for vehicle columns, skip columns starting with 'GPS'.
    """
    matched = {}
    available = list(df.columns)
    all_specs = {}
    all_specs.update(FEATURE_SPECS)
    all_specs.update(META_SPECS)

    print("=" * 60)
    print("COLUMN MATCHING (ML pipeline)")
    print("=" * 60)

    for key, keywords in all_specs.items():
        found = False
        is_vehicle = key.startswith("veh_")

        for col in available:
            col_lower = col.lower()
            if all(kw in col_lower for kw in keywords):
                if is_vehicle and col_lower.startswith("gps"):
                    continue
                matched[key] = col
                print(f"  {key:25s} -> {col}")
                found = True
                break
        if not found:
            print(f"\n  *** FAILED TO MATCH: {key}")
            print(f"      Keywords: {keywords}")
            sys.exit(1)

    print("=" * 60)
    return matched


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_data() -> Tuple[Dict[str, np.ndarray], Dict[str, str], pd.DataFrame]:
    """Load and validate S4_synced.csv, return (data_dict, col_map, df)."""
    print(f"\nLoading: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    cols = match_columns(df)

    data = {}
    for key in {**FEATURE_SPECS, **META_SPECS}:
        col = cols[key]  # use matched column name
        if key == "phone_date":
            data[key] = df[col].values
            continue
        try:
            data[key] = df[col].values.astype(np.float64)
        except (ValueError, TypeError):
            print(f"  Warning: {key} ({col}) not convertible to float")
            data[key] = df[col].values

    data["phone_date_str"] = df[cols["phone_date"]].values
    data["sync_time"] = df[cols["sync_time"]].values.astype(np.float64)

    print(f"\n  SYNC_TIME range: [{data['sync_time'][0]:.3f}, {data['sync_time'][-1]:.3f}] s")
    print(f"  Duration: {data['sync_time'][-1] - data['sync_time'][0]:.1f} s")

    return data, cols, df


# ──────────────────────────────────────────────────────────────
# SEGMENT DETECTION
# ──────────────────────────────────────────────────────────────
def detect_segments(sync_time: np.ndarray) -> List[dict]:
    """Detect segments from timestamp gaps > DT_MAX_GAP."""
    dt = np.diff(sync_time, prepend=sync_time[0] - DT_NOMINAL)
    gap_indices = np.where(dt > DT_MAX_GAP)[0]

    segments = []
    boundaries = [0] + list(gap_indices) + [len(sync_time)]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        gap_size = dt[boundaries[i]] if i > 0 else 0
        segments.append({
            "start": start,
            "end": end,
            "n_rows": end - start,
            "gap_before_s": gap_size,
            "t_start": sync_time[start],
            "t_end": sync_time[end - 1],
            "duration_s": sync_time[end - 1] - sync_time[start],
        })

    print(f"\n  Segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"    Seg {i}: [{seg['start']}, {seg['end']}) "
              f"({seg['n_rows']} rows, {seg['duration_s']:.1f} s)"
              f"  gap_before={seg['gap_before_s']:.3f} s")
    return segments


# ──────────────────────────────────────────────────────────────
# ENU COORDINATE SYSTEM
# ──────────────────────────────────────────────────────────────
def ll2enu(lat, lon, alt, lat0, lon0, alt0):
    """Local ENU via tangent-plane approximation."""
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east  = dlon * np.cos(lat0_rad) * R_EARTH
    north = dlat * R_EARTH
    up    = alt - alt0
    return east, north, up


def enu2ll(east, north, up, lat0, lon0, alt0):
    """Inverse ENU to lat/lon/alt."""
    lat0_rad = np.radians(lat0)
    lat = np.degrees(north / R_EARTH) + lat0
    lon = np.degrees(east / (np.cos(lat0_rad) * R_EARTH)) + lon0
    alt = up + alt0
    return lat, lon, alt


# ──────────────────────────────────────────────────────────────
# CLASSICAL DR VELOCITY PROPAGATION (A0)
# ──────────────────────────────────────────────────────────────
def propagate_classical_dr_velocity(
    gyro_pitch: np.ndarray,
    sync_time: np.ndarray,
    init_heading_rad: float,
    init_speed_ms: float,
    n_eval: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate classical A0 DR: heading from gyro, speed constant.

    Returns (headings_rad[n_eval], ve[n_eval], vn[n_eval]).
    ve/vn are the classical DR velocity components in ENU.
    """
    headings = np.zeros(n_eval)
    ve = np.zeros(n_eval)
    vn = np.zeros(n_eval)

    headings[0] = init_heading_rad
    ve[0] = init_speed_ms * np.cos(init_heading_rad)
    vn[0] = init_speed_ms * np.sin(init_heading_rad)

    h = init_heading_rad
    for k in range(1, n_eval):
        dt = sync_time[k] - sync_time[k - 1]
        if dt > DT_MAX_GAP or dt <= 0:
            headings[k] = h
            ve[k] = init_speed_ms * np.cos(h)
            vn[k] = init_speed_ms * np.sin(h)
            continue
        h = h + gyro_pitch[k - 1] * dt
        headings[k] = h
        ve[k] = init_speed_ms * np.cos(h)
        vn[k] = init_speed_ms * np.sin(h)

    return headings, ve, vn


def compute_reference_en_velocity(
    veh_lat: np.ndarray,
    veh_lon: np.ndarray,
    sync_time: np.ndarray,
    i0: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference (ground-truth) EN velocity via central differences.

    Returns (ve_ref[n], vn_ref[n]) in m/s.
    Boundary points use forward/backward differences.
    """
    ve = np.zeros(n)
    vn = np.zeros(n)
    lat0 = veh_lat[i0]
    lon0 = veh_lon[i0]

    for k in range(n):
        idx = i0 + k
        if k == 0:
            idx1 = idx
            idx2 = min(idx + 1, i0 + n - 1)
        elif k == n - 1:
            idx1 = max(idx - 1, i0)
            idx2 = idx
        else:
            idx1 = idx - 1
            idx2 = idx + 1

        dt = sync_time[idx2] - sync_time[idx1]
        if dt <= 0 or dt > DT_MAX_GAP * 2:
            continue

        e1, n1, _ = ll2enu(veh_lat[idx1], veh_lon[idx1], 0.0, lat0, lon0, 0.0)
        e2, n2, _ = ll2enu(veh_lat[idx2], veh_lon[idx2], 0.0, lat0, lon0, 0.0)

        ve[k] = (e2 - e1) / dt
        vn[k] = (n2 - n1) / dt

    return ve, vn


# ──────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────
def extract_features(data: dict) -> np.ndarray:
    """Build (N, N_FEATURES) feature array from raw data dict.

    Feature ordering follows FEATURE_KEYS.
    Phone GPS speed is converted from m/s to km/h for consistency.
    Heading is converted from degrees to radians.
    """
    n = len(data["sync_time"])
    features = np.zeros((n, N_FEATURES), dtype=np.float64)

    for i, key in enumerate(FEATURE_KEYS):
        raw = data[key].copy()

        if key == "phone_speed":
            raw = raw * GPS_SPEED_FACTOR  # m/s → km/h
        elif key in ("veh_heading",):
            raw = np.radians(raw)  # degrees → radians
        elif key in ("veh_yaw_rate",):
            raw = np.radians(raw)  # deg/s → rad/s

        features[:, i] = raw

    return features


def compute_velocity_targets(
    data: dict,
    segments: List[dict],
    offset_s: float = 1.81,
) -> np.ndarray:
    """Compute 2D EN velocity residual targets for the entire dataset.

    target[k] = (veh_velocity_ref_EN[k] - classical_DR_velocity_EN[k])

    Classical DR velocity is computed per-segment using initial conditions
    at segment start (no +1.81s offset — offset is for position evaluation only).

    Returns (N, 2) array: [delta_ve, delta_vn] in m/s.
    """
    n = len(data["sync_time"])
    targets = np.zeros((n, 2), dtype=np.float64)

    for seg in segments:
        s, e = seg["start"], seg["end"]
        seg_len = e - s

        # Reference EN velocity (central differences)
        ref_ve, ref_vn = compute_reference_en_velocity(
            data["veh_lat"], data["veh_lon"], data["sync_time"],
            s, seg_len,
        )

        # Classical DR velocity for this segment
        init_heading = np.radians(data["veh_heading"][s])
        init_speed = data["veh_velocity"][s] / 3.6  # km/h → m/s
        _, cls_ve, cls_vn = propagate_classical_dr_velocity(
            data["gyro_pitch"][s:e],
            data["sync_time"][s:e],
            init_heading,
            init_speed,
            seg_len,
        )

        targets[s:e, 0] = ref_ve - cls_ve
        targets[s:e, 1] = ref_vn - cls_vn

    return targets


# ──────────────────────────────────────────────────────────────
# WINDOWING
# ──────────────────────────────────────────────────────────────
def build_windows(
    features: np.ndarray,
    targets: np.ndarray,
    sync_time: np.ndarray,
    segments: List[dict],
    context_len: int = 20,
    stride: int = 5,
) -> List[dict]:
    """Build sliding-window training samples.

    Each window: features[i-context_len+1 : i+1] → target[i].

    Windows never cross segment boundaries. Returns list of dicts with:
      - feature_window: (context_len, N_FEATURES)
      - target: (2,)
      - segment_idx: int
      - abs_idx: int (index into full dataset)
      - t: float (sync_time at prediction point)
    """
    windows = []

    for seg_idx, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        seg_len = e - s

        if seg_len < context_len + 1:
            continue

        # Start from context_len to have full context
        for i in range(context_len, seg_len, stride):
            abs_idx = s + i
            feat_win = features[abs_idx - context_len + 1: abs_idx + 1]
            tgt = targets[abs_idx]

            # Skip windows with NaN/Inf in features or targets
            if not np.all(np.isfinite(feat_win)) or not np.all(np.isfinite(tgt)):
                continue

            windows.append({
                "feature_window": feat_win,
                "target": tgt,
                "segment_idx": seg_idx,
                "abs_idx": abs_idx,
                "t": sync_time[abs_idx],
            })

    print(f"\n  Windows built: {len(windows)} (context={context_len}, stride={stride})")
    for si in range(len(segments)):
        n_in_seg = sum(1 for w in windows if w["segment_idx"] == si)
        if n_in_seg > 0:
            print(f"    Seg {si}: {n_in_seg} windows")

    return windows


# ──────────────────────────────────────────────────────────────
# SEGMENT-LEVEL TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────────────────────────
def split_windows(
    windows: List[dict],
    train_segs: List[int] = None,
    val_segs: List[int] = None,
    test_segs: List[int] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split windows by segment index. Default: Seg0+Seg1=Train, Seg2=Val, Seg3=Test."""
    if train_segs is None:
        train_segs = [0, 1]
    if val_segs is None:
        val_segs = [2]
    if test_segs is None:
        test_segs = [3]

    train = [w for w in windows if w["segment_idx"] in train_segs]
    val   = [w for w in windows if w["segment_idx"] in val_segs]
    test  = [w for w in windows if w["segment_idx"] in test_segs]

    print(f"\n  Split: train={len(train)} (segs {train_segs}), "
          f"val={len(val)} (segs {val_segs}), "
          f"test={len(test)} (segs {test_segs})")
    return train, val, test


# ──────────────────────────────────────────────────────────────
# NORMALIZATION
# ──────────────────────────────────────────────────────────────
def compute_normalization(windows: List[dict]) -> dict:
    """Compute z-score normalization from training windows.

    Returns dict with 'feat_mean', 'feat_std', 'tgt_mean', 'tgt_std'.
    """
    feat_list = [w["feature_window"] for w in windows]
    tgt_list = [w["target"] for w in windows]

    # Stack: feat_list is list of (context_len, N_FEATURES)
    # Flatten all timesteps for feature stats
    all_feats = np.concatenate([f.reshape(-1, N_FEATURES) for f in feat_list], axis=0)
    all_tgts = np.stack(tgt_list, axis=0)  # (n_windows, 2)

    feat_mean = np.mean(all_feats, axis=0)
    feat_std = np.std(all_feats, axis=0)
    feat_std[feat_std < 1e-8] = 1.0  # avoid division by zero

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


def apply_normalization(windows: List[dict], norm: dict) -> List[dict]:
    """Apply z-score normalization to a list of windows. Returns new list."""
    normed = []
    for w in windows:
        fw = (w["feature_window"] - norm["feat_mean"]) / norm["feat_std"]
        tgt = (w["target"] - norm["tgt_mean"]) / norm["tgt_std"]
        normed.append({
            "feature_window": fw,
            "target": tgt,
            "segment_idx": w["segment_idx"],
            "abs_idx": w["abs_idx"],
            "t": w["t"],
        })
    return normed


def save_normalization(norm: dict, path: Optional[Path] = None):
    """Save normalization stats to .npz file."""
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
    """Load normalization stats from .npz file."""
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
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────
def make_dataset(windows: List[dict]):
    """Convert window list to PyTorch TensorDataset.

    Returns (dataset, feature_tensor, target_tensor).
    """
    import torch
    from torch.utils.data import TensorDataset

    feats = np.stack([w["feature_window"] for w in windows], axis=0)
    tgts = np.stack([w["target"] for w in windows], axis=0)

    feat_t = torch.tensor(feats, dtype=torch.float32)
    tgt_t = torch.tensor(tgts, dtype=torch.float32)

    dataset = TensorDataset(feat_t, tgt_t)
    print(f"  Dataset: {len(dataset)} samples, features={feat_t.shape}, targets={tgt_t.shape}")
    return dataset


# ──────────────────────────────────────────────────────────────
# BLACKOUT WINDOW SELECTION (for evaluation)
# ──────────────────────────────────────────────────────────────
MIN_MOTION_THRESHOLD = 2.0
MAX_GPS_ACCURACY = 50


def select_blackout_windows(
    data: dict,
    segments: List[dict],
    durations: List[int],
) -> List[dict]:
    """Select blackout windows matching classical_dr_baseline criteria.

    Each window has start_idx, blackout_end_idx, eval_end_idx, duration_s.
    """
    windows = []

    for duration in durations:
        for seg in segments:
            s, e = seg["start"], seg["end"]
            seg_vel = data["veh_velocity"][s:e]
            eval_margin = min(30, duration)
            needed_s = duration + eval_margin

            if seg["duration_s"] < needed_s + 5:
                continue

            valid_starts = []
            for i in range(s, e - 10):
                t_start = data["sync_time"][i]
                pre_mask = (
                    (data["sync_time"][s:e] >= t_start - 10)
                    & (data["sync_time"][s:e] < t_start)
                )
                if pre_mask.sum() < 10:
                    continue
                if np.median(seg_vel[pre_mask]) < MIN_MOTION_THRESHOLD:
                    continue
                pre_acc = data["phone_acc"][s:e][pre_mask]
                if np.median(pre_acc) > MAX_GPS_ACCURACY:
                    continue

                j = i
                while j < e and data["sync_time"][j] - t_start < duration:
                    j += 1
                if j >= e:
                    continue
                k = j
                while k < e and data["sync_time"][k] - data["sync_time"][j - 1] < eval_margin:
                    k += 1
                if k >= e:
                    continue

                actual_duration = data["sync_time"][j - 1] - t_start
                if actual_duration < duration * 0.9:
                    continue
                valid_starts.append(i)

            if not valid_starts:
                continue

            n_select = min(3, len(valid_starts))
            indices = np.linspace(0, len(valid_starts) - 1, n_select, dtype=int)

            for idx in indices:
                i = valid_starts[idx]
                t_start = data["sync_time"][i]
                j = i
                while j < e and data["sync_time"][j] - t_start < duration:
                    j += 1
                k = j
                while k < e and data["sync_time"][k] - data["sync_time"][j - 1] < eval_margin:
                    k += 1

                pre_mask = (
                    (data["sync_time"][s:e] >= t_start - 10)
                    & (data["sync_time"][s:e] < t_start)
                )

                windows.append({
                    "duration_s": duration,
                    "start_idx": i,
                    "blackout_end_idx": j,
                    "eval_end_idx": min(k, e),
                    "t_start": t_start,
                    "t_blackout_end": data["sync_time"][j - 1],
                    "t_eval_end": data["sync_time"][min(k - 1, e - 1)],
                    "segment": seg,
                    "pre_velocity_kmh": float(np.median(seg_vel[pre_mask])),
                })

    print(f"\n  Blackout windows selected: {len(windows)}")
    for w in windows:
        print(f"    {w['duration_s']:3d}s | rows [{w['start_idx']}, {w['blackout_end_idx']}) "
              f"| T={w['t_start']:.1f}-{w['t_blackout_end']:.1f} "
              f"| pre_vel={w['pre_velocity_kmh']:.1f} km/h")
    return windows


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: FULL DATA PREPARATION
# ──────────────────────────────────────────────────────────────
def prepare_ml_data(
    context_len: int = 20,
    stride: int = 5,
    train_segs: Optional[List[int]] = None,
    val_segs: Optional[List[int]] = None,
    test_segs: Optional[List[int]] = None,
) -> dict:
    """Full data preparation pipeline. Returns a dict with everything needed.

    Returns dict with keys:
      data, cols, df, segments,
      features, targets,
      windows_all, windows_train/val/test (raw),
      norm_windows_train/val/test (normalized),
      norm (normalization stats),
      train_dataset, val_dataset, test_dataset (PyTorch TensorDatasets)
    """
    print("\n" + "=" * 70)
    print("ML DATA PREPARATION")
    print("=" * 70)

    data, cols, df = load_data()
    segments = detect_segments(data["sync_time"])

    print("\nExtracting features...")
    features = extract_features(data)
    print(f"  Feature matrix: {features.shape}")

    print("\nComputing velocity targets...")
    targets = compute_velocity_targets(data, segments)
    print(f"  Target matrix: {targets.shape}")
    print(f"  Target mean: [{targets[:, 0].mean():.4f}, {targets[:, 1].mean():.4f}] m/s")
    print(f"  Target std:  [{targets[:, 0].std():.4f}, {targets[:, 1].std():.4f}] m/s")

    print("\nBuilding windows...")
    windows_all = build_windows(features, targets, data["sync_time"],
                                segments, context_len=context_len, stride=stride)

    train_raw, val_raw, test_raw = split_windows(windows_all, train_segs, val_segs, test_segs)

    print("\nComputing normalization from training data...")
    norm = compute_normalization(train_raw)
    save_normalization(norm)

    train_norm = apply_normalization(train_raw, norm)
    val_norm = apply_normalization(val_raw, norm)
    test_norm = apply_normalization(test_raw, norm)

    print("\nCreating PyTorch datasets...")
    train_dataset = make_dataset(train_norm)
    val_dataset = make_dataset(val_norm)
    test_dataset = make_dataset(test_norm)

    return {
        "data": data,
        "cols": cols,
        "df": df,
        "segments": segments,
        "features": features,
        "targets": targets,
        "windows_all": windows_all,
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
# MAIN (smoke test)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ML COMMON — SMOKE TEST")
    print("=" * 70)
    result = prepare_ml_data()
    print("\nSmoke test passed.")
    print(f"  Train: {len(result['train_dataset'])} samples")
    print(f"  Val:   {len(result['val_dataset'])} samples")
    print(f"  Test:  {len(result['test_dataset'])} samples")
