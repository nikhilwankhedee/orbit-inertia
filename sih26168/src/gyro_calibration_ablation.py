#!/usr/bin/env python3
"""
Gyro Calibration Ablation — Classical DR V0
=============================================

Three heading-rate variants:
  A0: yaw_rate = gyro_pitch                    (current baseline, s=1, b=0)
  A1: yaw_rate = s * gyro_pitch                (scale-calibrated)
  A2: yaw_rate = s * gyro_pitch + b            (scale + bias calibrated)

Purpose: determine how much of the classical DR failure comes from
simple gyro scale/bias mismatch versus irreducible/context-dependent error.

Calibration protocol:
  - Fit s, b on non-blackout data using least-squares regression.
  - The +1.81 s empirical inter-stream offset is used ONLY to align the
    reference yaw rate during calibration fitting (NOT during propagation).
  - Two calibration modes:
      (i)  in-sample: fit on ALL non-blackout data (diagnostic, has leakage)
      (ii) leave-one-window-out: for each blackout window, fit on all
           non-blackout data EXCEPT that window (leakage-safe)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT / "processed" / "S4_synced.csv"
OUT_DIR = SRC_DIR.parent / "outputs" / "dr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS (identical to classical_dr_baseline.py)
# ──────────────────────────────────────────────────────────────
DT_NOMINAL = 0.1
DT_MAX_GAP = 0.2
GPS_SPEED_FACTOR = 3.6
REF_ALIGNMENT_OFFSET_S = 1.81  # empirical inter-stream offset (evaluation/calibration only)

BLACKOUT_DURATIONS = [10, 30, 60, 120]
MIN_MOTION_THRESHOLD = 2.0
MAX_GPS_ACCURACY = 5.0
R_EARTH = 6371000.0


# ──────────────────────────────────────────────────────────────
# COLUMN MATCHING (robust, identical to baseline)
# ──────────────────────────────────────────────────────────────
REQUIRED_PHONE_COLS = {
    "accel_x": ["accelerometer x"],
    "accel_y": ["accelerometer y"],
    "accel_z": ["accelerometer z"],
    "gravity_x": ["gravity x"],
    "gravity_y": ["gravity y"],
    "gravity_z": ["gravity z"],
    "gyro_pitch": ["gyroscope pitch"],
    "sync_time": ["sync_time_s"],
    "phone_date": ["date (yyyy"],
    "phone_lat": ["gps latitude"],
    "phone_lon": ["gps longitude"],
    "phone_gps_speed": ["gps speed"],
    "phone_gps_accuracy": ["gps accuracy"],
}

REQUIRED_VEHICLE_COLS = {
    "veh_lat": ["latitude (degrees)"],
    "veh_lon": ["longitude (degrees)"],
    "veh_velocity": ["velocity (km/hr)"],
    "veh_heading": ["heading (degrees)"],
    "veh_yaw_rate": ["yaw rate (deg/sec)"],
}


def match_columns(df):
    matched = {}
    available = list(df.columns)
    all_required = {}
    all_required.update(REQUIRED_PHONE_COLS)
    all_required.update(REQUIRED_VEHICLE_COLS)

    print("=" * 60)
    print("COLUMN MATCHING")
    print("=" * 60)

    for key, keywords in all_required.items():
        found = False
        is_vehicle = key.startswith("veh_")
        for col in available:
            col_lower = col.lower()
            if all(kw in col_lower for kw in keywords):
                if is_vehicle and col_lower.startswith("gps"):
                    continue
                matched[key] = col
                print(f"  {key:25s} → {col}")
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
def load_data():
    print(f"\nLoading: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    cols = match_columns(df)

    data = {}
    for key, col in cols.items():
        if key == "phone_date":
            data[key] = df[col].values
            continue
        try:
            data[key] = df[col].values.astype(np.float64)
        except (ValueError, TypeError):
            print(f"  Warning: {key} ({col}) could not be converted to float")
            data[key] = df[col].values

    data["phone_date_str"] = df[cols["phone_date"]].values
    data["sync_time"] = df[cols["sync_time"]].values.astype(np.float64)

    print(f"\n  SYNC_TIME_S range: [{data['sync_time'][0]:.3f}, {data['sync_time'][-1]:.3f}] s")
    print(f"  Duration: {data['sync_time'][-1] - data['sync_time'][0]:.1f} s")

    return data, cols, df


# ──────────────────────────────────────────────────────────────
# SEGMENT DETECTION
# ──────────────────────────────────────────────────────────────
def detect_segments(sync_time):
    dt = np.diff(sync_time, prepend=sync_time[0] - DT_NOMINAL)
    gap_indices = np.where(dt > DT_MAX_GAP)[0]

    segments = []
    boundaries = [0] + list(gap_indices) + [len(sync_time)]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        gap_size = dt[boundaries[i]] if i > 0 else 0
        segments.append({
            "start": start, "end": end,
            "n_rows": end - start,
            "gap_before_s": gap_size,
            "t_start": sync_time[start],
            "t_end": sync_time[end - 1],
            "duration_s": sync_time[end - 1] - sync_time[start],
        })

    print(f"\n  Segments detected: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"    Seg {i}: rows [{seg['start']}, {seg['end']}) "
              f"({seg['n_rows']} rows, {seg['duration_s']:.1f} s)")
    return segments


# ──────────────────────────────────────────────────────────────
# COORDINATE SYSTEM: LOCAL ENU
# ──────────────────────────────────────────────────────────────
def ll2enu(lat, lon, alt, lat0, lon0, alt0):
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east = dlon * np.cos(lat0_rad) * R_EARTH
    north = dlat * R_EARTH
    up = alt - alt0
    return east, north, up


# ──────────────────────────────────────────────────────────────
# BLACKOUT WINDOW SELECTION (identical to baseline)
# ──────────────────────────────────────────────────────────────
def select_blackout_windows(data, segments, durations):
    windows = []

    for duration in durations:
        found_windows = []
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
                pre_mask = (data["sync_time"][s:e] >= t_start - 10) & \
                           (data["sync_time"][s:e] < t_start)
                if pre_mask.sum() < 10:
                    continue
                if np.median(seg_vel[pre_mask]) < MIN_MOTION_THRESHOLD:
                    continue
                pre_acc = data["phone_gps_accuracy"][s:e][pre_mask]
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
            if n_select == 1:
                indices = [0]
            else:
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

                pre_mask = (data["sync_time"][s:e] >= t_start - 10) & \
                           (data["sync_time"][s:e] < t_start)

                found_windows.append({
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

        windows.extend(found_windows)

    print(f"\n  Blackout windows selected: {len(windows)}")
    for w in windows:
        print(f"    {w['duration_s']:3d}s | rows [{w['start_idx']}, {w['blackout_end_idx']}) "
              f"| T={w['t_start']:.1f}-{w['t_blackout_end']:.1f} "
              f"| pre_vel={w['pre_velocity_kmh']:.1f} km/h")
    return windows


# ──────────────────────────────────────────────────────────────
# CALIBRATION FITTING
# ──────────────────────────────────────────────────────────────
def fit_calibration(gyro_pitch_cal, veh_yaw_rate_cal_deg, alignment_offset_s=0.0,
                    sync_time_cal=None):
    """Fit yaw_rate = s * gyro_pitch + b using least squares.

    If alignment_offset_s != 0 and sync_time_cal is provided, shift the
    reference yaw rate backward by alignment_offset_s seconds to align
    with the phone gyro signal BEFORE fitting. This is an evaluation/
    calibration alignment only — it does NOT alter stored timestamps.

    Returns dict with s, b, n_samples, calibration metrics, and the
    aligned calibration arrays.
    """
    gyro = np.asarray(gyro_pitch_cal, dtype=np.float64)
    veh_deg = np.asarray(veh_yaw_rate_cal_deg, dtype=np.float64)
    t_cal = np.asarray(sync_time_cal, dtype=np.float64) if sync_time_cal is not None else None

    # Optional alignment: shift reference yaw rate backward
    if alignment_offset_s != 0 and t_cal is not None:
        veh_deg_aligned = np.interp(
            t_cal - alignment_offset_s,  # query: shifted backward
            t_cal,                       # xp: original reference times
            veh_deg,                     # fp: original reference values
        )
    else:
        veh_deg_aligned = veh_deg

    # Convert to rad/s for fitting (same units as gyro_pitch)
    gyro_rad = gyro
    veh_rad = np.radians(veh_deg_aligned)

    valid = np.isfinite(gyro_rad) & np.isfinite(veh_rad)
    x = gyro_rad[valid]
    y = veh_rad[valid]
    n = int(valid.sum())

    if n < 2:
        return {"s": 1.0, "b": 0.0, "n_samples": n,
                "cal_rmse": float("nan"), "cal_mae": float("nan"),
                "cal_r": float("nan"), "sign_agreement": float("nan"),
                "x": x, "y": y, "veh_deg_aligned": veh_deg_aligned}

    # OLS: y = s*x + b
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-30:
        return {"s": 1.0, "b": 0.0, "n_samples": n,
                "cal_rmse": float("nan"), "cal_mae": float("nan"),
                "cal_r": float("nan"), "sign_agreement": float("nan"),
                "x": x, "y": y, "veh_deg_aligned": veh_deg_aligned}

    s = ss_xy / ss_xx
    b = y_mean - s * x_mean

    y_pred = s * x + b
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r = float(np.corrcoef(x, y)[0, 1])

    # Sign agreement during moderate motion
    moderate_mask = (np.abs(veh_deg_aligned) > 2) & (np.abs(gyro_rad) > 0.02) & valid
    if moderate_mask.sum() > 0:
        sign_agree = float(np.mean(
            np.sign(gyro_rad[moderate_mask]) == np.sign(np.radians(veh_deg_aligned[moderate_mask]))
        ))
    else:
        sign_agree = float("nan")

    return {
        "s": float(s), "b": float(b),
        "n_samples": n,
        "cal_rmse": rmse, "cal_mae": mae, "cal_r": r,
        "sign_agreement": sign_agree,
        "x": x, "y": y, "veh_deg_aligned": veh_deg_aligned,
    }


# ──────────────────────────────────────────────────────────────
# DR PROPAGATION (generalized for calibration variants)
# ──────────────────────────────────────────────────────────────
def run_dr_variant(data, window, cal_func):
    """Yaw-only kinematic DR with arbitrary yaw-rate calibration.

    cal_func: callable(gyro_pitch_raw, dt) → yaw_rate_rad_per_s
              Must be causal (uses only the current sample).

    Initialization and propagation structure identical to baseline A0.
    """
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]

    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]

    east0, north0, _ = ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0
    )
    heading0 = np.radians(data["veh_heading"][i0])
    speed0 = data["veh_velocity"][i0] / 3.6

    n_eval = i2 - i0
    headings = np.zeros(n_eval)
    positions_east = np.zeros(n_eval)
    positions_north = np.zeros(n_eval)

    headings[0] = heading0
    positions_east[0] = east0
    positions_north[0] = north0

    h = heading0
    e_pos = east0
    n_pos = north0
    v = speed0

    for k in range(1, n_eval):
        idx = i0 + k
        dt = data["sync_time"][idx] - data["sync_time"][idx - 1]

        if dt > DT_MAX_GAP or dt <= 0:
            headings[k] = h
            positions_east[k] = e_pos
            positions_north[k] = n_pos
            continue

        gyro_pitch = data["gyro_pitch"][idx - 1]
        yaw_rate = cal_func(gyro_pitch, dt)
        h = h + yaw_rate * dt

        e_pos = e_pos + v * np.cos(h) * dt
        n_pos = n_pos + v * np.sin(h) * dt

        headings[k] = h
        positions_east[k] = e_pos
        positions_north[k] = n_pos

    return {
        "headings": headings,
        "positions_east": positions_east,
        "positions_north": positions_north,
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
            ref_lat0, ref_lon0, 0.0
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


# ──────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ──────────────────────────────────────────────────────────────
def summarize(all_met):
    """Compute per-duration summary statistics across blackout windows."""
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
COLORS = {"A0": "#1f77b4", "A1": "#ff7f0e", "A2": "#2ca02c"}


def plot_gyro_calibration_fit(cal_all, cal_loo_list):
    """Plot 1: gyro_pitch vs vehicle yaw rate with fitted lines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_cal = np.degrees(cal_all["x"])
    y_cal = np.degrees(cal_all["y"])

    # Left: scatter + fits
    ax = axes[0]
    ax.scatter(x_cal, y_cal, s=1, alpha=0.15, c="gray", label="Calibration samples")
    x_range = np.linspace(x_cal.min(), x_cal.max(), 200)

    # A0: identity
    ax.plot(x_range, x_range, "b--", linewidth=1.5, label=f"A0: s=1.000, b=0.000 (identity)")

    # A1: scale only
    s1 = cal_all["s"]
    ax.plot(x_range, s1 * x_range, "r-", linewidth=2,
            label=f"A1: s={s1:.4f}, b=0.000")

    # A2: scale + bias
    b2_rad = cal_all["b"]
    ax.plot(x_range, s1 * x_range + np.degrees(b2_rad), "g-", linewidth=2,
            label=f"A2: s={s1:.4f}, b={np.degrees(b2_rad):.4f} °/s")

    ax.set_xlabel("Phone gyro pitch (°/s)", fontsize=11)
    ax.set_ylabel("Vehicle yaw rate (°/s)", fontsize=11)
    ax.set_title("Calibration Fit (all non-blackout data)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Right: residual analysis for A2
    ax = axes[1]
    residuals_deg = y_cal - (s1 * x_cal + np.degrees(b2_rad))
    ax.scatter(x_cal, residuals_deg, s=1, alpha=0.15, c="gray")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Phone gyro pitch (°/s)", fontsize=11)
    ax.set_ylabel("A2 residual (°/s)", fontsize=11)
    ax.set_title("A2 Calibration Residuals", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Stats text
    stats_text = (f"A2 fit: s={s1:.4f}, b={np.degrees(b2_rad):.4f} °/s\n"
                  f"n={cal_all['n_samples']:,}, "
                  f"cal RMSE={np.degrees(cal_all['cal_rmse']):.4f} °/s, "
                  f"r={cal_all['cal_r']:.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Gyro Calibration Fit — A0/A1/A2", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gyro_calibration_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'gyro_calibration_fit.png'}")


def plot_gyro_signal_comparison(data, segments):
    """Plot 2: gyro_pitch vs vehicle yaw rate over representative intervals."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Pick a representative interval from segment 2 (largest segment)
    seg = segments[2] if len(segments) > 2 else segments[0]
    # Find a 60s window with moderate turning
    s, e = seg["start"], seg["end"]
    t_seg = data["sync_time"][s:e]
    yaw_seg = data["veh_yaw_rate"][s:e]

    # Find interval with highest absolute yaw activity
    best_start = 0
    best_activity = 0
    window_samples = 600  # 60s at 10Hz
    for i in range(0, len(t_seg) - window_samples, 50):
        activity = np.mean(np.abs(yaw_seg[i:i + window_samples]))
        if activity > best_activity:
            best_activity = activity
            best_start = i

    idx0 = s + best_start
    idx1 = min(idx0 + window_samples, e)
    t_window = data["sync_time"][idx0:idx1] - data["sync_time"][idx0]
    gyro_window = data["gyro_pitch"][idx0:idx1]
    yaw_window = data["veh_yaw_rate"][idx0:idx1]

    # A0 (raw) and A2 (calibrated) — using calibration from fit on all non-blackout
    # We'll compute this inline
    all_nonbo_mask = np.ones(len(data["sync_time"]), dtype=bool)
    # (approximate: just show raw vs reference for this plot)
    ax = axes[0]
    ax.plot(t_window, gyro_window, "b-", alpha=0.7, linewidth=0.8, label="Phone gyro pitch")
    ax.plot(t_window, yaw_window, "r-", alpha=0.7, linewidth=0.8, label="Vehicle yaw rate")
    ax.set_ylabel("Rate (°/s)", fontsize=11)
    ax.set_title("Raw Signals — Representative Interval", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Cross-correlation snippet
    ax = axes[1]
    # Show the alignment effect
    gyro_aligned = np.interp(
        t_window + REF_ALIGNMENT_OFFSET_S,
        t_window,
        gyro_window,
    )
    ax.plot(t_window, gyro_aligned, "b-", alpha=0.7, linewidth=0.8,
            label=f"Phone gyro pitch (shifted +{REF_ALIGNMENT_OFFSET_S}s)")
    ax.plot(t_window, yaw_window, "r-", alpha=0.7, linewidth=0.8,
            label="Vehicle yaw rate")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Rate (°/s)", fontsize=11)
    ax.set_title(f"After +{REF_ALIGNMENT_OFFSET_S}s Reference Alignment", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "gyro_signal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'gyro_signal_comparison.png'}")


def plot_heading_error_comparison(all_metrics_by_variant, duration):
    """Plot 3: heading error vs blackout time for A0/A1/A2 at given duration."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, met_list in all_metrics_by_variant.items():
        if duration not in met_list or not met_list[duration]:
            continue
        for i, met in enumerate(met_list[duration]):
            t = met["t_since"]
            err = met["heading_error"]
            alpha = 0.5 if label == "A0" else 0.6
            lw = 0.8
            ax.plot(t, err, color=COLORS[label], alpha=alpha, linewidth=lw,
                    label=f"{label}" if i == 0 else None)

    ax.set_xlabel("Time since blackout start (s)", fontsize=12)
    ax.set_ylabel("Heading error (°)", fontsize=12)
    ax.set_title(f"Heading Error vs Time — {duration}s Blackout", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"heading_error_A0_A1_A2_{duration}s.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / f'heading_error_A0_A1_A2_{duration}s.png'}")


def plot_position_error_comparison(all_metrics_by_variant, duration):
    """Plot 4: position error vs blackout time for A0/A1/A2 at given duration."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, met_list in all_metrics_by_variant.items():
        if duration not in met_list or not met_list[duration]:
            continue
        for i, met in enumerate(met_list[duration]):
            t = met["t_since"]
            err = met["pos_error"]
            alpha = 0.5 if label == "A0" else 0.6
            lw = 0.8
            ax.plot(t, err, color=COLORS[label], alpha=alpha, linewidth=lw,
                    label=f"{label}" if i == 0 else None)

    ax.set_xlabel("Time since blackout start (s)", fontsize=12)
    ax.set_ylabel("Position error (m)", fontsize=12)
    ax.set_title(f"Position Error vs Time — {duration}s Blackout", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"position_error_A0_A1_A2_{duration}s.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / f'position_error_A0_A1_A2_{duration}s.png'}")


def plot_calibration_ablation_by_duration(summaries):
    """Plot 5: final position error for A0/A1/A2 at each blackout duration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    durations = sorted(set().union(*[s.keys() for s in summaries.values()]))
    durations = [d for d in durations if isinstance(d, (int, float))]

    variant_labels = list(summaries.keys())
    bar_width = 0.25
    x = np.arange(len(durations))

    for ax_idx, metric in enumerate(["final_m_mean", "mae_m_mean"]):
        ax = axes[ax_idx]
        for vi, label in enumerate(variant_labels):
            vals = []
            for d in durations:
                if d in summaries[label] and metric in summaries[label][d]:
                    vals.append(summaries[label][d][metric])
                else:
                    vals.append(0)
            offset = (vi - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=label, color=COLORS[label], alpha=0.8)
            # Add value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                            f"{val:.0f}", ha="center", va="bottom", fontsize=8)

        metric_label = "Final Position Error (m)" if "final" in metric else "Position MAE (m)"
        ax.set_xlabel("Blackout Duration", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}s" for d in durations])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Calibration Ablation — Position Error by Duration", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_ablation_by_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'calibration_ablation_by_duration.png'}")


def plot_heading_error_ablation_by_duration(summaries):
    """Plot 6: heading MAE for A0/A1/A2 at each blackout duration."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    durations = sorted(set().union(*[s.keys() for s in summaries.values()]))
    durations = [d for d in durations if isinstance(d, (int, float))]

    variant_labels = list(summaries.keys())
    bar_width = 0.25
    x = np.arange(len(durations))

    for vi, label in enumerate(variant_labels):
        vals = []
        for d in durations:
            if d in summaries[label] and "heading_mae_deg_mean" in summaries[label][d]:
                vals.append(summaries[label][d]["heading_mae_deg_mean"])
            else:
                vals.append(0)
        offset = (vi - 1) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=label, color=COLORS[label], alpha=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.1f}°", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Blackout Duration", fontsize=12)
    ax.set_ylabel("Heading MAE (°)", fontsize=12)
    ax.set_title("Heading MAE by Duration — A0/A1/A2", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in durations])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "heading_error_ablation_by_duration.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'heading_error_ablation_by_duration.png'}")


def plot_calibration_residuals(cal_all):
    """Plot 7: calibration residual analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_deg = np.degrees(cal_all["x"])
    y_deg = np.degrees(cal_all["y"])
    s = cal_all["s"]
    b_deg = np.degrees(cal_all["b"])
    y_pred_deg = s * x_deg + b_deg
    residuals_deg = y_deg - y_pred_deg

    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(y_pred_deg, residuals_deg, s=1, alpha=0.15, c="gray")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Fitted vehicle yaw rate (°/s)")
    ax.set_ylabel("Residual (°/s)")
    ax.set_title("Residuals vs Fitted Values")
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[0, 1]
    ax.hist(residuals_deg, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Residual (°/s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution (std={residuals_deg.std():.4f} °/s)")
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 0]
    from scipy.stats import norm
    sorted_res = np.sort(residuals_deg)
    n = len(sorted_res)
    theoretical = norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))
    theoretical = theoretical * residuals_deg.std()
    ax.scatter(theoretical, sorted_res, s=1, alpha=0.3, c="gray")
    lim = max(abs(theoretical.min()), abs(theoretical.max()), abs(sorted_res.min()), abs(sorted_res.max()))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_xlabel("Theoretical quantiles (°/s)")
    ax.set_ylabel("Sample quantiles (°/s)")
    ax.set_title("Q-Q Plot (normality check)")
    ax.grid(True, alpha=0.3)

    # Residuals vs time (using index as proxy)
    ax = axes[1, 1]
    step = max(1, len(residuals_deg) // 5000)
    idx = np.arange(0, len(residuals_deg), step)
    ax.scatter(idx, residuals_deg[::step], s=1, alpha=0.2, c="gray")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Sample index (sorted)")
    ax.set_ylabel("Residual (°/s)")
    ax.set_title("Residuals vs Sample Index")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Residual Analysis — A2 (s * gyro_pitch + b)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'calibration_residuals.png'}")


# ──────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────
def generate_report(cal_all, cal_loo_s, cal_loo_b, summaries, all_metrics, windows):
    lines = []

    def add(s=""):
        lines.append(s)

    add("=" * 80)
    add("GYRO CALIBRATION ABLATION — CLASSICAL DR V0")
    add("=" * 80)
    add()

    add("=" * 80)
    add("1. EXPERIMENT OBJECTIVE")
    add("=" * 80)
    add("  Determine how much of the catastrophic classical DR failure is")
    add("  caused by simple gyro scale/bias mismatch versus irreducible")
    add("  or context-dependent heading propagation error.")
    add()
    add("  Three variants tested:")
    add("    A0: yaw_rate = gyro_pitch                    (s=1, b=0)")
    add("    A1: yaw_rate = s * gyro_pitch                (scale only)")
    add("    A2: yaw_rate = s * gyro_pitch + b            (scale + bias)")
    add()

    add("=" * 80)
    add("2. CALIBRATION DATA SELECTION")
    add("=" * 80)
    add(f"  Calibration signal: phone gyroscope Pitch (rad/s)")
    add(f"  Reference signal: vehicle CAN yaw rate (deg/sec → rad/s)")
    add(f"  Alignment offset: +{REF_ALIGNMENT_OFFSET_S} s (applied to reference ONLY,")
    add(f"    for calibration fitting. NOT used during DR propagation.)")
    add()
    add("  In-sample calibration (diagnostic):")
    add(f"    ALL non-blackout data: {cal_all['n_samples']:,} samples")
    add(f"    NOTE: This has in-sample leakage — same data used for fit and eval.")
    add()
    add("  Leave-one-window-out calibration (leakage-safe):")
    add(f"    For each blackout window, fit on all non-blackout data")
    add(f"    EXCEPT that window. Mean s={cal_loo_s:.4f}, mean b={np.degrees(cal_loo_b):.4f} °/s")
    add()

    add("=" * 80)
    add("3. CAUSALITY / LEAKAGE PROTOCOL")
    add("=" * 80)
    add("  During blackout propagation:")
    add("    - NO future information is used.")
    add("    - NO vehicle heading/yaw rate is used.")
    add("    - NO GPS is used.")
    add("    - Only phone IMU + last-known state.")
    add()
    add("  Calibration parameters (s, b) are:")
    add("    - Fixed constants estimated BEFORE the blackout experiment.")
    add("    - Do NOT depend on the blackout being evaluated.")
    add("    - The leave-one-window-out protocol ensures no blackout window's")
    add("      data influences its own calibration parameters.")
    add()
    add("  The +1.81 s alignment is used ONLY to align the reference yaw rate")
    add("  during calibration fitting. It does NOT alter stored timestamps and")
    add("  is NOT used during DR propagation.")
    add()

    add("=" * 80)
    add("4. A0/A1/A2 DEFINITIONS")
    add("=" * 80)
    add("  All variants share identical:")
    add("    - Initialization (vehicle reference at blackout start)")
    add("    - Velocity model (constant, from vehicle reference)")
    add("    - Position propagation (heading + velocity)")
    add("    - Blackout selection and evaluation")
    add("    - ENU coordinate system")
    add()
    add("  The ONLY difference is the yaw rate computation:")
    add("    A0: yaw_rate = gyro_pitch")
    add("    A1: yaw_rate = s * gyro_pitch")
    add("    A2: yaw_rate = s * gyro_pitch + b")
    add()

    add("=" * 80)
    add("5. ESTIMATED CALIBRATION PARAMETERS")
    add("=" * 80)
    add()
    add("  In-sample (all non-blackout data):")
    add(f"    s = {cal_all['s']:.6f}")
    add(f"    b = {cal_all['b']:.6f} rad/s  ({np.degrees(cal_all['b']):.4f} °/s)")
    add(f"    n = {cal_all['n_samples']:,} samples")
    add()
    add("  Leave-one-window-out (mean across windows):")
    add(f"    s = {cal_loo_s:.6f}")
    add(f"    b = {cal_loo_b:.6f} rad/s  ({np.degrees(cal_loo_b):.4f} °/s)")
    add()

    add("=" * 80)
    add("6. CALIBRATION FIT QUALITY")
    add("=" * 80)
    add(f"  Pearson r:           {cal_all['cal_r']:.4f}")
    add(f"  Calibration RMSE:    {np.degrees(cal_all['cal_rmse']):.4f} °/s")
    add(f"  Calibration MAE:     {np.degrees(cal_all['cal_mae']):.4f} °/s")
    add(f"  Sign agreement:      {cal_all['sign_agreement']:.1%}")
    add(f"  Number of samples:   {cal_all['n_samples']:,}")
    add()
    add("  NOTE: These are in-sample fit metrics (same data used for fit and eval).")
    add("  They characterize the relationship quality, not DR performance.")
    add()

    add("=" * 80)
    add("7. RESULTS BY BLACKOUT DURATION")
    add("=" * 80)

    for dur in sorted([d for d in summaries["A0"].keys() if isinstance(d, (int, float))]):
        add(f"\n  --- {dur} s blackout ---")
        add(f"  {'Variant':>6s}  {'MAE (m)':>8s}  {'RMSE (m)':>8s}  {'Max (m)':>8s}  "
            f"{'Final (m)':>9s}  {'Head MAE':>9s}  {'Head Final':>10s}  {'N':>3s}")
        add(f"  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 10}  {'-' * 3}")
        for label in ["A0", "A1", "A2"]:
            if dur in summaries[label]:
                s = summaries[label][dur]
                n_win = int(s.get("n_windows", 0))
                add(f"  {label:>6s}  {s['mae_m_mean']:8.1f}  {s['rmse_m_mean']:8.1f}  "
                    f"{s['max_m_mean']:8.1f}  {s['final_m_mean']:9.1f}  "
                    f"{s['heading_mae_deg_mean']:8.1f}°  "
                    f"{s.get('heading_final_deg_mean', 0):9.1f}°  {n_win:3d}")
    add()

    add("=" * 80)
    add("8. PER-WINDOW STATISTICS")
    add("=" * 80)
    for label in ["A0", "A1", "A2"]:
        add(f"\n  {label}:")
        for dur in sorted([d for d in summaries[label].keys() if isinstance(d, (int, float))]):
            s = summaries[label][dur]
            add(f"    {dur:3d}s: final mean={s['final_m_mean']:.1f} m, "
                f"std={s.get('final_m_std', 0):.1f} m, "
                f"median={s.get('final_m_median', 0):.1f} m, "
                f"head_mae mean={s['heading_mae_deg_mean']:.1f}°")
    add()

    add("=" * 80)
    add("9. IMPROVEMENT PERCENTAGES (vs A0)")
    add("=" * 80)
    add(f"  {'Dur':>5s}  {'Metric':>12s}  {'A0':>8s}  {'A1':>8s}  {'A1 Δ%':>8s}  "
        f"{'A2':>8s}  {'A2 Δ%':>8s}")
    add(f"  {'-' * 5}  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for dur in sorted([d for d in summaries["A0"].keys() if isinstance(d, (int, float))]):
        for metric, mkey, unit in [
            ("Final Pos", "final_m_mean", "m"),
            ("MAE", "mae_m_mean", "m"),
            ("RMSE", "rmse_m_mean", "m"),
            ("Head MAE", "heading_mae_deg_mean", "°"),
        ]:
            a0_val = summaries["A0"][dur].get(mkey, 0)
            a1_val = summaries["A1"][dur].get(mkey, 0) if dur in summaries["A1"] else 0
            a2_val = summaries["A2"][dur].get(mkey, 0) if dur in summaries["A2"] else 0

            def pct_change(a0, cal):
                if a0 == 0:
                    return 0.0
                return (a0 - cal) / a0 * 100

            a1_pct = pct_change(a0_val, a1_val)
            a2_pct = pct_change(a0_val, a2_val)
            add(f"  {dur:4d}s  {metric:>12s}  {a0_val:7.1f}{unit[0]}  {a1_val:7.1f}{unit[0]}  "
                f"{a1_pct:+7.1f}%  {a2_val:7.1f}{unit[0]}  {a2_pct:+7.1f}%")
        add()

    add("=" * 80)
    add("10. FAILURE CASES")
    add("=" * 80)
    add("  Check per-window statistics for windows where calibrated variants")
    add("  perform WORSE than A0. This may indicate:")
    add("    - Calibration extrapolation to unusual operating conditions")
    add("    - Sign disagreements dominating in specific turn geometries")
    add("    - Non-stationary gyro behavior across trajectory segments")
    add()
    for dur in sorted([d for d in summaries["A0"].keys() if isinstance(d, (int, float))]):
        a0_final = summaries["A0"][dur].get("final_m_mean", 0)
        a2_final = summaries["A2"][dur].get("final_m_mean", 0) if dur in summaries["A2"] else 0
        if a2_final > a0_final:
            add(f"  WARNING: A2 ({a2_final:.1f} m) > A0 ({a0_final:.1f} m) at {dur}s blackout.")
    add()

    add("=" * 80)
    add("11. INTERPRETATION")
    add("=" * 80)
    add("  See answers to key questions below.")
    add()

    add("=" * 80)
    add("12. KEY QUESTIONS")
    add("=" * 80)

    # Compute answer to Q1
    a0_120 = summaries["A0"].get(120, {}).get("final_m_mean", 0)
    a2_120 = summaries["A2"].get(120, {}).get("final_m_mean", 0) if 120 in summaries["A2"] else 0
    a2_120_head = summaries["A2"].get(120, {}).get("heading_mae_deg_mean", 0) if 120 in summaries["A2"] else 0
    a0_10 = summaries["A0"].get(10, {}).get("final_m_mean", 0)
    a2_10 = summaries["A2"].get(10, {}).get("final_m_mean", 0) if 10 in summaries["A2"] else 0

    add()
    add("  QUESTION 1: Is the current classical DR failure substantially")
    add("  explained by gyro scale/bias mismatch?")
    add()
    if a0_120 > 0 and a2_120 > 0:
        reduction = (a0_120 - a2_120) / a0_120 * 100
        if reduction > 50:
            add(f"  ANSWER: YES — calibration reduces 120s final error by {reduction:.1f}%.")
            add(f"  Scale/bias mismatch was a major contributor to the failure.")
        elif reduction > 20:
            add(f"  ANSWER: PARTIALLY — calibration reduces 120s final error by {reduction:.1f}%.")
            add(f"  Scale/bias mismatch explains some but not all of the failure.")
        else:
            add(f"  ANSWER: NO — calibration reduces 120s final error by only {reduction:.1f}%.")
            add(f"  The failure is NOT primarily explained by scale/bias mismatch.")
    add()

    add("  QUESTION 2: After calibration, does heading still become unusable")
    add("  during long blackouts?")
    add()
    if a2_120_head > 45:
        add(f"  ANSWER: YES — heading MAE at 120s is {a2_120_head:.1f}° after calibration.")
        add(f"  Heading propagation remains catastrophic even with scale/bias correction.")
    else:
        add(f"  ANSWER: NO — heading MAE at 120s is {a2_120_head:.1f}° after calibration.")
        add(f"  Heading propagation is significantly improved.")
    add()

    add("  QUESTION 3: Does substantial position drift remain after calibration?")
    add()
    if a2_120 > 100:
        add(f"  ANSWER: YES — 120s final error is {a2_120:.1f} m after calibration.")
        add(f"  Substantial position drift persists.")
    elif a2_120 > 30:
        add(f"  ANSWER: MODERATE — 120s final error is {a2_120:.1f} m after calibration.")
        add(f"  Drift is reduced but still meaningful.")
    else:
        add(f"  ANSWER: NO — 120s final error is {a2_120:.1f} m after calibration.")
        add(f"  Position drift is largely resolved.")
    add()

    add("  QUESTION 4: What is the strongest evidence that a learned residual")
    add("  correction is worth testing?")
    add()
    a0_10_final = summaries["A0"].get(10, {}).get("final_m_mean", 0)
    a2_10_final = summaries["A2"].get(10, {}).get("final_m_mean", 0) if 10 in summaries["A2"] else 0
    a2_30_final = summaries["A2"].get(30, {}).get("final_m_mean", 0) if 30 in summaries["A2"] else 0

    if a2_120 > 100 or a2_30_final > 50:
        add(f"  ANSWER: The remaining drift after calibration ({a2_120:.1f} m at 120s)")
        add(f"  demonstrates that scale/bias correction alone is insufficient.")
        add(f"  Context-dependent, nonlinear heading error correction is warranted.")
        add(f"  A learned residual model can target the irreducible heading error")
        add(f"  that linear calibration cannot address.")
    else:
        add(f"  ANSWER: After calibration, residual drift is small ({a2_120:.1f} m at 120s).")
        add(f"  A learned correction may provide marginal additional benefit.")
    add()

    add("=" * 80)
    add("13. RECOMMENDED NEXT EXPERIMENT")
    add("=" * 80)
    if a2_120 > 100:
        add("  Proceed to causal residual-learning GRU.")
        add("  Calibration ablation confirms that scale/bias correction alone")
        add("  does not solve the navigation problem.")
        add("  The ML model should target context-dependent heading correction.")
    else:
        add("  Calibration substantially resolved the drift.")
        add("  Investigate whether the residual error is small enough to")
        add("  tolerate without ML, or proceed to ML for marginal improvement.")
    add()

    add("=" * 80)
    add("END OF REPORT")
    add("=" * 80)

    report_text = "\n".join(lines)
    report_path = OUT_DIR / "gyro_calibration_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n  Report saved: {report_path}")
    return report_text


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("GYRO CALIBRATION ABLATION — CLASSICAL DR V0")
    print("=" * 80)

    # ── Load data ──
    data, cols, df = load_data()

    # ── Detect segments ──
    segments = detect_segments(data["sync_time"])

    # ── Select blackout windows (same as baseline) ──
    windows = select_blackout_windows(data, segments, BLACKOUT_DURATIONS)
    if not windows:
        print("\n*** No valid blackout windows found. ***")
        sys.exit(1)

    # ── Build non-blackout mask for calibration ──
    n_total = len(data["sync_time"])
    nonblackout_mask = np.ones(n_total, dtype=bool)
    for w in windows:
        nonblackout_mask[w["start_idx"]:w["blackout_end_idx"]] = False

    # Also mask 1s around each blackout to avoid edge effects
    margin = 10  # samples
    for w in windows:
        s = max(0, w["start_idx"] - margin)
        e = min(n_total, w["blackout_end_idx"] + margin)
        nonblackout_mask[s:e] = False

    # ── Fit calibration: in-sample (all non-blackout data) ──
    print("\n" + "=" * 60)
    print("CALIBRATION FITTING")
    print("=" * 60)

    cal_all = fit_calibration(
        data["gyro_pitch"][nonblackout_mask],
        data["veh_yaw_rate"][nonblackout_mask],
        alignment_offset_s=REF_ALIGNMENT_OFFSET_S,
        sync_time_cal=data["sync_time"][nonblackout_mask],
    )

    print(f"\n  In-sample calibration (all non-blackout data):")
    print(f"    s = {cal_all['s']:.6f}")
    print(f"    b = {cal_all['b']:.6f} rad/s ({np.degrees(cal_all['b']):.4f} °/s)")
    print(f"    n = {cal_all['n_samples']:,}")
    print(f"    Pearson r = {cal_all['cal_r']:.4f}")
    print(f"    Cal RMSE = {np.degrees(cal_all['cal_rmse']):.4f} °/s")
    print(f"    Cal MAE = {np.degrees(cal_all['cal_mae']):.4f} °/s")
    print(f"    Sign agreement = {cal_all['sign_agreement']:.1%}")

    # ── Fit calibration: leave-one-window-out (leakage-safe) ──
    loo_s_list = []
    loo_b_list = []

    for w in windows:
        loo_mask = nonblackout_mask.copy()
        # Additionally exclude this window's blackout region (already excluded
        # in nonblackout_mask, but be explicit)
        loo_mask[w["start_idx"]:w["blackout_end_idx"]] = False

        cal_loo = fit_calibration(
            data["gyro_pitch"][loo_mask],
            data["veh_yaw_rate"][loo_mask],
            alignment_offset_s=REF_ALIGNMENT_OFFSET_S,
            sync_time_cal=data["sync_time"][loo_mask],
        )
        loo_s_list.append(cal_loo["s"])
        loo_b_list.append(cal_loo["b"])

    cal_loo_s = float(np.mean(loo_s_list))
    cal_loo_b = float(np.mean(loo_b_list))
    cal_loo_s_std = float(np.std(loo_s_list))
    cal_loo_b_std = float(np.std(loo_b_list))

    print(f"\n  Leave-one-window-out calibration (mean across {len(windows)} windows):")
    print(f"    s = {cal_loo_s:.6f} (std={cal_loo_s_std:.6f})")
    print(f"    b = {cal_loo_b:.6f} rad/s ({np.degrees(cal_loo_b):.4f} °/s) "
          f"(std={np.degrees(cal_loo_b_std):.4f} °/s)")

    # ── Define calibration functions ──
    cal_funcs = {
        "A0": lambda gp, dt: gp,
        "A1": lambda gp, dt, s=cal_all["s"]: s * gp,
        "A2": lambda gp, dt, s=cal_all["s"], b=cal_all["b"]: s * gp + b,
    }

    # Also create leakage-safe variants for comparison
    cal_funcs_loo = {
        "A0": lambda gp, dt: gp,
        "A1_loo": lambda gp, dt, s=cal_loo_s: s * gp,
        "A2_loo": lambda gp, dt, s=cal_loo_s, b=cal_loo_b: s * gp + b,
    }

    # ── Run all variants on all blackout windows ──
    print("\n" + "=" * 60)
    print("RUNNING BLACKOUT EXPERIMENTS")
    print("=" * 60)

    all_metrics = {}  # variant_label → {duration → [metric_dicts]}
    all_raw_results = {}  # variant_label → {duration → [raw DR results]}

    variant_labels = ["A0", "A1", "A2"]

    for w in windows:
        dur = w["duration_s"]
        print(f"\n  {dur}s blackout at T={w['t_start']:.1f}...")

        # Reference (same for all variants)
        ref_e, ref_n, ref_h = get_reference_enu(
            data, w["start_idx"], w["eval_end_idx"],
            data["veh_lat"][w["start_idx"]], data["veh_lon"][w["start_idx"]]
        )

        for label in variant_labels:
            res = run_dr_variant(data, w, cal_funcs[label])

            n_bo = w["blackout_end_idx"] - w["start_idx"]
            met = compute_metrics(
                res["positions_east"], res["positions_north"], res["headings"],
                ref_e, ref_n, ref_h,
                data["sync_time"][w["start_idx"]:w["eval_end_idx"]],
                n_bo,
            )
            all_metrics.setdefault(label, {}).setdefault(dur, []).append(met)
            all_raw_results.setdefault(label, {}).setdefault(dur, []).append(res)

    # ── Compute summaries ──
    summaries = {}
    for label in variant_labels:
        summaries[label] = summarize(all_metrics[label])

    # ── Generate plots ──
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_gyro_calibration_fit(cal_all, loo_s_list)
    plot_gyro_signal_comparison(data, segments)
    plot_calibration_residuals(cal_all)

    for dur in BLACKOUT_DURATIONS:
        plot_heading_error_comparison(all_metrics, dur)
        plot_position_error_comparison(all_metrics, dur)

    plot_calibration_ablation_by_duration(summaries)
    plot_heading_error_ablation_by_duration(summaries)

    # ── Generate report ──
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    generate_report(cal_all, cal_loo_s, cal_loo_b, summaries, all_metrics, windows)

    # ── Print terminal summary ──
    print("\n" + "=" * 80)
    print("GYRO CALIBRATION ABLATION — FINAL SUMMARY")
    print("=" * 80)

    print(f"\n  Files created:")
    print(f"    {OUT_DIR / 'gyro_calibration_report.txt'}")
    print(f"    {OUT_DIR / 'gyro_calibration_fit.png'}")
    print(f"    {OUT_DIR / 'gyro_signal_comparison.png'}")
    print(f"    {OUT_DIR / 'calibration_residuals.png'}")
    print(f"    {OUT_DIR / 'calibration_ablation_by_duration.png'}")
    print(f"    {OUT_DIR / 'heading_error_ablation_by_duration.png'}")
    for dur in BLACKOUT_DURATIONS:
        print(f"    {OUT_DIR / f'heading_error_A0_A1_A2_{dur}s.png'}")
        print(f"    {OUT_DIR / f'position_error_A0_A1_A2_{dur}s.png'}")

    print(f"\n  Calibration method: least-squares (y = s*x + b)")
    print(f"  Reference alignment: +{REF_ALIGNMENT_OFFSET_S} s (calibration only)")
    print(f"  In-sample: s={cal_all['s']:.4f}, b={np.degrees(cal_all['b']):.4f} °/s, "
          f"n={cal_all['n_samples']:,}")
    print(f"  LOO-safe:  s={cal_loo_s:.4f}, b={np.degrees(cal_loo_b):.4f} °/s "
          f"(mean across {len(windows)} windows)")

    print(f"\n  {'Dur':>5s}  {'A0 Final':>9s}  {'A1 Final':>9s}  {'A2 Final':>9s}  "
          f"{'A1 Δ%':>7s}  {'A2 Δ%':>7s}  {'A0 Head':>8s}  {'A2 Head':>8s}")
    print(f"  {'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 8}")

    for dur in sorted(summaries["A0"].keys()):
        if not isinstance(dur, (int, float)):
            continue
        a0 = summaries["A0"][dur]
        a1 = summaries["A1"].get(dur, {})
        a2 = summaries["A2"].get(dur, {})

        a0_f = a0.get("final_m_mean", 0)
        a1_f = a1.get("final_m_mean", 0)
        a2_f = a2.get("final_m_mean", 0)
        a0_h = a0.get("heading_mae_deg_mean", 0)
        a2_h = a2.get("heading_mae_deg_mean", 0)

        def pct(a0v, calv):
            return (a0v - calv) / a0v * 100 if a0v > 0 else 0

        print(f"  {dur:4d}s  {a0_f:8.1f}m  {a1_f:8.1f}m  {a2_f:8.1f}m  "
              f"{pct(a0_f, a1_f):+6.1f}%  {pct(a0_f, a2_f):+6.1f}%  "
              f"{a0_h:7.1f}°  {a2_h:7.1f}°")

    # Answer the key questions
    a0_120_final = summaries["A0"].get(120, {}).get("final_m_mean", 0)
    a2_120_final = summaries["A2"].get(120, {}).get("final_m_mean", 0) if 120 in summaries["A2"] else 0
    a2_120_head = summaries["A2"].get(120, {}).get("heading_mae_deg_mean", 0) if 120 in summaries["A2"] else 0
    a0_120_head = summaries["A0"].get(120, {}).get("heading_mae_deg_mean", 0)

    if a0_120_final > 0:
        reduction_120 = (a0_120_final - a2_120_final) / a0_120_final * 100
    else:
        reduction_120 = 0

    print(f"\n  120s final error reduction (A2 vs A0): {reduction_120:+.1f}%")
    print(f"  120s heading MAE: A0={a0_120_head:.1f}°, A2={a2_120_head:.1f}°")

    print(f"\n  Does calibration materially improve DR?")
    if reduction_120 > 30:
        print(f"    YES — {reduction_120:.1f}% reduction in 120s final error.")
    elif reduction_120 > 10:
        print(f"    PARTIALLY — {reduction_120:.1f}% reduction in 120s final error.")
    else:
        print(f"    NO — only {reduction_120:.1f}% reduction in 120s final error.")

    if a2_120_final > 100:
        print(f"  Large drift remains after calibration ({a2_120_final:.1f} m at 120s).")
        print(f"  → Proceed to residual-learning ML.")
    elif a2_120_final > 30:
        print(f"  Moderate drift remains after calibration ({a2_120_final:.1f} m at 120s).")
        print(f"  → ML may provide additional benefit.")
    else:
        print(f"  Drift largely resolved after calibration ({a2_120_final:.1f} m at 120s).")
        print(f"  → ML may be unnecessary.")

    print(f"\n  Unresolved methodological concerns:")
    print(f"    1. Phone-to-vehicle frame mapping still unresolved.")
    print(f"    2. Gyro Pitch has no proven physical basis as yaw rate.")
    print(f"    3. Calibration is in-sample for this single trajectory.")
    print(f"    4. +1.81 s alignment is empirical, not proven physical latency.")
    print(f"    5. Velocity model remains constant (no acceleration).")

    print(f"\n  Outputs: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
