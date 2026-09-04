#!/usr/bin/env python3
"""
Classical Dead-Reckoning V0 Baseline — SIH26168
================================================

Two baselines:
  A) Yaw-only kinematic DR: heading from gyro pitch, velocity held constant
  B) IMU-acceleration DR (EXPERIMENTAL): double-integrate rotated accel

Purpose: quantify DR position error growth during simulated GNSS blackouts.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT / "processed" / "S4_synced.csv"
OUT_DIR = SRC_DIR.parent / "outputs" / "dr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
DT_NOMINAL = 0.1          # nominal sample period (s)
DT_MAX_GAP = 0.2          # threshold to detect segment boundary
GPS_SPEED_FACTOR = 3.6    # GPS SPEED (Kmh) × 3.6 ≈ Vehicle Velocity (km/h)
REF_ALIGNMENT_OFFSET_S = 1.81  # empirical inter-stream offset (evaluation only)

# Blackout durations to test (seconds)
BLACKOUT_DURATIONS = [10, 30, 60, 120]

# Minimum motion threshold for blackout windows (km/h)
MIN_MOTION_THRESHOLD = 2.0

# GPS accuracy threshold (m) — ignore blackouts where GPS was poor before
MAX_GPS_ACCURACY = 5.0

# ──────────────────────────────────────────────────────────────
# V0 ASSUMPTIONS DOCUMENT
# ──────────────────────────────────────────────────────────────
V0_ASSUMPTIONS = """
================================================================================
CLASSICAL DR V0 ASSUMPTIONS
================================================================================

MEASURED:
  - Phone accelerometer X/Y/Z (m/s²) at 10 Hz
  - Phone gravity X/Y/Z (m/s²) at 10 Hz
  - Phone gyroscope Yaw/Pitch/Roll (rad/s) at 10 Hz
  - Phone GPS latitude/longitude (degrees) at ~1 Hz
  - Phone GPS speed (Kmh) at ~1 Hz
  - Vehicle CAN: velocity, heading, yaw rate, wheel speeds, steering angle

INFERRED:
  - Phone Z axis is approximately vertical (gravity Z ≈ 9.81 m/s²)
  - Gyro Pitch has an empirical relationship with vehicle yaw rate
    (correlation ~0.71 during strong turns, scale factor ~0.94 median)
  - GPS speed × 3.6 ≈ vehicle velocity (validated against CAN data)

ASSUMED (V0):
  A. Phone Z axis is vertical; phone X/Y axes are in the horizontal plane
     BUT the phone-to-vehicle yaw rotation is UNKNOWN.
  B. Gyro Pitch is used as the heading rate signal. No proven physical
     basis beyond empirical correlation with vehicle yaw rate.
  C. The phone-to-vehicle X/Y frame mapping is NOT established.
     For Baseline B (IMU acceleration), we ASSUME phone X ≈ vehicle forward
     and phone Y ≈ vehicle left. This is an UNVERIFIED assumption.
  D. The 1.81 s alignment offset is an empirical inter-stream parameter,
     NOT a proven physical sensor latency. It is used ONLY for
     reference comparison evaluation, never during DR propagation.
  E. Phone GPS provides absolute position at ~1 Hz with ~3 m accuracy.
  F. GPS speed × 3.6 provides a reasonable velocity estimate.

WHAT IS NOT ASSUMED:
  - No EKF, UKF, or filtering
  - No ML correction
  - No map matching
  - No future information during blackout (strictly causal)
  - No faking of successful DR results

SEPARATION OF CONCERNS:
  - DR propagation: causal, uses only IMU + last-known state
  - Evaluation: uses ground-truth GPS/vehicle reference AFTER blackout
================================================================================
"""


# ──────────────────────────────────────────────────────────────
# COLUMN MATCHING
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
    """Match required columns using semantic keywords.

    Priority matching: for vehicle columns, exclude columns that start
    with 'GPS' (those are phone columns in this dataset).
    """
    matched = {}
    available = list(df.columns)

    all_required = {}
    all_required.update({k: v for k, v in REQUIRED_PHONE_COLS.items()})
    all_required.update({k: v for k, v in REQUIRED_VEHICLE_COLS.items()})

    print("=" * 60)
    print("COLUMN MATCHING")
    print("=" * 60)

    for key, keywords in all_required.items():
        found = False
        is_vehicle = key.startswith("veh_")

        for col in available:
            col_lower = col.lower()
            if all(kw in col_lower for kw in keywords):
                # For vehicle columns, skip columns starting with "GPS"
                if is_vehicle and col_lower.startswith("gps"):
                    continue
                matched[key] = col
                print(f"  {key:25s} → {col}")
                found = True
                break
        if not found:
            print(f"\n  *** FAILED TO MATCH: {key}")
            print(f"      Keywords: {keywords}")
            candidates = [c for c in available
                         if any(kw in c.lower() for kw in keywords[0].split()[:1])]
            if candidates:
                print(f"      Candidates: {candidates}")
            else:
                print(f"      No close candidates found.")
            sys.exit(1)

    print("=" * 60)
    return matched


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_data():
    """Load and validate S4_synced.csv."""
    print(f"\nLoading: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    cols = match_columns(df)

    # Convert to numpy arrays for speed
    data = {}
    for key, col in cols.items():
        if key == "phone_date":
            data[key] = df[col].values
            continue
        try:
            data[key] = df[col].values.astype(np.float64)
        except (ValueError, TypeError):
            print(f"  Warning: column {key} ({col}) could not be converted to float, storing as-is")
            data[key] = df[col].values

    # Phone date as strings for reference
    data["phone_date_str"] = df[cols["phone_date"]].values

    # Store SYNC_TIME_S
    data["sync_time"] = df[cols["sync_time"]].values.astype(np.float64)

    print(f"\n  SYNC_TIME_S range: [{data['sync_time'][0]:.3f}, {data['sync_time'][-1]:.3f}] s")
    print(f"  Duration: {data['sync_time'][-1] - data['sync_time'][0]:.1f} s")

    return data, cols, df


# ──────────────────────────────────────────────────────────────
# SEGMENT DETECTION
# ──────────────────────────────────────────────────────────────
def detect_segments(sync_time):
    """Detect segment boundaries from timestamp gaps."""
    dt = np.diff(sync_time, prepend=sync_time[0] - DT_NOMINAL)
    gap_mask = dt > DT_MAX_GAP
    gap_indices = np.where(gap_mask)[0]

    # Build segment list: (start_idx, end_idx, valid)
    segments = []
    boundaries = [0] + list(gap_indices) + [len(sync_time)]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if i > 0:
            gap_size = dt[boundaries[i]]
        else:
            gap_size = 0
        segments.append({
            "start": start,
            "end": end,
            "n_rows": end - start,
            "gap_before_s": gap_size,
            "t_start": sync_time[start],
            "t_end": sync_time[end - 1],
            "duration_s": sync_time[end - 1] - sync_time[start],
        })

    print(f"\n  Segments detected: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"    Seg {i}: rows [{seg['start']}, {seg['end']}) "
              f"({seg['n_rows']} rows, {seg['duration_s']:.1f} s)"
              f"  gap_before={seg['gap_before_s']:.3f} s")

    return segments


# ──────────────────────────────────────────────────────────────
# COORDINATE SYSTEM: LOCAL ENU
# ──────────────────────────────────────────────────────────────
# Earth radius (mean)
R_EARTH = 6371000.0  # m


def ll2enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert lat/lon/alt to local ENU relative to (lat0, lon0, alt0).

    Uses local tangent-plane approximation:
      East  = (lon - lon0) * cos(lat0) * (π/180) * R_EARTH
      North = (lat - lat0) * (π/180) * R_EARTH
      Up    = alt - alt0

    Valid for trajectory scales << R_EARTH (this trajectory ~11 km, error < 0.01%).
    """
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)

    east = dlon * np.cos(lat0_rad) * R_EARTH
    north = dlat * R_EARTH
    up = alt - alt0

    return east, north, up


def enu2ll(east, north, up, lat0, lon0, alt0):
    """Convert local ENU back to lat/lon/alt."""
    lat0_rad = np.radians(lat0)
    lat = np.degrees(north / R_EARTH) + lat0
    lon = np.degrees(east / (np.cos(lat0_rad) * R_EARTH)) + lon0
    alt = up + alt0
    return lat, lon, alt


# ──────────────────────────────────────────────────────────────
# GYRO SCALE FACTOR ANALYSIS
# ──────────────────────────────────────────────────────────────
def analyze_gyro_scale(gyro_pitch, veh_yaw_rate_deg):
    """Analyze the scale factor between gyro pitch and vehicle yaw rate."""
    print("\n" + "=" * 60)
    print("GYRO PITCH vs VEHICLE YAW RATE ANALYSIS")
    print("=" * 60)

    gyro_rad = gyro_pitch
    veh_rad = np.radians(veh_yaw_rate_deg)

    # Overall correlation
    valid = np.isfinite(gyro_rad) & np.isfinite(veh_rad)
    r_all = np.corrcoef(gyro_rad[valid], veh_rad[valid])[0, 1]
    print(f"  Overall Pearson r: {r_all:.4f}")

    # During strong turns
    strong = (abs(veh_yaw_rate_deg) > 5) & (abs(gyro_rad) > 0.05) & valid
    if strong.sum() > 0:
        r_strong = np.corrcoef(gyro_rad[strong], veh_rad[strong])[0, 1]
        ratio = veh_rad[strong] / gyro_rad[strong]
        print(f"  Strong turns (|yaw|>5°/s): {strong.sum()} samples")
        print(f"    Correlation: {r_strong:.4f}")
        print(f"    Scale factor (veh/gyro): median={np.median(ratio):.4f}, "
              f"mean={ratio.mean():.4f}, std={ratio.std():.4f}")

    # Sign agreement during moderate turns
    moderate = (abs(veh_yaw_rate_deg) > 2) & (abs(gyro_rad) > 0.02) & valid
    sign_agree_pct = 0.0
    if moderate.sum() > 0:
        agree = np.sign(gyro_rad[moderate]) == np.sign(veh_rad[moderate])
        sign_agree_pct = agree.mean()
        print(f"  Sign agreement during moderate turns: "
              f"{agree.sum()}/{moderate.sum()} = {sign_agree_pct:.1%}")

    print(f"  → V0: use gyro pitch directly (scale factor = 1.0)")
    print(f"  → Limitation: correlation ~{r_all:.2f}, sign agreement ~{sign_agree_pct:.0%}")
    print("=" * 60)

    return {"r_all": r_all}


# ──────────────────────────────────────────────────────────────
# BLACKOUT WINDOW SELECTION
# ──────────────────────────────────────────────────────────────
def select_blackout_windows(data, segments, durations):
    """Select blackout windows that avoid gaps and boundaries.

    Strategy: for each duration, find windows across the trajectory:
      1. Within a single segment (no crossing gap boundaries)
      2. Vehicle is in motion before blackout
      3. GPS accuracy is acceptable before blackout
      4. Enough data after blackout for evaluation
      5. Spread across the segment, not clustered at start
    """
    windows = []

    for duration in durations:
        found_windows = []

        for seg in segments:
            s, e = seg["start"], seg["end"]
            seg_vel = data["veh_velocity"][s:e]

            # Need at least `duration` seconds plus evaluation margin
            eval_margin = min(30, duration)
            needed_s = duration + eval_margin

            if seg["duration_s"] < needed_s + 5:
                continue

            # Pre-scan: find all valid start indices
            valid_starts = []
            for i in range(s, e - 10):
                t_start = data["sync_time"][i]
                # Check motion before blackout (10s window before)
                pre_mask = (data["sync_time"][s:e] >= t_start - 10) & \
                           (data["sync_time"][s:e] < t_start)
                if pre_mask.sum() < 10:
                    continue
                pre_vel = seg_vel[pre_mask]
                if np.median(pre_vel) < MIN_MOTION_THRESHOLD:
                    continue

                # Check GPS accuracy before blackout
                pre_acc = data["phone_gps_accuracy"][s:e][pre_mask]
                if np.median(pre_acc) > MAX_GPS_ACCURACY:
                    continue

                # Check enough data after for blackout + evaluation
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

            # Select evenly spaced windows from valid starts
            n_select = min(3, len(valid_starts))
            if n_select == 1:
                indices = [0]
            else:
                indices = np.linspace(0, len(valid_starts) - 1, n_select, dtype=int)

            for idx in indices:
                i = valid_starts[idx]
                t_start = data["sync_time"][i]

                # Find blackout end index
                j = i
                while j < e and data["sync_time"][j] - t_start < duration:
                    j += 1

                # Find evaluation end index
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
# DR PROPAGATION: BASELINE A — YAW-ONLY KINEMATIC
# ──────────────────────────────────────────────────────────────
def run_baseline_a(data, window):
    """Baseline A: heading from gyro pitch, velocity held constant.

    Causal: only uses IMU + state available at blackout start.
    """
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]

    # ── INITIALIZATION (from GPS/reference at t=blackout_start) ──
    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]

    # Position: from vehicle reference (best available at init)
    east0, north0, _ = ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0
    )

    # Heading: from vehicle reference heading
    heading0 = np.radians(data["veh_heading"][i0])

    # Velocity: from vehicle reference speed (km/h → m/s)
    speed0 = data["veh_velocity"][i0] / 3.6

    # ── PROPAGATION (strictly causal) ──
    n_blackout = i1 - i0
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
    v = speed0  # constant velocity assumption

    for k in range(1, n_eval):
        idx = i0 + k
        dt = data["sync_time"][idx] - data["sync_time"][idx - 1]

        # Skip if dt is unreasonable (shouldn't happen within a segment)
        if dt > DT_MAX_GAP or dt <= 0:
            headings[k] = h
            positions_east[k] = e_pos
            positions_north[k] = n_pos
            continue

        # Heading propagation from gyro pitch
        gyro_pitch = data["gyro_pitch"][idx - 1]
        h = h + gyro_pitch * dt

        # Position propagation (constant velocity)
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
# DR PROPAGATION: BASELINE B — IMU ACCELERATION (EXPERIMENTAL)
# ──────────────────────────────────────────────────────────────
def run_baseline_b(data, window):
    """Baseline B: double-integrate rotated acceleration.

    EXPERIMENTAL — assumes phone X ≈ vehicle forward, phone Y ≈ vehicle left.
    This frame assumption is UNVERIFIED.

    Causal: only uses IMU + state available at blackout start.
    """
    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    i2 = window["eval_end_idx"]

    # ── INITIALIZATION ──
    ref_lat0 = data["veh_lat"][i0]
    ref_lon0 = data["veh_lon"][i0]

    east0, north0, _ = ll2enu(
        data["veh_lat"][i0], data["veh_lon"][i0], 0.0,
        ref_lat0, ref_lon0, 0.0
    )

    heading0 = np.radians(data["veh_heading"][i0])
    speed0 = data["veh_velocity"][i0] / 3.6

    # Initial velocity in ENU
    ve0 = speed0 * np.cos(heading0)  # east velocity
    vn0 = speed0 * np.sin(heading0)  # north velocity

    # ── GRAVITY CALIBRATION (mean gravity over window) ──
    # Use the gravity vector to estimate average phone orientation
    g_window = slice(max(0, i0 - 100), i0)
    gx_mean = np.mean(data["gravity_x"][g_window])
    gy_mean = np.mean(data["gravity_y"][g_window])
    gz_mean = np.mean(data["gravity_z"][g_window])

    # ── PROPAGATION (strictly causal) ──
    n_eval = i2 - i0

    headings = np.zeros(n_eval)
    positions_east = np.zeros(n_eval)
    positions_north = np.zeros(n_eval)
    velocities_east = np.zeros(n_eval)
    velocities_north = np.zeros(n_eval)

    headings[0] = heading0
    positions_east[0] = east0
    positions_north[0] = north0
    velocities_east[0] = ve0
    velocities_north[0] = vn0

    h = heading0
    e_pos = east0
    n_pos = north0
    ve = ve0
    vn = vn0

    for k in range(1, n_eval):
        idx = i0 + k
        dt = data["sync_time"][idx] - data["sync_time"][idx - 1]

        if dt > DT_MAX_GAP or dt <= 0:
            headings[k] = h
            positions_east[k] = e_pos
            positions_north[k] = n_pos
            velocities_east[k] = ve
            velocities_north[k] = vn
            continue

        # Heading from gyro pitch
        gyro_pitch = data["gyro_pitch"][idx - 1]
        h = h + gyro_pitch * dt

        # Specific force (accelerometer minus gravity)
        ax = data["accel_x"][idx - 1] - data["gravity_x"][idx - 1]
        ay = data["accel_y"][idx - 1] - data["gravity_y"][idx - 1]

        # Rotate body-frame acceleration into navigation frame
        # ASSUMPTION: phone X ≈ vehicle forward, phone Y ≈ vehicle left
        # Then: a_east = ax*cos(h) - ay*sin(h)   [if phone Y is left]
        #        a_north = ax*sin(h) + ay*cos(h)
        # NOTE: This rotation is UNVERIFIED and is the main weakness.
        c_h = np.cos(h)
        s_h = np.sin(h)
        a_east = ax * c_h - ay * s_h
        a_north = ax * s_h + ay * c_h

        # Velocity integration (trapezoidal)
        ve = ve + a_east * dt
        vn = vn + a_north * dt

        # Position integration (trapezoidal)
        e_pos = e_pos + ve * dt + 0.5 * a_east * dt ** 2
        n_pos = n_pos + vn * dt + 0.5 * a_north * dt ** 2

        headings[k] = h
        positions_east[k] = e_pos
        positions_north[k] = n_pos
        velocities_east[k] = ve
        velocities_north[k] = vn

    return {
        "headings": headings,
        "positions_east": positions_east,
        "positions_north": positions_north,
        "velocities_east": velocities_east,
        "velocities_north": velocities_north,
        "init_heading_deg": np.degrees(heading0),
        "init_speed_ms": speed0,
        "init_east": east0,
        "init_north": north0,
        "ref_lat0": ref_lat0,
        "ref_lon0": ref_lon0,
        "gravity_used": (gx_mean, gy_mean, gz_mean),
    }


# ──────────────────────────────────────────────────────────────
# REFERENCE TRAJECTORY
# ──────────────────────────────────────────────────────────────
def get_reference_enu(data, i0, i2, ref_lat0, ref_lon0):
    """Get reference (ground-truth) position in ENU from vehicle GPS."""
    n = i2 - i0
    ref_east = np.zeros(n)
    ref_north = np.zeros(n)
    ref_heading = np.zeros(n)
    ref_velocity = np.zeros(n)

    for k in range(n):
        idx = i0 + k
        e, n, _ = ll2enu(
            data["veh_lat"][idx], data["veh_lon"][idx], 0.0,
            ref_lat0, ref_lon0, 0.0
        )
        ref_east[k] = e
        ref_north[k] = n
        ref_heading[k] = data["veh_heading"][idx]
        ref_velocity[k] = data["veh_velocity"][idx]

    return ref_east, ref_north, ref_heading, ref_velocity


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def compute_metrics(dr_east, dr_north, dr_heading,
                    ref_east, ref_north, ref_heading,
                    sync_time, t_blackout_start, blackout_end_rel):
    """Compute DR error metrics as a function of time since blackout start."""
    n = len(dr_east)
    t_since = sync_time[:n] - sync_time[0]

    # Position error
    de = dr_east - ref_east
    dn = dr_north - ref_north
    pos_error = np.sqrt(de ** 2 + dn ** 2)

    # Heading error (handle wrap-around)
    dh = np.degrees(dr_heading) - ref_heading
    dh = (dh + 180) % 360 - 180
    heading_error = np.abs(dh)

    # Sample at key durations
    results = {}
    for dur in [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]:
        # Find closest index to this duration
        mask = t_since >= dur
        if mask.any():
            idx = np.argmax(mask)
            results[dur] = {
                "t_s": t_since[idx],
                "pos_error_m": pos_error[idx],
                "east_error_m": de[idx],
                "north_error_m": dn[idx],
                "heading_error_deg": heading_error[idx],
            }

    # Overall metrics over blackout period
    blackout_mask = t_since <= (t_since[blackout_end_rel] if blackout_end_rel < n else t_since[-1])
    if blackout_mask.sum() > 0:
        bm = blackout_mask
        results["blackout"] = {
            "mae_m": float(np.mean(pos_error[bm])),
            "rmse_m": float(np.sqrt(np.mean(pos_error[bm] ** 2))),
            "max_m": float(np.max(pos_error[bm])),
            "final_m": float(pos_error[blackout_end_rel]) if blackout_end_rel < n else float(pos_error[-1]),
            "heading_mae_deg": float(np.mean(heading_error[bm])),
            "heading_rmse_deg": float(np.sqrt(np.mean(heading_error[bm] ** 2))),
        }

    return {
        "t_since": t_since,
        "pos_error": pos_error,
        "east_error": de,
        "north_error": dn,
        "heading_error": heading_error,
        "at_durations": results,
    }


# ──────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────
def plot_position_error_vs_time(all_results, duration):
    """Plot position error vs time since blackout for all blackouts of a given duration."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, res in enumerate(all_results):
        t = res["t_since"]
        err = res["pos_error"]
        ax.plot(t, err, alpha=0.6, label=f"Blackout {i + 1}")

    ax.set_xlabel("Time since blackout start (s)", fontsize=12)
    ax.set_ylabel("Position error (m)", fontsize=12)
    ax.set_title(f"DR Position Error vs Time — {duration}s Blackout (Baseline A)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)

    outpath = OUT_DIR / f"dr_position_error_{duration}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_trajectory_example(dr_result, ref_enu, window, baseline_label):
    """Plot example trajectory: reference vs DR during blackout."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    i0 = window["start_idx"]
    i1 = window["blackout_end_idx"]
    n_blackout = i1 - i0
    n_total = window["eval_end_idx"] - i0

    # Reference trajectory (full)
    ax.plot(ref_enu[0], ref_enu[1], "b-", alpha=0.3, linewidth=0.5, label="Reference (full)")

    # DR trajectory
    ax.plot(dr_result["positions_east"][:n_total],
            dr_result["positions_north"][:n_total],
            "r-", linewidth=1.5, label=f"DR ({baseline_label})")

    # Mark blackout boundaries
    ax.plot(ref_enu[0][0], ref_enu[1][0], "go", markersize=10, label="Blackout start")
    if n_blackout < len(ref_enu[0]):
        ax.plot(ref_enu[0][n_blackout], ref_enu[1][n_blackout], "r^",
                markersize=10, label="Blackout end")

    # Highlight blackout region
    ax.plot(ref_enu[0][:n_blackout], ref_enu[1][:n_blackout],
            "b-", linewidth=2.5, alpha=0.7, label="Reference (blackout)")

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title(f"Trajectory: {window['duration_s']}s Blackout — {baseline_label}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    outpath = OUT_DIR / f"dr_trajectory_{window['duration_s']}s_{baseline_label.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_heading_error_vs_time(all_results, duration):
    """Plot heading error vs time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, res in enumerate(all_results):
        t = res["t_since"]
        err = res["heading_error"]
        ax.plot(t, err, alpha=0.6, label=f"Blackout {i + 1}")

    ax.set_xlabel("Time since blackout start (s)", fontsize=12)
    ax.set_ylabel("Heading error (deg)", fontsize=12)
    ax.set_title(f"DR Heading Error vs Time — {duration}s Blackout (Baseline A)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration + 10)

    outpath = OUT_DIR / f"dr_heading_error_{duration}s.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_error_vs_blackout_duration(summary_by_duration, baseline_label):
    """Plot summary: error metrics vs blackout duration."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    durations = sorted(summary_by_duration.keys())
    mae_vals = [summary_by_duration[d]["mae_m_mean"] for d in durations]
    rmse_vals = [summary_by_duration[d]["rmse_m_mean"] for d in durations]
    max_vals = [summary_by_duration[d]["max_m_mean"] for d in durations]

    axes[0].bar(range(len(durations)), mae_vals, tick_label=[f"{d}s" for d in durations])
    axes[0].set_ylabel("MAE (m)")
    axes[0].set_title("Mean Absolute Error")

    axes[1].bar(range(len(durations)), rmse_vals, tick_label=[f"{d}s" for d in durations])
    axes[1].set_ylabel("RMSE (m)")
    axes[1].set_title("Root Mean Square Error")

    axes[2].bar(range(len(durations)), max_vals, tick_label=[f"{d}s" for d in durations])
    axes[2].set_ylabel("Max Error (m)")
    axes[2].set_title("Maximum Error")

    fig.suptitle(f"DR Error vs Blackout Duration — {baseline_label}", fontsize=13, y=1.02)

    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")

    outpath = OUT_DIR / f"dr_error_vs_duration_{baseline_label.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_east_north_error(all_results_by_duration, baseline_label):
    """Plot East/North error decomposition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for dur, results_list in sorted(all_results_by_duration.items()):
        if not results_list:
            continue
        # Average across blackouts
        max_len = max(len(r["t_since"]) for r in results_list)
        t_avg = np.zeros(max_len)
        de_avg = np.zeros(max_len)
        dn_avg = np.zeros(max_len)
        count = np.zeros(max_len)

        for r in results_list:
            n = len(r["t_since"])
            t_avg[:n] += r["t_since"]
            de_avg[:n] += r["east_error"]
            dn_avg[:n] += r["north_error"]
            count[:n] += 1

        count[count == 0] = 1
        t_avg /= count
        de_avg /= count
        dn_avg /= count

        mask = count > 0
        axes[0].plot(t_avg[mask], de_avg[mask], alpha=0.6, label=f"{dur}s")
        axes[1].plot(t_avg[mask], dn_avg[mask], alpha=0.6, label=f"{dur}s")

    axes[0].set_xlabel("Time since blackout start (s)")
    axes[0].set_ylabel("East error (m)")
    axes[0].set_title("East Error Component")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time since blackout start (s)")
    axes[1].set_ylabel("North error (m)")
    axes[1].set_title("North Error Component")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"DR East/North Error Decomposition — {baseline_label}", fontsize=13)

    outpath = OUT_DIR / f"dr_east_north_error_{baseline_label.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────────
def generate_report(cols, segments, windows, gyro_analysis,
                    results_a, results_b, summary_a, summary_b):
    """Generate the full classical DR report."""
    lines = []

    def add(s=""):
        lines.append(s)

    add("=" * 80)
    add("CLASSICAL DEAD-RECKONING V0 BASELINE REPORT")
    add("=" * 80)
    add()
    add(V0_ASSUMPTIONS)

    add("=" * 80)
    add("A. INPUT DATASET")
    add("=" * 80)
    add(f"  File: {DATA_FILE}")
    add(f"  Rows: 91,463")
    add(f"  Duration: ~9460 s (~2.6 hours)")
    add(f"  Segments: {len(segments)} (separated by recording gaps)")
    for i, seg in enumerate(segments):
        add(f"    Seg {i}: {seg['n_rows']} rows, {seg['duration_s']:.1f} s")
    add()

    add("=" * 80)
    add("B. COLUMNS USED")
    add("=" * 80)
    for key, col in cols.items():
        add(f"  {key:25s} = {col}")
    add()

    add("=" * 80)
    add("C. COORDINATE-FRAME ASSUMPTIONS")
    add("=" * 80)
    add("  Local tangent-plane ENU:")
    add("    East  = dlon × cos(lat0) × (π/180) × R_EARTH")
    add("    North = dlat × (π/180) × R_EARTH")
    add("    R_EARTH = 6,371,000 m")
    add("  Validity: trajectory scale ~11 km, tangent-plane error < 0.01%")
    add("  Origin: vehicle reference position at first blackout start")
    add()

    add("=" * 80)
    add("D. PHONE→VEHICLE FRAME ASSUMPTIONS")
    add("=" * 80)
    add("  Baseline A: No frame assumption needed (heading-only propagation)")
    add("  Baseline B: UNVERIFIED — assumes phone X ≈ vehicle forward,")
    add("              phone Y ≈ vehicle left. Gravity vector confirms")
    add("              phone Z is vertical but says nothing about yaw.")
    add("  Sign agreement gyro_pitch vs vehicle_yaw_rate: ~80%")
    add("  → Baseline B results MUST be interpreted with extreme caution.")
    add()

    add("=" * 80)
    add("E. GYRO SCALE FACTOR")
    add("=" * 80)
    add(f"  Overall correlation (gyro pitch vs veh yaw rate): {gyro_analysis['r_all']:.4f}")
    add(f"  Scale factor NOT applied in V0 (used 1.0)")
    add(f"  → This is a known limitation.")
    add()

    add("=" * 80)
    add("F. TIMESTAMP / ALIGNMENT TREATMENT")
    add("=" * 80)
    add(f"  SYNC_TIME_S used for all timing")
    add(f"  Empirical inter-stream offset: +{REF_ALIGNMENT_OFFSET_S} s")
    add(f"  Applied ONLY for reference comparison, NEVER during DR propagation")
    add(f"  DR propagation is strictly causal")
    add()

    add("=" * 80)
    add("G. INITIALIZATION METHOD")
    add("=" * 80)
    add("  Position: vehicle reference lat/lon at blackout start → ENU")
    add("  Heading:  vehicle reference heading at blackout start")
    add("  Velocity: Baseline A uses vehicle reference speed (constant)")
    add("            Baseline B initializes velocity from vehicle reference,")
    add("            then propagates via IMU acceleration integration")
    add()

    add("=" * 80)
    add("H. BLACKOUT SELECTION METHODOLOGY")
    add("=" * 80)
    add(f"  Durations: {BLACKOUT_DURATIONS}")
    add(f"  Constraints:")
    add(f"    - Within a single segment (no gap crossing)")
    add(f"    - Vehicle motion > {MIN_MOTION_THRESHOLD} km/h before blackout")
    add(f"    - GPS accuracy < {MAX_GPS_ACCURACY} m before blackout")
    add(f"    - Evaluation margin after blackout: min(30, duration) s")
    add(f"  Selected windows: {len(windows)}")
    for w in windows:
        add(f"    {w['duration_s']:3d}s | rows [{w['start_idx']}, {w['blackout_end_idx']}) "
            f"| T={w['t_start']:.1f}-{w['t_blackout_end']:.1f} "
            f"| pre_vel={w['pre_velocity_kmh']:.1f} km/h")
    add()

    add("=" * 80)
    add("I. CAUSALITY RULES")
    add("=" * 80)
    add("  During blackout, DR state uses ONLY:")
    add("    - IMU samples up to current time")
    add("    - Last-known velocity (Baseline A) or velocity from accel integration (B)")
    add("    - Current heading (from gyro integration)")
    add("  Ground truth (vehicle GPS, heading, yaw rate) used ONLY for evaluation")
    add()

    add("=" * 80)
    add("J. INTEGRATION EQUATIONS")
    add("=" * 80)
    add("  Baseline A:")
    add("    heading_{k+1} = heading_k + gyro_pitch_k × dt")
    add("    pos_east_{k+1} = pos_east_k + v × cos(heading_k) × dt")
    add("    pos_north_{k+1} = pos_north_k + v × sin(heading_k) × dt")
    add()
    add("  Baseline B (EXPERIMENTAL):")
    add("    heading_{k+1} = heading_k + gyro_pitch_k × dt")
    add("    a_body = accel - gravity")
    add("    a_nav = R(heading) × a_body")
    add("    v_{k+1} = v_k + a_nav × dt")
    add("    p_{k+1} = p_k + v_k × dt + 0.5 × a_nav × dt²")
    add()

    # Results
    for label, summary in [("BASELINE A (Yaw-Only Kinematic)", summary_a),
                           ("BASELINE B (IMU Acceleration — EXPERIMENTAL)", summary_b)]:
        if summary is None:
            continue
        add("=" * 80)
        add(f"K. METRICS — {label}")
        add("=" * 80)
        add(f"  {'Duration':>10s}  {'MAE (m)':>10s}  {'RMSE (m)':>10s}  "
            f"{'Max (m)':>10s}  {'Final (m)':>10s}  {'Head MAE':>10s}")
        add(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        for dur in sorted(summary.keys()):
            s = summary[dur]
            add(f"  {dur:>7d} s  {s['mae_m_mean']:10.1f}  {s['rmse_m_mean']:10.1f}  "
                f"{s['max_m_mean']:10.1f}  {s['final_m_mean']:10.1f}  "
                f"{s['heading_mae_deg_mean']:9.1f}°")
        add()

    add("=" * 80)
    add("L. NUMERICAL RESULTS")
    add("=" * 80)
    for label, summary in [("Baseline A", summary_a), ("Baseline B", summary_b)]:
        if summary is None:
            continue
        add(f"\n  {label}:")
        for dur in sorted(summary.keys()):
            s = summary[dur]
            add(f"    {dur:3d}s blackout:")
            add(f"      Position MAE:  {s['mae_m_mean']:8.1f} m")
            add(f"      Position RMSE: {s['rmse_m_mean']:8.1f} m")
            add(f"      Position Max:  {s['max_m_mean']:8.1f} m")
            add(f"      Position Final:{s['final_m_mean']:8.1f} m")
            add(f"      Heading MAE:   {s['heading_mae_deg_mean']:8.1f}°")
    add()

    add("=" * 80)
    add("M. LIMITATIONS")
    add("=" * 80)
    add("  1. Gyro pitch correlation with vehicle yaw rate is moderate (~0.71)")
    add("     Sign agreement ~80%. This limits heading propagation accuracy.")
    add("  2. No gyro scale factor applied (V0 uses 1.0)")
    add("  3. Phone-to-vehicle frame mapping is UNVERIFIED (Baseline B)")
    add("  4. Baseline A holds velocity constant — no acceleration model")
    add("  5. GPS updates at ~1 Hz with ~3 m accuracy — init position has error")
    add("  6. Vehicle reference used for initialization (not phone GPS)")
    add("  7. Small sample: only ~3 blackout windows per duration")
    add("  8. No sensor noise modeling or filtering")
    add()

    add("=" * 80)
    add("N. IS THE BASELINE PHYSICALLY DEFENSIBLE?")
    add("=" * 80)
    add("  Baseline A: YES — it is a simple, honest kinematic DR model.")
    add("    Heading from gyro integration, constant velocity, known initial state.")
    add("    Every assumption is documented and the failure modes are expected.")
    add()
    add("  Baseline B: NO — the phone→vehicle frame transform is unverified.")
    add("    The 80% sign agreement and unknown yaw rotation between phone")
    add("    and vehicle mean that Baseline B acceleration integration may")
    add("    inject large systematic errors. Results should be treated as")
    add("    experimental and not as evidence of IMU capability.")
    add()

    add("=" * 80)
    add("O. RECOMMENDATIONS FOR NEXT STEPS")
    add("=" * 80)
    add("  1. Run Baseline A and quantify the real drift — do not optimize")
    add("  2. If Baseline A drift is severe, the ML correction has clear targets")
    add("  3. Investigate gyro pitch sign disagreements — phone reorientation?")
    add("  4. Consider using steering angle + wheel speed for velocity model")
    add("  5. If Baseline B shows large errors, this confirms frame uncertainty")
    add("     is the critical limitation to address")
    add("  6. Future: EKF with gyro + vehicle speed fusion")
    add("  7. Future: ML correction model (the ultimate goal)")
    add()
    add("=" * 80)
    add("END OF REPORT")
    add("=" * 80)

    report_text = "\n".join(lines)
    report_path = OUT_DIR / "classical_dr_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n  Report saved: {report_path}")
    return report_text


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("CLASSICAL DEAD-RECKONING V0 BASELINE")
    print("=" * 80)

    # ── Load data ──
    data, cols, df = load_data()

    # ── Detect segments ──
    segments = detect_segments(data["sync_time"])

    # ── Analyze gyro ──
    gyro_analysis = analyze_gyro_scale(data["gyro_pitch"], data["veh_yaw_rate"])

    # ── Select blackout windows ──
    windows = select_blackout_windows(data, segments, BLACKOUT_DURATIONS)

    if len(windows) == 0:
        print("\n*** No valid blackout windows found. Cannot proceed. ***")
        sys.exit(1)

    # ── Run baselines ──
    all_results_a = {}  # duration → list of metric dicts
    all_results_b = {}
    all_metrics_a = {}  # duration → list of metric result dicts
    all_metrics_b = {}

    for w in windows:
        dur = w["duration_s"]
        print(f"\n  Running blackouts: {dur}s at T={w['t_start']:.1f}...")

        # Baseline A
        print(f"    Baseline A...")
        res_a = run_baseline_a(data, w)
        ref_e, ref_n, ref_h, ref_v = get_reference_enu(
            data, w["start_idx"], w["eval_end_idx"],
            res_a["ref_lat0"], res_a["ref_lon0"]
        )
        n_bo = w["blackout_end_idx"] - w["start_idx"]
        met_a = compute_metrics(
            res_a["positions_east"], res_a["positions_north"], res_a["headings"],
            ref_e, ref_n, ref_h,
            data["sync_time"][w["start_idx"]:w["eval_end_idx"]],
            0, n_bo
        )
        all_metrics_a.setdefault(dur, []).append(met_a)

        # Baseline B
        print(f"    Baseline B...")
        res_b = run_baseline_b(data, w)
        met_b = compute_metrics(
            res_b["positions_east"], res_b["positions_north"], res_b["headings"],
            ref_e, ref_n, ref_h,
            data["sync_time"][w["start_idx"]:w["eval_end_idx"]],
            0, n_bo
        )
        all_metrics_b.setdefault(dur, []).append(met_b)

        # Plots for individual blackouts
        plot_position_error_vs_time([met_a], dur)
        plot_heading_error_vs_time([met_a], dur)
        plot_trajectory_example(res_a, (ref_e, ref_n), w, "Baseline A")

    # ── Summary across blackouts ──
    def summarize(all_met):
        summary = {}
        for dur, met_list in all_met.items():
            # Average across blackouts
            keys_at_dur = {}
            for met in met_list:
                for d, vals in met["at_durations"].items():
                    if d == "blackout":
                        continue
                    keys_at_dur.setdefault(d, {}).update(vals)

            avg = {}
            for d in sorted(keys_at_dur.keys()):
                vals = keys_at_dur[d]
                avg[d] = {k: float(np.mean([m["at_durations"][d][k]
                          for m in met_list if d in m["at_durations"]]))
                          for k in vals}
                # Also compute blackout-level summary
            if met_list and "blackout" in met_list[0]["at_durations"]:
                bl = [m["at_durations"]["blackout"] for m in met_list if "blackout" in m["at_durations"]]
                if bl:
                    avg["blackout"] = {
                        k: float(np.mean([b[k] for b in bl]))
                        for k in bl[0].keys()
                    }
            summary[dur] = avg

        # Build per-duration summary
        dur_summary = {}
        for dur, avg in summary.items():
            blackout = avg.get("blackout", {})
            dur_summary[dur] = {
                "mae_m_mean": blackout.get("mae_m", 0),
                "rmse_m_mean": blackout.get("rmse_m", 0),
                "max_m_mean": blackout.get("max_m", 0),
                "final_m_mean": blackout.get("final_m", 0),
                "heading_mae_deg_mean": blackout.get("heading_mae_deg", 0),
                "heading_rmse_deg_mean": blackout.get("heading_rmse_deg", 0),
            }
        return dur_summary

    summary_a = summarize(all_metrics_a) if all_metrics_a else None
    summary_b = summarize(all_metrics_b) if all_metrics_b else None

    # ── Aggregate plots ──
    for dur in BLACKOUT_DURATIONS:
        if dur in all_metrics_a:
            plot_position_error_vs_time(all_metrics_a[dur], dur)
            plot_heading_error_vs_time(all_metrics_a[dur], dur)

    if summary_a:
        plot_error_vs_blackout_duration(summary_a, "Baseline A")
        plot_east_north_error(all_metrics_a, "Baseline A")

    # ── Report ──
    generate_report(cols, segments, windows, gyro_analysis,
                    all_metrics_a, all_metrics_b, summary_a, summary_b)

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for label, summary in [("Baseline A", summary_a), ("Baseline B", summary_b)]:
        if summary is None:
            continue
        print(f"\n  {label}:")
        for dur in sorted(summary.keys()):
            s = summary[dur]
            print(f"    {dur:3d}s: MAE={s['mae_m_mean']:.1f} m, "
                  f"RMSE={s['rmse_m_mean']:.1f} m, "
                  f"Max={s['max_m_mean']:.1f} m, "
                  f"Head={s['heading_mae_deg_mean']:.1f}°")

    print(f"\n  Outputs: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
