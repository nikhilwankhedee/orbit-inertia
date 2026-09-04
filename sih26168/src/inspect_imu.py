#!/usr/bin/env python3
"""
inspect_imu.py — IMU characterization and phone→vehicle frame mapping.

Replaces the previous version with:
  - Semantic column discovery (no hardcoded mojibake strings)
  - Correct UTF-8 CSV reading
  - Full correlation analysis across ALL gyro axes vs vehicle yaw rate
  - Linear acceleration vs vehicle acceleration correlation
  - Turn-event identification and comparison plots
  - Frame-mapping hypothesis with evidence
  - Latency investigation distinguishing sync error from sensor latency
  - All diagnostic outputs saved to sih26168/outputs/

Usage:
    python inspect_imu.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FILE = ROOT / "processed" / "S4_synced.csv"

OUT_DIR = ROOT / "sih26168" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT LOGGING HELPER
# ============================================================

_text_lines = []


def log(msg=""):
    """Print to stdout and accumulate for file output."""
    print(msg)
    _text_lines.append(msg)


def save_text(filename, lines):
    path = OUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Saved: {path}")


# ============================================================
# LOAD CSV (UTF-8 — the file is valid UTF-8 with mojibake chars)
# ============================================================

log("Loading synchronized dataset...")

if not FILE.exists():
    raise FileNotFoundError(f"Dataset not found:\n{FILE}")

df = pd.read_csv(FILE)  # default UTF-8

df.columns = df.columns.str.strip()

log(f"Rows: {len(df):,}")
log(f"Columns: {len(df.columns)}")


# ============================================================
# SEMANTIC COLUMN DISCOVERY
# ============================================================

def find_column(prefix):
    """
    Locate a column by its stable semantic prefix.

    Avoids hardcoding corrupted Unicode (mojibake) such as
    Â°, Î¼, Â² etc.
    """
    prefix_upper = prefix.upper()
    matches = [
        c for c in df.columns if c.upper().startswith(prefix_upper)
    ]
    if len(matches) == 0:
        raise ValueError(
            f"No column starting with '{prefix}'\n"
            f"Available:\n" + "\n".join(f"  {c}" for c in df.columns)
        )
    if len(matches) > 1:
        log(f"  WARNING: prefix '{prefix}' matches multiple:")
        for m in matches:
            log(f"    {m}")
        log(f"    Using first: {matches[0]}")
    return matches[0]


# Phone IMU columns
ACCEL_X = find_column("ACCELEROMETER X")
ACCEL_Y = find_column("ACCELEROMETER Y")
ACCEL_Z = find_column("ACCELEROMETER Z")

GRAVITY_X = find_column("GRAVITY X")
GRAVITY_Y = find_column("GRAVITY Y")
GRAVITY_Z = find_column("GRAVITY Z")

GYRO_YAW = find_column("GYROSCOPE Yaw")
GYRO_PITCH = find_column("GYROSCOPE Pitch")
GYRO_ROLL = find_column("GYROSCOPE Roll")

ORIENT_YAW = find_column("ORIENTATION (Yaw)")
ORIENT_PITCH = find_column("ORIENTATION (Pitch)")
ORIENT_ROLL = find_column("ORIENTATION (Roll")

# Vehicle reference columns (exact names in this dataset)
VEH_VELOCITY = find_column("VELOCITY (")     # "Velocity (km/hr)" — avoids "Vertical velocity"
VEH_HEADING = find_column("HEADING (")       # "Heading (degrees)"
VEH_LONG_ACCEL = find_column("INDICATED LONGITUDINAL")  # "Indicated Longitudinal Acceleration (g)"
VEH_LAT_ACCEL = find_column("INDICATED LATERAL")        # "Indicated Lateral Acceleration (g)"
VEH_YAW_RATE = find_column("YAW RATE (")     # "Yaw Rate (deg/sec)"

# Timing
TIME = "SYNC_TIME_S"


# ============================================================
# PRINT SELECTED COLUMNS
# ============================================================

log()
log("=" * 80)
log("SELECTED COLUMNS (matched by prefix)")
log("=" * 80)

log(f"  Accel X:        {ACCEL_X}")
log(f"  Accel Y:        {ACCEL_Y}")
log(f"  Accel Z:        {ACCEL_Z}")
log(f"  Gravity X:      {GRAVITY_X}")
log(f"  Gravity Y:      {GRAVITY_Y}")
log(f"  Gravity Z:      {GRAVITY_Z}")
log(f"  Gyro Yaw:       {GYRO_YAW}")
log(f"  Gyro Pitch:     {GYRO_PITCH}")
log(f"  Gyro Roll:      {GYRO_ROLL}")
log(f"  Orient Yaw:     {ORIENT_YAW}")
log(f"  Orient Pitch:   {ORIENT_PITCH}")
log(f"  Orient Roll:    {ORIENT_ROLL}")
log(f"  Veh Velocity:   {VEH_VELOCITY}")
log(f"  Veh Heading:    {VEH_HEADING}")
log(f"  Veh Long Accel: {VEH_LONG_ACCEL}")
log(f"  Veh Lat Accel:  {VEH_LAT_ACCEL}")
log(f"  Veh Yaw Rate:   {VEH_YAW_RATE}")
log(f"  Time:           {TIME}")


# ============================================================
# NUMERIC CONVERSION
# ============================================================

all_cols = [
    ACCEL_X, ACCEL_Y, ACCEL_Z,
    GRAVITY_X, GRAVITY_Y, GRAVITY_Z,
    GYRO_YAW, GYRO_PITCH, GYRO_ROLL,
    ORIENT_YAW, ORIENT_PITCH, ORIENT_ROLL,
    VEH_VELOCITY, VEH_HEADING, VEH_LONG_ACCEL,
    VEH_LAT_ACCEL, VEH_YAW_RATE,
    TIME,
]

for col in all_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# VALID ROWS — require all IMU + vehicle columns present
# ============================================================

valid = df[all_cols].notna().all(axis=1)
imu = df.loc[valid].copy().reset_index(drop=True)

log()
log(f"Valid rows (all columns present): {len(imu):,}")
log(f"Dropped {len(df) - len(imu):,} rows with NaN in key columns")


# ============================================================
# BUILD NUMPY ARRAYS
# ============================================================

t = imu[TIME].to_numpy()

accel = imu[[ACCEL_X, ACCEL_Y, ACCEL_Z]].to_numpy()
gravity = imu[[GRAVITY_X, GRAVITY_Y, GRAVITY_Z]].to_numpy()
gyro = imu[[GYRO_YAW, GYRO_PITCH, GYRO_ROLL]].to_numpy()
orientation = imu[[ORIENT_YAW, ORIENT_PITCH, ORIENT_ROLL]].to_numpy()

veh_velocity = imu[VEH_VELOCITY].to_numpy()
veh_heading = imu[VEH_HEADING].to_numpy()
veh_long_accel = imu[VEH_LONG_ACCEL].to_numpy()
veh_lat_accel = imu[VEH_LAT_ACCEL].to_numpy()
veh_yaw_rate = imu[VEH_YAW_RATE].to_numpy()

linear_accel = accel - gravity  # candidate — see Section 4


# ============================================================
# SECTION 1: DATASET SHAPE AND TIMING
# ============================================================

log()
log("=" * 80)
log("1. DATASET SHAPE AND TIMING")
log("=" * 80)

duration = t[-1] - t[0]
dt = np.diff(t)
dt_valid = dt[(dt > 0) & (dt < 1.0)]

log(f"  Rows:          {len(imu):,}")
log(f"  Time range:    {t[0]:.3f} s  to  {t[-1]:.3f} s")
log(f"  Duration:      {duration:.3f} s  ({duration/60:.1f} min)")
log()
log(f"  Median dt:     {np.median(dt_valid)*1000:.2f} ms")
log(f"  Mean dt:       {np.mean(dt_valid)*1000:.2f} ms")
log(f"  Std dt:        {np.std(dt_valid)*1000:.2f} ms")
log(f"  P95 dt:        {np.percentile(dt_valid, 95)*1000:.2f} ms")
log(f"  Max dt:        {np.max(dt_valid)*1000:.2f} ms")
log(f"  Estimated freq:{1/np.median(dt_valid):.1f} Hz")

# Large gaps
gap_threshold = 0.5  # seconds
large_gaps = dt[dt > gap_threshold]
log()
log(f"  Gaps > {gap_threshold} s:  {len(large_gaps)}")
if len(large_gaps) > 0:
    for i, gap in enumerate(large_gaps[:20]):
        idx = np.where(dt > gap_threshold)[0][i]
        log(f"    t={t[idx]:.2f} s  gap={gap:.3f} s")
    if len(large_gaps) > 20:
        log(f"    ... and {len(large_gaps)-20} more")


# ============================================================
# SECTION 2: ACCELEROMETER STATISTICS
# ============================================================

log()
log("=" * 80)
log("2. ACCELEROMETER STATISTICS (m/s²)")
log("=" * 80)

axis_labels = ["X", "Y", "Z"]
accel_col_names = [ACCEL_X, ACCEL_Y, ACCEL_Z]

for i, (label, col) in enumerate(zip(axis_labels, accel_col_names)):
    v = accel[:, i]
    log(f"  Axis {label} ({col}):")
    log(f"    Mean:  {np.mean(v):10.4f}")
    log(f"    Std:   {np.std(v):10.4f}")
    log(f"    Min:   {np.min(v):10.4f}")
    log(f"    P01:   {np.percentile(v, 1):10.4f}")
    log(f"    P50:   {np.median(v):10.4f}")
    log(f"    P99:   {np.percentile(v, 99):10.4f}")
    log(f"    Max:   {np.max(v):10.4f}")

accel_mag = np.linalg.norm(accel, axis=1)
log()
log(f"  Magnitude |a|:")
log(f"    Mean:  {np.mean(accel_mag):10.4f} m/s²")
log(f"    Std:   {np.std(accel_mag):10.4f} m/s²")
log(f"    Min:   {np.min(accel_mag):10.4f} m/s²")
log(f"    P01:   {np.percentile(accel_mag, 1):10.4f}")
log(f"    P50:   {np.median(accel_mag):10.4f}")
log(f"    P99:   {np.percentile(accel_mag, 99):10.4f}")
log(f"    Max:   {np.max(accel_mag):10.4f}")


# ============================================================
# SECTION 3: GRAVITY STATISTICS
# ============================================================

log()
log("=" * 80)
log("3. GRAVITY VECTOR STATISTICS (m/s²)")
log("=" * 80)

for i, label in enumerate(axis_labels):
    v = gravity[:, i]
    log(f"  Axis {label}:")
    log(f"    Mean:  {np.mean(v):10.4f}")
    log(f"    Std:   {np.std(v):10.4f}")
    log(f"    Min:   {np.min(v):10.4f}")
    log(f"    Max:   {np.max(v):10.4f}")

gravity_mag = np.linalg.norm(gravity, axis=1)
log()
log(f"  Magnitude |g|:")
log(f"    Mean:  {np.mean(gravity_mag):10.6f} m/s²")
log(f"    Std:   {np.std(gravity_mag):10.6f} m/s²")
log(f"    Min:   {np.min(gravity_mag):10.6f} m/s²")
log(f"    Max:   {np.max(gravity_mag):10.6f} m/s²")
log(f"    (Expected: ~9.81 m/s²)")


# ============================================================
# SECTION 4: LINEAR ACCELERATION (CANDIDATE)
# ============================================================

log()
log("=" * 80)
log("4. LINEAR ACCELERATION — CANDIDATE QUANTITY")
log("=" * 80)
log("  Formula: linear_accel = accelerometer - gravity")
log("  WARNING: This is a DIAGNOSTIC estimate only.")
log("  We do NOT yet know if the phone reports true")
log("  linear accel, or if gravity subtraction is valid")
log("  for this device/sensor fusion pipeline.")
log()

for i, label in enumerate(axis_labels):
    v = linear_accel[:, i]
    log(f"  Axis {label}:")
    log(f"    Mean:  {np.mean(v):10.4f}")
    log(f"    Std:   {np.std(v):10.4f}")
    log(f"    Min:   {np.min(v):10.4f}")
    log(f"    P01:   {np.percentile(v, 1):10.4f}")
    log(f"    P50:   {np.median(v):10.4f}")
    log(f"    P99:   {np.percentile(v, 99):10.4f}")
    log(f"    Max:   {np.max(v):10.4f}")

lin_mag = np.linalg.norm(linear_accel, axis=1)
log()
log(f"  Candidate |a_linear| magnitude:")
log(f"    Mean:  {np.mean(lin_mag):10.4f} m/s²")
log(f"    Std:   {np.std(lin_mag):10.4f} m/s²")
log(f"    Min:   {np.min(lin_mag):10.4f} m/s²")
log(f"    P50:   {np.median(lin_mag):10.4f}")
log(f"    Max:   {np.max(lin_mag):10.4f}")


# ============================================================
# SECTION 5: GYROSCOPE STATISTICS
# ============================================================

log()
log("=" * 80)
log("5. GYROSCOPE STATISTICS")
log("=" * 80)

gyro_names = ["Yaw", "Pitch", "Roll"]
gyro_cols = [GYRO_YAW, GYRO_PITCH, GYRO_ROLL]

for i, (name, col) in enumerate(zip(gyro_names, gyro_cols)):
    v = gyro[:, i]
    vd = np.degrees(v)
    log(f"  {name} ({col}):")
    log(f"    Mean:  {np.mean(v):10.5f} rad/s  ({np.mean(vd):10.3f} deg/s)")
    log(f"    Std:   {np.std(v):10.5f} rad/s  ({np.std(vd):10.3f} deg/s)")
    log(f"    Min:   {np.min(v):10.5f} rad/s  ({np.min(vd):10.3f} deg/s)")
    log(f"    P01:   {np.percentile(v, 1):10.5f}")
    log(f"    P50:   {np.median(v):10.5f}")
    log(f"    P99:   {np.percentile(v, 99):10.5f}")
    log(f"    Max:   {np.max(v):10.5f} rad/s  ({np.max(vd):10.3f} deg/s)")

gyro_mag = np.linalg.norm(gyro, axis=1)
log()
log(f"  |gyro| magnitude:")
log(f"    Mean:  {np.mean(gyro_mag):.5f} rad/s")
log(f"    Max:   {np.max(gyro_mag):.5f} rad/s")


# ============================================================
# SECTION 6: PHONE ORIENTATION
# ============================================================

log()
log("=" * 80)
log("6. PHONE ORIENTATION (degrees)")
log("=" * 80)
log("  WARNING: Orientation convention is NOT yet known.")
log("  Yaw/Pitch/Roll labels may not map to vehicle axes.")
log()

for i, name in enumerate(["Yaw", "Pitch", "Roll"]):
    v = orientation[:, i]
    log(f"  {name}:")
    log(f"    Mean:  {np.mean(v):10.3f}")
    log(f"    Std:   {np.std(v):10.3f}")
    log(f"    Min:   {np.min(v):10.3f}")
    log(f"    P01:   {np.percentile(v, 1):10.3f}")
    log(f"    P50:   {np.median(v):10.3f}")
    log(f"    P99:   {np.percentile(v, 99):10.3f}")
    log(f"    Max:   {np.max(v):10.3f}")


# ============================================================
# SECTION 7: VEHICLE REFERENCE SIGNALS
# ============================================================

log()
log("=" * 80)
log("7. VEHICLE REFERENCE SIGNALS")
log("=" * 80)

veh_stats = {
    "Velocity (km/h)": veh_velocity,
    "Heading (deg)": veh_heading,
    "Long Accel (g)": veh_long_accel,
    "Lat Accel (g)": veh_lat_accel,
    "Yaw Rate (deg/s)": veh_yaw_rate,
}

for name, v in veh_stats.items():
    log(f"  {name}:")
    log(f"    Mean:  {np.mean(v):10.4f}")
    log(f"    Std:   {np.std(v):10.4f}")
    log(f"    Min:   {np.min(v):10.4f}")
    log(f"    P01:   {np.percentile(v, 1):10.4f}")
    log(f"    P50:   {np.median(v):10.4f}")
    log(f"    P99:   {np.percentile(v, 99):10.4f}")
    log(f"    Max:   {np.max(v):10.4f}")


# ============================================================
# HELPER: CROSS-CORRELATION WITH BEST LAG
# ============================================================

def best_lag_correlation(
    ref, signal, max_lag_s, dt_s, label_a="ref", label_b="signal"
):
    """
    Find the lag that maximises Pearson correlation.

    Convention:
      best_lag_s > 0  →  signal LAGS ref  (signal arrives later)
      best_lag_s < 0  →  signal LEADS ref (signal arrives earlier)

    Physical expectation for sensor latency: best_lag_s > 0
    (the phone reports values that occurred in the past).

    Returns (best_lag_s, best_corr, best_corr_sign_flipped)
    where the sign-flipped version tests -signal.
    """
    max_lag_n = int(round(max_lag_s / dt_s))
    n = len(ref)

    best_corr = -np.inf
    best_lag = 0

    for lag_n in range(-max_lag_n, max_lag_n + 1):
        if lag_n >= 0:
            a = ref[lag_n:]
            b = signal[:n - lag_n] if lag_n > 0 else signal
        else:
            a = ref[:n + lag_n]
            b = signal[-lag_n:]
        if len(a) < 200:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if not np.isnan(c) and c > best_corr:
            best_corr = c
            best_lag = lag_n

    best_lag_s = best_lag * dt_s

    # Also test sign flip
    best_corr_flip = -np.inf
    best_lag_flip = 0
    signal_neg = -signal

    for lag_n in range(-max_lag_n, max_lag_n + 1):
        if lag_n >= 0:
            a = ref[lag_n:]
            b = signal_neg[:n - lag_n] if lag_n > 0 else signal_neg
        else:
            a = ref[:n + lag_n]
            b = signal_neg[-lag_n:]
        if len(a) < 200:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if not np.isnan(c) and c > best_corr_flip:
            best_corr_flip = c
            best_lag_flip = lag_n

    best_lag_flip_s = best_lag_flip * dt_s

    return {
        "best_lag_s": best_lag_s,
        "best_corr": best_corr,
        "flip_best_lag_s": best_lag_flip_s,
        "flip_best_corr": best_corr_flip,
        "sign_flipped": best_corr_flip > best_corr,
    }


def compute_full_xcorr(ref, signal, max_lag_s, dt_s):
    """Compute normalised cross-correlation for plotting."""
    max_lag_n = int(round(max_lag_s / dt_s))
    n = min(len(ref), len(signal))
    ref = ref[:n] - ref[:n].mean()
    sig = signal[:n] - signal[:n].mean()

    ref_std = ref.std()
    sig_std = sig.std()
    if ref_std == 0 or sig_std == 0:
        return np.array([0]), np.array([0])

    lags = np.arange(-max_lag_n, max_lag_n + 1)
    cc = np.array([
        np.mean(ref[max(0, lag):n - max(0, -lag)]
                * sig[max(0, -lag):n - max(0, lag)])
        for lag in lags
    ]) / (ref_std * sig_std)

    return lags * dt_s, cc


# ============================================================
# SECTION 8: CORRELATION EXPERIMENTS
# ============================================================

log()
log("=" * 80)
log("8. CORRELATION EXPERIMENTS")
log("=" * 80)

dt_median = np.median(dt_valid)
MAX_LAG_S = 5.0

# --- 8a: Gyro axes vs vehicle yaw rate ---

log()
log("--- 8a: Gyro axes vs Vehicle Yaw Rate ---")
log(f"  Lag window: +/- {MAX_LAG_S} s")
log(f"  Median dt:  {dt_median*1000:.2f} ms")
log()

# Vehicle yaw rate in rad/s for comparison with gyro
veh_yaw_rad = np.radians(veh_yaw_rate)

gyro_axis_names = ["GYRO_Yaw (rad/s)", "GYRO_Pitch (rad/s)", "GYRO_ROLL (rad/s)"]
gyro_axis_cols = [GYRO_YAW, GYRO_PITCH, GYRO_ROLL]

gyro_results = {}

for i, (gname, gcol) in enumerate(zip(gyro_axis_names, gyro_axis_cols)):
    gv = gyro[:, i]

    # Zero-lag correlation
    zerolag = np.corrcoef(veh_yaw_rad, gv)[0, 1]

    # Best lag (no sign flip)
    res = best_lag_correlation(veh_yaw_rad, gv, MAX_LAG_S, dt_median)

    log(f"  {gcol}:")
    log(f"    Zero-lag corr:       {zerolag:+.4f}")
    log(f"    Best corr (no flip): {res['best_corr']:+.4f}  at lag {res['best_lag_s']:+.3f} s")
    log(f"    Best corr (flipped): {res['flip_best_corr']:+.4f}  at lag {res['flip_best_lag_s']:+.3f} s")
    log(f"    Sign flip helps:     {res['sign_flipped']}")
    log()

    gyro_results[gcol] = res

# --- 8b: Candidate linear accel vs vehicle accelerations ---

log("--- 8b: Candidate Linear Accel vs Vehicle Acceleration ---")
log()

veh_long_ms2 = veh_long_accel * 9.80665  # g → m/s²
veh_lat_ms2 = veh_lat_accel * 9.80665

accel_ref_pairs = [
    ("Vehicle Longitudinal Accel (m/s²)", veh_long_ms2, "longitudinal"),
    ("Vehicle Lateral Accel (m/s²)", veh_lat_ms2, "lateral"),
]

accel_results = {}

for ref_name, ref_signal, ref_type in accel_ref_pairs:
    log(f"  Against {ref_name}:")
    for i, alabel in enumerate(["Accel X", "Accel Y", "Accel Z"]):
        av = accel[:, i]
        zerolag = np.corrcoef(ref_signal, av)[0, 1]

        res = best_lag_correlation(ref_signal, av, MAX_LAG_S, dt_median)

        log(f"    {alabel} zero-lag: {zerolag:+.4f}  "
            f"best: {res['best_corr']:+.4f} @ {res['best_lag_s']:+.3f} s  "
            f"flip: {res['flip_best_corr']:+.4f} @ {res['flip_best_lag_s']:+.3f} s"
            f"  [flip helps: {res['sign_flipped']}]")
        accel_results[(ref_type, alabel)] = res
    log()

# --- 8c: Candidate linear accel vs vehicle accelerations ---

log("--- 8c: Candidate Linear Accel vs Vehicle Acceleration ---")
log()

for ref_name, ref_signal, ref_type in accel_ref_pairs:
    log(f"  Against {ref_name}:")
    for i, alabel in enumerate(["LinearAccel X", "LinearAccel Y", "LinearAccel Z"]):
        lv = linear_accel[:, i]
        zerolag = np.corrcoef(ref_signal, lv)[0, 1]

        res = best_lag_correlation(ref_signal, lv, MAX_LAG_S, dt_median)

        log(f"    {alabel} zero-lag: {zerolag:+.4f}  "
            f"best: {res['best_corr']:+.4f} @ {res['best_lag_s']:+.3f} s  "
            f"flip: {res['flip_best_corr']:+.4f} @ {res['flip_best_lag_s']:+.3f} s"
            f"  [flip helps: {res['sign_flipped']}]")
    log()

# --- 8d: ALL pairwise gyro vs yaw rate with full xcorr ---

log("--- 8d: Full cross-correlation curves (for plotting) ---")

xcorr_data = {}
for i, gcol in enumerate(gyro_axis_cols):
    lags_arr, cc_arr = compute_full_xcorr(veh_yaw_rad, gyro[:, i], MAX_LAG_S, dt_median)
    xcorr_data[gcol] = (lags_arr, cc_arr)
    best_idx = np.argmax(np.abs(cc_arr))
    log(f"  {gcol}: peak |corr| = {np.abs(cc_arr[best_idx]):.4f} at lag {lags_arr[best_idx]:+.3f} s")


# ============================================================
# SECTION 9: TURN-EVENT ANALYSIS
# ============================================================

log()
log("=" * 80)
log("9. TURN-EVENT ANALYSIS")
log("=" * 80)

# Find significant yaw-rate events
yaw_rate_abs = np.abs(veh_yaw_rate)
yaw_threshold = np.percentile(yaw_rate_abs, 90)

peaks, properties = find_peaks(
    yaw_rate_abs,
    height=yaw_threshold,
    distance=int(2.0 / dt_median),  # at least 2 s apart
    prominence=5.0,  # deg/s
)

log(f"  Yaw rate |threshold|: {yaw_threshold:.2f} deg/s (P90)")
log(f"  Turn events found: {len(peaks)}")

# Select representative events spread across the dataset
n_events = min(8, len(peaks))
if n_events > 0 and len(peaks) >= n_events:
    # Spread events across the time range
    event_indices = np.linspace(0, len(peaks) - 1, n_events, dtype=int)
    selected_peaks = peaks[event_indices]
elif len(peaks) > 0:
    selected_peaks = peaks[:n_events]
else:
    selected_peaks = np.array([], dtype=int)

window_s = 5.0  # seconds around each event
window_n = int(window_s / dt_median)

log(f"  Selected {len(selected_peaks)} representative events")
log()

for ev_idx, pk in enumerate(selected_peaks):
    t_event = t[pk]
    veh_yr = veh_yaw_rate[pk]
    log(f"  Event {ev_idx+1}: t={t_event:.2f} s, "
        f"veh yaw rate={veh_yr:+.2f} deg/s, "
        f"veh vel={veh_velocity[pk]:.1f} km/h")

    # Show all three gyro values at this point
    for gi, gname in enumerate(["Yaw", "Pitch", "Roll"]):
        gv_deg = np.degrees(gyro[pk, gi])
        log(f"    Phone gyro {gname}: {gv_deg:+.4f} deg/s")


# ============================================================
# PLOT: TURN EVENT GYRO COMPARISON
# ============================================================

log()
log("  Generating turn-event comparison plots...")

if len(selected_peaks) > 0:
    n_ev = len(selected_peaks)
    fig, axes = plt.subplots(n_ev, 1, figsize=(14, 3.5 * n_ev), sharex=False)
    if n_ev == 1:
        axes = [axes]

    for ev_idx, pk in enumerate(selected_peaks):
        ax = axes[ev_idx]
        lo = max(0, pk - window_n)
        hi = min(len(t), pk + window_n)
        t_win = t[lo:hi] - t[pk]

        veh_yr_win = veh_yaw_rate[lo:hi]
        ax.plot(t_win, veh_yr_win, "k-", linewidth=2.0,
                label="Vehicle Yaw Rate (deg/s)")
        ax.plot(t_win, np.degrees(gyro[lo:hi, 0]), "r-",
                linewidth=1.0, alpha=0.8, label="Gyro Yaw (deg/s)")
        ax.plot(t_win, np.degrees(gyro[lo:hi, 1]), "g-",
                linewidth=1.0, alpha=0.8, label="Gyro Pitch (deg/s)")
        ax.plot(t_win, np.degrees(gyro[lo:hi, 2]), "b-",
                linewidth=1.0, alpha=0.8, label="Gyro Roll (deg/s)")

        ax.set_ylabel("deg/s")
        ax.set_title(
            f"Event {ev_idx+1}: t={t[pk]:.1f}s  "
            f"veh_yaw={veh_yaw_rate[pk]:+.1f} deg/s"
        )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        if ev_idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time from event center (s)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "turn_event_gyro_comparison.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: {OUT_DIR / 'turn_event_gyro_comparison.png'}")
else:
    log("  No turn events found — skipping turn-event plot.")


# ============================================================
# PLOT: GYRO AXIS CROSS-CORRELATION
# ============================================================

log("  Generating gyro cross-correlation plot...")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colors = ["r", "g", "b"]
gyro_short = ["Gyro Yaw", "Gyro Pitch", "Gyro Roll"]

for i, (gcol, color, gshort) in enumerate(
    zip(gyro_axis_cols, colors, gyro_short)
):
    ax = axes[i]
    lags_arr, cc_arr = xcorr_data[gcol]
    ax.plot(lags_arr, cc_arr, color=color, linewidth=1.2)
    ax.set_ylabel("Correlation")
    ax.set_title(f"{gshort} vs Vehicle Yaw Rate")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-MAX_LAG_S, MAX_LAG_S)

    # Mark the peak
    best_idx = np.argmax(np.abs(cc_arr))
    ax.plot(lags_arr[best_idx], cc_arr[best_idx], "ko", markersize=6)
    ax.annotate(
        f"lag={lags_arr[best_idx]:+.2f}s\n|corr|={np.abs(cc_arr[best_idx]):.3f}",
        xy=(lags_arr[best_idx], cc_arr[best_idx]),
        xytext=(lags_arr[best_idx] + 1.5, cc_arr[best_idx]),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

axes[-1].set_xlabel("Lag (s)  [negative = phone lags vehicle]")
plt.tight_layout()
fig.savefig(OUT_DIR / "gyro_cross_correlation.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'gyro_cross_correlation.png'}")


# ============================================================
# PLOT: GYRO AXIS COMPARISON (TIME SERIES SNIPPET)
# ============================================================

log("  Generating gyro axis time-series comparison...")

# Show a representative segment
mid = len(t) // 2
snippet_n = min(int(30.0 / dt_median), len(t))  # 30 seconds
lo = mid - snippet_n // 2
hi = mid + snippet_n // 2
t_snip = t[lo:hi] - t[lo]

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(t_snip, veh_yaw_rate[lo:hi], "k-", linewidth=1.5)
axes[0].set_ylabel("deg/s")
axes[0].set_title("Vehicle Yaw Rate")
axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")

for i, (gname, color) in enumerate(
    zip(["Gyro Yaw", "Gyro Pitch", "Gyro Roll"], ["r", "g", "b"])
):
    axes[i + 1].plot(t_snip, np.degrees(gyro[lo:hi, i]), color=color, linewidth=1.0)
    axes[i + 1].set_ylabel("deg/s")
    axes[i + 1].set_title(gname)
    axes[i + 1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(OUT_DIR / "gyro_axis_comparison.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'gyro_axis_comparison.png'}")


# ============================================================
# PLOT: ACCEL AXIS COMPARISON
# ============================================================

log("  Generating accel axis time-series comparison...")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(t_snip, veh_long_accel[lo:hi] * 9.80665, "k-",
             linewidth=1.5, label="Vehicle Long Accel")
axes[0].plot(t_snip, veh_lat_accel[lo:hi] * 9.80665, "m--",
             linewidth=1.0, label="Vehicle Lat Accel")
axes[0].set_ylabel("m/s²")
axes[0].set_title("Vehicle Accelerations")
axes[0].legend(fontsize=8)

for i, (aname, color) in enumerate(
    zip(["Accel X", "Accel Y", "Accel Z"], ["r", "g", "b"])
):
    axes[i + 1].plot(t_snip, accel[lo:hi, i], color=color, linewidth=1.0)
    axes[i + 1].set_ylabel("m/s²")
    axes[i + 1].set_title(aname)
    axes[i + 1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(OUT_DIR / "accel_axis_comparison.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'accel_axis_comparison.png'}")


# ============================================================
# SECTION 10: FRAME-MAPPING HYPOTHESIS
# ============================================================

log()
log("=" * 80)
log("10. FRAME-MAPPING HYPOTHESIS")
log("=" * 80)

# Gather evidence from Section 8

log()
log("  === Gyro vs Vehicle Yaw Rate Evidence ===")
log()
for gcol, res in gyro_results.items():
    log(f"  {gcol}:")
    log(f"    Best (no flip): r={res['best_corr']:+.4f} at lag {res['best_lag_s']:+.3f} s")
    log(f"    Best (flipped): r={res['flip_best_corr']:+.4f} at lag {res['flip_best_lag_s']:+.3f} s")
    log(f"    Flip helps: {res['sign_flipped']}")
    log()

log("  === Candidate Linear Accel vs Vehicle Longitudinal ===")
log()
for i, alabel in enumerate(["Accel X", "Accel Y", "Accel Z"]):
    res = accel_results.get(("longitudinal", alabel))
    if res:
        log(f"    {alabel}: best r={res['best_corr']:+.4f} @ {res['best_lag_s']:+.3f} s"
            f"  flip r={res['flip_best_corr']:+.4f} @ {res['flip_best_lag_s']:+.3f} s")

log()
log("  === Candidate Linear Accel vs Vehicle Lateral ===")
log()
for i, alabel in enumerate(["Accel X", "Accel Y", "Accel Z"]):
    res = accel_results.get(("lateral", alabel))
    if res:
        log(f"    {alabel}: best r={res['best_corr']:+.4f} @ {res['best_lag_s']:+.3f} s"
            f"  flip r={res['flip_best_corr']:+.4f} @ {res['flip_best_lag_s']:+.3f} s")

log()
log("  === Gravity Vector at Stationary Periods ===")
log()
stationary_mask = veh_velocity < 1.0
n_stat = stationary_mask.sum()
log(f"  Stationary samples (veh speed < 1 km/h): {n_stat:,}")

if n_stat > 100:
    grav_stat = gravity[stationary_mask]
    log(f"  Gravity mean at rest:")
    log(f"    X: {np.mean(grav_stat[:, 0]):.4f} m/s²")
    log(f"    Y: {np.mean(grav_stat[:, 1]):.4f} m/s²")
    log(f"    Z: {np.mean(grav_stat[:, 2]):.4f} m/s²")
    log(f"    |g|: {np.mean(np.linalg.norm(grav_stat, axis=1)):.4f} m/s²")
    log()
    log(f"  Interpretation: If phone is flat on dash,")
    log(f"  the axis with |mean gravity| closest to 9.81")
    log(f"  is the phone's vertical (Z-up or Z-down).")
    log(f"  The two near-zero gravity axes are the")
    log(f"  horizontal plane (phone X-Y).")

log()
log("  === Orientation Ranges ===")
log()
for i, name in enumerate(["Yaw", "Pitch", "Roll"]):
    v = orientation[:, i]
    log(f"    {name}: range = {np.ptp(v):.1f} deg, "
        f"std = {np.std(v):.1f} deg")


# ============================================================
# SECTION 10b: PROPOSED MAPPING
# ============================================================

log()
log("  --- PROPOSED FRAME MAPPING (provisional) ---")
log()

# Determine which gyro axis best matches vehicle yaw rate
best_gyro_match = None
best_abs_corr = 0
for gcol, res in gyro_results.items():
    c = max(res["best_corr"], res["flip_best_corr"])
    if abs(c) > best_abs_corr:
        best_abs_corr = abs(c)
        best_gyro_match = gcol

log(f"  Best gyro match to vehicle yaw rate:")
log(f"    {best_gyro_match}  (|r| = {best_abs_corr:.4f})")
log()

# Determine which accel axis best matches longitudinal
best_long_match = None
best_long_corr = 0
for i, alabel in enumerate(["Accel X", "Accel Y", "Accel Z"]):
    res = accel_results.get(("longitudinal", alabel))
    if res:
        c = max(res["best_corr"], res["flip_best_corr"])
        if abs(c) > abs(best_long_corr):
            best_long_corr = c
            best_long_match = alabel

best_lat_match = None
best_lat_corr = 0
for i, alabel in enumerate(["Accel X", "Accel Y", "Accel Z"]):
    res = accel_results.get(("lateral", alabel))
    if res:
        c = max(res["best_corr"], res["flip_best_corr"])
        if abs(c) > abs(best_lat_corr):
            best_lat_corr = c
            best_lat_match = alabel

log(f"  Best accel match to vehicle longitudinal accel:")
log(f"    {best_long_match}  (r = {best_long_corr:+.4f})")
log()
log(f"  Best accel match to vehicle lateral accel:")
log(f"    {best_lat_match}  (r = {best_lat_corr:+.4f})")

log()
log("  NOTE: The phone is mounted in an unknown orientation.")
log("  The axis labels (X/Y/Z) refer to the PHONE frame,")
log("  not the vehicle frame. The mapping must be inferred")
log("  from correlations. Correlation alone cannot distinguish")
log("  between e.g. phone-X→vehicle-longitudinal vs")
log("  phone-X→vehicle-lateral if both correlations are")
log("  similar. We need gravity + dynamic evidence combined.")


# ============================================================
# SECTION 11: LATENCY INVESTIGATION
# ============================================================

log()
log("=" * 80)
log("11. LATENCY INVESTIGATION")
log("=" * 80)

log()
log("  Known timestamp synchronization error:")
log(f"    Median: ~41 ms")
log(f"    Mean:   ~25.6 ms")
log(f"    P95:    ~42 ms")
log(f"    Max:    ~50 ms")
log()
log("  The cross-correlation lag reflects the COMBINED effect of:")
log("    1. Timestamp synchronization error (~41 ms)")
log("    2. Sensor processing latency (unknown)")
log("    3. Physical sensor response time (typically < 1 ms)")
log("    4. Android sensor fusion latency (unknown)")
log()

log("  --- Best gyro candidate lags ---")
for gcol, res in gyro_results.items():
    lag = res["best_lag_s"]
    flip_lag = res["flip_best_lag_s"]
    best_lag_use = flip_lag if res["sign_flipped"] else lag
    est_sensor_latency = abs(best_lag_use) * 1000 - 41  # subtract sync error
    log(f"  {gcol}:")
    log(f"    Total lag:    {best_lag_use*1000:+.1f} ms")
    log(f"    Est. sensor latency (total - sync): "
        f"{abs(best_lag_use)*1000 - 41:+.1f} ms (if total > sync)")
    log()

log("  --- Best accel candidate lags ---")
for (ref_type, alabel), res in accel_results.items():
    lag = res["best_lag_s"]
    flip_lag = res["flip_best_lag_s"]
    best_lag_use = flip_lag if res["sign_flipped"] else lag
    log(f"  {alabel} vs vehicle {ref_type}: "
        f"total lag {best_lag_use*1000:+.1f} ms")

log()
log("  INTERPRETATION:")
log("  If |best_lag| is much larger than ~41 ms, the excess")
log("  likely represents genuine sensor processing latency.")
log("  Android's SENSOR_DELAY_GAME typically adds 5-20 ms.")
log("  Gravity/linear-accel fusion may add more.")
log("  If the lag is < 50 ms, it is dominated by sync error.")


# ============================================================
# SECTION 12: SAVE ALL TEXT OUTPUTS
# ============================================================

log()
log("=" * 80)
log("12. SAVING OUTPUTS")
log("=" * 80)

save_text("imu_statistics.txt", _text_lines)

# Save frame mapping summary separately
fm_lines = []
fm_lines.append("FRAME MAPPING SUMMARY")
fm_lines.append("=" * 60)
fm_lines.append("")
fm_lines.append(f"Best gyro ↔ vehicle yaw rate: {best_gyro_match}")
fm_lines.append(f"  |correlation| = {best_abs_corr:.4f}")
fm_lines.append("")
fm_lines.append(f"Best accel ↔ vehicle longitudinal: {best_long_match}")
fm_lines.append(f"  correlation = {best_long_corr:+.4f}")
fm_lines.append("")
fm_lines.append(f"Best accel ↔ vehicle lateral: {best_lat_match}")
fm_lines.append(f"  correlation = {best_lat_corr:+.4f}")
fm_lines.append("")

if n_stat > 100:
    fm_lines.append("Gravity at stationary periods:")
    for i, ax in enumerate(["X", "Y", "Z"]):
        fm_lines.append(f"  {ax}: {np.mean(grav_stat[:, i]):.4f} m/s²")
    fm_lines.append("")

fm_lines.append("Latency estimates:")
for gcol, res in gyro_results.items():
    lag = res["best_lag_s"]
    flip_lag = res["flip_best_lag_s"]
    best_lag_use = flip_lag if res["sign_flipped"] else lag
    fm_lines.append(f"  {gcol}: {best_lag_use*1000:+.1f} ms total")

save_text("frame_mapping_summary.txt", fm_lines)


# ============================================================
# FINAL SUMMARY
# ============================================================

log()
log("=" * 80)
log("ANALYSIS COMPLETE")
log("=" * 80)
log()
log("Outputs saved to:")
log(f"  {OUT_DIR / 'imu_statistics.txt'}")
log(f"  {OUT_DIR / 'frame_mapping_summary.txt'}")
log(f"  {OUT_DIR / 'gyro_axis_comparison.png'}")
log(f"  {OUT_DIR / 'accel_axis_comparison.png'}")
log(f"  {OUT_DIR / 'turn_event_gyro_comparison.png'}")
log(f"  {OUT_DIR / 'gyro_cross_correlation.png'}")
log()
log("No filtering, clipping, smoothing, or outlier removal performed.")
log("The synchronized dataset was NOT modified.")
log("All analysis is diagnostic only.")
