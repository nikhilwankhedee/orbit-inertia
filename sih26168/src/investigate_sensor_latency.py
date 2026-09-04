#!/usr/bin/env python3
"""
investigate_sensor_latency.py — Targeted temporal-alignment experiment.

Determines whether the apparent ~1.8–2.2 s lag is a common systematic
timing offset across independent sensor modalities, or merely
signal-specific cross-correlation behaviour.

CRITICAL: Does NOT modify S4_synced.csv. Does NOT apply any correction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "processed" / "S4_synced.csv"
OUT_DIR = ROOT / "sih26168" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT LOGGING
# ============================================================

_text = []


def log(msg=""):
    print(msg)
    _text.append(msg)


def save_text(name, lines):
    p = OUT_DIR / name
    p.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Saved: {p}")


# ============================================================
# COLUMN DISCOVERY
# ============================================================

log("Loading dataset...")
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
log(f"Rows: {len(df):,}")


def find_col(prefix):
    matches = [c for c in df.columns if c.upper().startswith(prefix.upper())]
    if not matches:
        raise ValueError(f"No column starting with '{prefix}'")
    return matches[0]


# Phone columns
C_ACCEL_X = find_col("ACCELEROMETER X")
C_ACCEL_Y = find_col("ACCELEROMETER Y")
C_ACCEL_Z = find_col("ACCELEROMETER Z")
C_GYRO_PITCH = find_col("GYROSCOPE Pitch")
C_GYRO_YAW = find_col("GYROSCOPE Yaw")
C_GYRO_ROLL = find_col("GYROSCOPE Roll")
C_GPS_SPEED = find_col("GPS SPEED")

# Vehicle columns
C_VEH_VEL = find_col("VELOCITY (")
C_VEH_HEADING = find_col("HEADING (")
C_VEH_LONG = find_col("INDICATED LONGITUDINAL")
C_VEH_LAT = find_col("INDICATED LATERAL")
C_VEH_YAW = find_col("YAW RATE (")

TIME = "SYNC_TIME_S"

ALL_COLS = [
    C_ACCEL_X, C_ACCEL_Y, C_ACCEL_Z,
    C_GYRO_PITCH, C_GYRO_YAW, C_GYRO_ROLL,
    C_GPS_SPEED,
    C_VEH_VEL, C_VEH_HEADING, C_VEH_LONG, C_VEH_LAT, C_VEH_YAW,
    TIME,
]

for c in ALL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

valid = df[ALL_COLS].notna().all(axis=1)
imu = df.loc[valid].copy().reset_index(drop=True)
log(f"Valid rows: {len(imu):,}")

t = imu[TIME].to_numpy()


# ============================================================
# SEGMENT DEFINITION
# ============================================================
# The 312 s gap runs from ~3518 s to ~3831 s.
# Define segments that avoid this region entirely.
# Each segment must be long enough for meaningful correlation
# over a ±5 s lag window at 10 Hz.

gap_start = 3518.0
gap_end = 3831.0
MARGIN = 30.0  # keep 30 s clear of gap edges

segments = {
    "early (0–1500 s)":    (0, 1500),
    "pre-gap (1500–3480 s)": (1500, gap_start - MARGIN),
    "post-gap (3870–5500 s)": (gap_end + MARGIN, 5500),
    "mid (5500–7500 s)":   (5500, 7500),
    "late (7500–9380 s)":  (7500, 9380),
}

log()
log("Segment definitions:")
for name, (t0, t1) in segments.items():
    mask = (t >= t0) & (t < t1)
    n = mask.sum()
    dur = t1 - t0
    log(f"  {name}: t=[{t0}, {t1}), {n:,} samples, {dur:.0f} s")


# ============================================================
# CROSS-CORRELATION ENGINE
# ============================================================

def lag_analysis(ref, sig, max_lag_s, dt):
    """
    For each lag in ±max_lag_s, compute Pearson r.

    Convention:
      best_lag_s > 0  →  sig LAGS ref  (sig arrives later)
      best_lag_s < 0  →  sig LEADS ref (sig arrives earlier)

    Returns dict with zero-lag, best (no flip), best (flipped),
    and peak-width metric.
    """
    max_lag_n = int(round(max_lag_s / dt))
    n = len(ref)

    # --- zero-lag ---
    zl_r = np.corrcoef(ref, sig)[0, 1]

    # --- no-flip sweep ---
    best_r = -np.inf
    best_lag_n = 0
    for lag_n in range(-max_lag_n, max_lag_n + 1):
        if lag_n >= 0:
            a = ref[lag_n:]
            b = sig[:n - lag_n] if lag_n > 0 else sig
        else:
            a = ref[:n + lag_n]
            b = sig[-lag_n:]
        if len(a) < 100:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if not np.isnan(r) and r > best_r:
            best_r = r
            best_lag_n = lag_n

    # --- flipped sweep ---
    sig_neg = -sig
    best_r_flip = -np.inf
    best_lag_n_flip = 0
    for lag_n in range(-max_lag_n, max_lag_n + 1):
        if lag_n >= 0:
            a = ref[lag_n:]
            b = sig_neg[:n - lag_n] if lag_n > 0 else sig_neg
        else:
            a = ref[:n + lag_n]
            b = sig_neg[-lag_n:]
        if len(a) < 100:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if not np.isnan(r) and r > best_r_flip:
            best_r_flip = r
            best_lag_n_flip = lag_n

    # --- peak width: find where |r| > 0.9 * peak_r for no-flip best ---
    # Recompute full curve around the peak to measure width
    full_lags = np.arange(-max_lag_n, max_lag_n + 1)
    full_r = np.empty(len(full_lags))
    for idx, lag_n in enumerate(full_lags):
        if lag_n >= 0:
            a = ref[lag_n:]
            b = sig[:n - lag_n] if lag_n > 0 else sig
        else:
            a = ref[:n + lag_n]
            b = sig[-lag_n:]
        if len(a) < 100:
            full_r[idx] = 0.0
        else:
            full_r[idx] = np.corrcoef(a, b)[0, 1]

    # Peak width at 0.9 × peak
    threshold = 0.9 * best_r if best_r > 0.05 else best_r + 0.01
    above = full_r >= threshold
    if above.any():
        above_idx = np.where(above)[0]
        width_samples = above_idx[-1] - above_idx[0] + 1
        width_s = width_samples * dt
    else:
        width_s = float("inf")

    # Peak width at 0.5 × peak (half-max)
    threshold_half = 0.5 * best_r if best_r > 0.05 else best_r + 0.01
    above_half = full_r >= threshold_half
    if above_half.any():
        above_idx_h = np.where(above_half)[0]
        width_half_s = (above_idx_h[-1] - above_idx_h[0] + 1) * dt
    else:
        width_half_s = float("inf")

    # Flatness metric: ratio of peak to second-highest near peak
    # (within ±1 s of peak)
    nearby_mask = np.abs(full_lags - best_lag_n) <= int(1.0 / dt)
    nearby_r = full_r.copy()
    nearby_r[~nearby_mask] = -np.inf
    sorted_r = np.sort(nearby_r)[::-1]
    if sorted_r[1] > 0:
        peak_prominence = sorted_r[0] / sorted_r[1]
    else:
        peak_prominence = float("inf")

    # Sign-flip judgement
    use_flip = best_r_flip > best_r
    effective_r = best_r_flip if use_flip else best_r
    effective_lag_n = best_lag_n_flip if use_flip else best_lag_n
    effective_lag_s = effective_lag_n * dt

    return {
        "zero_lag_r": zl_r,
        "best_r": best_r,
        "best_lag_s": best_lag_n * dt,
        "flip_r": best_r_flip,
        "flip_lag_s": best_lag_n_flip * dt,
        "sign_flipped": use_flip,
        "effective_r": effective_r,
        "effective_lag_s": effective_lag_s,
        "width_90_s": width_s,
        "width_50_s": width_half_s,
        "peak_prominence": peak_prominence,
        "full_lags": full_lags * dt,
        "full_r": full_r,
    }


# ============================================================
# SIGNAL PAIR DEFINITIONS
# ============================================================

MAX_LAG_S = 5.0

# All pairs: (label, ref_array_getter, signal_array_getter, flip_ok)
# Using lambda to lazily build arrays from imu DataFrame

signal_pairs = {
    "Gyro Pitch vs Veh Yaw Rate": {
        "ref": lambda: np.radians(imu[C_VEH_YAW].to_numpy()),
        "sig": lambda: imu[C_GYRO_PITCH].to_numpy(),
        "unit": "rad/s",
        "ref_name": "Vehicle Yaw Rate",
        "sig_name": "Gyro Pitch",
    },
    "Gyro Yaw vs Veh Yaw Rate": {
        "ref": lambda: np.radians(imu[C_VEH_YAW].to_numpy()),
        "sig": lambda: imu[C_GYRO_YAW].to_numpy(),
        "unit": "rad/s",
        "ref_name": "Vehicle Yaw Rate",
        "sig_name": "Gyro Yaw",
    },
    "Gyro Roll vs Veh Yaw Rate": {
        "ref": lambda: np.radians(imu[C_VEH_YAW].to_numpy()),
        "sig": lambda: imu[C_GYRO_ROLL].to_numpy(),
        "unit": "rad/s",
        "ref_name": "Vehicle Yaw Rate",
        "sig_name": "Gyro Roll",
    },
    "GPS Speed vs Veh Velocity": {
        "ref": lambda: imu[C_VEH_VEL].to_numpy(),
        "sig": lambda: imu[C_GPS_SPEED].to_numpy() * 3.6,
        "unit": "km/h",
        "ref_name": "Vehicle Velocity (km/h)",
        "sig_name": "Phone GPS Speed ×3.6 (km/h)",
    },
    "Accel X vs Veh Long Accel": {
        "ref": lambda: imu[C_VEH_LONG].to_numpy() * 9.80665,
        "sig": lambda: imu[C_ACCEL_X].to_numpy(),
        "unit": "m/s²",
        "ref_name": "Vehicle Long Accel",
        "sig_name": "Accel X",
    },
    "Accel Y vs Veh Long Accel": {
        "ref": lambda: imu[C_VEH_LONG].to_numpy() * 9.80665,
        "sig": lambda: imu[C_ACCEL_Y].to_numpy(),
        "unit": "m/s²",
        "ref_name": "Vehicle Long Accel",
        "sig_name": "Accel Y",
    },
    "Accel X vs Veh Lat Accel": {
        "ref": lambda: imu[C_VEH_LAT].to_numpy() * 9.80665,
        "sig": lambda: imu[C_ACCEL_X].to_numpy(),
        "unit": "m/s²",
        "ref_name": "Vehicle Lat Accel",
        "sig_name": "Accel X",
    },
    "Accel Y vs Veh Lat Accel": {
        "ref": lambda: imu[C_VEH_LAT].to_numpy() * 9.80665,
        "sig": lambda: imu[C_ACCEL_Y].to_numpy(),
        "unit": "m/s²",
        "ref_name": "Vehicle Lat Accel",
        "sig_name": "Accel Y",
    },
}

dt_median = np.median(np.diff(t))


# ============================================================
# RUN ANALYSIS: ALL PAIRS × ALL SEGMENTS
# ============================================================

log()
log("=" * 80)
log("SEGMENT-WISE LAG ANALYSIS")
log("=" * 80)

# Master results table
rows = []

# For plotting, store full xcorr curves for the primary pair
primary_pair = "Gyro Pitch vs Veh Yaw Rate"
primary_curves = {}

for seg_name, (t0, t1) in segments.items():
    seg_mask = (t >= t0) & (t < t1)
    seg_idx = np.where(seg_mask)[0]

    if len(seg_idx) < 200:
        log(f"\n  {seg_name}: SKIPPED (only {len(seg_idx)} samples)")
        continue

    seg_t = t[seg_idx]
    seg_dur = seg_t[-1] - seg_t[0]
    log(f"\n  {'=' * 70}")
    log(f"  SEGMENT: {seg_name}  ({len(seg_idx):,} samples, {seg_dur:.0f} s)")
    log(f"  {'=' * 70}")

    for pair_name, pair_def in signal_pairs.items():
        ref_full = pair_def["ref"]()
        sig_full = pair_def["sig"]()

        ref_seg = ref_full[seg_idx]
        sig_seg = sig_full[seg_idx]

        res = lag_analysis(ref_seg, sig_seg, MAX_LAG_S, dt_median)

        rows.append({
            "pair": pair_name,
            "segment": seg_name,
            "n_samples": len(seg_idx),
            "zero_lag_r": res["zero_lag_r"],
            "best_r": res["best_r"],
            "best_lag_s": res["best_lag_s"],
            "flip_r": res["flip_r"],
            "flip_lag_s": res["flip_lag_s"],
            "sign_flipped": res["sign_flipped"],
            "effective_r": res["effective_r"],
            "effective_lag_s": res["effective_lag_s"],
            "width_90_s": res["width_90_s"],
            "width_50_s": res["width_50_s"],
            "peak_prominence": res["peak_prominence"],
        })

        flip_note = " [FLIP]" if res["sign_flipped"] else ""
        log(
            f"  {pair_name:40s}  "
            f"zl={res['zero_lag_r']:+.4f}  "
            f"best_r={res['best_r']:+.4f} @ {res['best_lag_s']:+.2f}s  "
            f"flip_r={res['flip_r']:+.4f} @ {res['flip_lag_s']:+.2f}s"
            f"{flip_note}"
        )

        # Store full curve for primary pair
        if pair_name == primary_pair:
            primary_curves[seg_name] = (
                res["full_lags"],
                res["full_r"],
                res["best_lag_s"],
                res["best_r"],
            )

results = pd.DataFrame(rows)


# ============================================================
# STATISTICAL SUMMARY BY SIGNAL PAIR
# ============================================================

log()
log("=" * 80)
log("LAG STATISTICS ACROSS SEGMENTS")
log("=" * 80)

summary_rows = []

for pair_name in signal_pairs:
    sub = results[results["pair"] == pair_name].copy()

    if len(sub) == 0:
        continue

    lags = sub["effective_lag_s"].to_numpy()
    rs = sub["effective_r"].to_numpy()

    # Filter out degenerate cases where r is very low
    reliable = sub[sub["effective_r"] > 0.1]
    lags_rel = reliable["effective_lag_s"].to_numpy() if len(reliable) > 0 else lags

    median_lag = np.median(lags_rel)
    iqr_lag = np.percentile(lags_rel, 75) - np.percentile(lags_rel, 25)
    std_lag = np.std(lags_rel)
    range_lag = np.ptp(lags_rel) if len(lags_rel) > 1 else 0.0

    # Peak width statistics
    widths_90 = sub["width_90_s"].to_numpy()
    widths_50 = sub["width_50_s"].to_numpy()
    median_w90 = np.median(widths_90[np.isfinite(widths_90)])
    median_w50 = np.median(widths_50[np.isfinite(widths_50)])

    summary_rows.append({
        "pair": pair_name,
        "n_segments": len(sub),
        "n_reliable": len(reliable),
        "median_lag_s": median_lag,
        "iqr_lag_s": iqr_lag,
        "std_lag_s": std_lag,
        "range_lag_s": range_lag,
        "median_width90_s": median_w90,
        "median_width50_s": median_w50,
        "lags_per_seg": lags_rel.tolist(),
    })

    log(f"\n  {pair_name}")
    log(f"    Segments with r>0.1: {len(reliable)}/{len(sub)}")
    log(f"    Median lag:  {median_lag:+.3f} s")
    log(f"    IQR lag:     {iqr_lag:.3f} s")
    log(f"    Std lag:     {std_lag:.3f} s")
    log(f"    Range lag:   {range_lag:.3f} s")
    log(f"    Per-segment: {[f'{l:+.2f}' for l in lags_rel]}")
    log(f"    Median peak width (90%): {median_w90:.2f} s")
    log(f"    Median peak width (50%): {median_w50:.2f} s")


# ============================================================
# CROSS-MODALITY COMPARISON
# ============================================================

log()
log("=" * 80)
log("CROSS-MODALITY LAG COMPARISON")
log("=" * 80)

# Focus on the key pairs that test whether the lag is common
key_pairs = [
    "Gyro Pitch vs Veh Yaw Rate",
    "GPS Speed vs Veh Velocity",
    "Accel X vs Veh Lat Accel",
    "Accel Y vs Veh Long Accel",
]

log()
log("  Comparing effective best lags across modalities:")
log()

for pair_name in key_pairs:
    sub = results[results["pair"] == pair_name]
    reliable = sub[sub["effective_r"] > 0.1]
    if len(reliable) > 0:
        lags = reliable["effective_lag_s"].to_numpy()
        log(f"  {pair_name:40s}  "
            f"median={np.median(lags):+.3f}  "
            f"std={np.std(lags):.3f}  "
            f"range=[{np.min(lags):+.3f}, {np.max(lags):+.3f}]")
    else:
        log(f"  {pair_name:40s}  (no reliable segments)")


# ============================================================
# PEAK SHARPNESS / BROADNESS ASSESSMENT
# ============================================================

log()
log("=" * 80)
log("PEAK SHARPNESS ASSESSMENT")
log("=" * 80)

for sr in summary_rows:
    pair_name = sr["pair"]
    w90 = sr["median_width90_s"]
    w50 = sr["median_width50_s"]
    median_lag = sr["median_lag_s"]

    if w50 < 0.5:
        sharpness = "SHARP"
    elif w50 < 1.5:
        sharpness = "MODERATE"
    else:
        sharpness = "BROAD"

    log(f"\n  {pair_name}")
    log(f"    Peak width at 90% of max: {w90:.2f} s")
    log(f"    Peak width at 50% of max: {w50:.2f} s")
    log(f"    Assessment: {sharpness}")
    if sharpness == "BROAD":
        log(f"    WARNING: Broad peak — lag estimate has high uncertainty")
        log(f"    The 'best lag' could shift substantially with noise.")


# ============================================================
# SAVE RESULTS TABLE
# ============================================================

log()
log("=" * 80)
log("SAVING RESULTS")
log("=" * 80)

# Save the full per-segment results
results_path = OUT_DIR / "sensor_latency_report.txt"

report_lines = []
report_lines.append("SENSOR LATENCY INVESTIGATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Dataset: S4_synced.csv (NOT MODIFIED)")
report_lines.append(f"Total valid rows: {len(imu):,}")
report_lines.append(f"Duration: {t[-1] - t[0]:.1f} s")
report_lines.append(f"Median dt: {dt_median*1000:.1f} ms ({1/dt_median:.1f} Hz)")
report_lines.append("")
report_lines.append("LAG CONVENTION:")
report_lines.append("  best_lag_s > 0  →  phone signal LAGS vehicle signal")
report_lines.append("  best_lag_s < 0  →  phone signal LEADS vehicle signal")
report_lines.append("")
report_lines.append("SEGMENTS:")
for seg_name, (t0, t1) in segments.items():
    seg_mask = (t >= t0) & (t < t1)
    report_lines.append(f"  {seg_name}: t=[{t0}, {t1}), {seg_mask.sum():,} samples")
report_lines.append("")

# Per-segment table
report_lines.append("-" * 80)
report_lines.append("PER-SEGMENT RESULTS")
report_lines.append("-" * 80)
report_lines.append("")

header = (
    f"{'Signal Pair':42s} | {'Segment':26s} | "
    f"{'zl_r':>7s} | {'best_r':>7s} | {'lag_s':>7s} | "
    f"{'flip_r':>7s} | {'flip_lag':>8s} | {'w90':>5s} | {'w50':>5s}"
)
report_lines.append(header)
report_lines.append("-" * len(header))

for _, row in results.iterrows():
    report_lines.append(
        f"{row['pair']:42s} | {row['segment']:26s} | "
        f"{row['zero_lag_r']:+7.4f} | {row['best_r']:+7.4f} | "
        f"{row['best_lag_s']:+7.3f} | {row['flip_r']:+7.4f} | "
        f"{row['flip_lag_s']:+8.3f} | "
        f"{row['width_90_s']:5.2f} | {row['width_50_s']:5.2f}"
    )

# Summary table
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("LAG STATISTICS ACROSS SEGMENTS")
report_lines.append("-" * 80)
report_lines.append("")

for sr in summary_rows:
    report_lines.append(f"  {sr['pair']}")
    report_lines.append(f"    Reliable segments: {sr['n_reliable']}/{sr['n_segments']}")
    report_lines.append(f"    Median lag:  {sr['median_lag_s']:+.3f} s")
    report_lines.append(f"    IQR lag:     {sr['iqr_lag_s']:.3f} s")
    report_lines.append(f"    Std lag:     {sr['std_lag_s']:.3f} s")
    report_lines.append(f"    Range:       {sr['range_lag_s']:.3f} s")
    report_lines.append(f"    Median peak width (90%): {sr['median_width90_s']:.2f} s")
    report_lines.append(f"    Median peak width (50%): {sr['median_width50_s']:.2f} s")
    report_lines.append(f"    Per-segment lags: "
                        f"{[f'{l:+.2f}' for l in sr['lags_per_seg']]}")
    report_lines.append("")

# Cross-modality comparison
report_lines.append("-" * 80)
report_lines.append("CROSS-MODALITY COMPARISON")
report_lines.append("-" * 80)
report_lines.append("")
for pair_name in key_pairs:
    sub = results[results["pair"] == pair_name]
    reliable = sub[sub["effective_r"] > 0.1]
    if len(reliable) > 0:
        lags = reliable["effective_lag_s"].to_numpy()
        report_lines.append(
            f"  {pair_name:40s}  "
            f"median={np.median(lags):+.3f}  "
            f"std={np.std(lags):.3f}  "
            f"range=[{np.min(lags):+.3f}, {np.max(lags):+.3f}]"
        )

# Interpretation
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("INTERPRETATION")
report_lines.append("-" * 80)
report_lines.append("")

# Collect median lags for key pairs
pair_lag_map = {}
for sr in summary_rows:
    if sr["pair"] in key_pairs:
        pair_lag_map[sr["pair"]] = sr["median_lag_s"]

if pair_lag_map:
    vals = list(pair_lag_map.values())
    overall_median = np.median(vals)
    overall_range = np.ptp(vals) if len(vals) > 1 else 0

    report_lines.append(
        f"  Median lags across key modalities: "
        f"{[f'{k}: {v:+.3f}s' for k, v in pair_lag_map.items()]}"
    )
    report_lines.append(
        f"  Overall median: {overall_median:+.3f} s"
    )
    report_lines.append(
        f"  Cross-modality range: {overall_range:.3f} s"
    )
    report_lines.append("")

    if overall_range < 0.5:
        report_lines.append(
            "  RESULT A: Lags are CONSISTENT across modalities."
        )
        report_lines.append(
            f"  All independent pairs show lag ≈ {overall_median:+.2f} s."
        )
        report_lines.append(
            "  This supports a common systematic timing offset."
        )
        report_lines.append(
            "  A single global correction may be justified."
        )
    elif overall_range < 1.5:
        report_lines.append(
            "  RESULT B: Lags show MODERATE variation across modalities."
        )
        report_lines.append(
            "  A single global correction may be partially effective"
        )
        report_lines.append(
            "  but residual modality-specific offsets will remain."
        )
    else:
        report_lines.append(
            "  RESULT C: Lags are INCONSISTENT across modalities."
        )
        report_lines.append(
            "  Different sensor pairs show materially different lags."
        )
        report_lines.append(
            "  A single global timing correction is NOT justified."
        )
        report_lines.append(
            "  Investigate modality-specific latency."
        )

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

save_text("sensor_latency_report.txt", report_lines)
results.to_csv(OUT_DIR / "sensor_latency_detail.csv", index=False)
log(f"  Saved: {OUT_DIR / 'sensor_latency_detail.csv'}")


# ============================================================
# PLOT 1: Correlation vs lag — Gyro Pitch vs Veh Yaw Rate
# ============================================================

log("  Generating plots...")

fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(segments)))

for i, (seg_name, _) in enumerate(segments.items()):
    if seg_name in primary_curves:
        lags_arr, cc_arr, best_l, best_r = primary_curves[seg_name]
        ax.plot(lags_arr, cc_arr, color=colors[i], linewidth=1.2,
                label=f"{seg_name} (best={best_l:+.2f}s, r={best_r:.3f})")
        ax.plot(best_l, best_r, "o", color=colors[i], markersize=5)

ax.set_xlabel("Lag (s)  [positive = phone lags vehicle]")
ax.set_ylabel("Pearson r")
ax.set_title("Gyro Pitch vs Vehicle Yaw Rate — Correlation vs Lag by Segment")
ax.set_xlim(-MAX_LAG_S, MAX_LAG_S)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(OUT_DIR / "latency_gyro_pitch.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'latency_gyro_pitch.png'}")


# ============================================================
# PLOT 2: Correlation vs lag — GPS Speed vs Vehicle Velocity
# ============================================================

speed_curves = {}

# Recompute speed curves (they weren't stored earlier)
for seg_name, (t0, t1) in segments.items():
    seg_mask = (t >= t0) & (t < t1)
    seg_idx = np.where(seg_mask)[0]
    if len(seg_idx) < 200:
        continue

    ref = imu[C_VEH_VEL].to_numpy()[seg_idx]
    sig = imu[C_GPS_SPEED].to_numpy()[seg_idx] * 3.6

    res = lag_analysis(ref, sig, MAX_LAG_S, dt_median)
    speed_curves[seg_name] = (
        res["full_lags"], res["full_r"],
        res["best_lag_s"], res["best_r"],
    )

fig, ax = plt.subplots(figsize=(12, 5))

for i, (seg_name, _) in enumerate(segments.items()):
    if seg_name in speed_curves:
        lags_arr, cc_arr, best_l, best_r = speed_curves[seg_name]
        ax.plot(lags_arr, cc_arr, color=colors[i], linewidth=1.2,
                label=f"{seg_name} (best={best_l:+.2f}s, r={best_r:.3f})")
        ax.plot(best_l, best_r, "o", color=colors[i], markersize=5)

ax.set_xlabel("Lag (s)  [positive = phone GPS lags vehicle]")
ax.set_ylabel("Pearson r")
ax.set_title("Phone GPS Speed vs Vehicle Velocity — Correlation vs Lag by Segment")
ax.set_xlim(-MAX_LAG_S, MAX_LAG_S)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(OUT_DIR / "latency_speed.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'latency_speed.png'}")


# ============================================================
# PLOT 3: Correlation vs lag — Best acceleration pairs
# ============================================================

accel_pairs_to_plot = [
    "Accel X vs Veh Lat Accel",
    "Accel Y vs Veh Long Accel",
]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax_idx, pair_name in enumerate(accel_pairs_to_plot):
    ax = axes[ax_idx]
    accel_curves = {}

    for seg_name, (t0, t1) in segments.items():
        seg_mask = (t >= t0) & (t < t1)
        seg_idx = np.where(seg_mask)[0]
        if len(seg_idx) < 200:
            continue

        pair_def = signal_pairs[pair_name]
        ref = pair_def["ref"]()[seg_idx]
        sig = pair_def["sig"]()[seg_idx]

        res = lag_analysis(ref, sig, MAX_LAG_S, dt_median)
        lags_arr = res["full_lags"]
        cc_arr = res["full_r"]

        ax.plot(lags_arr, cc_arr, color=colors[list(segments.keys()).index(seg_name)],
                linewidth=1.2,
                label=f"{seg_name} (best={res['best_lag_s']:+.2f}s)")
        ax.plot(res["best_lag_s"], res["best_r"], "o",
                color=colors[list(segments.keys()).index(seg_name)], markersize=5)

    ax.set_xlabel("Lag (s)  [positive = phone lags vehicle]")
    ax.set_ylabel("Pearson r")
    ax.set_title(pair_name)
    ax.set_xlim(-MAX_LAG_S, MAX_LAG_S)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=6, loc="best")

plt.tight_layout()
fig.savefig(OUT_DIR / "latency_acceleration.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'latency_acceleration.png'}")


# ============================================================
# PLOT 4: Segment-wise lag summary bar chart
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: lag values per segment for key pairs
ax = axes[0]
seg_names = list(segments.keys())
x_pos = np.arange(len(seg_names))
bar_width = 0.18

for idx, pair_name in enumerate(key_pairs):
    sub = results[results["pair"] == pair_name]
    lags_for_plot = []
    for sn in seg_names:
        row = sub[sub["segment"] == sn]
        if len(row) > 0 and row["effective_r"].values[0] > 0.1:
            lags_for_plot.append(row["effective_lag_s"].values[0])
        else:
            lags_for_plot.append(0)

    ax.bar(x_pos + idx * bar_width, lags_for_plot, bar_width,
           label=pair_name.split(" vs ")[0], alpha=0.8)

ax.set_ylabel("Effective best lag (s)")
ax.set_title("Best Lag by Segment and Signal Pair")
ax.legend(fontsize=7)
ax.axhline(0, color="gray", linewidth=0.5)

# Bottom: peak width (uncertainty)
ax = axes[1]
for idx, pair_name in enumerate(key_pairs):
    sub = results[results["pair"] == pair_name]
    widths_for_plot = []
    for sn in seg_names:
        row = sub[sub["segment"] == sn]
        if len(row) > 0:
            w = row["width_50_s"].values[0]
            widths_for_plot.append(w if np.isfinite(w) else 0)
        else:
            widths_for_plot.append(0)

    ax.bar(x_pos + idx * bar_width, widths_for_plot, bar_width,
           label=pair_name.split(" vs ")[0], alpha=0.8)

ax.set_ylabel("Peak width at 50% (s)")
ax.set_title("Peak Width (uncertainty proxy) by Segment")
ax.set_xticks(x_pos + bar_width * 1.5)
ax.set_xticklabels([s.replace(" ", "\n") for s in seg_names], fontsize=7)
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUT_DIR / "latency_segment_summary.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'latency_segment_summary.png'}")


# ============================================================
# FINAL SUMMARY
# ============================================================

log()
log("=" * 80)
log("INVESTIGATION COMPLETE")
log("=" * 80)
log()
log("Outputs:")
log(f"  {OUT_DIR / 'sensor_latency_report.txt'}")
log(f"  {OUT_DIR / 'sensor_latency_detail.csv'}")
log(f"  {OUT_DIR / 'latency_gyro_pitch.png'}")
log(f"  {OUT_DIR / 'latency_speed.png'}")
log(f"  {OUT_DIR / 'latency_acceleration.png'}")
log(f"  {OUT_DIR / 'latency_segment_summary.png'}")
log()
log("S4_synced.csv was NOT modified.")
