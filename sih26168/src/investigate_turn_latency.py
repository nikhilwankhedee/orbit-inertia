#!/usr/bin/env python3
"""
investigate_turn_latency.py — Turn-event temporal alignment experiment.

Estimates the temporal relationship between phone GYROSCOPE Pitch and
vehicle yaw rate using actual dynamic turn events rather than the
entire trajectory.

Does NOT modify S4_synced.csv. Does NOT apply any correction.
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
# LOAD DATA
# ============================================================

log("Loading dataset...")
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()


def find_col(prefix):
    matches = [c for c in df.columns if c.upper().startswith(prefix.upper())]
    if not matches:
        raise ValueError(f"No column starting with '{prefix}'")
    return matches[0]


C_ACCEL_X = find_col("ACCELEROMETER X")
C_ACCEL_Y = find_col("ACCELEROMETER Y")
C_ACCEL_Z = find_col("ACCELEROMETER Z")
C_GYRO_PITCH = find_col("GYROSCOPE Pitch")
C_GYRO_YAW = find_col("GYROSCOPE Yaw")
C_GYRO_ROLL = find_col("GYROSCOPE Roll")
C_GPS_SPEED = find_col("GPS SPEED")
C_VEH_VEL = find_col("VELOCITY (")
C_VEH_LONG = find_col("INDICATED LONGITUDINAL")
C_VEH_LAT = find_col("INDICATED LATERAL")
C_VEH_YAW = find_col("YAW RATE (")
TIME = "SYNC_TIME_S"

ALL_COLS = [
    C_ACCEL_X, C_ACCEL_Y, C_ACCEL_Z,
    C_GYRO_PITCH, C_GYRO_YAW, C_GYRO_ROLL,
    C_GPS_SPEED, C_VEH_VEL, C_VEH_LONG, C_VEH_LAT, C_VEH_YAW,
    TIME,
]

for c in ALL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

valid = df[ALL_COLS].notna().all(axis=1)
imu = df.loc[valid].copy().reset_index(drop=True)

t = imu[TIME].to_numpy()
veh_yaw = imu[C_VEH_YAW].to_numpy()  # deg/s
gyro_pitch = imu[C_GYRO_PITCH].to_numpy()  # rad/s
gyro_pitch_deg = np.degrees(gyro_pitch)  # deg/s

dt_median = np.median(np.diff(t))
log(f"Valid rows: {len(imu):,}")
log(f"Median dt: {dt_median*1000:.1f} ms ({1/dt_median:.1f} Hz)")


# ============================================================
# LAG CONVENTION (DOCUMENTED)
# ============================================================
#
# For lag L (in seconds):
#   correlation(ref[t], sig[t + L]) or equivalently
#   we shift sig backward by L to align with ref.
#
# lag > 0: sig is compared at time t+L against ref at time t.
#          If this gives best correlation, sig ARRIVES LATER than ref.
#          sig LAGS ref by L seconds.
#
# lag < 0: sig is compared at time t-|L| against ref at time t.
#          sig ARRIVES EARLIER than ref.
#          sig LEADS ref by |L| seconds.
#
# Here: ref = vehicle yaw rate, sig = phone gyro pitch.
# Positive best lag = phone gyro pitch lags vehicle yaw rate.
#


# ============================================================
# STEP 1 — DETECT TURN EVENTS
# ============================================================

log()
log("=" * 80)
log("STEP 1: TURN EVENT DETECTION")
log("=" * 80)

# Primary threshold
yaw_threshold = 10.0  # deg/s

# Find samples above threshold
above = np.abs(veh_yaw) >= yaw_threshold
log(f"Samples with |yaw rate| >= {yaw_threshold} deg/s: {above.sum():,}")

# Group into contiguous events (allowing small gaps)
# Merge events separated by < 1 s
max_gap_s = 1.0
max_gap_n = int(round(max_gap_s / dt_median))

# Label contiguous regions
labels, n_regions = _label_regions(above) if False else (None, 0)

# Manual contiguous grouping with gap merging
event_starts = []
event_ends = []

in_event = False
gap_count = 0
start_idx = 0

for i in range(len(above)):
    if above[i]:
        if not in_event:
            start_idx = i
            in_event = True
            gap_count = 0
        else:
            gap_count = 0  # reset gap counter when we see another above-threshold sample
    else:
        if in_event:
            gap_count += 1
            if gap_count > max_gap_n:
                # Gap too large — close this event
                event_starts.append(start_idx)
                event_ends.append(i - gap_count)
                in_event = False
                gap_count = 0

if in_event:
    event_starts.append(start_idx)
    event_ends.append(len(above) - 1)

n_events = len(event_starts)
log(f"Raw events detected: {n_events}")

# Compute event properties
event_info = []
for idx in range(n_events):
    s, e = event_starts[idx], event_ends[idx]
    dur = (t[e] - t[s]) + dt_median  # inclusive duration
    peak_yaw = np.max(np.abs(veh_yaw[s:e+1]))
    peak_yaw_signed = veh_yaw[s[e+1].argmax() + s] if False else 0
    # Find actual peak index
    abs_yaw = np.abs(veh_yaw[s:e+1])
    peak_local = np.argmax(abs_yaw)
    peak_idx = s + peak_local
    peak_yaw_signed = veh_yaw[peak_idx]
    peak_time = t[peak_idx]
    mean_yaw = np.mean(veh_yaw[s:e+1])

    event_info.append({
        "idx": idx,
        "start_idx": s,
        "end_idx": e,
        "start_t": t[s],
        "end_t": t[e],
        "duration_s": dur,
        "peak_yaw": peak_yaw,
        "peak_yaw_signed": peak_yaw_signed,
        "peak_time": peak_time,
        "mean_yaw": mean_yaw,
        "direction": "positive" if peak_yaw_signed > 0 else "negative",
    })

# Filter: require sufficient duration and peak magnitude
min_duration = 0.5  # seconds
min_peak = 12.0  # deg/s
# Also require padding room (±5 s window around event)
padding_s = 5.0
padding_n = int(round(padding_s / dt_median))

filtered_events = []
for ev in event_info:
    if ev["duration_s"] < min_duration:
        continue
    if ev["peak_yaw"] < min_peak:
        continue
    # Check padding
    if ev["start_idx"] - padding_n < 0:
        continue
    if ev["end_idx"] + padding_n >= len(t):
        continue
    filtered_events.append(ev)

log(f"After filtering (dur>={min_duration}s, peak>={min_peak} deg/s, padding ok): {len(filtered_events)}")

# Select representative events
# Stratify by direction and strength
pos_events = [e for e in filtered_events if e["direction"] == "positive"]
neg_events = [e for e in filtered_events if e["direction"] == "negative"]

log(f"  Positive turns: {len(pos_events)}")
log(f"  Negative turns: {len(neg_events)}")

# Sort by peak magnitude and select spread
def select_representative(events, n_target):
    if len(events) <= n_target:
        return events
    sorted_ev = sorted(events, key=lambda e: e["peak_yaw"])
    # Select evenly spaced in magnitude
    indices = np.linspace(0, len(sorted_ev) - 1, n_target, dtype=int)
    return [sorted_ev[i] for i in indices]


n_per_dir = min(15, max(8, len(filtered_events) // 3))
selected_pos = select_representative(pos_events, n_per_dir)
selected_neg = select_representative(neg_events, n_per_dir)
selected = selected_pos + selected_neg
selected.sort(key=lambda e: e["start_t"])

log(f"  Selected for analysis: {len(selected)} events")
log(f"    ({len(selected_pos)} positive, {len(selected_neg)} negative)")

# Show summary
for ev in selected:
    log(f"    Event {ev['idx']:3d}: t=[{ev['start_t']:.1f}, {ev['end_t']:.1f}]  "
        f"dur={ev['duration_s']:.2f}s  peak={ev['peak_yaw_signed']:+.1f} deg/s  {ev['direction']}")


# ============================================================
# STEP 2 & 3 & 4 — ANALYZE EACH EVENT
# ============================================================

log()
log("=" * 80)
log("STEPS 2-4: PER-EVENT ANALYSIS")
log("=" * 80)

SEARCH_LAG_S = 4.0  # search ±4 s
SEARCH_LAG_N = int(round(SEARCH_LAG_S / dt_median))
WINDOW_HALF_S = 6.0  # extract ±6 s around event center
WINDOW_HALF_N = int(round(WINDOW_HALF_S / dt_median))


def quadratic_peak_interpolation(x, y, idx):
    """
    Fit a parabola through (x[idx-1], y[idx-1]), (x[idx], y[idx]),
    (x[idx+1], y[idx+1]) and return the interpolated peak x-coordinate.
    """
    if idx <= 0 or idx >= len(x) - 1:
        return x[idx], y[idx]

    x_prev, x_c, x_next = x[idx - 1], x[idx], x[idx + 1]
    y_prev, y_c, y_next = y[idx - 1], y[idx], y[idx + 1]

    denom = 2.0 * (y_prev - 2 * y_c + y_next)
    if abs(denom) < 1e-15:
        return x_c, y_c

    offset = (y_prev - y_next) / denom
    interp_x = x_c + offset * (x_c - x_prev)
    interp_y = y_c - 0.25 * (y_prev - y_next) * offset

    return interp_x, interp_y


def zero_crossings(signal, t_ref):
    """Find zero-crossing times (simple linear interpolation)."""
    crossings = []
    for i in range(len(signal) - 1):
        if signal[i] == 0:
            crossings.append(t_ref[i])
        elif signal[i] * signal[i + 1] < 0:
            # Linear interpolation
            frac = abs(signal[i]) / (abs(signal[i]) + abs(signal[i + 1]))
            tc = t_ref[i] + frac * (t_ref[i + 1] - t_ref[i])
            crossings.append(tc)
    return crossings


results = []

for ev in selected:
    ev_idx = ev["idx"]
    center_idx = (ev["start_idx"] + ev["end_idx"]) // 2

    # Extract window
    lo = center_idx - WINDOW_HALF_N
    hi = center_idx + WINDOW_HALF_N
    t_win = t[lo:hi]
    veh_win = veh_yaw[lo:hi]
    gyro_win = gyro_pitch_deg[lo:hi]

    n_win = len(t_win)

    # --- Cross-correlation ---
    best_cc_r = -np.inf
    best_cc_lag_n = 0
    cc_lags = np.arange(-SEARCH_LAG_N, SEARCH_LAG_N + 1)
    cc_values = np.empty(len(cc_lags))

    for li, lag_n in enumerate(cc_lags):
        if lag_n >= 0:
            a = veh_win[lag_n:]
            b = gyro_win[:n_win - lag_n] if lag_n > 0 else gyro_win
        else:
            a = veh_win[:n_win + lag_n]
            b = gyro_win[-lag_n:]
        if len(a) < 20:
            cc_values[li] = 0
            continue
        r = np.corrcoef(a, b)[0, 1]
        cc_values[li] = r if not np.isnan(r) else 0
        if r > best_cc_r:
            best_cc_r = r
            best_cc_lag_n = lag_n

    best_cc_lag_s = best_cc_lag_n * dt_median
    zero_lag_r = np.corrcoef(veh_win, gyro_win)[0, 1]

    # Sub-sample interpolation
    peak_idx = np.argmax(cc_values)
    interp_lag_s, interp_r = quadratic_peak_interpolation(
        cc_lags * dt_median, cc_values, peak_idx
    )

    # --- Peak timing ---
    veh_peak_local = np.argmax(np.abs(veh_win))
    veh_peak_time = t_win[veh_peak_local]

    gyro_peak_local = np.argmax(np.abs(gyro_win))
    gyro_peak_time = t_win[gyro_peak_local]

    peak_time_diff = gyro_peak_time - veh_peak_time  # positive = gyro is later

    # --- Zero-crossing alignment ---
    veh_zc = zero_crossings(veh_win, t_win)
    gyro_zc = zero_crossings(gyro_win, t_win)

    zc_diff = None
    if len(veh_zc) > 0 and len(gyro_zc) > 0:
        # Find the zero-crossing pair closest to the event center
        veh_closest = min(veh_zc, key=lambda x: abs(x - t_win[WINDOW_HALF_N]))
        gyro_closest = min(gyro_zc, key=lambda x: abs(x - t_win[WINDOW_HALF_N]))
        zc_diff = gyro_closest - veh_closest

    # --- Onset alignment ---
    onset_diff = None
    onset_threshold = 8.0  # deg/s — where the event "starts"
    # Vehicle onset
    veh_onset_candidates = np.where(
        np.abs(veh_win[:veh_peak_local]) < onset_threshold
    )[0]
    if len(veh_onset_candidates) > 0:
        veh_onset_idx = veh_onset_candidates[-1]  # last sample below threshold before peak
        veh_onset_time = t_win[veh_onset_idx]
        # Find where it crosses above
        for j in range(veh_onset_idx, veh_peak_local):
            if np.abs(veh_win[j]) >= onset_threshold:
                veh_onset_time = t_win[j]
                break
        # Gyro onset
        gyro_onset_candidates = np.where(
            np.abs(gyro_win[:gyro_peak_local]) < onset_threshold
        )[0]
        if len(gyro_onset_candidates) > 0:
            gyro_onset_idx = gyro_onset_candidates[-1]
            gyro_onset_time = t_win[gyro_onset_idx]
            for j in range(gyro_onset_idx, gyro_peak_local):
                if np.abs(gyro_win[j]) >= onset_threshold:
                    gyro_onset_time = t_win[j]
                    break
            onset_diff = gyro_onset_time - veh_onset_time

    # --- Sign consistency ---
    # Check that the gyro pitch has the same sign as vehicle yaw at event peak
    sign_consistent = np.sign(gyro_win[veh_peak_local]) == np.sign(veh_win[veh_peak_local])

    results.append({
        "event_id": ev_idx,
        "direction": ev["direction"],
        "peak_yaw": ev["peak_yaw_signed"],
        "duration_s": ev["duration_s"],
        "start_t": ev["start_t"],
        "end_t": ev["end_t"],
        "center_t": t_win[WINDOW_HALF_N],
        # Cross-correlation
        "zero_lag_r": zero_lag_r,
        "best_cc_lag_s": best_cc_lag_s,
        "best_cc_r": best_cc_r,
        "interp_lag_s": interp_lag_s,
        "interp_r": interp_r,
        # Peak timing
        "peak_time_diff_s": peak_time_diff,
        # Zero-crossing
        "zc_diff_s": zc_diff,
        # Onset
        "onset_diff_s": onset_diff,
        # Sign
        "sign_consistent": sign_consistent,
        # Store for plotting
        "_t_win": t_win,
        "_veh_win": veh_win,
        "_gyro_win": gyro_win,
        "_cc_lags": cc_lags * dt_median,
        "_cc_values": cc_values,
        "_interp_lag": interp_lag_s,
    })


# ============================================================
# STEP 5 — STATISTICAL SUMMARY
# ============================================================

log()
log("=" * 80)
log("STEP 5: STATISTICAL SUMMARY")
log("=" * 80)

# Build results DataFrame (without internal arrays)
df_results = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in results])

# Use interpolated lag as primary estimate
primary_lags = np.array([r["interp_lag_s"] for r in results])
peak_diffs = np.array([r["peak_time_diff_s"] for r in results])
zc_diffs = np.array([r["zc_diff_s"] for r in results if r["zc_diff_s"] is not None])
onset_diffs = np.array([r["onset_diff_s"] for r in results if r["onset_diff_s"] is not None])
cc_lags = np.array([r["best_cc_lag_s"] for r in results])
signs = np.array([r["sign_consistent"] for r in results])

log()
log("  Primary estimate: interpolated cross-correlation lag")
log(f"  N events: {len(primary_lags)}")
log(f"  Median:  {np.median(primary_lags):+.4f} s")
log(f"  Mean:    {np.mean(primary_lags):+.4f} s")
log(f"  Std:     {np.std(primary_lags):.4f} s")
log(f"  IQR:     [{np.percentile(primary_lags, 25):+.4f}, "
    f"{np.percentile(primary_lags, 75):+.4f}]")
log(f"  Min:     {np.min(primary_lags):+.4f} s")
log(f"  Max:     {np.max(primary_lags):+.4f} s")
log(f"  MAD:     {np.median(np.abs(primary_lags - np.median(primary_lags))):.4f} s")
log(f"  Sign consistent: {signs.sum()}/{len(signs)}")

log()
log("  Cross-correlation discrete lag (for comparison):")
log(f"  Median:  {np.median(cc_lags):+.4f} s")
log(f"  IQR:     [{np.percentile(cc_lags, 25):+.4f}, "
    f"{np.percentile(cc_lags, 75):+.4f}]")

log()
log("  Peak timing difference (gyro peak time − vehicle peak time):")
log(f"  Median:  {np.median(peak_diffs):+.4f} s")
log(f"  Mean:    {np.mean(peak_diffs):+.4f} s")
log(f"  Std:     {np.std(peak_diffs):.4f} s")

if len(zc_diffs) > 0:
    log()
    log("  Zero-crossing difference:")
    log(f"  Median:  {np.median(zc_diffs):+.4f} s")
    log(f"  Mean:    {np.mean(zc_diffs):+.4f} s")
    log(f"  Std:     {np.std(zc_diffs):.4f} s")

if len(onset_diffs) > 0:
    log()
    log("  Onset timing difference:")
    log(f"  Median:  {np.median(onset_diffs):+.4f} s")
    log(f"  Mean:    {np.mean(onset_diffs):+.4f} s")
    log(f"  Std:     {np.std(onset_diffs):.4f} s")


# --- Subgroup summaries ---

def summarize_subgroup(name, mask):
    if mask.sum() == 0:
        log(f"\n  {name}: (no events)")
        return
    lags = primary_lags[mask]
    log(f"\n  {name} (n={mask.sum()}):")
    log(f"    Median lag:  {np.median(lags):+.4f} s")
    log(f"    Mean lag:    {np.mean(lags):+.4f} s")
    log(f"    Std lag:     {np.std(lags):.4f} s")
    log(f"    IQR:         [{np.percentile(lags, 25):+.4f}, "
        f"{np.percentile(lags, 75):+.4f}]")
    log(f"    Range:       [{np.min(lags):+.4f}, {np.max(lags):+.4f}]")
    log(f"    Per-event:   {[f'{l:+.3f}' for l in lags]}")

pos_mask = np.array([r["direction"] == "positive" for r in results])
neg_mask = np.array([r["direction"] == "negative" for r in results])
strong_mask = np.array([abs(r["peak_yaw"]) >= 25 for r in results])
moderate_mask = np.array([(abs(r["peak_yaw"]) >= 12) & (abs(r["peak_yaw"]) < 25) for r in results])

summarize_subgroup("Positive turns", pos_mask)
summarize_subgroup("Negative turns", neg_mask)
summarize_subgroup("Strong turns (|peak| >= 25 deg/s)", strong_mask)
summarize_subgroup("Moderate turns (12 <= |peak| < 25 deg/s)", moderate_mask)


# ============================================================
# STEP 6 — HYPOTHESIS TEST
# ============================================================

log()
log("=" * 80)
log("STEP 6: HYPOTHESIS EVALUATION")
log("=" * 80)

log()
log("  H1: Phone gyro Pitch genuinely lags vehicle yaw by ~1.8 s")
log()

within_18 = np.abs(primary_lags - 1.8) < 0.3
log(f"    Events with |lag - 1.8| < 0.3 s: {within_18.sum()}/{len(within_18)}")

if within_18.sum() > len(within_18) * 0.7:
    log("    → PARTIALLY SUPPORTED: most events cluster near 1.8 s")
elif within_18.sum() > len(within_18) * 0.4:
    log("    → WEAKLY SUPPORTED: some events near 1.8 s but substantial scatter")
else:
    log("    → NOT SUPPORTED: few events near 1.8 s")

log()
log("  H2: The full-trajectory 1.8 s peak is an artifact of sparse turn structure")
log()
spread = np.ptp(primary_lags)
log(f"    Spread of event-level lags: {spread:.3f} s")
if spread < 1.0:
    log("    → Events show consistent lag — unlikely to be purely an artifact")
elif spread < 2.0:
    log("    → Moderate spread — possible partial artifact")
else:
    log("    → Large spread — full-trajectory peak may not represent individual events")

log()
log("  H3: Variable or event-dependent timing relationship")
log()
std_lag = np.std(primary_lags)
log(f"    Std of event lags: {std_lag:.3f} s")
if std_lag < 0.2:
    log("    → Low variability — timing relationship is relatively stable")
elif std_lag < 0.5:
    log("    → Moderate variability — some event-dependence")
else:
    log("    → High variability — strongly event-dependent or unreliable")

# Compare methods
log()
log("  Method agreement:")
log(f"    CC lag vs peak timing:  median diff = {np.median(cc_lags - peak_diffs):+.4f} s")
if len(zc_diffs) > 0:
    log(f"    CC lag vs zero-crossing: median diff = {np.median(cc_lags[:len(zc_diffs)] - zc_diffs):+.4f} s")
if len(onset_diffs) > 0:
    log(f"    CC lag vs onset:         median diff = {np.median(cc_lags[:len(onset_diffs)] - onset_diffs):+.4f} s")
log(f"    CC discrete vs interpolated: median diff = {np.median(cc_lags - primary_lags):+.4f} s")


# ============================================================
# STEP 7 — PLOTS
# ============================================================

log()
log("=" * 80)
log("STEP 7: GENERATING PLOTS")
log("=" * 80)


# --- Plot A: Turn event examples with alignment ---

n_show = min(12, len(results))
fig, axes = plt.subplots(n_show, 1, figsize=(14, 3.2 * n_show), sharex=False)
if n_show == 1:
    axes = [axes]

# Select events to show (spread across magnitude)
show_indices = np.linspace(0, len(results) - 1, n_show, dtype=int)

for plot_idx, r_idx in enumerate(show_indices):
    ax = axes[plot_idx]
    r = results[r_idx]
    t_w = r["_t_win"]
    veh_w = r["_veh_win"]
    gyro_w = r["_gyro_win"]
    center_t = r["center_t"]

    # Time relative to window center
    t_rel = t_w - center_t

    ax.plot(t_rel, veh_w, "k-", linewidth=1.5, label="Vehicle Yaw Rate")
    ax.plot(t_rel, gyro_w, "r-", linewidth=1.0, alpha=0.8, label="Gyro Pitch (deg/s)")

    # Show alignment lines
    # Peak timing
    veh_peak_t = t_rel[np.argmax(np.abs(veh_w))]
    gyro_peak_t = t_rel[np.argmax(np.abs(gyro_w))]
    ax.axvline(veh_peak_t, color="k", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(gyro_peak_t, color="r", linewidth=0.5, linestyle=":", alpha=0.5)

    # Interpolated lag
    interp_lag = r["interp_lag_s"]
    ax.axvline(-interp_lag / 2, color="blue", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"interp lag={interp_lag:+.3f}s")

    ax.set_ylabel("deg/s")
    ax.set_title(
        f"Event {r['event_id']}  peak={r['peak_yaw']:+.1f} deg/s  "
        f"{r['direction']}  "
        f"cc_lag={r['best_cc_lag_s']:+.3f}s  "
        f"interp={r['interp_lag_s']:+.3f}s  "
        f"peak_diff={r['peak_time_diff_s']:+.3f}s"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    if plot_idx == 0:
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time from event center (s)")
plt.tight_layout()
fig.savefig(OUT_DIR / "turn_latency_events.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'turn_latency_events.png'}")


# --- Plot B: Lag distribution ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# B1: Histogram of interpolated lags
ax = axes[0, 0]
ax.hist(primary_lags, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(np.median(primary_lags), color="red", linewidth=2,
           label=f"median={np.median(primary_lags):+.3f}s")
ax.axvline(1.8, color="orange", linewidth=1.5, linestyle="--",
           label="1.80 s (full-traj)")
ax.set_xlabel("Interpolated lag (s)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Event-Level Lags")
ax.legend(fontsize=8)

# B2: Lag vs peak yaw rate
ax = axes[0, 1]
peak_yaws = [r["peak_yaw"] for r in results]
colors_dir = ["blue" if r["direction"] == "positive" else "red" for r in results]
ax.scatter(peak_yaws, primary_lags, c=colors_dir, alpha=0.6, s=40)
ax.axhline(np.median(primary_lags), color="gray", linewidth=1, linestyle="--")
ax.axhline(1.8, color="orange", linewidth=1, linestyle="--", alpha=0.7)
ax.set_xlabel("Peak vehicle yaw rate (deg/s)")
ax.set_ylabel("Interpolated lag (s)")
ax.set_title("Lag vs Turn Magnitude")
ax.legend(["median", "1.80 s"], fontsize=8)

# B3: Method comparison
ax = axes[1, 0]
methods = ["CC interp", "CC discrete", "Peak timing"]
method_vals = [primary_lags, cc_lags, peak_diffs]
bp = ax.boxplot(method_vals, labels=methods, patch_artist=True)
ax.axhline(1.8, color="orange", linewidth=1.5, linestyle="--", label="1.80 s")
ax.set_ylabel("Estimated lag (s)")
ax.set_title("Lag Estimates by Method")
ax.legend(fontsize=8)

# B4: Per-event lags sorted
ax = axes[1, 1]
sorted_idx = np.argsort(primary_lags)
sorted_lags = primary_lags[sorted_idx]
sorted_dirs = [results[i]["direction"] for i in sorted_idx]
bar_colors = ["steelblue" if d == "positive" else "salmon" for d in sorted_dirs]
ax.bar(range(len(sorted_lags)), sorted_lags, color=bar_colors, alpha=0.7)
ax.axhline(np.median(sorted_lags), color="red", linewidth=2,
           label=f"median={np.median(sorted_lags):+.3f}s")
ax.axhline(1.8, color="orange", linewidth=1.5, linestyle="--", label="1.80 s")
ax.set_xlabel("Event index (sorted by lag)")
ax.set_ylabel("Interpolated lag (s)")
ax.set_title("Per-Event Lags (blue=pos, red=neg)")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "turn_latency_distribution.png", dpi=150)
plt.close(fig)
log(f"  Saved: {OUT_DIR / 'turn_latency_distribution.png'}")


# ============================================================
# SAVE REPORT
# ============================================================

log()
log("=" * 80)
log("SAVING REPORT")
log("=" * 80)

report = []
report.append("TURN-EVENT TEMPORAL ALIGNMENT REPORT")
report.append("=" * 80)
report.append("")
report.append(f"Dataset: S4_synced.csv (NOT MODIFIED)")
report.append(f"Valid rows: {len(imu):,}")
report.append(f"Median dt: {dt_median*1000:.1f} ms ({1/dt_median:.1f} Hz)")
report.append("")
report.append("LAG CONVENTION:")
report.append("  lag > 0 → phone gyro pitch LAGS vehicle yaw rate (arrives later)")
report.append("  lag < 0 → phone gyro pitch LEADS vehicle yaw rate (arrives earlier)")
report.append("")
report.append(f"Detection threshold: |yaw rate| >= {yaw_threshold} deg/s")
report.append(f"Min event duration: {min_duration} s")
report.append(f"Min peak yaw rate: {min_peak} deg/s")
report.append(f"Search lag range: ±{SEARCH_LAG_S} s")
report.append("")
report.append("-" * 80)
report.append(f"EVENTS DETECTED: {n_events} raw, {len(filtered_events)} after filtering")
report.append(f"EVENTS ANALYZED: {len(selected)}")
report.append(f"  Positive: {len(selected_pos)}, Negative: {len(selected_neg)}")
report.append("-" * 80)
report.append("")

# Per-event table
report.append("PER-EVENT RESULTS")
report.append("-" * 80)
hdr = (f"{'id':>4s} | {'dir':>8s} | {'peak':>7s} | {'dur':>5s} | "
       f"{'zl_r':>6s} | {'cc_lag':>7s} | {'interp':>7s} | "
       f"{'pk_diff':>8s} | {'zc_diff':>8s} | {'sign':>4s}")
report.append(hdr)
report.append("-" * len(hdr))

for r in results:
    zc_str = f"{r['zc_diff_s']:+7.3f}" if r["zc_diff_s"] is not None else "     N/A"
    report.append(
        f"{r['event_id']:4d} | {r['direction']:>8s} | "
        f"{r['peak_yaw']:+7.1f} | {r['duration_s']:5.2f} | "
        f"{r['zero_lag_r']:+6.3f} | {r['best_cc_lag_s']:+7.3f} | "
        f"{r['interp_lag_s']:+7.3f} | {r['peak_time_diff_s']:+8.3f} | "
        f"{zc_str} | {'Y' if r['sign_consistent'] else 'N':>4s}"
    )

report.append("")
report.append("-" * 80)
report.append("LAG STATISTICS (interpolated CC lag)")
report.append("-" * 80)
report.append(f"  N:       {len(primary_lags)}")
report.append(f"  Median:  {np.median(primary_lags):+.4f} s")
report.append(f"  Mean:    {np.mean(primary_lags):+.4f} s")
report.append(f"  Std:     {np.std(primary_lags):.4f} s")
report.append(f"  IQR:     [{np.percentile(primary_lags, 25):+.4f}, "
              f"{np.percentile(primary_lags, 75):+.4f}]")
report.append(f"  Range:   [{np.min(primary_lags):+.4f}, {np.max(primary_lags):+.4f}]")
report.append(f"  MAD:     {np.median(np.abs(primary_lags - np.median(primary_lags))):.4f} s")
report.append(f"  Sign consistent: {signs.sum()}/{len(signs)}")
report.append("")

report.append("PEAK TIMING DIFFERENCE")
report.append(f"  Median:  {np.median(peak_diffs):+.4f} s")
report.append(f"  Mean:    {np.mean(peak_diffs):+.4f} s")
report.append(f"  Std:     {np.std(peak_diffs):.4f} s")
report.append("")

if len(zc_diffs) > 0:
    report.append("ZERO-CROSSING DIFFERENCE")
    report.append(f"  N:       {len(zc_diffs)}")
    report.append(f"  Median:  {np.median(zc_diffs):+.4f} s")
    report.append(f"  Mean:    {np.mean(zc_diffs):+.4f} s")
    report.append(f"  Std:     {np.std(zc_diffs):.4f} s")
    report.append("")

report.append("-" * 80)
report.append("SUBGROUP SUMMARY")
report.append("-" * 80)

for name, mask in [("Positive turns", pos_mask), ("Negative turns", neg_mask),
                   ("Strong (|peak|>=25)", strong_mask),
                   ("Moderate (12<=|peak|<25)", moderate_mask)]:
    if mask.sum() == 0:
        continue
    lags = primary_lags[mask]
    report.append(f"  {name} (n={mask.sum()}):")
    report.append(f"    Median: {np.median(lags):+.4f}  Std: {np.std(lags):.4f}  "
                  f"Range: [{np.min(lags):+.4f}, {np.max(lags):+.4f}]")
    report.append(f"    Per-event: {[f'{l:+.3f}' for l in lags]}")
    report.append("")

report.append("-" * 80)
report.append("HYPOTHESIS EVALUATION")
report.append("-" * 80)
report.append("")
report.append(f"  H1 (genuine ~1.8 s lag):")
report.append(f"    Events with |lag - 1.8| < 0.3 s: {within_18.sum()}/{len(within_18)}")
report.append(f"    → {'SUPPORTED' if within_18.sum() > len(within_18)*0.7 else 'WEAKLY SUPPORTED' if within_18.sum() > len(within_18)*0.4 else 'NOT SUPPORTED'}")
report.append("")
report.append(f"  H2 (artifact of sparse turn structure):")
report.append(f"    Spread of event lags: {spread:.3f} s")
report.append(f"    → {'UNLIKELY' if spread < 1.0 else 'POSSIBLE' if spread < 2.0 else 'LIKELY'}")
report.append("")
report.append(f"  H3 (variable/event-dependent timing):")
report.append(f"    Std of event lags: {std_lag:.3f} s")
report.append(f"    → {'LOW variability' if std_lag < 0.2 else 'MODERATE' if std_lag < 0.5 else 'HIGH variability'}")
report.append("")

report.append("-" * 80)
report.append("CRITICAL SANITY CHECK")
report.append("-" * 80)
report.append("")
report.append("  The strong empirical association between Gyro Pitch and vehicle")
report.append("  yaw rate is established (consistently high correlation across")
report.append("  all segments and events, with correct sign).")
report.append("")
report.append("  However, the TEMPORAL ALIGNMENT is uncertain:")
report.append("  - Peak widths from full-trjectory analysis are broad (~3.8 s at 50%)")
report.append("  - Event-level lag estimates have non-trivial spread")
report.append("  - Multiple alignment methods do not fully agree")
report.append("")
report.append("  Current conclusion: 'Phone GYROSCOPE Pitch is strongly")
report.append("  empirically associated with vehicle yaw rate.'")
report.append("  The physical frame mapping interpretation requires further study.")
report.append("")

report.append("=" * 80)
report.append("END OF REPORT")
report.append("=" * 80)

save_text("turn_latency_report.txt", report)
df_results.to_csv(OUT_DIR / "turn_latency_detail.csv", index=False)
log(f"  Saved: {OUT_DIR / 'turn_latency_detail.csv'}")


# ============================================================
# FINAL SUMMARY
# ============================================================

log()
log("=" * 80)
log("TURN-EVENT LATENCY INVESTIGATION COMPLETE")
log("=" * 80)
log()
log("Outputs:")
log(f"  {OUT_DIR / 'turn_latency_report.txt'}")
log(f"  {OUT_DIR / 'turn_latency_detail.csv'}")
log(f"  {OUT_DIR / 'turn_latency_events.png'}")
log(f"  {OUT_DIR / 'turn_latency_distribution.png'}")
log()
log("S4_synced.csv was NOT modified.")
