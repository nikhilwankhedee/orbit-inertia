from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PHONE_FILE = (
    ROOT
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "S (Driver A)"
    / "S4"
    / "S-S4.csv"
)

VEHICLE_FILE = (
    ROOT
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "S (Driver A)"
    / "S4"
    / "V-S4.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OFFSET_JUMP_THRESHOLD = 0.5
PHONE_GAP_THRESHOLD = 1.0
CONTEXT_ROWS = 10


# ============================================================
# HELPER
# ============================================================

def header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CHECK FILES
# ============================================================

print("Loading data...")

if not PHONE_FILE.exists():
    raise FileNotFoundError(
        f"Smartphone file not found:\n{PHONE_FILE}"
    )

if not VEHICLE_FILE.exists():
    raise FileNotFoundError(
        f"Vehicle file not found:\n{VEHICLE_FILE}"
    )

print(f"Phone file:   {PHONE_FILE}")
print(f"Vehicle file: {VEHICLE_FILE}")


# ============================================================
# LOAD DATA
# ============================================================

phone = pd.read_csv(
    PHONE_FILE,
    encoding="cp1252",
)

vehicle = pd.read_csv(
    VEHICLE_FILE,
    encoding="cp1252",
)

phone.columns = phone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()

print()
print(f"Phone rows:    {len(phone):,}")
print(f"Vehicle rows:  {len(vehicle):,}")


# ============================================================
# ACTUAL COLUMN NAMES
# ============================================================

PHONE_DATE_COLUMN = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

VEHICLE_TIME_COLUMN = (
    "Time Since Start of Day (seconds)"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

if PHONE_DATE_COLUMN not in phone.columns:

    print("\nERROR: Smartphone timestamp column not found.")
    print(f"Expected: {PHONE_DATE_COLUMN}")

    print("\nAvailable smartphone columns:")
    for col in phone.columns:
        print(f"  {col}")

    raise SystemExit(1)


if VEHICLE_TIME_COLUMN not in vehicle.columns:

    print("\nERROR: Vehicle timestamp column not found.")
    print(f"Expected: {VEHICLE_TIME_COLUMN}")

    print("\nAvailable vehicle columns:")
    for col in vehicle.columns:
        print(f"  {col}")

    raise SystemExit(1)


# ============================================================
# PARSE SMARTPHONE DATE
# ============================================================

phone["DATE_PARSED"] = pd.to_datetime(
    phone[PHONE_DATE_COLUMN],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)

invalid_dates = phone["DATE_PARSED"].isna().sum()

if invalid_dates > 0:

    print(
        f"\nERROR: {invalid_dates:,} smartphone "
        "timestamps could not be parsed."
    )

    raise SystemExit(1)


# ============================================================
# CREATE ELAPSED TIME
# ============================================================

phone_time = (
    phone["DATE_PARSED"]
    - phone["DATE_PARSED"].iloc[0]
).dt.total_seconds().to_numpy()


vehicle_raw_time = pd.to_numeric(
    vehicle[VEHICLE_TIME_COLUMN],
    errors="coerce",
)

invalid_vehicle_time = vehicle_raw_time.isna().sum()

if invalid_vehicle_time > 0:

    print(
        f"\nERROR: {invalid_vehicle_time:,} invalid "
        "vehicle timestamp values."
    )

    raise SystemExit(1)


vehicle_time = (
    vehicle_raw_time
    - vehicle_raw_time.iloc[0]
).to_numpy()


# ============================================================
# MATCH LENGTH
# ============================================================

n_phone = len(phone)
n_vehicle = len(vehicle)

if n_phone != n_vehicle:

    print()
    print("WARNING: Phone and vehicle have different row counts.")

    print(f"Phone:   {n_phone:,}")
    print(f"Vehicle: {n_vehicle:,}")

n = min(n_phone, n_vehicle)

phone_time = phone_time[:n]
vehicle_time = vehicle_time[:n]


# ============================================================
# 1. SMARTPHONE TIMING
# ============================================================

header("1. SMARTPHONE DATE TIMING")

phone_dt = np.diff(phone_time)

print(
    f"First timestamp: {phone['DATE_PARSED'].iloc[0]}"
)

print(
    f"Last timestamp:  {phone['DATE_PARSED'].iloc[n - 1]}"
)

print()

print(
    f"Mean dt:         {np.mean(phone_dt):.6f} s"
)

print(
    f"Median dt:       {np.median(phone_dt):.6f} s"
)

print(
    f"Std dt:          {np.std(phone_dt):.6f} s"
)

print(
    f"Min dt:          {np.min(phone_dt):.6f} s"
)

print(
    f"Max dt:          {np.max(phone_dt):.6f} s"
)

print(
    f"Monotonic:       {np.all(phone_dt > 0)}"
)

print(
    f"Duplicates:      {np.sum(phone_dt == 0):,}"
)

print(
    f"Duration:        {phone_time[-1]:.3f} s"
)

print(
    f"Duration:        {phone_time[-1] / 60:.2f} min"
)

print(
    f"Frequency:       "
    f"{1 / np.median(phone_dt):.3f} Hz"
)


# ============================================================
# 2. VEHICLE TIMING
# ============================================================

header("2. VEHICLE TIMING")

vehicle_dt = np.diff(vehicle_time)

print(
    f"First elapsed:   {vehicle_time[0]:.6f} s"
)

print(
    f"Last elapsed:    {vehicle_time[-1]:.6f} s"
)

print()

print(
    f"Mean dt:         {np.mean(vehicle_dt):.6f} s"
)

print(
    f"Median dt:       {np.median(vehicle_dt):.6f} s"
)

print(
    f"Std dt:          {np.std(vehicle_dt):.6f} s"
)

print(
    f"Min dt:          {np.min(vehicle_dt):.6f} s"
)

print(
    f"Max dt:          {np.max(vehicle_dt):.6f} s"
)

print(
    f"Monotonic:       {np.all(vehicle_dt > 0)}"
)

print(
    f"Duplicates:      {np.sum(vehicle_dt == 0):,}"
)

print(
    f"Duration:        {vehicle_time[-1]:.3f} s"
)

print(
    f"Duration:        {vehicle_time[-1] / 60:.2f} min"
)

print(
    f"Frequency:       "
    f"{1 / np.median(vehicle_dt):.3f} Hz"
)


# ============================================================
# 3. PHONE ↔ VEHICLE OFFSET
# ============================================================

header("3. PHONE ↔ VEHICLE TIME OFFSET")

# This is the key calculation:
#
# offset[i] =
#     smartphone elapsed time[i]
#     -
#     vehicle elapsed time[i]
#
# If rows remain synchronized:
#
#     offset ≈ constant
#
# If one stream jumps ahead/behind:
#
#     offset changes suddenly.

offset = phone_time - vehicle_time

print(
    f"Initial offset:  {offset[0]:.6f} s"
)

print(
    f"Final offset:    {offset[-1]:.6f} s"
)

print(
    f"Minimum offset:  {np.min(offset):.6f} s"
)

print(
    f"Maximum offset:  {np.max(offset):.6f} s"
)


# ============================================================
# 4. OFFSET AT REGULAR INTERVALS
# ============================================================

header("4. OFFSET THROUGHOUT RECORDING")

selected_rows = [
    0,
    1,
    10,
    100,
    1_000,
    5_000,
    10_000,
    20_000,
    30_000,
    40_000,
    50_000,
    60_000,
    70_000,
    80_000,
    90_000,
    n - 1,
]

selected_rows = sorted(
    set(
        i for i in selected_rows
        if 0 <= i < n
    )
)

print(
    f"{'ROW':>10} | "
    f"{'PHONE TIME':>14} | "
    f"{'VEHICLE TIME':>14} | "
    f"{'OFFSET':>14}"
)

print("-" * 65)

for i in selected_rows:

    print(
        f"{i:10,d} | "
        f"{phone_time[i]:14.3f} | "
        f"{vehicle_time[i]:14.3f} | "
        f"{offset[i]:14.3f}"
    )


# ============================================================
# 5. FIND SUDDEN OFFSET CHANGES
# ============================================================

header("5. LARGE SYNCHRONIZATION CHANGES")

offset_change = np.diff(offset)

break_indices = np.where(
    np.abs(offset_change) > OFFSET_JUMP_THRESHOLD
)[0]

print(
    f"Threshold:          "
    f"{OFFSET_JUMP_THRESHOLD:.3f} s"
)

print(
    f"Detected changes:   "
    f"{len(break_indices):,}"
)


if len(break_indices) == 0:

    print()
    print("No large synchronization changes detected.")

else:

    for idx in break_indices:

        print()
        print(
            f"ROW {idx:,} → {idx + 1:,}"
        )

        print("-" * 60)

        print(
            f"Phone time:      "
            f"{phone_time[idx]:.6f} → "
            f"{phone_time[idx + 1]:.6f} s"
        )

        print(
            f"Vehicle time:    "
            f"{vehicle_time[idx]:.6f} → "
            f"{vehicle_time[idx + 1]:.6f} s"
        )

        print(
            f"Offset:          "
            f"{offset[idx]:.6f} → "
            f"{offset[idx + 1]:.6f} s"
        )

        print(
            f"Offset change:   "
            f"{offset_change[idx]:+.6f} s"
        )

        print(
            f"Phone DATE:      "
            f"{phone['DATE_PARSED'].iloc[idx]}"
        )

        print(
            f"Phone DATE next: "
            f"{phone['DATE_PARSED'].iloc[idx + 1]}"
        )


# ============================================================
# 6. CONTEXT AROUND BREAKS
# ============================================================

header("6. CONTEXT AROUND SYNCHRONIZATION BREAKS")

if len(break_indices) == 0:

    print("No breaks to inspect.")

else:

    for idx in break_indices:

        start = max(
            0,
            idx - CONTEXT_ROWS
        )

        end = min(
            n,
            idx + CONTEXT_ROWS + 2
        )

        print()
        print(
            f"--- Around row {idx:,} ---"
        )

        print(
            f"{'ROW':>8} | "
            f"{'PHONE':>12} | "
            f"{'VEHICLE':>12} | "
            f"{'OFFSET':>12} | "
            f"DATE"
        )

        print("-" * 95)

        for i in range(start, end):

            marker = ""

            if i == idx:
                marker = " <-- BEFORE"

            elif i == idx + 1:
                marker = " <-- AFTER"

            print(
                f"{i:8,d} | "
                f"{phone_time[i]:12.3f} | "
                f"{vehicle_time[i]:12.3f} | "
                f"{offset[i]:12.3f} | "
                f"{phone['DATE_PARSED'].iloc[i]}"
                f"{marker}"
            )


# ============================================================
# 7. OFFSET DISTRIBUTION
# ============================================================

header("7. OFFSET DISTRIBUTION")

rounded_offset = np.round(
    offset,
    decimals=1
)

unique_offsets, counts = np.unique(
    rounded_offset,
    return_counts=True
)

order = np.argsort(counts)[::-1]

print(
    "Most common offset values:"
)

print()

for i in order[:20]:

    print(
        f"Offset ≈ "
        f"{unique_offsets[i]:8.1f} s"
        f"    rows = "
        f"{counts[i]:8,d}"
    )


# ============================================================
# 8. SMOOTH OFFSET
# ============================================================

header("8. SMOOTHED OFFSET")

# 101 samples at 10 Hz ≈ 10.1 seconds.
#
# Rolling median suppresses tiny timing noise and makes
# genuine synchronization segments easier to see.

offset_series = pd.Series(offset)

smooth_offset = (
    offset_series
    .rolling(
        window=101,
        center=True,
        min_periods=1,
    )
    .median()
    .to_numpy()
)

smooth_change = np.abs(
    np.diff(smooth_offset)
)

SMOOTH_THRESHOLD = 1.0

smooth_breaks = np.where(
    smooth_change > SMOOTH_THRESHOLD
)[0]

print(
    f"Threshold:          "
    f"{SMOOTH_THRESHOLD:.3f} s"
)

print(
    f"Detected boundaries: "
    f"{len(smooth_breaks):,}"
)

if len(smooth_breaks) == 0:

    print(
        "\nNo clear smoothed synchronization "
        "boundaries detected."
    )

else:

    for idx in smooth_breaks:

        print()

        print(
            f"Boundary near row {idx:,}"
        )

        print(
            f"Offset before: "
            f"{smooth_offset[idx]:.3f} s"
        )

        print(
            f"Offset after:  "
            f"{smooth_offset[idx + 1]:.3f} s"
        )


# ============================================================
# 9. GROUP NEARBY OFFSET EVENTS
# ============================================================

header("9. SYNCHRONIZATION EVENT SUMMARY")

if len(break_indices) == 0:

    print("No synchronization events detected.")

else:

    groups = []

    current_group = [
        break_indices[0]
    ]

    for idx in break_indices[1:]:

        # Events within 5 rows belong to the same event.
        if idx - current_group[-1] <= 5:

            current_group.append(idx)

        else:

            groups.append(current_group)

            current_group = [idx]

    groups.append(current_group)

    print(
        f"Grouped events: "
        f"{len(groups):,}"
    )

    for event_number, group in enumerate(
        groups,
        start=1
    ):

        first = group[0]
        last = group[-1]

        before = offset[first]
        after = offset[last + 1]

        print()
        print(
            f"EVENT {event_number}"
        )

        print("-" * 60)

        print(
            f"Rows:             "
            f"{first:,} → {last + 1:,}"
        )

        print(
            f"Phone DATE:       "
            f"{phone['DATE_PARSED'].iloc[first]}"
        )

        print(
            f"Phone DATE after: "
            f"{phone['DATE_PARSED'].iloc[last + 1]}"
        )

        print(
            f"Offset before:    "
            f"{before:.6f} s"
        )

        print(
            f"Offset after:     "
            f"{after:.6f} s"
        )

        print(
            f"Net offset shift: "
            f"{after - before:+.6f} s"
        )


# ============================================================
# 10. SMARTPHONE INTERNAL TIMESTAMP GAPS
# ============================================================

header("10. SMARTPHONE TIMESTAMP GAPS")

phone_gap_indices = np.where(
    phone_dt > PHONE_GAP_THRESHOLD
)[0]

print(
    f"Threshold:       "
    f"{PHONE_GAP_THRESHOLD:.3f} s"
)

print(
    f"Gaps detected:   "
    f"{len(phone_gap_indices):,}"
)

if len(phone_gap_indices) == 0:

    print(
        "\nNo smartphone timestamp gaps "
        "greater than 1 second."
    )

else:

    for idx in phone_gap_indices[:50]:

        print()

        print(
            f"Row {idx:,} → {idx + 1:,}"
        )

        print(
            f"dt: "
            f"{phone_dt[idx]:.3f} s"
        )

        print(
            f"DATE: "
            f"{phone['DATE_PARSED'].iloc[idx]}"
        )

        print(
            f"DATE next: "
            f"{phone['DATE_PARSED'].iloc[idx + 1]}"
        )

    if len(phone_gap_indices) > 50:

        print(
            f"\n... and "
            f"{len(phone_gap_indices) - 50:,} more."
        )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

header("11. FINAL SUMMARY")

print(
    f"""
Samples analysed:        {n:,}

SMARTPHONE
    Duration:             {phone_time[-1]:.3f} s
    Duration:             {phone_time[-1] / 60:.2f} min
    Median dt:            {np.median(phone_dt):.6f} s
    Frequency:            {1 / np.median(phone_dt):.3f} Hz
    Monotonic:            {np.all(phone_dt > 0)}
    Large gaps:           {len(phone_gap_indices):,}

VEHICLE
    Duration:             {vehicle_time[-1]:.3f} s
    Duration:             {vehicle_time[-1] / 60:.2f} min
    Median dt:            {np.median(vehicle_dt):.6f} s
    Frequency:            {1 / np.median(vehicle_dt):.3f} Hz
    Monotonic:            {np.all(vehicle_dt > 0)}

PHONE ↔ VEHICLE
    Initial offset:       {offset[0]:.6f} s
    Final offset:         {offset[-1]:.6f} s
    Minimum offset:       {np.min(offset):.6f} s
    Maximum offset:       {np.max(offset):.6f} s
    Large offset events:  {len(break_indices):,}
    Smoothed boundaries:  {len(smooth_breaks):,}
"""
)

print("=" * 80)
print("SYNC BREAK ANALYSIS COMPLETE")
print("=" * 80)
