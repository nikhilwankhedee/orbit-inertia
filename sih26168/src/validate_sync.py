from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SYNCED_FILE = (
    ROOT
    / "processed"
    / "S4_synced.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

# Smartphone
PHONE_LAT = "GPS LATITUDE (degrees)"
PHONE_LON = "GPS LONGITUDE (degrees)"

# Vehicle
VEHICLE_LAT = "Latitude (degrees)"
VEHICLE_LON = "Longitude (degrees)"

# Synchronization metadata
SYNC_TIME = "SYNC_TIME_S"
MATCH_ERROR = "match_error_ms"


# ============================================================
# LOAD
# ============================================================

print("Loading synchronized dataset...")

if not SYNCED_FILE.exists():
    raise FileNotFoundError(
        f"Could not find synchronized dataset:\n"
        f"{SYNCED_FILE}"
    )

df = pd.read_csv(
    SYNCED_FILE,
    encoding="cp1252",
)

df.columns = df.columns.str.strip()

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    PHONE_LAT,
    PHONE_LON,
    VEHICLE_LAT,
    VEHICLE_LON,
    SYNC_TIME,
    MATCH_ERROR,
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    print("\nERROR: Missing required columns:")

    for column in missing:
        print(f"  {column}")

    print("\nAvailable columns:")

    for column in df.columns:
        print(f"  {column}")

    raise SystemExit(1)


# ============================================================
# CONVERT GPS COLUMNS TO NUMERIC
# ============================================================

for column in [
    PHONE_LAT,
    PHONE_LON,
    VEHICLE_LAT,
    VEHICLE_LON,
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# REMOVE ROWS WITH INVALID GPS
# ============================================================

valid_gps = (
    df[PHONE_LAT].notna()
    & df[PHONE_LON].notna()
    & df[VEHICLE_LAT].notna()
    & df[VEHICLE_LON].notna()
)

gps_df = df.loc[
    valid_gps
].copy()

print()

print(
    f"Rows with valid GPS on both streams: "
    f"{len(gps_df):,}"
)


# ============================================================
# GPS DISTANCE FUNCTION
# ============================================================

def gps_distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate approximate horizontal distance between
    two latitude/longitude coordinates.

    Uses a local equirectangular approximation, which is
    sufficient for this trajectory-scale validation.
    """

    earth_radius = 6_371_000.0

    lat_mean = np.radians(
        (lat1 + lat2) / 2.0
    )

    dlat = np.radians(
        lat1 - lat2
    )

    dlon = np.radians(
        lon1 - lon2
    )

    north = (
        dlat
        * earth_radius
    )

    east = (
        dlon
        * earth_radius
        * np.cos(lat_mean)
    )

    return np.sqrt(
        north ** 2
        + east ** 2
    )


# ============================================================
# CALCULATE GPS DIFFERENCE
# ============================================================

gps_df["gps_difference_m"] = gps_distance_m(
    gps_df[PHONE_LAT].to_numpy(),
    gps_df[PHONE_LON].to_numpy(),
    gps_df[VEHICLE_LAT].to_numpy(),
    gps_df[VEHICLE_LON].to_numpy(),
)

error = (
    gps_df["gps_difference_m"]
    .to_numpy()
)


# ============================================================
# 1. GPS POSITION AGREEMENT
# ============================================================

print()
print("=" * 80)
print("1. GPS POSITION AGREEMENT AFTER SYNCHRONIZATION")
print("=" * 80)

print(
    f"Mean:              {np.mean(error):.3f} m"
)

print(
    f"Median:            {np.median(error):.3f} m"
)

print(
    f"Std:               {np.std(error):.3f} m"
)

print(
    f"90th percentile:   "
    f"{np.percentile(error, 90):.3f} m"
)

print(
    f"95th percentile:   "
    f"{np.percentile(error, 95):.3f} m"
)

print(
    f"99th percentile:   "
    f"{np.percentile(error, 99):.3f} m"
)

print(
    f"Maximum:           {np.max(error):.3f} m"
)


# ============================================================
# 2. GPS ERROR THROUGHOUT RECORDING
# ============================================================

print()
print("=" * 80)
print("2. GPS ERROR THROUGHOUT RECORDING")
print("=" * 80)

check_rows = [
    0,
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
    len(gps_df) - 1,
]

check_rows = sorted(
    set(
        i
        for i in check_rows
        if 0 <= i < len(gps_df)
    )
)

print(
    f"{'ROW':>10} | "
    f"{'TIME':>12} | "
    f"{'ERROR (m)':>12}"
)

print("-" * 42)

for i in check_rows:

    row = gps_df.iloc[i]

    print(
        f"{i:10,d} | "
        f"{row[SYNC_TIME]:12.3f} | "
        f"{row['gps_difference_m']:12.3f}"
    )


# ============================================================
# 3. GPS ERROR DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("3. GPS ERROR DISTRIBUTION")
print("=" * 80)

bins = [
    0,
    2,
    5,
    10,
    20,
    50,
    100,
    500,
    1000,
    np.inf,
]

labels = [
    "<2 m",
    "2-5 m",
    "5-10 m",
    "10-20 m",
    "20-50 m",
    "50-100 m",
    "100-500 m",
    "500-1000 m",
    ">1000 m",
]

categories = pd.cut(
    error,
    bins=bins,
    labels=labels,
    right=False,
)

distribution = (
    pd.Series(categories)
    .value_counts(
        sort=False
    )
)

for label, count in distribution.items():

    percentage = (
        count
        / len(error)
        * 100.0
    )

    print(
        f"{str(label):>12}: "
        f"{count:8,d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 4. TIMESTAMP MATCH QUALITY
# ============================================================

print()
print("=" * 80)
print("4. TIMESTAMP MATCH QUALITY")
print("=" * 80)

match_error = pd.to_numeric(
    df[MATCH_ERROR],
    errors="coerce",
).dropna().to_numpy()

print(
    f"Median:            "
    f"{np.median(match_error):.3f} ms"
)

print(
    f"Mean:              "
    f"{np.mean(match_error):.3f} ms"
)

print(
    f"95th percentile:   "
    f"{np.percentile(match_error, 95):.3f} ms"
)

print(
    f"Maximum:           "
    f"{np.max(match_error):.3f} ms"
)


# ============================================================
# 5. TIMESTAMP MATCH ERROR DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("5. TIMESTAMP MATCH ERROR DISTRIBUTION")
print("=" * 80)

match_bins = [
    0,
    1,
    5,
    10,
    25,
    50,
    100,
    np.inf,
]

match_labels = [
    "<1 ms",
    "1-5 ms",
    "5-10 ms",
    "10-25 ms",
    "25-50 ms",
    "50-100 ms",
    ">100 ms",
]

match_categories = pd.cut(
    match_error,
    bins=match_bins,
    labels=match_labels,
    right=False,
)

match_distribution = (
    pd.Series(match_categories)
    .value_counts(
        sort=False
    )
)

for label, count in match_distribution.items():

    percentage = (
        count
        / len(match_error)
        * 100.0
    )

    print(
        f"{str(label):>12}: "
        f"{count:8,d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 6. REMAINING TIME GAPS
# ============================================================

print()
print("=" * 80)
print("6. REMAINING PHONE TIME GAPS")
print("=" * 80)

sync_time = pd.to_numeric(
    df[SYNC_TIME],
    errors="coerce",
).to_numpy()

dt = np.diff(sync_time)

gap_indices = np.where(
    dt > 1.0
)[0]

print(
    f"Gaps > 1 second: "
    f"{len(gap_indices)}"
)

if len(gap_indices) == 0:

    print(
        "No remaining gaps > 1 second."
    )

else:

    for idx in gap_indices:

        print()

        print(
            f"row {idx:,} → {idx + 1:,}"
        )

        print(
            f"Gap: "
            f"{dt[idx]:.3f} s"
        )

        print(
            f"Before: "
            f"{sync_time[idx]:.3f} s"
        )

        print(
            f"After:  "
            f"{sync_time[idx + 1]:.3f} s"
        )


# ============================================================
# 7. FINAL VERDICT
# ============================================================

print()
print("=" * 80)
print("7. SYNCHRONIZATION VERDICT")
print("=" * 80)

median_gps_error = np.median(error)
p95_gps_error = np.percentile(error, 95)

if (
    median_gps_error < 10
    and p95_gps_error < 25
):

    print(
        """
PASS

The smartphone and vehicle GPS trajectories agree
closely after timestamp-based synchronization.

The previous kilometre-scale row-by-row mismatch
was therefore primarily caused by temporal
misalignment.

We can proceed to trajectory reconstruction and
dead-reckoning experiments.
"""
    )

elif (
    median_gps_error < 50
    and p95_gps_error < 100
):

    print(
        """
PLAUSIBLE — INVESTIGATE

Timestamp synchronization substantially improved
the agreement between the two GPS streams, but
the remaining discrepancy deserves inspection
before building the navigation baseline.
"""
    )

else:

    print(
        """
FAIL / INVESTIGATE

Timestamp synchronization did not produce strong
GPS agreement.

Do NOT proceed to ML yet.

The GPS streams or their correspondence require
further investigation.
"""
    )


# ============================================================
# DONE
# ============================================================

print("=" * 80)
print("SYNC VALIDATION COMPLETE")
print("=" * 80)
