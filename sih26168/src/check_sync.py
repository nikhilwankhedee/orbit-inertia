from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    ROOT
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "S (Driver A)"
    / "S4"
)

SMARTPHONE_PATH = DATASET_ROOT / "S-S4.csv"
VEHICLE_PATH = DATASET_ROOT / "V-S4.csv"


# ============================================================================
# Configuration
# ============================================================================

CSV_ENCODING = "cp1252"


# ============================================================================
# Load
# ============================================================================

print("Loading data...")

smartphone = pd.read_csv(
    SMARTPHONE_PATH,
    encoding=CSV_ENCODING,
)

vehicle = pd.read_csv(
    VEHICLE_PATH,
    encoding=CSV_ENCODING,
)

# Normalize whitespace around column names.
smartphone.columns = smartphone.columns.str.strip()
vehicle.columns = vehicle.columns.str.strip()


# ============================================================================
# 1. Smartphone DATE timestamp
# ============================================================================

print("\n" + "=" * 80)
print("1. SMARTPHONE DATE TIMESTAMP")
print("=" * 80)

date_column = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

dates = pd.to_datetime(
    smartphone[date_column],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)

print(f"Invalid dates: {dates.isna().sum():,}")

date_dt = dates.diff().dt.total_seconds().dropna()

print(f"First timestamp: {dates.iloc[0]}")
print(f"Last timestamp:  {dates.iloc[-1]}")

print(f"\nMean dt:         {date_dt.mean():.6f} s")
print(f"Median dt:       {date_dt.median():.6f} s")
print(f"Std dt:          {date_dt.std():.6f} s")
print(f"Min dt:          {date_dt.min():.6f} s")
print(f"Max dt:          {date_dt.max():.6f} s")

print(
    f"Monotonic:       {dates.is_monotonic_increasing}"
)

print(
    f"Duplicates:      {dates.duplicated().sum():,}"
)

duration = (
    dates.iloc[-1] - dates.iloc[0]
).total_seconds()

print(f"Duration:        {duration:.3f} s")
print(f"Duration:        {duration / 60:.2f} min")


# ============================================================================
# 2. Smartphone TIME SINCE START anomalies
# ============================================================================

print("\n" + "=" * 80)
print("2. SMARTPHONE TIME SINCE START")
print("=" * 80)

phone_time = (
    pd.to_numeric(
        smartphone["TIME SINCE START (ms)"],
        errors="coerce",
    )
    / 1000.0
)

phone_dt = phone_time.diff()

print(f"Invalid values:   {phone_time.isna().sum():,}")
print(f"Duplicates:       {phone_time.duplicated().sum():,}")
print(f"Negative dt:      {(phone_dt < 0).sum():,}")
print(f"Zero dt:          {(phone_dt == 0).sum():,}")
print(f"dt > 1 sec:       {(phone_dt > 1).sum():,}")


# Show suspicious transitions.

bad_indices = np.where(
    (phone_dt < 0) | (phone_dt > 1)
)[0]

print("\nFirst 20 suspicious timestamp transitions:")

if len(bad_indices) == 0:
    print("  None")
else:
    for idx in bad_indices[:20]:

        print(f"\nrow {idx:,} → {idx + 1:,}")

        print(
            f"  TIME: "
            f"{phone_time.iloc[idx]:.3f} → "
            f"{phone_time.iloc[idx + 1]:.3f}"
        )

        print(
            f"  DATE: "
            f"{dates.iloc[idx]} → "
            f"{dates.iloc[idx + 1]}"
        )


# ============================================================================
# 3. Compare smartphone DATE against vehicle time
# ============================================================================

print("\n" + "=" * 80)
print("3. SMARTPHONE DATE ↔ VEHICLE TIME")
print("=" * 80)

vehicle_time = pd.to_numeric(
    vehicle["Time Since Start of Day (seconds)"],
    errors="coerce",
)

print(
    f"Vehicle start: "
    f"{vehicle_time.iloc[0]:.3f} s"
)

print(
    f"Vehicle end:   "
    f"{vehicle_time.iloc[-1]:.3f} s"
)


# Convert both timelines to elapsed time from their own starts.

phone_date_elapsed = (
    dates - dates.iloc[0]
).dt.total_seconds()

vehicle_elapsed = (
    vehicle_time - vehicle_time.iloc[0]
)


# IMPORTANT:
# We compare the timestamps by ROW INDEX.
#
# This is a diagnostic only. We are testing whether the synchronized
# files appear to have been constructed with corresponding rows.

time_difference = (
    phone_date_elapsed - vehicle_elapsed
)

print("\nElapsed-time difference at corresponding rows:")

print(
    f"Mean:       {time_difference.mean():.6f} s"
)

print(
    f"Median:     {time_difference.median():.6f} s"
)

print(
    f"Std:        {time_difference.std():.6f} s"
)

print(
    f"Min:        {time_difference.min():.6f} s"
)

print(
    f"Max:        {time_difference.max():.6f} s"
)


# ============================================================================
# 4. GPS position comparison
# ============================================================================

print("\n" + "=" * 80)
print("4. ROW-BY-ROW GPS POSITION CHECK")
print("=" * 80)

phone_lat = pd.to_numeric(
    smartphone["GPS LATITUDE (degrees)"],
    errors="coerce",
)

phone_lon = pd.to_numeric(
    smartphone["GPS LONGITUDE (degrees)"],
    errors="coerce",
)

vehicle_lat = pd.to_numeric(
    vehicle["Latitude (degrees)"],
    errors="coerce",
)

vehicle_lon = pd.to_numeric(
    vehicle["Longitude (degrees)"],
    errors="coerce",
)


# Approximate conversion from degrees to metres.
#
# This is ONLY a sanity check.
# We will use a proper local coordinate conversion later.

mean_lat = np.deg2rad(
    vehicle_lat.mean()
)

meters_per_lat = 111_320.0

meters_per_lon = (
    111_320.0 * np.cos(mean_lat)
)


lat_error = (
    phone_lat - vehicle_lat
) * meters_per_lat

lon_error = (
    phone_lon - vehicle_lon
) * meters_per_lon

horizontal_error = np.sqrt(
    lat_error**2 + lon_error**2
)


print(
    f"Mean horizontal difference:   "
    f"{horizontal_error.mean():.3f} m"
)

print(
    f"Median horizontal difference: "
    f"{horizontal_error.median():.3f} m"
)

print(
    f"95th percentile:              "
    f"{np.percentile(horizontal_error.dropna(), 95):.3f} m"
)

print(
    f"Maximum difference:           "
    f"{horizontal_error.max():.3f} m"
)


# ============================================================================
# 5. Selected row inspection
# ============================================================================

print("\n" + "=" * 80)
print("5. SELECTED ROWS")
print("=" * 80)

selected_indices = [
    0,
    1,
    10,
    100,
    1_000,
    10_000,
    50_000,
    90_000,
    len(smartphone) - 1,
]

for idx in selected_indices:

    if idx >= len(smartphone):
        continue

    print(f"\nrow {idx:,}")

    print(
        f"  Phone DATE: "
        f"{dates.iloc[idx]}"
    )

    print(
        f"  Phone elapsed: "
        f"{phone_date_elapsed.iloc[idx]:.3f} s"
    )

    print(
        f"  Vehicle elapsed: "
        f"{vehicle_elapsed.iloc[idx]:.3f} s"
    )

    print(
        f"  Time difference: "
        f"{time_difference.iloc[idx]:.3f} s"
    )

    print(
        f"  Phone GPS: "
        f"{phone_lat.iloc[idx]:.8f}, "
        f"{phone_lon.iloc[idx]:.8f}"
    )

    print(
        f"  Vehicle GPS: "
        f"{vehicle_lat.iloc[idx]:.8f}, "
        f"{vehicle_lon.iloc[idx]:.8f}"
    )

    print(
        f"  GPS position difference: "
        f"{horizontal_error.iloc[idx]:.3f} m"
    )


# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "=" * 80)
print("6. SUMMARY")
print("=" * 80)

print(
    f"""
Smartphone samples: {len(smartphone):,}
Vehicle samples:    {len(vehicle):,}
Same row count:     {len(smartphone) == len(vehicle)}

Smartphone DATE:
    Duration:        {duration:.3f} s
    Median dt:       {date_dt.median():.6f} s
    Frequency:       {1.0 / date_dt.median():.3f} Hz
    Monotonic:       {dates.is_monotonic_increasing}
    Duplicates:      {dates.duplicated().sum():,}

Vehicle:
    Duration:        {vehicle_elapsed.iloc[-1]:.3f} s
    Median dt:       {vehicle_elapsed.diff().median():.6f} s
    Frequency:       {1.0 / vehicle_elapsed.diff().median():.3f} Hz
    Monotonic:       {vehicle_time.is_monotonic_increasing}
    Duplicates:      {vehicle_time.duplicated().sum():,}

GPS position difference:
    Median:          {horizontal_error.median():.3f} m
    95th percentile: {np.percentile(horizontal_error.dropna(), 95):.3f} m
    Maximum:         {horizontal_error.max():.3f} m
"""
)

print("=" * 80)
print("SYNC CHECK COMPLETE")
print("=" * 80)
