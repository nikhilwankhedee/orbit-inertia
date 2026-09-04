from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FILE = (
    ROOT
    / "processed"
    / "S4_synced.csv"
)


# ============================================================
# COLUMNS
# ============================================================

PHONE_SPEED = "GPS SPEED (Kmh)"
VEHICLE_SPEED = "Velocity (km/hr)"
TIME = "SYNC_TIME_S"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    FILE,
    encoding="cp1252",
)

df.columns = df.columns.str.strip()

print(f"Rows: {len(df):,}")


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

phone_speed = pd.to_numeric(
    df[PHONE_SPEED],
    errors="coerce",
)

vehicle_speed = pd.to_numeric(
    df[VEHICLE_SPEED],
    errors="coerce",
)

time = pd.to_numeric(
    df[TIME],
    errors="coerce",
)


# ============================================================
# VALID ROWS
# ============================================================

valid = (
    phone_speed.notna()
    & vehicle_speed.notna()
    & time.notna()
)

phone_speed = phone_speed[valid].to_numpy()
vehicle_speed = vehicle_speed[valid].to_numpy()
time = time[valid].to_numpy()


# ============================================================
# TEST BOTH INTERPRETATIONS
# ============================================================

# Interpretation A:
# Phone value is already km/h

phone_as_kmh = phone_speed

# Interpretation B:
# Phone value is actually m/s
# Convert to km/h.

phone_converted = phone_speed * 3.6


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    reference,
):

    error = prediction - reference

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    bias = np.mean(error)

    correlation = np.corrcoef(
        prediction,
        reference,
    )[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Correlation": correlation,
    }


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("PHONE GPS SPEED vs VEHICLE VELOCITY")
print("=" * 80)

print()
print("A) Treat phone value as km/h")

metrics_a = calculate_metrics(
    phone_as_kmh,
    vehicle_speed,
)

for name, value in metrics_a.items():

    print(
        f"{name:15}: {value:.4f}"
    )


print()
print("B) Treat phone value as m/s → km/h")

metrics_b = calculate_metrics(
    phone_converted,
    vehicle_speed,
)

for name, value in metrics_b.items():

    print(
        f"{name:15}: {value:.4f}"
    )


# ============================================================
# BASIC STATISTICS
# ============================================================

print()
print("=" * 80)
print("RAW SPEED STATISTICS")
print("=" * 80)

print()
print("Phone raw values:")

print(
    f"Mean:   {np.mean(phone_speed):.3f}"
)

print(
    f"Median: {np.median(phone_speed):.3f}"
)

print(
    f"Max:    {np.max(phone_speed):.3f}"
)


print()
print("Phone × 3.6:")

print(
    f"Mean:   {np.mean(phone_converted):.3f} km/h"
)

print(
    f"Median: {np.median(phone_converted):.3f} km/h"
)

print(
    f"Max:    {np.max(phone_converted):.3f} km/h"
)


print()
print("Vehicle:")

print(
    f"Mean:   {np.mean(vehicle_speed):.3f} km/h"
)

print(
    f"Median: {np.median(vehicle_speed):.3f} km/h"
)

print(
    f"Max:    {np.max(vehicle_speed):.3f} km/h"
)


# ============================================================
# SAMPLE COMPARISON
# ============================================================

print()
print("=" * 80)
print("SAMPLE-BY-SAMPLE COMPARISON")
print("=" * 80)

sample_indices = [
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
    len(phone_speed) - 1,
]

sample_indices = [
    i
    for i in sample_indices
    if 0 <= i < len(phone_speed)
]

print(
    f"{'TIME':>10} | "
    f"{'PHONE RAW':>12} | "
    f"{'PHONE ×3.6':>12} | "
    f"{'VEHICLE':>12}"
)

print("-" * 56)

for i in sample_indices:

    print(
        f"{time[i]:10.1f} | "
        f"{phone_speed[i]:12.3f} | "
        f"{phone_converted[i]:12.3f} | "
        f"{vehicle_speed[i]:12.3f}"
    )


# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 80)
print("VERDICT")
print("=" * 80)

if (
    metrics_b["RMSE"]
    < metrics_a["RMSE"]
    and
    metrics_b["Correlation"]
    > metrics_a["Correlation"]
):

    print(
        """
Phone × 3.6 agrees substantially better with the
vehicle velocity.

This strongly supports interpreting the smartphone
GPS SPEED values as m/s despite the dataset column
label saying Kmh.
"""
    )

else:

    print(
        """
The ×3.6 conversion did not clearly improve the
agreement.

Do not assume the phone GPS speed is m/s yet.
"""
    )

print("=" * 80)
