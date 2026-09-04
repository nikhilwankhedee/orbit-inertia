from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

# Repository root:
# IO-VNBD/
# ├── Synchronised V abd S datasets/
# └── sih26168/
#     └── src/
#         └── inspect_dataset.py
ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    ROOT
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "S (Driver A)"
    / "S4"
)

SMARTPHONE = DATASET_ROOT / "S-S4.csv"
VEHICLE = DATASET_ROOT / "V-S4.csv"


# ============================================================================
# Configuration
# ============================================================================

# The IO-VNBD CSV files contain characters such as "²" in their headers.
# They are not UTF-8 encoded.
CSV_ENCODING = "cp1252"


# ============================================================================
# Loading
# ============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load an IO-VNBD CSV and normalize its column names."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")

    df = pd.read_csv(
        path,
        encoding=CSV_ENCODING,
    )

    # Remove accidental whitespace around column names.
    df.columns = df.columns.str.strip()

    return df


# ============================================================================
# General inspection
# ============================================================================

def inspect_dataframe(name: str, df: pd.DataFrame) -> None:
    """Print basic information about a dataframe."""

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for i, column in enumerate(df.columns):
        print(f"  [{i:02d}] {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")
    else:
        print(missing.to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


# ============================================================================
# Timing analysis
# ============================================================================

def analyze_timing(
    time_seconds: pd.Series,
    name: str,
) -> None:
    """Analyze timestamp spacing and effective sampling rate."""

    time_seconds = pd.to_numeric(
        time_seconds,
        errors="coerce",
    )

    valid_time = time_seconds.dropna()

    dt = valid_time.diff().dropna()

    print("\n" + "=" * 80)
    print(f"{name} TIMING")
    print("=" * 80)

    print(f"Start time:       {valid_time.iloc[0]:.6f} s")
    print(f"End time:         {valid_time.iloc[-1]:.6f} s")

    duration = valid_time.iloc[-1] - valid_time.iloc[0]

    print(f"Duration:         {duration:.3f} s")
    print(f"Duration:         {duration / 60:.2f} min")

    print("\nSample interval (dt):")

    print(f"  Mean:           {dt.mean():.6f} s")
    print(f"  Median:         {dt.median():.6f} s")
    print(f"  Std:            {dt.std():.6f} s")
    print(f"  Min:            {dt.min():.6f} s")
    print(f"  Max:            {dt.max():.6f} s")

    if dt.median() > 0:
        frequency = 1.0 / dt.median()
        print(f"\nApprox frequency: {frequency:.3f} Hz")

    print(f"Monotonic:        {valid_time.is_monotonic_increasing}")
    print(f"Duplicate times:  {valid_time.duplicated().sum()}")

    # Count unusually large time gaps.
    large_gaps = dt[dt > 0.2]

    print(f"Gaps > 200 ms:    {len(large_gaps):,}")


# ============================================================================
# Sensor statistics
# ============================================================================

def print_statistics(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
) -> None:
    """Print descriptive statistics for selected columns."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for column in columns:

        if column not in df.columns:
            print(f"\n[WARNING] Missing column: {column}")
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        print(f"\n{column}")

        print(f"  mean:   {values.mean():.6f}")
        print(f"  std:    {values.std():.6f}")
        print(f"  min:    {values.min():.6f}")
        print(f"  max:    {values.max():.6f}")


# ============================================================================
# GPS analysis
# ============================================================================

def analyze_gps(
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    speed_column: str,
    accuracy_column: str | None,
    name: str,
) -> None:
    """Print basic GPS statistics."""

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    latitude = pd.to_numeric(
        df[latitude_column],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df[longitude_column],
        errors="coerce",
    )

    speed = pd.to_numeric(
        df[speed_column],
        errors="coerce",
    )

    print(
        f"Latitude range:   "
        f"{latitude.min():.8f} → {latitude.max():.8f}"
    )

    print(
        f"Longitude range:  "
        f"{longitude.min():.8f} → {longitude.max():.8f}"
    )

    print(
        f"Speed range:      "
        f"{speed.min():.3f} → {speed.max():.3f}"
    )

    print(
        f"Speed mean:       "
        f"{speed.mean():.3f}"
    )

    print(
        f"Speed median:     "
        f"{speed.median():.3f}"
    )

    if accuracy_column is not None and accuracy_column in df.columns:

        accuracy = pd.to_numeric(
            df[accuracy_column],
            errors="coerce",
        )

        print(
            f"GPS accuracy mean: "
            f"{accuracy.mean():.3f} m"
        )

        print(
            f"GPS accuracy median: "
            f"{accuracy.median():.3f} m"
        )


# ============================================================================
# Smartphone ↔ vehicle comparison
# ============================================================================

def compare_datasets(
    smartphone: pd.DataFrame,
    vehicle: pd.DataFrame,
) -> None:
    """Compare the basic structure of smartphone and vehicle data."""

    print("\n" + "=" * 80)
    print("SMARTPHONE ↔ VEHICLE")
    print("=" * 80)

    print(
        f"Smartphone samples: {len(smartphone):,}"
    )

    print(
        f"Vehicle samples:    {len(vehicle):,}"
    )

    print(
        f"Same length:        {len(smartphone) == len(vehicle)}"
    )

    # Starting positions
    smartphone_lat = float(
        smartphone.iloc[0]["GPS LATITUDE (degrees)"]
    )

    smartphone_lon = float(
        smartphone.iloc[0]["GPS LONGITUDE (degrees)"]
    )

    vehicle_lat = float(
        vehicle.iloc[0]["Latitude (degrees)"]
    )

    vehicle_lon = float(
        vehicle.iloc[0]["Longitude (degrees)"]
    )

    print("\nStarting GPS position:")

    print(
        f"  Smartphone: "
        f"{smartphone_lat:.8f}, "
        f"{smartphone_lon:.8f}"
    )

    print(
        f"  Vehicle:    "
        f"{vehicle_lat:.8f}, "
        f"{vehicle_lon:.8f}"
    )

    # Rough geographic difference.
    #
    # This is NOT our final coordinate conversion.
    # It is only a sanity check.
    lat_difference = smartphone_lat - vehicle_lat
    lon_difference = smartphone_lon - vehicle_lon

    print("\nStarting coordinate difference:")

    print(
        f"  Latitude:  {lat_difference:.10f} degrees"
    )

    print(
        f"  Longitude: {lon_difference:.10f} degrees"
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    # ------------------------------------------------------------------------
    # Verify files
    # ------------------------------------------------------------------------

    print("=" * 80)
    print("IO-VNBD DATASET INSPECTION")
    print("=" * 80)

    print(f"\nDataset root:")
    print(f"  {DATASET_ROOT}")

    print("\nSmartphone file:")
    print(f"  {SMARTPHONE}")

    print("\nVehicle file:")
    print(f"  {VEHICLE}")

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    smartphone = load_csv(SMARTPHONE)
    vehicle = load_csv(VEHICLE)

    # ------------------------------------------------------------------------
    # Basic inspection
    # ------------------------------------------------------------------------

    inspect_dataframe(
        "SMARTPHONE DATA",
        smartphone,
    )

    inspect_dataframe(
        "VEHICLE DATA",
        vehicle,
    )

    # ------------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------------

    smartphone_time = (
        pd.to_numeric(
            smartphone["TIME SINCE START (ms)"],
            errors="coerce",
        )
        / 1000.0
    )

    vehicle_time = pd.to_numeric(
        vehicle["Time Since Start of Day (seconds)"],
        errors="coerce",
    )

    analyze_timing(
        smartphone_time,
        "SMARTPHONE",
    )

    analyze_timing(
        vehicle_time,
        "VEHICLE",
    )

    # ------------------------------------------------------------------------
    # Smartphone sensor statistics
    # ------------------------------------------------------------------------

    smartphone_accelerometer = [
        "ACCELEROMETER X (m/s²)",
        "ACCELEROMETER Y (m/s²)",
        "ACCELEROMETER Z (m/s²)",
    ]

    smartphone_gravity = [
        "GRAVITY X (m/s²)",
        "GRAVITY Y (m/s²)",
        "GRAVITY Z (m/s²)",
    ]

    smartphone_gyro = [
        "GYROSCOPE Yaw (rad/s)",
        "GYROSCOPE Pitch (rad/s)",
        "GYROSCOPE Roll (rad/s)",
    ]

    print_statistics(
        smartphone,
        smartphone_accelerometer,
        "SMARTPHONE ACCELEROMETER",
    )

    print_statistics(
        smartphone,
        smartphone_gravity,
        "SMARTPHONE GRAVITY",
    )

    print_statistics(
        smartphone,
        smartphone_gyro,
        "SMARTPHONE GYROSCOPE",
    )

    # ------------------------------------------------------------------------
    # Smartphone GPS
    # ------------------------------------------------------------------------

    analyze_gps(
        smartphone,
        latitude_column="GPS LATITUDE (degrees)",
        longitude_column="GPS LONGITUDE (degrees)",
        speed_column="GPS SPEED (Kmh)",
        accuracy_column="GPS ACCURACY (m)",
        name="SMARTPHONE GPS",
    )

    # ------------------------------------------------------------------------
    # Vehicle measurements
    # ------------------------------------------------------------------------

    vehicle_motion_columns = [
        "Velocity (km/hr)",
        "Heading (degrees)",
        "Yaw Rate (deg/sec)",
        "Indicated Vehicle Speed (km/hr)",
        "Indicated Longitudinal Acceleration (g)",
        "Indicated Lateral Acceleration (g)",
    ]

    print_statistics(
        vehicle,
        vehicle_motion_columns,
        "VEHICLE MOTION",
    )

    # ------------------------------------------------------------------------
    # Vehicle GPS
    # ------------------------------------------------------------------------

    analyze_gps(
        vehicle,
        latitude_column="Latitude (degrees)",
        longitude_column="Longitude (degrees)",
        speed_column="Velocity (km/hr)",
        accuracy_column=None,
        name="VEHICLE GPS",
    )

    # ------------------------------------------------------------------------
    # Smartphone ↔ vehicle
    # ------------------------------------------------------------------------

    compare_datasets(
        smartphone,
        vehicle,
    )

    # ------------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
