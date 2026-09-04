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

OUTPUT_DIR = (
    ROOT
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "S4_synced.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

PHONE_DATE_COLUMN = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

VEHICLE_TIME_COLUMN = (
    "Time Since Start of Day (seconds)"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum allowed timestamp difference when matching
# smartphone samples to vehicle samples.
#
# At 10 Hz, normal samples are ~100 ms apart.
# A 50 ms tolerance means we accept the closest sample
# only when it is within half a normal sampling period.

MATCH_TOLERANCE_SECONDS = 0.050


# ============================================================
# HELPER
# ============================================================

def print_header(title):
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

print(f"Phone:   {PHONE_FILE}")
print(f"Vehicle: {VEHICLE_FILE}")


# ============================================================
# LOAD
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
print(f"Phone rows:   {len(phone):,}")
print(f"Vehicle rows: {len(vehicle):,}")


# ============================================================
# VERIFY COLUMNS
# ============================================================

if PHONE_DATE_COLUMN not in phone.columns:
    raise ValueError(
        f"Phone timestamp column not found:\n"
        f"{PHONE_DATE_COLUMN}"
    )

if VEHICLE_TIME_COLUMN not in vehicle.columns:
    raise ValueError(
        f"Vehicle timestamp column not found:\n"
        f"{VEHICLE_TIME_COLUMN}"
    )


# ============================================================
# PARSE PHONE TIMESTAMP
# ============================================================

print_header("1. PARSING TIMESTAMPS")

phone["timestamp"] = pd.to_datetime(
    phone[PHONE_DATE_COLUMN],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)

invalid_phone = phone["timestamp"].isna().sum()

if invalid_phone > 0:
    raise ValueError(
        f"{invalid_phone:,} invalid smartphone timestamps."
    )


# ============================================================
# VEHICLE TIMESTAMP
# ============================================================

vehicle["vehicle_elapsed_s"] = pd.to_numeric(
    vehicle[VEHICLE_TIME_COLUMN],
    errors="coerce",
)

invalid_vehicle = (
    vehicle["vehicle_elapsed_s"].isna().sum()
)

if invalid_vehicle > 0:
    raise ValueError(
        f"{invalid_vehicle:,} invalid vehicle timestamps."
    )


# Vehicle time is "time since start of day".
#
# We don't actually need the calendar date for matching.
# We can simply convert both streams into elapsed seconds
# relative to their respective starts.
#
# The initial phone and vehicle timestamps were verified
# to correspond, so their elapsed timelines can be compared.

phone["phone_elapsed_s"] = (
    phone["timestamp"] - phone["timestamp"].iloc[0]
).dt.total_seconds()

vehicle["vehicle_elapsed_s"] = (
    vehicle["vehicle_elapsed_s"]
    - vehicle["vehicle_elapsed_s"].iloc[0]
)


# ============================================================
# INITIAL TIMELINE CHECK
# ============================================================

print(
    f"Phone start:   {phone['timestamp'].iloc[0]}"
)

print(
    f"Phone end:     {phone['timestamp'].iloc[-1]}"
)

print(
    f"Phone duration:"
    f" {phone['phone_elapsed_s'].iloc[-1]:.3f} s"
)

print()

print(
    f"Vehicle duration:"
    f" {vehicle['vehicle_elapsed_s'].iloc[-1]:.3f} s"
)


# ============================================================
# IMPORTANT:
# HANDLE PHONE RECORDING GAPS
# ============================================================

print_header("2. DETECTING PHONE RECORDING GAPS")

phone_time = (
    phone["phone_elapsed_s"]
    .to_numpy()
)

phone_dt = np.diff(phone_time)

gap_indices = np.where(
    phone_dt > 1.0
)[0]

print(
    f"Phone gaps > 1 second: "
    f"{len(gap_indices)}"
)

for idx in gap_indices:

    print()
    print(
        f"Gap: row {idx:,} → {idx + 1:,}"
    )

    print(
        f"Duration: "
        f"{phone_dt[idx]:.3f} s"
    )

    print(
        f"Before: "
        f"{phone['timestamp'].iloc[idx]}"
    )

    print(
        f"After:  "
        f"{phone['timestamp'].iloc[idx + 1]}"
    )


# ============================================================
# BUILD SYNCHRONIZED TIMESTAMP
# ============================================================

# The phone's absolute timestamp is our reference.
#
# We construct a relative timestamp for the vehicle using
# its clean 10 Hz elapsed timeline.
#
# Because the first samples were aligned, we can express
# both streams in the same elapsed-time coordinate system.

vehicle_time = (
    vehicle["vehicle_elapsed_s"]
    .to_numpy()
)

phone_time = (
    phone["phone_elapsed_s"]
    .to_numpy()
)


# ============================================================
# TIMESTAMP-BASED MATCHING
# ============================================================

print_header("3. TIMESTAMP-BASED MATCHING")

print(
    "Matching each smartphone sample to the nearest "
    "vehicle sample..."
)

# np.searchsorted finds the insertion position of every
# phone timestamp in the sorted vehicle timeline.

right_indices = np.searchsorted(
    vehicle_time,
    phone_time,
    side="left",
)

# Clamp indices so we don't access vehicle[n]
right_indices = np.clip(
    right_indices,
    0,
    len(vehicle_time) - 1,
)

left_indices = np.clip(
    right_indices - 1,
    0,
    len(vehicle_time) - 1,
)

# Distance to right-side vehicle sample
right_distance = np.abs(
    vehicle_time[right_indices]
    - phone_time
)

# Distance to left-side vehicle sample
left_distance = np.abs(
    vehicle_time[left_indices]
    - phone_time
)

# Pick whichever vehicle sample is closer
use_right = right_distance < left_distance

matched_vehicle_indices = np.where(
    use_right,
    right_indices,
    left_indices,
)

matched_vehicle_times = (
    vehicle_time[matched_vehicle_indices]
)

match_error = np.abs(
    phone_time
    - matched_vehicle_times
)


# ============================================================
# MATCH QUALITY
# ============================================================

print_header("4. MATCH QUALITY")

valid_matches = (
    match_error <= MATCH_TOLERANCE_SECONDS
)

print(
    f"Tolerance: "
    f"{MATCH_TOLERANCE_SECONDS * 1000:.1f} ms"
)

print(
    f"Total phone samples: "
    f"{len(phone):,}"
)

print(
    f"Valid matches:       "
    f"{valid_matches.sum():,}"
)

print(
    f"Rejected matches:    "
    f"{(~valid_matches).sum():,}"
)

print()

print(
    f"Median match error:  "
    f"{np.median(match_error) * 1000:.3f} ms"
)

print(
    f"Mean match error:    "
    f"{np.mean(match_error) * 1000:.3f} ms"
)

print(
    f"95th percentile:     "
    f"{np.percentile(match_error, 95) * 1000:.3f} ms"
)

print(
    f"Maximum match error: "
    f"{np.max(match_error) * 1000:.3f} ms"
)


# ============================================================
# CHECK WHERE MATCHES FAIL
# ============================================================

print_header("5. MATCH REJECTIONS")

rejected_indices = np.where(
    ~valid_matches
)[0]

if len(rejected_indices) == 0:

    print(
        "Excellent: every smartphone sample "
        "has a vehicle match within tolerance."
    )

else:

    print(
        f"Rejected samples: {len(rejected_indices):,}"
    )

    print()
    print("First rejected samples:")

    for i in rejected_indices[:30]:

        print(
            f"row {i:8,d} | "
            f"phone={phone_time[i]:10.3f}s | "
            f"nearest_vehicle="
            f"{matched_vehicle_times[i]:10.3f}s | "
            f"error="
            f"{match_error[i]:8.3f}s"
        )


# ============================================================
# BUILD SYNCHRONIZED DATASET
# ============================================================

print_header("6. BUILDING SYNCHRONIZED DATASET")

# Keep only valid phone samples.

phone_valid = phone.loc[
    valid_matches
].copy()

vehicle_indices_valid = (
    matched_vehicle_indices[valid_matches]
)

vehicle_valid = (
    vehicle.iloc[vehicle_indices_valid]
    .copy()
)

# Reset indexes so rows line up exactly.

phone_valid.reset_index(
    drop=True,
    inplace=True,
)

vehicle_valid.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# REMOVE HELPER COLUMNS FROM VEHICLE
# ============================================================

# We don't want duplicate timing/helper columns in the
# final dataset unless we explicitly add our own clean ones.

vehicle_valid = vehicle_valid.drop(
    columns=[
        "vehicle_elapsed_s",
    ],
    errors="ignore",
)


# ============================================================
# REMOVE ORIGINAL PHONE HELPER TIMESTAMP COLUMNS
# ============================================================

# Keep the original DATE column because it is useful.
#
# Remove our temporary parsed timestamp column later.

phone_valid["match_error_ms"] = (
    match_error[valid_matches] * 1000
)

phone_valid["matched_vehicle_row"] = (
    vehicle_indices_valid
)


# ============================================================
# PREFIX VEHICLE COLUMNS
# ============================================================

# This prevents ambiguous duplicate column names such as:
#
# Latitude (degrees)
#
# because both phone and vehicle have GPS latitude.

vehicle_columns = {}

for column in vehicle_valid.columns:

    if column in phone_valid.columns:

        vehicle_columns[column] = (
            f"VEHICLE_{column}"
        )

vehicle_valid = vehicle_valid.rename(
    columns=vehicle_columns
)


# ============================================================
# COMBINE
# ============================================================

synced = pd.concat(
    [
        phone_valid,
        vehicle_valid,
    ],
    axis=1,
)


# ============================================================
# ADD CLEAN TIMELINE
# ============================================================

synced["SYNC_TIME_S"] = (
    synced[PHONE_DATE_COLUMN]
    .apply(
        lambda x: (
            pd.to_datetime(
                x,
                format="%Y-%m-%d %H:%M:%S:%f"
            )
            - phone["timestamp"].iloc[0]
        ).total_seconds()
    )
)


# ============================================================
# CLEAN TEMP COLUMN
# ============================================================

synced = synced.drop(
    columns=[
        "timestamp",
        "phone_elapsed_s",
    ],
    errors="ignore",
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE
# ============================================================

print_header("7. SAVING")

synced.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"Saved synchronized dataset:"
)

print(
    OUTPUT_FILE
)

print()

print(
    f"Rows:    {len(synced):,}"
)

print(
    f"Columns: {len(synced.columns):,}"
)

print(
    f"File size: "
    f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print_header("8. FINAL VALIDATION")

print(
    f"Original phone rows:     {len(phone):,}"
)

print(
    f"Synchronized rows:       {len(synced):,}"
)

print(
    f"Original vehicle rows:   {len(vehicle):,}"
)

print(
    f"Rejected phone samples:  "
    f"{(~valid_matches).sum():,}"
)

print()

print(
    f"Maximum match error:     "
    f"{np.max(match_error) * 1000:.3f} ms"
)

print(
    f"Median match error:      "
    f"{np.median(match_error) * 1000:.3f} ms"
)

print()

print(
    "First synchronized sample:"
)

print(
    f"  Phone time:   "
    f"{phone_valid[PHONE_DATE_COLUMN].iloc[0]}"
)

print(
    f"  Vehicle row:  "
    f"{vehicle_indices_valid[0]:,}"
)

print(
    f"  Error:        "
    f"{phone_valid['match_error_ms'].iloc[0]:.3f} ms"
)

print()

print(
    "Last synchronized sample:"
)

print(
    f"  Phone time:   "
    f"{phone_valid[PHONE_DATE_COLUMN].iloc[-1]}"
)

print(
    f"  Vehicle row:  "
    f"{vehicle_indices_valid[-1]:,}"
)

print(
    f"  Error:        "
    f"{phone_valid['match_error_ms'].iloc[-1]:.3f} ms"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 80)
print("SYNCHRONIZED DATASET CREATION COMPLETE")
print("=" * 80)
