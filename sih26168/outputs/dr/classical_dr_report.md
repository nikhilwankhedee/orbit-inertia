# Classical Dead-Reckoning V0 Baseline Report

---

## Classical DR V0 Assumptions

### Measured
- Phone accelerometer X/Y/Z (m/s²) at 10 Hz
- Phone gravity X/Y/Z (m/s²) at 10 Hz
- Phone gyroscope Yaw/Pitch/Roll (rad/s) at 10 Hz
- Phone GPS latitude/longitude (degrees) at ~1 Hz
- Phone GPS speed (Kmh) at ~1 Hz
- Vehicle CAN: velocity, heading, yaw rate, wheel speeds, steering angle

### Inferred
- Phone Z axis is approximately vertical (gravity Z ≈ 9.81 m/s²)
- Gyro Pitch has an empirical relationship with vehicle yaw rate (correlation ~0.71 during strong turns, scale factor ~0.94 median)
- GPS speed × 3.6 ≈ vehicle velocity (validated against CAN data)

### Assumed (V0)
- **A.** Phone Z axis is vertical; phone X/Y axes are in the horizontal plane BUT the phone-to-vehicle yaw rotation is UNKNOWN.
- **B.** Gyro Pitch is used as the heading rate signal. No proven physical basis beyond empirical correlation with vehicle yaw rate.
- **C.** The phone-to-vehicle X/Y frame mapping is NOT established. For Baseline B (IMU acceleration), we ASSUME phone X ≈ vehicle forward and phone Y ≈ vehicle left. This is an UNVERIFIED assumption.
- **D.** The 1.81 s alignment offset is an empirical inter-stream parameter, NOT a proven physical sensor latency. It is used ONLY for reference comparison evaluation, never during DR propagation.
- **E.** Phone GPS provides absolute position at ~1 Hz with ~3 m accuracy.
- **F.** GPS speed × 3.6 provides a reasonable velocity estimate.

### What Is Not Assumed
- No EKF, UKF, or filtering
- No ML correction
- No map matching
- No future information during blackout (strictly causal)
- No faking of successful DR results

### Separation of Concerns
- **DR propagation:** causal, uses only IMU + last-known state
- **Evaluation:** uses ground-truth GPS/vehicle reference AFTER blackout

---

## A. Input Dataset

| Property | Value |
|----------|-------|
| File | `S4_synced.csv` |
| Rows | 91,463 |
| Duration | ~9,460 s (~2.6 hours) |
| Segments | 4 (separated by recording gaps) |

### Segments

| Segment | Rows | Duration |
|---------|------|----------|
| 0 | 35,186 | 3,518.5 s |
| 1 | 13,001 | 1,300.1 s |
| 2 | 42,775 | 4,277.7 s |
| 3 | 501 | 50.0 s |

---

## B. Columns Used

| Key | Column |
|-----|--------|
| `accel_x` | ACCELEROMETER X (m/s²) |
| `accel_y` | ACCELEROMETER Y (m/s²) |
| `accel_z` | ACCELEROMETER Z (m/s²) |
| `gravity_x` | GRAVITY X (m/s²) |
| `gravity_y` | GRAVITY Y (m/s²) |
| `gravity_z` | GRAVITY Z (m/s²) |
| `gyro_pitch` | GYROSCOPE Pitch (rad/s) |
| `sync_time` | SYNC_TIME_S |
| `phone_date` | DATE (YYYY-MO-DD HH-MI-SS_SSS) |
| `phone_lat` | GPS LATITUDE (degrees) |
| `phone_lon` | GPS LONGITUDE (degrees) |
| `phone_gps_speed` | GPS SPEED (Kmh) |
| `phone_gps_accuracy` | GPS ACCURACY (m) |
| `veh_lat` | Latitude (degrees) |
| `veh_lon` | Longitude (degrees) |
| `veh_velocity` | Velocity (km/hr) |
| `veh_heading` | Heading (degrees) |
| `veh_yaw_rate` | Yaw Rate (deg/sec) |

---

## C. Coordinate-Frame Assumptions

Local tangent-plane ENU:

$$
\text{East} = \Delta\text{lon} \times \cos(\text{lat}_0) \times \frac{\pi}{180} \times R_{\text{Earth}}
$$

$$
\text{North} = \Delta\text{lat} \times \frac{\pi}{180} \times R_{\text{Earth}}
$$

- **R_Earth** = 6,371,000 m
- **Validity:** trajectory scale ~11 km, tangent-plane error < 0.01%
- **Origin:** vehicle reference position at first blackout start

---

## D. Phone → Vehicle Frame Assumptions

| Baseline | Frame Assumption |
|----------|-----------------|
| A | No frame assumption needed (heading-only propagation) |
| B | **UNVERIFIED** — assumes phone X ≈ vehicle forward, phone Y ≈ vehicle left. Gravity vector confirms phone Z is vertical but says nothing about yaw. |

- Sign agreement (gyro_pitch vs vehicle_yaw_rate): **~80%**
- **Baseline B results MUST be interpreted with extreme caution.**

---

## E. Gyro Scale Factor

- Overall correlation (gyro pitch vs veh yaw rate): **0.5073**
- Scale factor NOT applied in V0 (used 1.0)
- This is a known limitation.

---

## F. Timestamp / Alignment Treatment

- `SYNC_TIME_S` used for all timing
- Empirical inter-stream offset: **+1.81 s**
- Applied ONLY for reference comparison, NEVER during DR propagation
- DR propagation is **strictly causal**

---

## G. Initialization Method

| State | Source |
|-------|--------|
| Position | Vehicle reference lat/lon at blackout start → ENU |
| Heading | Vehicle reference heading at blackout start |
| Velocity (A) | Vehicle reference speed (constant) |
| Velocity (B) | Vehicle reference at init, then propagated via IMU acceleration integration |

---

## H. Blackout Selection Methodology

- **Durations:** 10s, 30s, 60s, 120s
- **Constraints:**
  - Within a single segment (no gap crossing)
  - Vehicle motion > 2.0 km/h before blackout
  - GPS accuracy < 5.0 m before blackout
  - Evaluation margin after blackout: min(30, duration) s
- **Selected windows:** 39

### Selected Windows

| Dur | Rows | Time Range | Pre-Vel (km/h) |
|-----|------|------------|-----------------|
| 10s | [10, 110) | T=1.0–10.9 | 9.8 |
| 10s | [18410, 18510) | T=1841.0–1850.9 | 24.9 |
| 10s | [34986, 35086) | T=3498.6–3508.5 | 4.5 |
| 10s | [40239, 40339) | T=4335.9–4345.8 | 2.1 |
| 10s | [44146, 44246) | T=4726.7–4736.6 | 19.7 |
| 10s | [47986, 48086) | T=5110.7–5120.6 | 3.9 |
| 10s | [48197, 48297) | T=5131.9–5141.8 | 13.9 |
| 10s | [69939, 70039) | T=7306.4–7316.3 | 23.0 |
| 10s | [90761, 90861) | T=9388.6–9398.5 | 36.6 |
| 10s | [91006, 91106) | T=9414.3–9424.2 | 28.0 |
| 10s | [91109, 91209) | T=9424.6–9434.5 | 33.4 |
| 10s | [91212, 91312) | T=9434.9–9444.8 | 36.6 |
| 30s | [10, 310) | T=1.0–30.9 | 9.8 |
| 30s | [18318, 18618) | T=1831.8–1861.7 | 2.1 |
| 30s | [34559, 34860) | T=3455.9–3485.9 | 2.1 |
| 30s | [40239, 40539) | T=4335.9–4365.8 | 2.1 |
| 30s | [43947, 44247) | T=4706.8–4736.7 | 3.8 |
| 30s | [47587, 47887) | T=5070.8–5100.7 | 33.5 |
| 30s | [48197, 48497) | T=5131.9–5161.8 | 13.9 |
| 30s | [69381, 69681) | T=7250.6–7280.5 | 82.0 |
| 30s | [90361, 90661) | T=9348.6–9378.5 | 16.6 |
| 60s | [10, 610) | T=1.0–60.9 | 9.8 |
| 60s | [18100, 18701) | T=1810.0–1870.0 | 41.8 |
| 60s | [34286, 34886) | T=3428.6–3488.5 | 34.5 |
| 60s | [40239, 40839) | T=4335.9–4395.8 | 2.1 |
| 60s | [43797, 44396) | T=4691.7–4751.6 | 31.4 |
| 60s | [47287, 47887) | T=5040.8–5100.7 | 17.0 |
| 60s | [48197, 48797) | T=5131.9–5191.8 | 13.9 |
| 60s | [69231, 69831) | T=7235.6–7295.5 | 87.2 |
| 60s | [90061, 90661) | T=9318.6–9378.5 | 3.3 |
| 120s | [10, 1210) | T=1.0–120.9 | 9.8 |
| 120s | [17663, 18863) | T=1766.3–1886.2 | 32.3 |
| 120s | [33686, 34886) | T=3368.6–3488.5 | 27.2 |
| 120s | [40239, 41440) | T=4335.9–4455.9 | 2.1 |
| 120s | [43497, 44697) | T=4661.7–4781.7 | 26.1 |
| 120s | [46687, 47887) | T=4980.8–5100.7 | 29.9 |
| 120s | [48197, 49396) | T=5131.9–5251.8 | 13.9 |
| 120s | [68931, 70132) | T=7205.6–7325.6 | 88.7 |
| 120s | [89461, 90661) | T=9258.6–9378.5 | 30.7 |

---

## I. Causality Rules

During blackout, DR state uses **ONLY**:
- IMU samples up to current time
- Last-known velocity (Baseline A) or velocity from accel integration (B)
- Current heading (from gyro integration)

Ground truth (vehicle GPS, heading, yaw rate) used **ONLY** for evaluation.

---

## J. Integration Equations

### Baseline A

$$
\text{heading}_{k+1} = \text{heading}_k + \text{gyro\_pitch}_k \times dt
$$

$$
\text{pos\_east}_{k+1} = \text{pos\_east}_k + v \times \cos(\text{heading}_k) \times dt
$$

$$
\text{pos\_north}_{k+1} = \text{pos\_north}_k + v \times \sin(\text{heading}_k) \times dt
$$

### Baseline B (Experimental)

$$
\text{heading}_{k+1} = \text{heading}_k + \text{gyro\_pitch}_k \times dt
$$

$$
a_{\text{body}} = \text{accel} - \text{gravity}
$$

$$
a_{\text{nav}} = R(\text{heading}) \times a_{\text{body}}
$$

$$
v_{k+1} = v_k + a_{\text{nav}} \times dt
$$

$$
p_{k+1} = p_k + v_k \times dt + 0.5 \times a_{\text{nav}} \times dt^2
$$

---

## K. Metrics

### Baseline A — Yaw-Only Kinematic

| Duration | MAE (m) | RMSE (m) | Max (m) | Final (m) | Heading MAE |
|----------|---------|----------|---------|-----------|-------------|
| 10 s | 41.3 | 48.0 | 83.0 | 83.0 | 77.7° |
| 30 s | 137.6 | 161.6 | 288.5 | 288.5 | 93.0° |
| 60 s | 254.6 | 294.1 | 500.9 | 486.1 | 87.2° |
| 120 s | 440.1 | 504.5 | 872.4 | 850.2 | 90.1° |

### Baseline B — IMU Acceleration (Experimental)

| Duration | MAE (m) | RMSE (m) | Max (m) | Final (m) | Heading MAE |
|----------|---------|----------|---------|-----------|-------------|
| 10 s | 45.0 | 53.2 | 95.6 | 95.6 | 77.7° |
| 30 s | 161.4 | 189.5 | 341.0 | 325.7 | 93.0° |
| 60 s | 343.3 | 411.1 | 758.9 | 758.9 | 87.2° |
| 120 s | 756.2 | 898.5 | 1719.1 | 1719.1 | 90.1° |

---

## L. Numerical Results

### Baseline A

| Blackout | Position MAE | Position RMSE | Position Max | Position Final | Heading MAE |
|----------|-------------|---------------|-------------|----------------|-------------|
| 10 s | 41.3 m | 48.0 m | 83.0 m | 83.0 m | 77.7° |
| 30 s | 137.6 m | 161.6 m | 288.5 m | 288.5 m | 93.0° |
| 60 s | 254.6 m | 294.1 m | 500.9 m | 486.1 m | 87.2° |
| 120 s | 440.1 m | 504.5 m | 872.4 m | 850.2 m | 90.1° |

### Baseline B

| Blackout | Position MAE | Position RMSE | Position Max | Position Final | Heading MAE |
|----------|-------------|---------------|-------------|----------------|-------------|
| 10 s | 45.0 m | 53.2 m | 95.6 m | 95.6 m | 77.7° |
| 30 s | 161.4 m | 189.5 m | 341.0 m | 325.7 m | 93.0° |
| 60 s | 343.3 m | 411.1 m | 758.9 m | 758.9 m | 87.2° |
| 120 s | 756.2 m | 898.5 m | 1719.1 m | 1719.1 m | 90.1° |

---

## M. Limitations

1. Gyro pitch correlation with vehicle yaw rate is moderate (~0.71). Sign agreement ~80%. This limits heading propagation accuracy.
2. No gyro scale factor applied (V0 uses 1.0).
3. Phone-to-vehicle frame mapping is UNVERIFIED (Baseline B).
4. Baseline A holds velocity constant — no acceleration model.
5. GPS updates at ~1 Hz with ~3 m accuracy — init position has error.
6. Vehicle reference used for initialization (not phone GPS).
7. Small sample: only ~3 blackout windows per duration.
8. No sensor noise modeling or filtering.

---

## N. Is the Baseline Physically Defensible?

**Baseline A: YES** — it is a simple, honest kinematic DR model. Heading from gyro integration, constant velocity, known initial state. Every assumption is documented and the failure modes are expected.

**Baseline B: NO** — the phone→vehicle frame transform is unverified. The 80% sign agreement and unknown yaw rotation between phone and vehicle mean that Baseline B acceleration integration may inject large systematic errors. Results should be treated as experimental and not as evidence of IMU capability.

---

## O. Recommendations for Next Steps

1. Run Baseline A and quantify the real drift — do not optimize.
2. If Baseline A drift is severe, the ML correction has clear targets.
3. Investigate gyro pitch sign disagreements — phone reorientation?
4. Consider using steering angle + wheel speed for velocity model.
5. If Baseline B shows large errors, this confirms frame uncertainty is the critical limitation to address.
6. **Future:** EKF with gyro + vehicle speed fusion.
7. **Future:** ML correction model (the ultimate goal).
