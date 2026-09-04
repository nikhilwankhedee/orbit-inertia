# Orbit-Inertia — AI-ML Intelligent Dead Reckoning (SIH26168)

**Smartphone IMU + vehicle CAN fusion for GNSS-blackout navigation.**

This pipeline learns a corrective model on top of classical dead reckoning (DR): rather than
predicting absolute position, we predict a **velocity residual** (Δv = reference − classical DR)
with a small GRU, then apply it to correct the DR trajectory under GPS outage.

## Motivation

Classical DR fails fast under GNSS blackout because neither the phone gyro nor a constant-velocity
model is accurate enough. Our earlier experiments quantified the failure:

| Duration | A0 DR Final Error |
|----------|-------------------|
| 10 s     | 83 m              |
| 30 s     | 289 m             |
| 60 s     | 486 m             |
| 120 s    | 850 m             |

Linear gyro calibration (scale/bias) made it **worse** (r=0.17 fit dominated by straight-line
noise), confirming that a **nonlinear, context-aware learned correction** is the right next step.

## Approach

- **Target:** 2D East/North velocity residual between reference velocity and classical A0 DR velocity
- **Model:** 1-layer GRU (hidden=32, ~4,900 params) over a 20-sample (2 s) context window
- **Features (16):** phone accel/gravity/gyro, phone GPS, vehicle CAN (velocity, heading, yaw rate,
  steering, wheel speeds)
- **Correction:** `v_corrected = v_classical + Δv_pred`; position integrated recursively
- **Strictly causal:** never uses future GPS/vehicle data during a blackout
- **Segment-level split:** train (Se0+Se1) / val (Se2) / test (Se3) — no data leakage

## Pipeline

```
ml_common.py            Data loading, feature extraction, windowing, normalization
train_velocity_residual_gru.py   GRU training (early stopping, checkpointing)
evaluate_velocity_residual_gru.py  Recursive rollout + A0 comparison + reports
```

```
S4_synced.csv
     │
     ▼
ml_common.py ──► windows (16 features × 20 samples) ──► z-score normalize
     │                                                     │
     ▼                                                     ▼
targets: Δv = v_ref − v_dr                          train GRU (hidden=32)
     │                                                     │
     └──────────► evaluate: recursive rollout ──► A0 vs ML comparison/report
```

## Results

Local 3-epoch smoke test (not scientific — for validation only):

| Duration | A0 MAE | ML MAE | Improvement |
|----------|--------|--------|-------------|
| 10 s     | 49.8 m | 40.7 m | +18.2%      |
| 30 s     | 82.7 m | 61.7 m | +25.5%      |
| 60 s     | 244 m  | 177 m  | +27.8%      |
| 120 s    | 537 m  | 368 m  | +31.5%      |

Full 200-epoch training (on GPU) expected to improve further. Pipeline is Kaggle-ready, configured
for the challenge's S4 sequence.

## Usage

All code expects `processed/S4_synced.csv` (the synchronized S4 dataset), path overridable via
`--data-root`.

```bash
# Train (GPU recommended, e.g. Kaggle notebook)
python sih26168/src/train_velocity_residual_gru.py --epochs 200 --batch-size 256

# Evaluate
python sih26168/src/evaluate_velocity_residual_gru.py
```

Outputs land in `sih26168/outputs/ml/`. See `sih26168/KAGGLE_ML_RUN.md` for full instructions.

## Repo Layout

```
sih26168/src/            pipeline source (analysis, DR baselines, ML)
sih26168/outputs/        experiment reports and figures
sih26168/KAGGLE_ML_RUN.md   Kaggle execution guide
sih26168/OBSERVATION_LOG.md  dated research log
```

## Notes

- The raw IO-VNBD dataset is **not** included in this repo (licensing/size). Download it separately
  and produce `processed/S4_synced.csv`.
- Phone GPS speed field is m/s despite a "Kmh" label — auto-corrected in `ml_common.py`.
- The empirical +1.81 s inter-stream offset is evaluation-alignment only, never used during DR.
