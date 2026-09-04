# SIH26168 — ML Residual GRU: Kaggle Evaluation Report

**Date:** 2026-09-04
**Environment:** Kaggle (NVIDIA GPU / CUDA)
**Run status:** COMPLETE — training + full evaluation (39 blackout windows)

---

## Summary

A 1-layer GRU (hidden=32) was trained to predict the East/North velocity
residual `Δv = v_reference − v_classical` of the A0 classical dead-reckoning
pipeline. During blackout inference, the corrected velocity
`v_corrected = v_classical + Δv_pred` is integrated to propagate position.

On the held-out evaluation windows, the ML-corrected DR reduces mean absolute
error by **34–45%** versus the A0 classical baseline across all blackout
durations (10 s → 120 s). This substantially exceeds the local 3-epoch smoke
test (18–31%), confirming that full GPU training materially helped.

| Duration | A0 MAE | ML MAE | Δ MAE (m) | Improvement |
|---------:|-------:|-------:|----------:|------------:|
| 10 s     |  49.8 m |  30.4 m |   +19.4 m |  **+38.9%** |
| 30 s     |  82.7 m |  45.7 m |   +37.0 m |  **+44.7%** |
| 60 s     | 244.4 m | 161.3 m |   +83.2 m |  **+34.0%** |
| 120 s    | 536.6 m | 298.2 m |  +238.4 m |  **+44.4%** |

---

## Model & Training

| Config | Value |
|--------|-------|
| Architecture | 1-layer GRU → Linear → 2 outputs (ΔEast, ΔNorth) |
| Parameters | 4,866 |
| Input features | 16 (accel, gravity, gyro pitch, phone speed, vehicle velocity/heading/yaw-rate/steering, wheel speeds) |
| Context window | 20 samples (stride 5) |
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-4 |
| LR schedule | ReduceLROnPlateau (→ 5e-4 at epoch 52) |
| Early stopping | patience = 20 |
| Epochs trained | 60 (best val @ epoch 40) |
| Best val loss | **0.5702** |
| Batch size | 256 |
| Seed | 42 |
| Device | CUDA (GPU) |

## Data Split (segment-level, no leakage)

| Split | Segments | Windows | Rows |
|-------|----------|---------|------|
| Train | Seg0 + Seg1 | 9,631 | 48,187 |
| Val   | Seg2       | 8,551 | 42,775 |
| Test  | Seg3       |    97 |    501 |
| **Total** | —    | **18,279** | 91,463 |

Normalization (z-score) computed **from training windows only**:
feature means/std from 9,631 train windows; target mean
`[−0.508, +0.417] m/s`, target std `[6.10, 6.75] m/s`.

## Training Dynamics

- Train loss collapsed 0.906 → 0.038 (epoch 1 → 60), still decreasing slowly;
  suggests capacity is the limiting factor, not optimization.
- Validation loss bottomed at 0.570 @ epoch 40; fluctuated between 0.57–0.72
  afterwards (best checkpoint saved).
- Early stopping triggered at epoch 60 (patience 20) — by design.

## Evaluation Protocol

- 39 blackout windows, matched to `classical_dr_baseline.py` criteria
  (durations 10 / 30 / 60 / 120 s; only windows fully contained within a
  segment).
- Causal: each step's GRU input uses only sensor data up to (and including)
  that timestep; +1.81 s inter-stream offset is **not** applied during
  propagation.
- A0 baseline reproduced identically inside the eval (same windows, same
  metric) — comparison is apples-to-apples.
- Metrics: MAE and RMSE of position error over the blackout period.

## Artifacts (in `outputs/ml/`)

| File | Description |
|------|-------------|
| `best_model.pt` | Trained GRU checkpoint (epoch 40, val 0.5702) |
| `normalization.npz` | Z-score stats from training data |
| `train_config.json` | Reproducible training config |
| `training_log.csv` | Per-epoch train/val loss + lr |
| `training_curves.png` | Loss curves |
| `ml_evaluation_report.txt` | ASCII results table |
| `ml_error_vs_duration.png` | Error vs blackout duration |
| `ml_improvement.png` | Improvement vs A0 |
| `ml_position_error_{10,30,60,120}s.png` | Error over time per duration |
| `ml_trajectory_{10,30,60,120}s.png` | Sample trajectory vs A0 vs reference |
| `ml_velocity_residuals_30s.png` | Example velocity residual fit |

## Notes & Caveats

1. **Baseline metric difference:** A0 MAE here (e.g. 49.8 m @ 10 s) differs
   from the classical report's headline numbers (83 m @ 10 s) because the
   eval reports mean error over the whole blackout, not a single end-point
   value. The A0 vs ML comparison within this report is internally consistent.
2. **Rollout mode:** this evaluation uses full-context batched inference per
   blackout (teacher-style). A fully recursive / deployment-like rollout
   (predicted states fed back as inputs) is beyond V0 scope.
3. **Frame mapping** (phone vs vehicle EN frame) remains unresolved; the
   2-D EN velocity target is thus an empirical construct.
4. **Not integrated** with the rejected calibration ablation (A1/A2) — A0
   remains the sole baseline.
5. This report reflects the completed Kaggle run at commit `c4d4deb`
   (device-mismatch fix included).