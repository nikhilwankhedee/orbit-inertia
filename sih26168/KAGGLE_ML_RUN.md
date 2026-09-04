# Kaggle ML Run — Velocity Residual GRU (SIH26168)

## Setup

1. **Kernel type:** Python + GPU (T4 recommended)
2. **Dataset:** Upload `processed/S4_synced.csv` as a Kaggle dataset, OR mount from an existing dataset
3. **Source files:** Upload the `sih26168/src/` directory to Kaggle notebook

## Dataset Path

Adjust `--data-root` to match your Kaggle dataset mount path, e.g.:

```
/kaggle/input/sih26168/processed
```

## Commands

### Step 1 — Train (on GPU)

```bash
python /kaggle/working/sih26168/src/train_velocity_residual_gru.py \
    --data-root /kaggle/input/sih26168 \
    --context-len 20 \
    --stride 5 \
    --hidden-size 32 \
    --n-layers 1 \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 20 \
    --seed 42
```

Expected output in `outputs/ml/`:
- `best_model.pt` — trained checkpoint
- `normalization.npz` — z-score stats
- `training_log.csv` — per-epoch losses
- `training_curves.png` — loss plot
- `train_config.json` — hyperparameters

### Step 2 — Evaluate

```bash
python /kaggle/working/sih26168/src/evaluate_velocity_residual_gru.py \
    --model-path /kaggle/working/outputs/ml/best_model.pt \
    --norm-path /kaggle/working/outputs/ml/normalization.npz \
    --context-len 20 \
    --seed 42
```

Expected output in `outputs/ml/`:
- `ml_evaluation_report.txt` — text comparison table
- `ml_error_vs_duration.png` — A0 vs ML bar chart
- `ml_improvement.png` — percentage improvement chart
- `ml_position_error_{10,30,60,120}s.png` — per-duration error curves
- `ml_trajectory_{10,30,60,120}s.png` — trajectory comparison plots
- `ml_velocity_residuals_30s.png` — correction signal analysis

### Step 3 — Recursive (deployment-like) evaluation

```bash
python /kaggle/working/sih26168/src/evaluate_recursive_gru.py \
    --model-path /kaggle/working/outputs/ml/best_model.pt \
    --norm-path /kaggle/working/outputs/ml/normalization.npz \
    --context-len 20 \
    --seed 42
```

Expected output in `outputs/ml/recursive/`:
- `recursive_v0_report.txt` — full experiment report (12-question checklist)
- `recursive_comparison.csv` — per-window A0 vs teacher vs recursive metrics
- `plots/...` — position error vs time (per duration), teacher-vs-recursive,
  classical-vs-teacher-vs-recursive, recursive velocity error

> **Recursive = deployment-like.** During the blackout the vehicle-CAN and
> GNSS-derived feature channels are NOT read from the recording; they are
> replaced by recursively-maintained internal state (see
> `RECURSIVE_FEATURE_SUBSTITUTION` in the script). Same 39 windows as Step 2.

### One-shot (train + teacher eval + recursive eval)

```bash
python /kaggle/working/sih26168/src/run_all.py \
    --data-root /kaggle/input/sih26168
```

Runs Step 1 → 2 → 3 sequentially.

## Architecture

```
16 features → GRU (1 layer, hidden=32) → Linear(2)
```

- **4,866 parameters**
- Input: 20-sample context window (2 s at 10 Hz)
- Output: 2D EN velocity residual [Δve, Δvn] in m/s
- Target: v_reference_EN − v_classical_DR_EN

## Feature Set (16)

| # | Feature | Source |
|---|---------|--------|
| 1–3 | accel_x/y/z | Phone accelerometer |
| 4–6 | gravity_x/y/z | Phone gravity |
| 7 | gyro_pitch | Phone gyroscope |
| 8 | phone_speed | Phone GPS speed (km/h) |
| 9 | phone_acc | Phone GPS accuracy (m) |
| 10 | veh_velocity | Vehicle CAN (km/h) |
| 11 | veh_heading | Vehicle CAN (rad) |
| 12 | veh_yaw_rate | Vehicle CAN (rad/s) |
| 13 | steering | Vehicle CAN steering angle |
| 14–16 | whl_fl/fr/rl | Vehicle wheel speeds |

## Data Split

| Split | Segments | Samples | Purpose |
|-------|----------|---------|---------|
| Train | Seg 0 + Seg 1 | 9,631 | Model fitting |
| Val | Seg 2 | 8,551 | Early stopping / model selection |
| Test | Seg 3 | 97 | Final evaluation |

Segment-level split — no data leakage across segments.

## Key Design Decisions

1. **Velocity residual target:** Model learns Δv = v_ref − v_dr, not absolute velocity
2. **Causal evaluation:** No future data during blackout inference
3. **A0 as baseline:** Classical DR with raw gyro pitch (no calibration)
4. **Batched inference:** Context windows stacked for single forward pass per blackout
5. **Z-score normalization:** Fit on training data only

## Smoke Test (local, NOT scientific)

3-epoch training on CPU yielded:

| Duration | A0 MAE | ML MAE | Improvement |
|----------|--------|--------|-------------|
| 10 s | 49.8 m | 40.7 m | +18.2% |
| 30 s | 82.7 m | 61.7 m | +25.5% |
| 60 s | 244.4 m | 176.6 m | +27.8% |
| 120 s | 536.6 m | 367.5 m | +31.5% |

Full 200-epoch training on GPU is expected to improve significantly.

## Notes

- Phone GPS speed is in m/s despite "Kmh" label — auto-corrected by `ml_common.py`
- Gyro Pitch is radians/s, heading is degrees — auto-converted to radians
- +1.81 s offset is NOT used during training or DR propagation (evaluation alignment only)
- Segments separated by gaps >0.2 s — never integrate across segment boundaries
