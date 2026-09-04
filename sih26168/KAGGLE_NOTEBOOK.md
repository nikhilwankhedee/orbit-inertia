# Orbit-Inertia — Kaggle Notebook

Run this as a single notebook. It clones the code from GitHub (code only, no dataset)
and trains/evaluates on the S4 dataset you upload.

## 1. Upload the dataset

Create a Kaggle Dataset named e.g. `sih26168` containing:

```
sih26168/S4_synced.csv     ← the 37MB synchronized S4 file (required)
```

Alternate layout `sih26168/processed/S4_synced.csv` is also auto-detected.

## 2. Notebook cell (one cell, accelerator = GPU)

```python
# Clone code from GitHub (code only — no dataset in repo)
!git clone --depth 1 https://github.com/nikhilwankhedee/orbit-inertia.git

# Train + evaluate in one shot
!python orbit-inertia/sih26168/src/run_all.py \
    --data-root /kaggle/input/sih26168 \
    --epochs 200 \
    --batch-size 256 \
    --hidden-size 32 \
    --context-len 20 \
    --stride 5 \
    --seed 42
```

Replace `/kaggle/input/sih26168` with your actual dataset mount path
(it usually matches the dataset slug).

## 3. Expected outputs

After the run, `orbit-inertia/outputs/ml/` contains:

```
best_model.pt            trained GRU checkpoint
normalization.npz        z-score stats
training_log.csv         per-epoch train/val loss
training_curves.png      loss curves
ml_evaluation_report.txt A0 vs ML comparison table
ml_error_vs_duration.png A0 vs ML bar chart
ml_improvement.png       % improvement chart
ml_position_error_{10,30,60,120}s.png
ml_trajectory_{10,30,60,120}s.png
ml_velocity_residuals_30s.png
recursive/               recursive (deployment-like) rollout results:
  recursive_v0_report.txt  full 12-question experiment report
  recursive_comparison.csv per-window A0/teacher/recursive metrics
  plots/*.png              position error vs time + comparisons
```

`run_all.py` runs three steps: **1) train → 2) teacher-style evaluation →
3) recursive (deployment-like) evaluation**. The final cell prints the
A0-vs-ML error table; the recursive report/plots land in `outputs/ml/recursive/`.

## Options

To run training and evaluation separately (e.g. to tweak evaluation):

```python
!python orbit-inertia/sih26168/src/train_velocity_residual_gru.py \
    --data-root /kaggle/input/sih26168 --epochs 200 --batch-size 256

!python orbit-inertia/sih26168/src/evaluate_velocity_residual_gru.py \
    --data-root /kaggle/input/sih26168
```

## Requirements

- Python 3.10+
- torch, numpy, pandas, matplotlib, scikit-learn (all standard on Kaggle)
- Internet access (for the git clone) — Kaggle notebooks have it by default
