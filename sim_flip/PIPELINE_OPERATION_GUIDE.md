# Fixed Pipeline Operation Guide

This file is the practical handbook for:
- experiment operation,
- data saving rules,
- fast pipeline execution,
- train/val/test organization,
- supplemental experiment design for paper-quality evidence.

---

## 1. Goal and Scope

Current objective:
- Start from raw free-decay experiment txt files.
- Generate standardized run CSV and segment CSV automatically.
- Run identification/evaluation with manifest-driven split control.
- Produce reproducible outputs for paper figures and tables.

This guide is for the current paper scope:
- horizontal launch to vertical stabilization stage,
- free-decay based identification and sim-real validation.

---

## 2. Environment Setup (Conda: `pytorchlearning`)

From repo root (`C:\AD_v2_stage1_analysis`):

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt
```

Optional test tool:

```powershell
python -m pip install pytest
```

Sanity check:

```powershell
python -m unittest discover -s sim_flip/tests -v
```

---

## 3. Raw Experiment Data: How to Save

### 3.1 Directory

Put raw experiment files here:

`sim_flip/data/raw/runs/`

### 3.2 File naming (stable ID)

Each experiment record (one excitation -> one free-decay-to-stable process) is one txt file:

`RYYYYMMDD_##.txt`

Examples:
- `R20260209_01.txt`
- `R20260209_02.txt`

### 3.3 Raw txt column contract (required)

Raw file must contain exactly these physical channels:
- `angleX`
- `angleY`
- `gyroX`
- `gyroY`
- `time`

`time` must be monotonic increasing.

---

## 4. What the Pipeline Does Automatically

For each `R*.txt`, the pipeline executes:

1. Read raw txt (`angleX angleY gyroX gyroY time`).
2. PCA projection to principal oscillation axis.
3. Coordinate alignment to dynamics convention.
4. Resample to uniform grid (`dt=0.001 s`, `fs=1000 Hz`).
5. Savitzky-Golay smoothing.
6. Adaptive 4th-order Butterworth + `filtfilt` zero-phase filtering.
7. Numerical differentiation of `q(t)` -> `q_dot(t)`.
8. Auto lock-phase segmentation:
   - start: valley phase locking,
   - end: stability criterion.
9. Export segment CSV files.
10. Auto update `experiment_manifest.csv` with measured initial state and segment window.

---

## 5. Core Config Files You Will Edit

### 5.1 Protocol config

`sim_flip/configs/id_protocol.yaml`

Contains:
- resampling,
- S-G and Butterworth settings,
- segmentation parameters,
- Step3/Step4 settings,
- bootstrap options,
- gates and reproducibility seed.

### 5.2 Manifest

`sim_flip/configs/experiment_manifest.csv`

Minimum fields:
- `run_id`
- `segment_id`
- `repeat_id`
- `theta0_meas_deg`
- `q0_meas_rad_s`
- `t_start_s`
- `t_end_s`
- `segmentation_mode`
- `split_tag`
- `cv_fold`
- `notes`

---

## 6. Output Artifacts and Where They Go

For each run:

- Processed run CSV:
  - `sim_flip/data/derived/run_csv/{run_id}.csv`
- QC report:
  - `sim_flip/data/derived/run_csv/{run_id}_qc.json`

For each segment:

- Segment CSV:
  - `sim_flip/data/derived/segments/{segment_id}.csv`
- Segment metadata:
  - `sim_flip/data/derived/segments/{run_id}_segments.json`

Global:

- Manifest auto update:
  - `sim_flip/configs/experiment_manifest.csv`
- Identification outputs:
  - `sim_flip/results/identification/{run_id}/identified_params.json`
  - `sim_flip/results/identification/{run_id}/identified_params.yaml`
  - `sim_flip/results/identification/{run_id}/multistart_convergence.csv`
  - `sim_flip/results/identification/{run_id}/bootstrap_*.{json,csv,png}` (when `bootstrap.n_boot > 0`)
  - `sim_flip/results/identification/{run_id}/protocol_snapshot.yaml`
  - `sim_flip/results/identification/{run_id}/git_commit.txt`
  - `sim_flip/results/identification/{run_id}/python_env.txt`
  - `sim_flip/results/identification/identification_summary.csv`
- Sim-real evaluation outputs:
  - `sim_flip/results/sim_real_eval/`
- Sensitivity outputs:
  - `sim_flip/results/sensitivity/`

---

## 7. split_tag and cv_fold: Meaning and Use

### 7.1 `split_tag`

Dataset role tag for each segment:
- `train`: used for parameter fitting.
- `val`: used for model selection / tuning checks.
- `test`: used only for final unbiased report.

### 7.2 `cv_fold`

Fold label to organize different split plans.

Examples:
- `holdout_v1`
- `fold_01`
- `fold_02`

If you do fixed holdout, use one fold label for all rows.
If you do CV, use different manifest files or fold labels per experiment plan.

### 7.3 Current script behavior

- Identification script filters by:
  - `--fit-split-tag`
  - `--cv-fold` (optional)
- Evaluation script filters by:
  - `--eval-split-tags`
  - `--cv-fold` (optional)

---

## 8. Fast-Run Command Set (Recommended)

### 8.1 Train fit

```powershell
python sim_flip/scripts/run_identification_cv.py `
  --fit-split-tag train `
  --cv-fold holdout_v1
```

### 8.2 Validation evaluation

```powershell
python sim_flip/scripts/run_sim_real_eval.py `
  --eval-split-tags val `
  --cv-fold holdout_v1
```

### 8.3 Final test evaluation

```powershell
python sim_flip/scripts/run_sim_real_eval.py `
  --eval-split-tags test `
  --cv-fold holdout_v1
```

### 8.4 Sensitivity and figures

```powershell
python sim_flip/scripts/run_sensitivity_suite.py
python sim_flip/scripts/build_paper_figures.py
```

---

## 9. How to Organize train / val / test in Practice

Use condition-block split (anti-leakage):
- do NOT randomly split samples inside one condition.
- keep same condition and repeats in one split.

Recommended workflow:

1. Run pipeline once to auto-generate segments and manifest rows.
2. Open `experiment_manifest.csv`.
3. Assign each segment:
   - `split_tag` = train/val/test
   - `cv_fold` = e.g. holdout_v1
4. Re-run scripts with split filters.

---

## 10. Supplemental Experiments (Based on Current Paper Plan)

You currently have only 3 free-decay initial pitch conditions.
To reach robust evidence for the paper, use the following expansion.

### 10.1 Minimum condition matrix (P0)

Target 12 condition cells:

- `theta0` in `{5 deg, 30 deg, 55 deg}`
- `q0` in `{-0.5, 0.0, +0.5} rad/s`

This gives 9 cells; plus 3 anchor cells near your existing data, total 12.

### 10.2 Repeats

At least 2 repeats per condition cell.
Preferred total:
- 12 conditions x 2 repeats = 24 runs.

If budget is limited:
- Phase A: cover all 12 conditions with 1 repeat each.
- Phase B: add second repeats for high-impact cells (large theta0 and nonzero q0).

### 10.3 Suggested split strategy

Fixed holdout example:
- train: 8 condition cells
- val: 2 condition cells
- test: 2 condition cells

Keep all repeats of same condition in same split.

For stronger extrapolation claim:
- leave-one-theta-level-out protocol:
  - fold_01: hold out theta0=5 deg as val/test,
  - fold_02: hold out theta0=30 deg,
  - fold_03: hold out theta0=55 deg.

### 10.4 What to report after supplementation

At minimum report:
- per-segment frequency locking consistency,
- Step3/Step4 identified parameter distribution,
- in-sample vs out-of-sample sim-real metrics,
- GapScore ranking,
- bootstrap confidence intervals,
- sensitivity maps (BG, damping scale, initial envelope).

---

## 11. CFD Pipeline Usage (for model evidence layer)

Prepare CFD files in a source directory:
- `C_0_overmesh.csv`, `C_5_overmesh.csv`, ..., `C_180_overmesh.csv`
- each file contains iterative history columns: `Cx, Cz, Cm`

Run:

```powershell
python sim_flip/scripts/run_cfd_review_pipeline.py --source-dir <your_cfd_dir>
```

Rules:
- runtime table must be strict 0..180 coverage,
- steady value uses last 500 iterations mean,
- no interpolated fake truth points in final table.

---

## 12. Common Failure Modes and Quick Fixes

1. No segments detected:
- relax `segmentation.valley_prominence_deg`,
- check raw signal quality and units,
- manually override `t_start_s/t_end_s` in manifest.

2. Identification skipped:
- check if `split_tag/cv_fold` filter matches any segment.

3. Manifest split leakage:
- ensure one condition family does not appear in multiple splits.

4. CFD contract error:
- verify `cfd_table_clean.csv` covers 0..180 with fixed step and strict monotonic `alpha_deg`.

---

## 13. Daily Operation Checklist

1. Save raw run txt with stable run ID.
2. Run identification script once to generate derived files.
3. Edit manifest split tags and fold label.
4. Re-run train fit.
5. Run val/test evaluation.
6. Export figures and update paper artifacts.
7. Record protocol snapshot and commit hash for reproducibility.

---

## 14. Quick Start (single block)

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt

python sim_flip/scripts/run_identification_cv.py --fit-split-tag train --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
python sim_flip/scripts/run_sensitivity_suite.py
python sim_flip/scripts/build_paper_figures.py
```
