# sim_flip

Fixed, script-first pipeline for:
- free-decay experiment preprocessing and segmentation,
- parameter identification and sim-real evaluation,
- CFD evidence review and full-angle coefficient table export.

This repository has moved from notebook-centric workflow to manifest/protocol-driven pipeline.

## Quick Links

- Operation handbook: `sim_flip/PIPELINE_OPERATION_GUIDE.md`
- Migration notes: `sim_flip/README_MIGRATION.md`
- Main protocol config: `sim_flip/configs/id_protocol.yaml`
- Main manifest: `sim_flip/configs/experiment_manifest.csv`

---

## 1. Scope

Current paper scope:
- horizontal launch to vertical stabilization stage,
- free-decay identification and validation,
- CFD as model-evidence layer (not standalone CFD novelty claim).

Out of scope:
- depth-hold closed-loop control implementation.

---

## 2. Data Contract

### Raw experiment files

- Directory: `sim_flip/data/raw/runs/`
- Naming: `RYYYYMMDD_##.txt`
- Required columns: `angleX angleY gyroX gyroY time`

### Derived files

- Run-level CSV: `sim_flip/data/derived/run_csv/{run_id}.csv`
- Run QC JSON: `sim_flip/data/derived/run_csv/{run_id}_qc.json`
- Segment CSV: `sim_flip/data/derived/segments/{segment_id}.csv`
- Segment metadata: `sim_flip/data/derived/segments/{run_id}_segments.json`
- Identification per-run artifacts: `sim_flip/results/identification/{run_id}/`

### Control files

- Protocol (single source of truth): `sim_flip/configs/id_protocol.yaml`
- Manifest (single source of truth): `sim_flip/configs/experiment_manifest.csv`

---

## 3. Core Pipeline Behavior

For each raw run txt:

1. Read raw signals.
2. PCA projection to principal oscillation axis.
3. Coordinate alignment to dynamics convention.
4. Resample to `dt=0.001s` (`fs=1000Hz`).
5. Savitzky-Golay smoothing.
6. Adaptive 4th-order Butterworth + `filtfilt` zero-phase filtering.
7. Differentiate `q(t)` to obtain `q_dot(t)`.
8. Auto lock-phase segmentation:
   - start at valley phase,
   - end by stability criterion.
9. Export segment CSV files.
10. Update manifest rows (auto-detected mode; can be overridden).

---

## 4. Split and CV Control

Manifest fields:
- `split_tag`: `train` / `val` / `test`
- `cv_fold`: fold label (e.g., `holdout_v1`, `fold_01`)

Script filtering:

- Identification uses:
  - `--fit-split-tag` (default `train`)
  - `--cv-fold` (optional)

- Sim-real evaluation uses:
  - `--eval-split-tags` (default `val,test`)
  - `--cv-fold` (optional)

Recommended anti-leakage rule:
- split by condition blocks, not random sample points.

---

## 5. CFD Runtime Contract (Important)

Runtime CFD table must be full-angle:
- File: `sim_flip/data/cfd_table_clean.csv`
- Columns: `alpha_deg,Cx,Cz,Cm`
- Coverage: strict `0..180` (default 5 deg step)
- `alpha_deg` strictly increasing

No runtime auto-completion from 0..90 is allowed.

---

## 6. Main Entrypoints

- Identification:
  - `python sim_flip/scripts/run_identification_cv.py`
  - Bootstrap is enabled when `bootstrap.n_boot > 0` in `sim_flip/configs/id_protocol.yaml`
- Sim-real evaluation:
  - `python sim_flip/scripts/run_sim_real_eval.py`
- Sensitivity:
  - `python sim_flip/scripts/run_sensitivity_suite.py`
- Figure export:
  - `python sim_flip/scripts/build_paper_figures.py`
- CFD review pipeline:
  - `python sim_flip/scripts/run_cfd_review_pipeline.py --source-dir <dir>`

---

## 7. Environment

Recommended (your current setup):

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt
```

Run test suite:

```powershell
python -m unittest discover -s sim_flip/tests -v
```

---

## 8. Repository Structure

```text
sim_flip/
  analysis/                     # fixed pipeline modules
  scripts/                      # CLI entrypoints
  src/                          # dynamics / CFD interpolator core
  configs/
    id_protocol.yaml
    experiment_manifest.csv
    params_nominal.yaml
  data/
    raw/runs/                   # raw txt inputs
    derived/run_csv/            # processed run outputs
    derived/segments/           # segment outputs
    cfd_table_clean.csv         # runtime CFD table (0..180)
  results/                      # identification/evaluation outputs
  tests/                        # unittest-based validation
```

---

## 9. Notes

- Legacy notebook files are kept for analysis/reference, but authoritative logic is in `sim_flip/analysis/`.
- Legacy script `sim_flip/scripts/sim_real_compare_multi.py` has been removed intentionally.
- For day-to-day execution and experiment operation details, use `sim_flip/PIPELINE_OPERATION_GUIDE.md`.
