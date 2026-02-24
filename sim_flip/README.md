# sim_flip Module Documentation

This directory is the implementation and operation center for the fixed identification pipeline used by the manuscript.

## What this module does

- Raw free-decay data preprocessing and segmentation
- Manifest-driven global joint identification
- Simulation-vs-experiment evaluation
- Sensitivity analysis and paper-figure data generation
- CFD table review/processing pipeline support

## Documentation Hierarchy (inside `sim_flip/`)

- `sim_flip/README.md` (this file): module overview, entrypoints, migration summary
- `sim_flip/PIPELINE_OPERATION_GUIDE.md`: detailed operation handbook (commands, outputs, troubleshooting)

Project-level navigation lives in `docs/README.md`.

## Canonical Usage Rules

- Use `sim_flip/PIPELINE_OPERATION_GUIDE.md` for exact commands and troubleshooting.
- Keep this file short and index-oriented; do not duplicate the operation handbook.
- Pipeline-related manuscript policy belongs in `docs/AI_WRITING_REFERENCE.md` only when it affects claims/evidence wording.

## Entrypoints (summary)

Primary scripts:
- `python sim_flip/scripts/run_identification_cv.py`
- `python sim_flip/scripts/run_sim_real_eval.py`
- `python sim_flip/scripts/run_sensitivity_suite.py`
- `python sim_flip/scripts/build_paper_figures.py`
- `python sim_flip/scripts/run_cfd_review_pipeline.py --source-dir <path>`

Split/CV filtering (summary):
- Identification: `--fit-split-tag train` and optional `--cv-fold <fold_id>`
- Evaluation: `--eval-split-tags val,test` and optional `--cv-fold <fold_id>`

For exact invocation patterns and two-pass workflow, use `sim_flip/PIPELINE_OPERATION_GUIDE.md`.

## Data Contract (summary)

- Raw run files: `sim_flip/data/raw/runs/RYYYYMMDD_##.txt`
- Processed run CSV: `sim_flip/data/derived/run_csv/{run_id}.csv`
- Segment CSV: `sim_flip/data/derived/segments/{segment_id}.csv`
- Manifest: `sim_flip/configs/experiment_manifest.csv`
- Protocol: `sim_flip/configs/id_protocol.yaml`

Detailed output file list and semantics are documented in `sim_flip/PIPELINE_OPERATION_GUIDE.md`.

## Legacy Migration Summary (merged from former `README_MIGRATION.md`)

Legacy notebook workflow mapping:
- Legacy notebook preprocessing (`dqcalc.ipynb`) -> `sim_flip/analysis/raw_preprocess.py`
- Legacy segmentation (`dqcalc.ipynb`) -> `sim_flip/analysis/segment_lock.py`
- Frequency locking -> `sim_flip/analysis/frequency.py`
- Step3 energy damping -> `sim_flip/analysis/id_step3_energy.py`
- Step4 ODE fit -> `sim_flip/analysis/id_step4_ode.py`
- Bootstrap -> `sim_flip/analysis/bootstrap.py`
- Sim-real metrics -> `sim_flip/analysis/metrics.py`
- CFD review notebook (`CFD_data_review.ipynb`) -> `sim_flip/analysis/cfd_pipeline.py`

Removed legacy script:
- `sim_flip/scripts/sim_real_compare_multi.py` (removed by design)

Migration note:
- The detailed fixed-pipeline operating procedure is now centralized in `sim_flip/PIPELINE_OPERATION_GUIDE.md` to avoid duplicate command and output documentation.
