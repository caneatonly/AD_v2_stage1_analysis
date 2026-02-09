# Pipeline Migration Notes

Detailed operation handbook:
- `sim_flip/PIPELINE_OPERATION_GUIDE.md`

## Legacy to Fixed Pipeline Mapping

- Legacy notebook preprocessing (`dqcalc.ipynb`) -> `sim_flip/analysis/raw_preprocess.py`
- Legacy segmentation (`dqcalc.ipynb`) -> `sim_flip/analysis/segment_lock.py`
- Frequency locking -> `sim_flip/analysis/frequency.py`
- Step3 energy damping -> `sim_flip/analysis/id_step3_energy.py`
- Step4 ODE fit -> `sim_flip/analysis/id_step4_ode.py`
- Bootstrap -> `sim_flip/analysis/bootstrap.py`
- Sim-real metrics -> `sim_flip/analysis/metrics.py`
- CFD review notebook (`CFD_data_review.ipynb`) -> `sim_flip/analysis/cfd_pipeline.py`

## New Entrypoints

- `python sim_flip/scripts/run_identification_cv.py`
- `python sim_flip/scripts/run_sim_real_eval.py`
- `python sim_flip/scripts/run_sensitivity_suite.py`
- `python sim_flip/scripts/build_paper_figures.py`
- `python sim_flip/scripts/run_cfd_review_pipeline.py --source-dir <path>`

### Split / CV filtering

- Identification:
  - `--fit-split-tag train` (default)
  - `--cv-fold <fold_id>` (optional)
- Sim-real evaluation:
  - `--eval-split-tags val,test` (default)
  - `--cv-fold <fold_id>` (optional)

## Data Contract

- Raw run files: `sim_flip/data/raw/runs/RYYYYMMDD_##.txt`
- Processed run CSV: `sim_flip/data/derived/run_csv/{run_id}.csv`
- Segment CSV: `sim_flip/data/derived/segments/{run_id}_S##.csv`
- Manifest: `sim_flip/configs/experiment_manifest.csv`
- Protocol: `sim_flip/configs/id_protocol.yaml`

## Removed Legacy Script

- `sim_flip/scripts/sim_real_compare_multi.py` is removed by design.
