# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repository serves an Ocean Engineering journal paper about an open-mouth underwater platform for inflatable flexible corner-reflector deployment. All code, experiments, figures, and writing should support the paper's central storyline:

1. A miniature mission-oriented underwater deployment platform carries an inflatable flexible corner reflector.
2. After near-horizontal release, the platform must passively transition to a near-vertical working attitude before fairing separation and reflector inflation.
3. The fairing-attached, reflector-uninflated transition stage is modeled with a hybrid longitudinal 3-DOF reduced-order dynamic model.
4. The model closes parameters through permeability-corrected added inertia, CFD-derived angle-of-attack maps, and pitch free-decay rotational identification.
5. Validation uses out-of-sample tank release experiments, a CFD-identified diagonal Fossen-type baseline, ablation studies, and feasible release-envelope analysis.

Do not treat this as a generic AUV/control repository. The work is for paper experiment analysis, figure generation, and manuscript drafting.

## Common commands

Use PowerShell from the repository root on Windows.

Environment setup:

```powershell
conda activate pytorchlearning
python -m pip install -r sim_flip/requirements.txt
```

Run all tests:

```powershell
python -m unittest discover -s sim_flip/tests -v
```

Run a single test module:

```powershell
python -m unittest sim_flip.tests.test_dynamics_sanity -v
```

Free-decay identification pipeline, recommended two-pass workflow:

```powershell
python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
```

Evaluation against held-out split tags:

```powershell
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
```

Sensitivity analysis and paper figure data:

```powershell
python sim_flip/scripts/run_sensitivity_suite.py
python sim_flip/scripts/build_paper_figures.py
```

CFD table review/processing:

```powershell
python sim_flip/scripts/run_cfd_review_pipeline.py --source-dir <path>
```

Before a formal pipeline rerun, old derived data may need manual cleanup as described in `sim_flip/PIPELINE_OPERATION_GUIDE.md`; do not delete data unless the user explicitly approves.

## Repository structure

- `sim_flip/` is the main implementation and operation center for the fixed identification pipeline used by the manuscript.
- `sim_flip/PIPELINE_OPERATION_GUIDE.md` is the authoritative operation handbook for exact commands, outputs, and troubleshooting.
- `sim_flip/scripts/` contains entrypoints for identification, evaluation, sensitivity analysis, paper-figure generation, CFD review, and cable comparison.
- `sim_flip/analysis/` contains the pipeline stages: raw preprocessing, segmentation, frequency locking, Step 3 energy damping, Step 4 ODE fitting, bootstrap, metrics, and CFD processing.
- `sim_flip/src/` contains core modeling utilities: conventions, kinematics, dynamics, added-mass handling, and CFD coefficient-table support.
- `sim_flip/configs/experiment_manifest.csv` controls which segments enter train/val/test and CV folds.
- `sim_flip/configs/id_protocol.yaml` controls segmentation, identification, Step 3/Step 4, and bootstrap behavior.
- `sim_flip/data/raw/runs/` is the expected location for raw free-decay TXT runs named like `RYYYYMMDD_##.txt`.
- `sim_flip/data/derived/` stores generated run CSVs and segment CSVs.
- `sim_flip/results/` stores identification and evaluation outputs.

At the time this file was created, many manuscript/document files appeared deleted in git status. Be careful with any rollback, clean, reset, or harness operation until the user confirms the intended repository state.

## Obsidian source-of-truth documents

When paper logic, parameter interpretation, or writing scope is unclear, consult the Obsidian vault before drafting or editing. Current durable source priorities are:

- `00_总控/主线总控.md`: highest-level paper storyline, theory boundaries, terminology, section guidance, and reviewer-risk rules.
- `02_研究内容1_姿态翻转/03_参数与操作/01_基础静态参数卡.md`: static parameter card, unit interpretation, implementation values, and unresolved parameter checks.
- `02_研究内容1_姿态翻转/02_辨识与实验/01_free_decay辨识主流程.md`: only valid Step0–Step4 free-decay identification method document.
- `02_研究内容1_姿态翻转/03_参数与操作/02_CFD重算操作指南.md`: STAR-CCM+ rerun data-delivery and provenance contract for AoA CFD coefficient cleanup.
- `01_论文主线/02_Section2信息清单.md`: confirmed platform facts, Section 2 boundaries, hardware modules, and remaining gaps.

Conflict rule: if a lower-level note conflicts with `00_总控/主线总控.md` on theory scope, terminology, or equations, follow the master-control note. For free-decay methodology, follow `01_free_decay辨识主流程.md` unless it conflicts with the master-control note's model boundary.

## Data and pipeline contracts

Raw free-decay TXT files must contain these columns:

- `angleX`
- `angleY`
- `gyroX`
- `gyroY`
- `time`

`time` is a millisecond tick and must be strictly increasing. One TXT should contain exactly one excitation followed by one free decay and stabilization period.

The identification workflow is manifest-driven:

1. `--prepare-only` generates/updates run CSVs, segment CSVs, and manifest rows.
2. The user labels `split_tag` and optionally `cv_fold` in `sim_flip/configs/experiment_manifest.csv`.
3. `--fit-only` uses the existing manifest and segment CSVs for global joint identification.

Filtering is by manifest fields, not by filename. New manifest rows default to `split_tag=train`, so avoid one-shot full runs when a condition-level split is required.

## Paper storyline and section plan

Recommended main paper structure:

1. Introduction
2. Platform realization and transition-stage problem
3. Transition-stage hybrid dynamic model
4. Parameter closure and identification
5. Validation, baseline comparison, ablation, and release envelope
6. Conclusions

The paper should be written as a mission-oriented platform and transition-stage reduced-order dynamics paper, not as a generic AUV 6-DOF model, pure CFD paper, acoustic scattering paper, conventional control paper, or hardware-only report.

Recommended title direction:

`A Hybrid Transition-Stage Dynamic Model for a Miniature Underwater Corner-Reflector Deployment Platform with an Open-Mouth Structure`

Use these contribution categories consistently:

1. Hardware/platform contribution: miniature underwater deployment platform with open-mouth detachable fairing and tail-heavy mass-buoyancy layout.
2. Hybrid modeling and parameter closure: permeability-corrected added inertia, CFD AoA maps, and free-decay rotational damping.
3. Validation and baseline comparison: CFD-identified diagonal Fossen-type baseline, out-of-sample tank tests, and ablation variants.
4. Engineering output: feasible release envelope for deployment guidance.

## Modeling scope and terminology

The main model covers only the fairing-attached, reflector-uninflated transition stage from near-horizontal release to near-vertical quasi-steady working attitude. It does not model fairing separation, reflector inflation, post-deployment depth control, full 6-DOF coupling, environmental waves/currents, or full fluid-structure interaction.

Preferred terminology:

- `inflatable flexible corner reflector`, not airbag or buoyancy bag.
- `deployment platform`, not generic AUV.
- `open-mouth structure` or `open-mouth fairing`.
- `detachable fairing` or `upper fairing`.
- `transition stage`.
- `passive transition` or `passive reorientation`.
- `hybrid reduced-order dynamic model`, not full Fossen model.
- `parameter closure`, not all-parameter identification.
- `CFD-derived AoA coefficient maps`, not hydrostatic CFD forces.
- `internal-water participation`, not rigidly entrapped water except when explicitly discussing the limiting case.
- `feasible release envelope`, not just a sensitivity plot.

## Current static parameter facts

Use these as current Obsidian-recorded values, but do not present unresolved values as final paper results without user confirmation:

- `m_dry = 2.55 kg`; `m_wet = 2.76 kg`; `m_water = 0.21 kg`.
- `I_yy,dry = 0.05741 kg m^2`; `I_water = 0.01119 kg m^2`.
- `X_dot_u,outer = -0.0806 kg`; `Z_dot_w,outer = -3.54 kg`; `M_dot_q,outer = -0.115 kg m^2`.
- `rho = 1000 kg/m^3`; `g = 9.81 m/s^2`; `A = 5.6745e-3 m^2`; `L = 0.625 m`.
- `x_b = 0.02535 m`, `z_b = 0`, with the body-frame origin defined at `CG`.
- The recorded `B = 2.551 kg` should be treated as buoyancy-equivalent mass; convert to force before using in equations: `B = 2.551 * 9.81 = 25.02531 N`.
- `W = 2.55 * 9.81 = 25.01550 N`, so the current record implies very weak positive buoyancy of about `0.00981 N`.
- Current implementation damping values in the static parameter card (`d_q ≈ 0.005021`, `d_q|q| ≈ 0.279837`) are not final manuscript values; the free-decay workflow currently reports Step4 values `mu_theta = 0.5837`, `d_q = 0.003599`, `d_q|q| = 0.355264`, `K_cable = 0.047737`, `I_id = 0.178831`.
- Current permeability initial guesses (`mu_x = 0.95`, `mu_z = 0.20`, `mu_theta = 0.60`) are modeling priors/initial values, not final identification conclusions.

Open parameter checks from Obsidian: confirm the buoyancy-equivalent interpretation of `B`, reference area `A`, reference length `L`, outer added-inertia derivation, and final damping units/condition IDs before final manuscript tables.

## Theory constraints for writing and code interpretation

Preserve these constraints when editing manuscript text, equations, analysis scripts, figure captions, and result summaries:

- Pitch free-decay tests identify pitch-related rotational dynamics only: `mu_theta`, `d_q`, `d_q|q|`, and the free-decay setup compensation `K_cable`.
- Pitch free-decay must not be claimed to identify translational parameters `X_dot_u^t` or `Z_dot_w^t`.
- Translational loads are represented by CFD-derived `C_X(alpha)` and `C_Z(alpha)` maps; translational added inertia requires directional added-inertia closure or bounded physical correction.
- `K_cable` is an experimental compensation term for the free-decay setup only and must not enter the real untethered release model.
- Do not add the analytical pitch Munk term `(X_dot_u^t - Z_dot_w^t) u w` to the release model when `Q A L C_m(alpha)` is already used; this would double-count large-angle translational pitching moment.
- CFD AoA maps are quasi-static hydrodynamic closures scaled by dynamic pressure, not hydrostatic forces.
- The passive transition baseline sets axial thrust `T = 0`; assisted transition is a possible extension, not the main validation model.
- Do not insert unconfirmed numerical results such as RMSE, transition-time boundaries, critical release velocities, or release-envelope limits. Use `[TBD]` or refer to future tables until results are confirmed.

## Canonical model framing

The Section 3 model should be introduced from the mission transition problem, not from a textbook Fossen derivation.

Use longitudinal 3-DOF state:

```text
nu = [u, w, q]^T
V = sqrt(u^2 + w^2)
alpha = atan2(w, u)
Q = 0.5 rho V^2
```

The successful transition criterion should be defined in Section 3.1 using a near-vertical angle window, pitch-rate threshold, and hold duration. Section 5 should cite this definition rather than redefining success.

The hybrid model partitions physics as follows:

- dry-body inertia and restoring structure from measurement/CAD/hydrostatics;
- permeability-corrected added inertia for internal-water participation;
- CFD AoA maps for large-angle translational loads and attitude-dependent pitch moment;
- free-decay identification for rotational damping;
- no analytical `uw`-type Munk pitch term when CFD `C_m(alpha)` is active.

## Section 2 confirmed platform facts

For platform-realization writing, use these confirmed facts unless newer Obsidian notes supersede them:

- The platform is horizontally launched by an AUV/ROV or carrier and leaves the launching constraint with near-horizontal attitude and nonzero initial velocity.
- The platform and corner reflector do not separate after deployment; the platform continues to provide gas pressure and depth/working support.
- Mission sequence: compact storage in carrier, horizontal launch, free passive transition, near-vertical quasi-steady state, fairing separation, reflector inflation, depth-near working stage.
- Tank tests are validation environments, not the mission scenario itself.
- The detachable fairing remains attached during the modeled transition stage; fairing separation and reflector inflation are outside the model.
- The hardware includes STM32 controller, IMU, depth sensor, electromagnetic fairing release, single axial thruster, status light, Bluetooth module, two-position three-way solenoid valve, pressure reducer, check valve, high-pressure gas bottle, pressure sensor, 3S battery, buck module, ESC, and ballast.
- The single thruster axis coincides with the body axis but is not the main transition driver; the main model keeps `T = 0`.
- The platform is tail-heavy, weakly positively buoyant, and uses CG-CB layout plus hydrodynamic loads for passive transition.
- The inflatable flexible corner reflector is in the head fairing and remains uninflated during transition; the gas bottle and ballast are near the tail.
- Section 2 needs a CG/CB/body-frame/working-attitude schematic because `x_b` produces restoring moment in the near-vertical working attitude and may otherwise confuse readers.
- The minimum evidence path for the 3-DOF longitudinal assumption is an IMU three-axis appendix figure showing roll/yaw amplitudes or energy are much smaller than pitch.

## Validation and paper claims

Validation must be condition-level, not random repeated-sample splitting. Separate calibration, out-of-sample validation, and boundary cases where possible.

Section 5 should evaluate:

- `theta(t)` trajectory;
- `q(t)` trajectory;
- transition time;
- final attitude;
- success/failure classification;
- false positives and false negatives, with false positives treated as more serious for deployment risk.

The CFD-identified diagonal Fossen-type baseline is a reasonable comparison method, not a strawman. The comparison should focus on the global validity of constant hydrodynamic derivatives for large-AoA open-mouth transition, not on ODE integration cost.

Ablations should include, when data/model state permits:

- dry model;
- fully entrapped model;
- CFD-identified diagonal Fossen baseline;
- no CFD AoA maps;
- no free-decay damping;
- analytical Munk term added;
- no directional surge/heave added-inertia closure.

The release envelope should scan at least `theta_0`, `u_0`, and `q_0`; `w_0` can be added when data supports it. Output success/failure maps, transition-time contours, overshoot contours, boundary cases, and deployment guidance.

## CFD rerun and coefficient-table evidence contract

For STAR-CCM+ AoA CFD cleanup, follow `02_CFD重算操作指南.md` rather than inventing data-processing assumptions.

Current CFD rerun context:

- Initial `0°–180°` AoA sweep used 5° steps; problematic separated region is concentrated in `85°–125°`.
- Keep original CSVs untouched; all reruns must use tagged filenames and provenance rows.
- Critical rerun angles: `85°`, `90°`, `100°`, and `125°`; `90°` and `125°` require URANS evidence anchors.
- Geometry baseline: robot diameter `85 mm`, length `625 mm`.
- Current STAR-CCM+ model baseline: surface remesher + polyhedral mesh + prism layers, SST `k-omega`, all-y+ wall treatment, overset conservation, constant-density liquid, 3D.
- When changing AoA in STAR-CCM+, rotate both the overset region and the `Body frame`; otherwise force/moment report directions are invalid.
- For `125°`, preserve two continuation branches (`120° -> 125°` and `130° -> 125°`) and do not manually choose one if they differ; PartB/URANS should adjudicate.
- URANS standard: preferably start from steady solution, use `dt = 0.002 s`, run at least 15 estimated cycles, discard first 5 cycles, average last 10 cycles.
- Required monitor columns include `Cx Monitor`, `Cz Monitor`, and `Cm Monitor`; URANS also needs physical time or a recorded timestep.
- Required provenance file: `rerun_provenance_log_manual.csv` with angle, run type, variant tag, seed source, pseudo-transient/URF notes, timestep, total iterations/time, discard/average windows, monitor file, residual file, sim filename, and comments.
- Do not use CFD rerun results in manuscript claims unless monitor CSV, residual CSV, and provenance are all traceable.

## Obsidian and project memory

The user wants to use the Obsidian vault at:

```text
C:\Users\22296\iCloudDrive\iCloud~md~obsidian\Obsidian_git\Thsis
```

as the human-readable project memory repository for research notes, experiment records, figure provenance, decisions, and manuscript planning. Claude memory should only keep durable entry points and collaboration preferences; detailed research state should live in Obsidian Markdown when requested.

Suggested Obsidian organization:

- `00_Project_Index.md`
- `01_Research_Questions/`
- `02_Literature/`
- `03_Experiments/`
- `04_Data_and_Methods/`
- `05_Figures_and_Tables/`
- `06_Manuscript/`
- `07_Decisions/`
- `08_TODO_and_Review/`
- `_Templates/`

## Harness usage

Harness files are initialized in this repository and ignored by git. Use harness for recoverable, objectively validated tasks such as running experiment pipelines, generating figures, checking manuscript consistency, or applying section-specific revisions.

Do not run harness on broad subjective tasks like “improve the whole paper” without splitting them into small tasks with validation commands. Because harness rollback can involve `git reset --hard` and `git clean -fd`, confirm git state and user intent before any harness run that could discard uncommitted or untracked work.

Good harness tasks for this project should specify:

- target paper section or experiment matrix;
- input files and Obsidian notes;
- required changes;
- objective validation command, such as unit tests, identification/evaluation script, manuscript build, figure generation, or custom consistency check.
