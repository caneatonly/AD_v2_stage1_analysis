# Figure Requirements and Placeholders

Date: 2026-02-12
Target journal style: Ocean Engineering-oriented engineering narrative

## 1. Purpose

This document freezes figure planning for manuscript drafting before full dataset expansion is complete. It defines:

- figure numbering and titles,
- in-text placeholder locations,
- required information content,
- visual design instructions,
- data/script ownership,
- and literature-informed rationale.

All figure labels in plots should use English.

## 2. Master Figure List

| ID | Figure title (manuscript) | In-text placeholder | Must show | Design notes | Data/script source |
|---|---|---|---|---|---|
| Fig. 1 | Mission-oriented platform architecture and deployment workflow | `paper/sections/01_introduction.md` | platform photo/CAD block, deployment sequence from horizontal launch to vertical stabilization, mission role abstraction | 2-panel layout: (a) hardware architecture, (b) mission workflow timeline; use arrows and stage labels | platform media + manual diagram; export via `sim_flip/scripts/build_paper_figures.py` |
| Fig. 2 | Body-axis definition, sign convention, and state interface | `paper/sections/02_system_and_mission_scenario.md` (Sec. 2.2) | `x_b/z_b` axes, positive `theta` and `q`, `nu=[u,w,q]^T`, `eta=[theta]`, `alpha=atan2(w,u)` | single clean schematic; include symbols table inset | conventions from `sim_flip/src/conventions.py` |
| Fig. 3 | Experimental condition matrix, repeat policy, and anti-leakage split strategy | `paper/sections/02_system_and_mission_scenario.md` | `theta_0`-`q_0` matrix (12-condition target), current available runs, train/val/test block split, leave-one-theta-level-out | heatmap or grid with markers for available/planned/repeated; add split legend | `sim_flip/configs/experiment_manifest.csv` + protocol |
| Fig. 4 | Model-term map from physical mechanisms to equation blocks and code interfaces | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | rigid-body terms, restoring terms, damping terms, permeability-corrected added mass/inertia, CFD table inputs | block-and-arrow map; left physical source, middle equation block, right code/config interface | `sim_flip/src/dynamics.py`, `sim_flip/src/added_mass.py`, `sim_flip/configs/params_nominal.yaml` |
| Fig. 5 | Free-body force and moment decomposition for the transition-phase model | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | body-frame force/moment components, hydrostatic and optional actuation contributions | vector diagram aligned with state and sign conventions; annotate positive directions | model equations + conventions |
| Fig. 6 | Conceptual interpretation of anisotropic internal-water coupling parameters (`mu_x`, `mu_z`, `mu_theta`) | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | directional coupling mechanism for surge/heave/pitch channels | conceptual sectional sketch with channel-specific coupling arrows | modeling assumptions + parameter definitions |
| Fig. 7 | CFD verification results: mesh-independence and residual convergence | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | mesh-independence trend, selected mesh rationale, residual histories | 2-panel: (a) coefficient vs mesh level, (b) residual vs iteration/time; annotate acceptance rule | CFD post-processing outputs |
| Fig. 8 | Hydrodynamic coefficient mappings versus attack angle (`C_X`, `C_Z`, `C_m`) | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | full-angle coefficient curves and high-angle detail | 3+1 panels: global curves + high-angle zoom; show raw marker support points | `sim_flip/data/cfd_table_clean.csv` |
| Fig. 9 | Pressure-center migration and static-stability-region transition versus attack angle | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | `X_cp` migration, stability-region marks, transition-relevant critical angles | single curve with shaded regions and critical-angle callouts | CFD-derived CP data |
| Fig. 10 | Representative pressure and velocity field patterns at selected attack angles | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | flow topology and pressure distribution evidence at key AoA points | multi-panel with consistent colorbar and selected-angle labels | CFD fields from solver exports |
| Fig. 11 | Parameter-identification workflow from segmented free-decay data to calibrated model parameters | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | preprocessing -> segmentation -> objective construction -> calibration chain | workflow diagram with data/parameter artifacts | `sim_flip/scripts/run_identification_cv.py` |
| Fig. 12 | Identification diagnostics and uncertainty summaries (residuals, multi-start convergence, bootstrap intervals) | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | residual distribution, multi-start scatter, bootstrap CI bars/violin | 3-panel diagnostics figure; include convergence success rate text box | `sim_flip/results/identification/global/*` |
| Fig. 13 | Validation dataset coverage and split topology across operating conditions | `paper/sections/04_model_validation_against_free_decay_experiments.md` (Sec. 4.1) | condition coverage, in/out split, fold structure, repeats per condition | matrix + split topology overlay; visually separate calibration/validation regions | `sim_flip/configs/experiment_manifest.csv` |
| Fig. 14 | Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`) | `paper/sections/04_model_validation_against_free_decay_experiments.md` (Sec. 4.2-4.3) | trajectory overlays for representative runs, fixed ordering and split tags | stacked subplots with matched time windows and legend style | `sim_flip/scripts/run_sim_real_eval.py` outputs |
| Fig. 15 | Time-domain trajectory-error traces with zero-reference and uncertainty bands | `paper/sections/04_model_validation_against_free_decay_experiments.md` (Sec. 4.3) | error series, zero line, RMSE/CI bands, optional phase-aligned comparison track | one row per representative condition; aligned y-axis limits | sim-real evaluation artifacts |
| Fig. 16 | Condition-wise validation metrics and GapScore ranking map | `paper/sections/04_model_validation_against_free_decay_experiments.md` (Sec. 4.4) | normalized and absolute metrics, ranking consistency across conditions | left heatmap + right ranking bars; include metric direction arrows | `sim_flip/results/sim_real_eval/*` |
| Fig. 17 | Discrepancy-mode decomposition for representative validation cases | `paper/sections/04_model_validation_against_free_decay_experiments.md` (Sec. 4.5) | phase mismatch, amplitude mismatch, terminal-bias decomposition | 3-panel error-mode decomposition with representative cases | discrepancy analysis artifacts |
| Fig. 18 | Operational success-envelope map in the initial-condition space | `paper/sections/05_parametric_analysis_and_design_guidance.md` (Sec. 5.1) | success/failure regions and transition-completion boundary | 2D map with pass/fail shading and boundary annotation | `sim_flip/scripts/run_sensitivity_suite.py` |
| Fig. 19 | Control-strategy objective surface `J(u_0, theta_off)` and feasible valley region | `paper/sections/05_parametric_analysis_and_design_guidance.md` (Sec. 5.2) | objective surface under actuation timing sweep and feasible valley | contour with valley highlight and optional 3D inset | strategy sweep outputs |
| Fig. 20 | Sensitivity envelopes under `BG` shift and damping uncertainty | `paper/sections/05_parametric_analysis_and_design_guidance.md` (Sec. 5.3) | envelope shift under `BG` and damping-factor variation | multi-panel envelopes with nominal operating point marker | `sim_flip/scripts/run_sensitivity_suite.py` |
| Fig. 21 | Integrated operating-guidance chart combining feasibility, robustness, and strategy preference | `paper/sections/05_parametric_analysis_and_design_guidance.md` (Sec. 5.4) | integrated recommendation map under bounded confidence region | decision chart combining envelope, sensitivity, and strategy results | synthesized from Sec. 5 outputs |

## 3. In-Text Placeholder Policy

- Placeholders in section drafts must use the exact format:
  - `[Fig. X. <Title>. Insert here.]`
- Placeholder order must remain monotonic from Fig. 1 to Fig. 21.
- If a figure is split later (e.g., 10a/10b), keep the same root ID and update this document first.

## 4. Visual Design Rules (Frozen)

- Serif font family and consistent font size across all figures.
- Colorblind-safe palette.
- Thin major grid lines only.
- Remove top/right spines for 2D plots where appropriate.
- Subpanel tags must be `(a)`, `(b)`, `(c)` ...
- Downsampling is allowed only for rendering speed, never for metric computation.

## 5. OE Figure and Table Compliance Rules

Apply the following journal-aligned rules to every final figure/table export:

- Captions in LaTeX must be self-contained: define symbols, panel roles, and key acceptance criterion when used.
- Axes must use explicit units and engineering notation, e.g. `Pitch rate, q (rad/s)` and `Angle of attack, alpha (deg)`.
- Validation graphics must separate in-sample and out-of-sample evidence clearly in panel layout and legends.
- Error graphics must include zero-reference baselines; uncertainty must be shown with CI bands/bars where available.
- CFD credibility figures must include both mesh-dependence evidence and convergence behavior.
- Sensitivity/strategy figures must show feasibility boundaries and recommended operating region marks.
- Tables must use booktabs style (`toprule/midrule/bottomrule`) and decimal-aligned numeric columns.

Common reviewer-triggered failures to avoid:

- No explicit uncertainty representation in validation plots.
- Curves shown without visible data support markers in critical ranges.
- Captions that describe only what is drawn but not why the panel matters for the claim.
- Mixing validated region and extrapolated region without boundary annotation.

## 6. Literature-Informed Rationale (Related Journal Papers)

The following papers are used as style and evidence-structure references (not copied content):

1. Ocean Engineering guide for authors (concise, factual, engineering-focused presentation)  
   https://www.sciencedirect.com/journal/ocean-engineering/publish/guide-for-authors
2. Ocean Engineering (2023): Integrated dynamic modeling and analysis of a novel man-jellyfish autonomous underwater vehicle  
   https://www.sciencedirect.com/science/article/abs/pii/S002980182300571X
3. Applied Ocean Research (2021): Determination of hydrodynamic derivatives using CFD synthetic dynamic tests  
   https://www.sciencedirect.com/science/article/abs/pii/S014111872100016X
4. Applied Ocean Research (2023): Nonlinear dynamics of novel flight-style AUV, Part I/II (coefficient estimation + experimental validation chain)  
   https://www.sciencedirect.com/science/article/abs/pii/S0141118723002808
5. Ocean Engineering (2021): Verification and validation of hydrodynamic coefficients by CFD and model tests  
   https://www.sciencedirect.com/science/article/pii/S0029801821011666
6. Ocean Engineering (2021): CFD verification and validation under pitch free-decay motion  
   https://www.sciencedirect.com/science/article/abs/pii/S0029801821013329

How these references affect figure design:

- include geometry/coordinate figure early,
- provide explicit CFD credibility plots (not only final coefficients),
- use simulation-vs-experiment overlays as core evidence,
- report sensitivity and robustness plots as engineering implication layer.

## 7. Execution Checklist

- [ ] Generate raw plotting artifacts with script version and config snapshot.
- [ ] Verify unit consistency in labels (`deg`, `rad/s`, SI units).
- [ ] Verify split tags in legends (`in-sample`, `out-of-sample`).
- [ ] Check all placeholder IDs in manuscript sections match this file.
- [ ] Export print and editable source versions for each figure.
