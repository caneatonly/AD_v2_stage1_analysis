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
| Fig. 2 | Body-axis definition, sign convention, and state interface | `paper/sections/02_system_and_mission_scenario.md` | `x_b/z_b` axes, positive `theta` and `q`, `nu=[u,w,q]^T`, `eta=[theta]`, `alpha=atan2(w,u)` | single clean schematic; include symbols table inset | conventions from `sim_flip/src/conventions.py` |
| Fig. 3 | Experimental condition matrix, repeat policy, and anti-leakage split strategy | `paper/sections/02_system_and_mission_scenario.md` | `theta_0`-`q_0` matrix (12-condition target), current available runs, train/val/test block split, leave-one-theta-level-out | heatmap or grid with markers for available/planned/repeated; add split legend | `sim_flip/configs/experiment_manifest.csv` + protocol |
| Fig. 4 | Model-term map from physical mechanisms to equation blocks and code interfaces | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | rigid-body terms, restoring terms, damping terms, permeability-corrected added mass/inertia, CFD table inputs | block-and-arrow map; left physical source, middle equation block, right code/config interface | `sim_flip/src/dynamics.py`, `sim_flip/src/added_mass.py`, `sim_flip/configs/params_nominal.yaml` |
| Fig. 5 | CFD credibility summary: mesh refinement and residual convergence | `paper/sections/04_cfd_to_model_mapping_and_validation.md` | mesh-independence trend, selected mesh rationale, residual histories | 2-panel: (a) coefficient vs mesh level, (b) residual vs iteration/time; annotate acceptance rule | CFD post-processing outputs |
| Fig. 6 | Full-AoA `C_X/C_Z/C_m` curves with high-AoA local enlargement | `paper/sections/04_cfd_to_model_mapping_and_validation.md` | full-angle coefficient curves and high-angle zoom (`75-90 deg` recommended) | 3+1 panels: three global curves + one zoom panel; consistent axis ranges | `sim_flip/data/cfd_table_clean.csv` |
| Fig. 7 | Pressure-center migration `X_cp(alpha)` with static stability-region annotations | `paper/sections/04_cfd_to_model_mapping_and_validation.md` | `X_cp` migration, stable/unstable region marks, transition-relevant critical angles | single curve with shaded regions and critical-angle callouts | CFD-derived CP data |
| Fig. 8 | Representative flow and pressure snapshots at `0/30/60/90 deg` AoA | `paper/sections/04_cfd_to_model_mapping_and_validation.md` | flow topology and pressure distribution evidence at key AoA points | 2x2 layout, same colorbar per field type, one explanatory annotation per panel | CFD fields from solver exports |
| Fig. 9 | Identification diagnostics: residuals, multi-start convergence, and bootstrap confidence intervals | `paper/sections/05_parameter_identification_free_decay.md` | residual distribution, multi-start scatter, bootstrap CI bars/violin | 3-panel diagnostics figure; include convergence success rate text box | `sim_flip/results/identification/global/*` |
| Fig. 10 | Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`) | `paper/sections/06_sim_real_validation.md` | trajectory overlays for representative runs, fixed ordering (in-sample then out-of-sample) | stacked subplots with matched time windows and legend style | `sim_flip/scripts/run_sim_real_eval.py` outputs |
| Fig. 11 | Time-domain error traces with zero-reference and RMSE bands | `paper/sections/06_sim_real_validation.md` | error series, zero line, RMSE band, optional phase-aligned comparison track | one row per representative condition; absolute and phase-aligned versions separated by style | sim-real evaluation artifacts |
| Fig. 12 | Condition-normalized metric heatmap and GapScore ranking | `paper/sections/07_sensitivity_and_design_implications.md` | normalized metrics and rank order across conditions | left heatmap + right ranking bars | `sim_flip/results/sim_real_eval/*` |
| Fig. 13 | Sensitivity envelopes for initial energy, `BG`, and damping uncertainty | `paper/sections/07_sensitivity_and_design_implications.md` | success/failure envelope maps, BG +/-30%, damping factor scan | multi-panel with shared style; include clear pass/fail boundaries | `sim_flip/scripts/run_sensitivity_suite.py` |
| Fig. 14 | Strategy surface `J(u_0, theta_off)` with feasible valley regions | `paper/sections/07_sensitivity_and_design_implications.md` | 3D or contour objective surface, valley (recommended policy) region | primary contour plot + optional 3D inset; mark chosen operation point | sensitivity sweep outputs |

## 3. In-Text Placeholder Policy

- Placeholders in section drafts must use the exact format:
  - `[Fig. X. <Title>. Insert here.]`
- Placeholder order must remain monotonic from Fig. 1 to Fig. 14.
- If a figure is split later (e.g., 10a/10b), keep the same root ID and update this document first.

## 4. Visual Design Rules (Frozen)

- Serif font family and consistent font size across all figures.
- Colorblind-safe palette.
- Thin major grid lines only.
- Remove top/right spines for 2D plots where appropriate.
- Subpanel tags must be `(a)`, `(b)`, `(c)` ...
- Downsampling is allowed only for rendering speed, never for metric computation.

## 5. Literature-Informed Rationale (Related Journal Papers)

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

## 6. Execution Checklist

- [ ] Generate raw plotting artifacts with script version and config snapshot.
- [ ] Verify unit consistency in labels (`deg`, `rad/s`, SI units).
- [ ] Verify split tags in legends (`in-sample`, `out-of-sample`).
- [ ] Check all placeholder IDs in manuscript sections match this file.
- [ ] Export print and editable source versions for each figure.
