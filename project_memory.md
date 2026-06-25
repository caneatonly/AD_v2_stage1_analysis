# Project Memory: Ocean Engineering Transition-Stage Study

Last updated: 2026-06-15

## Primary Working Documents

The project's main notes and paper-planning documents live in:

```text
C:\Users\22296\iCloudDrive\iCloud~md~obsidian\Obsidian_git\Thsis
```

The current AD_v2 / Ocean Engineering working center inside this vault is:

```text
C:\Users\22296\iCloudDrive\iCloud~md~obsidian\Obsidian_git\Thsis\Transition Stage
```

When project background, current writing direction, or progress needs to be understood, read the `Transition Stage` folder first. Highest-priority documents:

1. `Transition Stage\00_总控\主线总控-latest.md`
   - Current writing controller and single source of truth.
   - Supersedes older `docs/主线总控.md` and historical permeability-model notes.
2. `Transition Stage\00_总控\论文idea梳理 5.24.md`
   - 2026-05-24 pivot note.
   - Locks the change away from revolution-body empirical formulas and permeability correction.
3. `Transition Stage\00_总控\论文idea梳理 6.4.md`
   - 2026-06-04 reviewer-risk correction note.
   - Tightens contribution wording, quasi-steady boundary, Munk-term treatment, free-decay boundary, and Section 5 evidence requirements.
4. `Transition Stage\01_论文主线\section45证据矩阵.md`
   - Current Section 4/5 claim-evidence matrix.
   - Tracks the evidence gaps for parameter identification methods and results/discussion.
5. `Transition Stage\01_论文主线\Section4_4.1-4.3_AI写作说明.md`
   - External-AI handoff for Section 4.1--4.3 writing.
   - Records current CFD setup, steady angle-of-attack maps, forced-acceleration `m_x,m_z` values, evidence anchors, and writing boundaries.
6. `Transition Stage\辨识方法\全CFD常系数Fossen水动力模型基准构建方案文档.md`
   - Detailed all-CFD constant-coefficient Fossen baseline scheme.
   - Defines the fair comparison baseline against the proposed hybrid model.

Supporting current notes:

| Folder | Role |
|---|---|
| `Transition Stage\01_论文主线` | Section 4/5 structure, claim-evidence matrix, writing/evidence planning |
| `Transition Stage\StarCCM` | STAR-CCM+ steady angle-of-attack CFD setup and current `.sim` implementation notes |
| `Transition Stage\辨识方法` | `m_x,m_z` forced-acceleration CFD, all-CFD Fossen baseline, and free-decay identification workflows |
| `Transition Stage\成果性文件` | Reviewer-facing result/evidence notes such as parameter-source table and mesh-independence result |
| `Transition Stage\思路与审稿攻防` | Reviewer-risk arguments and modeling-boundary memos |

The vault-level `90_归档` folder and old root-level manuscript folders are historical context only. Do not treat them as current manuscript direction unless a current `Transition Stage` controller explicitly points back to them.

Repo files under `sim_flip/`, `docs/`, `drafts/`, and `figures/` are implementation/artifact context, but the current manuscript strategy is controlled by the `Transition Stage` Obsidian documents above.

## Current Position

The paper is now a **Fossen-based 3DOF hybrid modeling, hybrid CFD-experimental hydrodynamic parameter-identification, release-validation, model-comparison, release-envelope, and CG--CB layout-guidance paper** for a miniature underwater platform that deploys an inflatable flexible corner reflector.

The model boundary is the fairing-attached, reflector-uninflated transition stage: the platform is released near-horizontal and passively reorients toward a near-vertical working attitude before fairing separation and reflector inflation.

Current technical spine:

```text
mission-oriented miniature deployment platform
-> passive transition-stage problem
-> Fossen-based underactuated surge--heave--pitch 3DOF hybrid model
-> steady angle-of-attack CFD coefficient maps for large-angle quasi-static loads
-> CFD virtual captive / forced-acceleration tests for m_x and m_z
-> pitch free-decay tests for I_theta_eff, d_q, and d_q_abs
-> tank release experiments and validation + model comparison
-> feasible/robust release envelope and CG--CB layout guidance
```

The paper should not be framed as a full 6-DOF AUV paper, a full transient/free-running CFD paper, a control paper, an acoustic scattering paper, or a hardware-only engineering note.

## 2026-06-08 Obsidian Vault Restructure

The Obsidian project vault was reorganized for easier reading. The active AD_v2 working notes are now under:

```text
Thsis\Transition Stage
```

Current folder roles:

| Path | Current role |
|---|---|
| `Transition Stage\00_总控` | current controller and project idea pivots |
| `Transition Stage\01_论文主线` | Section 4/5 structure, content planning, and claim-evidence matrix |
| `Transition Stage\StarCCM` | STAR-CCM+ setup notes and `.sim` implementation state |
| `Transition Stage\辨识方法` | free-decay, `m_x,m_z` forced-acceleration CFD, and all-CFD baseline methods |
| `Transition Stage\成果性文件` | result/evidence notes intended to support the manuscript |
| `Transition Stage\思路与审稿攻防` | reviewer-risk and modeling-boundary memos |
| `90_归档` | archived historical notes, including deprecated permeability-route materials |

Practical routing rule:

- For current manuscript direction, read `Transition Stage\00_总控\主线总控-latest.md` first.
- For Section 4/5 work, read `Transition Stage\01_论文主线\section45证据矩阵.md`, `section4 内容构思.md`, `section5 内容构思.md`, and for Section 4.1--4.3 handoff read `Section4_4.1-4.3_AI写作说明.md`.
- For CFD operations, read `Transition Stage\StarCCM\当前SIM文件实现.md` and the relevant note under `Transition Stage\辨识方法`.
- For reviewer-facing evidence, read `Transition Stage\成果性文件` and `Transition Stage\思路与审稿攻防`.
- Treat old root-level paths and `90_归档` as historical unless explicitly reactivated by the current controller.

## 2026-06-09 Section 3-4 Terminology and Structure Revision

Current contribution terminology:

```text
hybrid CFD-experimental hydrodynamic parameter identification
```

This supersedes `source-partitioned hydrodynamic closure` as the public-facing manuscript contribution term. The underlying logic remains source-specific: different hydrodynamic and effective-inertia quantities are obtained from the evidence source that best isolates their physical contribution.

Section 3 / Section 4 boundary:

- Section 3 should formulate the Fossen-based 3DOF transition-stage model and place one framework figure at the end of the model section.
- Section 3 should not include a source-allocation / model-term allocation table.
- Section 4 should not contain a numbered `Source allocation` or `Identification roadmap` subsection.
- Section 4 should start with a short unnumbered bridge paragraph, then move directly into concrete CFD and free-decay methods.
- `Static oblique-flow simulations` is no longer the current section title. Use `Steady angle-of-attack CFD simulations` because the sweep includes axial, normal, and reverse-axial cases, not only oblique inflow.

Current Section 4 organization:

```text
4. Parameter identification
  [unnumbered bridge paragraph]
  4.1 CFD setup
  4.2 Steady angle-of-attack CFD simulations
  4.3 Forced-acceleration CFD simulations
  4.4 Pitch free-decay test
  4.5 Verification and uncertainty
  [closed parameter summary table]
```

Recommended bridge logic:

```text
Section 3 defines the transition-stage model and summarizes the source-specific role of each hydrodynamic quantity in a framework figure. Section 4 describes how these quantities are obtained and directly reports the parameter-closure outputs: common CFD setup, steady angle-of-attack CFD coefficient maps, forced-acceleration CFD for m_x,m_z, pitch free-decay for I_theta_eff,d_q,d_q_abs, and verification/uncertainty treatment.
```

## 2026-06-15 Section 4-5 Structure Lock

Current Section 4 and Section 5 organization:

```text
4. Parameter identification
  [unnumbered bridge paragraph]
  4.1 CFD setup
  4.2 Steady angle-of-attack CFD simulations
  4.3 Forced-acceleration CFD simulations
  4.4 Pitch free-decay test
  4.5 Verification and uncertainty
  [closed parameter summary table]

5. Results and discussion
  5.1 Release experiments and validation
  5.2 Model comparison
  5.3 Release-envelope and layout analysis

6. Conclusions
  include applicability and limitations in the final paragraph
```

Writing decisions:

- Use `Parameter identification` for Section 4. Use `hybrid CFD-experimental hydrodynamic parameter identification` as the public-facing contribution concept.
- Section 4 may report direct parameter-closure outputs: coefficient maps, `m_x,m_z`, free-decay in-sample fit and parameters, verification/uncertainty, and the final closed parameter table.
- Use `Results and discussion` for Section 5. Section 5 starts from out-of-sample release validation and then reports model comparison and engineering analysis.
- Do not keep a standalone `Identification results` subsection in Section 5; those values are closed in Section 4.
- Do not keep a standalone `Ablation study` subsection. Necessary model-variant checks, including no-AoA-map, no-free-decay-pitch-closure, no-translational-inertia, CAD-only pitch inertia, and analytical Munk-term variants, are folded into `5.2 Model comparison`.
- Do not keep a standalone `Operating range` subsection. The quasi-steady metric `k_q=|q|L/(2V)` is introduced methodologically in `4.2 Steady angle-of-attack CFD simulations`; numerical distributions and interpretation are reported in `5.1` or `5.2`, depending on whether they are tied to release-validation trajectories or model-comparison trajectories.
- Merge release envelope and CG--CB layout sensitivity into `5.3 Release-envelope and layout analysis`.
- Keep applicability and limitations in the final paragraph of Section 6, not as an extra Section 5 subsection.

## 2026-05-24 Main Pivot

The old route is discarded in the manuscript:

- revolution-body empirical formula;
- permeability correction model;
- `mu_x`, `mu_z`, `mu_theta`;
- internal-water participation factor;
- dry / fully entrapped limits as primary model variants;
- indirect derivation of pitch effective inertia through `mu_theta`;
- permeability model as contribution, baseline, or ablation.

The new manuscript variables are:

```text
m_x, m_z, I_theta_eff, d_q, d_q_abs, C_X(alpha), C_Z(alpha), C_m(alpha)
```

Parameter-source rule:

| Parameter | Source | Manuscript role |
|---|---|---|
| `C_X(alpha), C_Z(alpha), C_m(alpha)` | steady angle-of-attack CFD maps | large-angle quasi-static hydrodynamic loads |
| `m_x, m_z` | CFD virtual captive / forced-acceleration tests | effective translational inertia |
| `I_theta_eff` | pitch free-decay tests | directly identified effective pitch inertia |
| `d_q, d_q_abs` | pitch free-decay tests | residual pitch damping |
| `K_cable` | free-decay setup only | compensation term, excluded from real release simulation |

Steady angle-of-attack CFD simulations must not be described as identifying added mass or damping. Free-decay must not be described as identifying surge/heave parameters.

## Final Hybrid Model

The original diagonal-added-mass Fossen 3DOF model may be shown only as the reference model:

```text
m_x u_dot = -m_z w q - (d_u + d_u_abs |u|) u - (W-B) sin(theta) + T
m_z w_dot =  m_x u q - (d_w + d_w_abs |w|) w + (W-B) cos(theta)
I_theta q_dot = (m_z-m_x) u w - (d_q + d_q_abs |q|) q
                + B (z_b sin(theta) + x_b cos(theta))
```

The final proposed hybrid model is:

```text
m_x u_dot =
  -m_z w q + Q A C_X(alpha) - (W-B) sin(theta) + T

m_z w_dot =
   m_x u q + Q A C_Z(alpha) + (W-B) cos(theta)

I_theta_eff q_dot =
  Q A L C_m(alpha)
  - (d_q + d_q_abs |q|) q
  + B (z_b sin(theta) + x_b cos(theta))
```

For the passive-transition main model, use `T = 0`. Assisted transition with a surge thruster is only an extension.

Anti-double-counting rule:

- If the pitch equation uses `Q A L C_m(alpha)`, do not also include the analytical Munk-type pitch term `(m_z-m_x) u w`.
- The steady angle-of-attack CFD moment coefficient already contains the attitude-dependent pitching moment associated with large-angle translational motion.

## All-CFD Fossen Baseline

The all-CFD baseline is not full transient CFD and not free-running CFD. It is a low-order constant-coefficient Fossen-type ODE model whose hydrodynamic derivatives are obtained from numerical captive tests.

Use this as the fair baseline against the proposed hybrid model:

| Direction | CFD numerical test | Target parameters |
|---|---|---|
| Surge damping | steady surge towing | `d_u`, `d_u_abs` |
| Surge added mass | pure surge VPMM / forced acceleration | `X_udot`, `m_x` |
| Heave damping | steady vertical towing | `d_w`, `d_w_abs` |
| Heave added mass | pure heave VPMM / forced acceleration | `Z_wdot`, `m_z` |
| Pitch damping | rotating-arm / constant pitch-rate test | `d_q`, `d_q_abs` |
| Pitch added inertia | pure pitch VPMM | `M_qdot`, `I_theta` |

Important post-processing rule: in VPMM/forced-oscillation tests, subtract the already identified velocity-dependent damping contribution first, then estimate the acceleration-dependent added-mass or added-inertia derivative from the residual force/moment.

Baseline set for Section 5:

| ID | Model | Purpose |
|---|---|---|
| B1 | original constant-coefficient Fossen-type quadratic damping model | check constant-coefficient limitation |
| B2 | all-CFD low-order parameter identification | check fully CFD-identified low-order baseline |
| B3 | proposed CFD/free-decay hybrid model | main model |

## Model Comparison Variants

Delete old dry / fully entrapped / permeability-correction variants.

Current model-comparison variants may be used inside `5.2 Model comparison`; they should not form a standalone Section 5 ablation subsection:

| ID | Variant | Purpose |
|---|---|---|
| M0 | proposed hybrid model | main model |
| M1 | no CFD AoA maps | prove large-angle quasi-static maps are needed |
| M2 | no free-decay pitch closure | prove `I_theta_eff`, `d_q`, and `d_q_abs` need physical identification |
| M3 | analytical Munk term added | prove double counting worsens prediction |
| M4 | constant-coefficient Fossen baseline | prove constant coefficients are insufficient |
| M5 | no translational added inertia | prove `m_x`, `m_z` closure is needed |
| M6 | CAD-only pitch inertia | prove effective pitch inertia correction is needed |

## Release Envelope And Layout Guidance

Successful transition must be defined once in Section 3 and reused in Section 5. Candidate metrics:

- final attitude near `90 deg`;
- pitch rate decays below a threshold;
- transition completes within the allowed time;
- no excessive overshoot or failed reversal.

Envelope definition:

```text
E = { (theta0, u0, w0, q0) | successful transition }
```

Minimum scan variables:

- `theta0`;
- `u0`;
- `q0`;
- optional `w0`.

Required outputs:

- success/failure map;
- transition-time contour;
- maximum-overshoot contour;
- boundary cases;
- feasible release envelope;
- robust release envelope;
- deployment guidance table.

CG--CB layout guidance should scan or discuss:

- `x_b`;
- `z_b`;
- `B-W`;
- `I_theta_eff`;
- `d_q`, `d_q_abs`.

This is the engineering output of the paper, not a separate control study.

## Current Repo Audit

Checked on 2026-05-25:

1. `sim_flip/src/dynamics.py` already uses the hybrid-style pitch structure:
   - `q_dot = (M_cfd + M_damp + M_bg + M_cable + M_thruster) / eff.I_y`;
   - no analytical `(m_z-m_x) u w` pitch term is present in this active equation.
2. `sim_flip/data/coefficients_final_lookup.csv` is now the runtime default steady angle-of-attack lookup table.
   - It is generated from `static_aoa_data/Hydrodynamic_Coefficients_Corrected.csv` by `sim_flip/scripts/build_static_aoa_final_lookup.py`.
   - It preserves `alpha_deg,Cx,Cz,Cm` for the ODE and adds source, monitor, residual, raw-std, processing-status, and provenance fields for Section 4.2 evidence tracking.
   - `sim_flip/data/cfd_table_clean.csv` is deprecated as a historical intermediate table.
3. Three legacy segment CSVs still exist under `sim_flip/data/`.
4. The fixed manifest pipeline is still not populated:
   - `sim_flip/configs/experiment_manifest.csv` contains only the header;
   - `sim_flip/data/derived/run_csv/` contains only `.gitkeep`;
   - `sim_flip/data/derived/segments/` contains only `.gitkeep`;
   - `sim_flip/results/` contains only `.gitkeep`.
5. Several implementation/config files still reflect the old route:
   - `sim_flip/configs/params_nominal.yaml` still has `permeability.mu_x`, `mu_z`, and `mu_theta`;
   - `sim_flip/src/added_mass.py` still computes permeability-scaled added mass/inertia;
   - `sim_flip/configs/id_protocol.yaml` and `sim_flip/scripts/run_identification_cv.py` still use `mu_theta`;
   - `params_nominal.yaml` still has `cable.enabled: true`.

Therefore, the **writing mainline has moved ahead of the code/artifact pipeline**. Before using repo outputs as final evidence, align the scripts/configs with the v2 parameterization or clearly mark old outputs as historical diagnostics.

Last known test status from the previous audit: `python -m unittest discover -s sim_flip/tests -v` passed with 27 tests on 2026-05-20. This was not rerun during the 2026-05-25 memory update.

## Milestones

| ID | Milestone | Status | Evidence / blocker | Exit criterion |
|---|---|---|---|---|
| M0 | v2 storyline locked | Done | Obsidian `Transition Stage\00_总控\主线总控-latest.md`, `Transition Stage\00_总控\论文idea梳理 5.24.md`, `Transition Stage\00_总控\论文idea梳理 6.4.md` | old permeability route no longer used in manuscript |
| M1 | Platform/problem narrative | Partly done | `drafts/main5.8.pdf`, Section 2 notes | source images and final captions organized |
| M2 | Hybrid model equation | Partly done | `sim_flip/src/dynamics.py`, v2 controller notes | code, symbols, and Section 3 wording aligned |
| M3 | Steady angle-of-attack CFD maps | Mostly closed | `sim_flip/data/coefficients_final_lookup.csv`; `sim_flip/results/static_aoa_review/coefficients_final_lookup_metadata.json` | complete Fig. 8/Table 4 packaging; resolve or document missing `Residuals_15.csv` |
| M4 | Translational effective inertia `m_x,m_z` | Mostly closed | `mx_mz_calc/mx_data/mx_result_summary.csv`; `mx_mz_calc/mz_data/mz_result_summary.csv`; `mx_mz_calc/*endpoint_sensitivity.csv`; `mx_mz_calc/*inner_iteration_convergence.csv` | report `m_x=3.5932 kg`, `m_z=7.0351 kg`; either add acceleration-amplitude sensitivity or clearly state current single-acceleration boundary |
| M5 | Pitch free-decay identification | Needs refactor | old scripts still identify `mu_theta` | direct `I_theta_eff,d_q,d_q_abs` output |
| M6 | Manifest-driven pipeline | Not populated | manifest and derived/results folders empty | reproducible outputs under `sim_flip/results/` |
| M7 | Tank release validation | Notebook/legacy only | legacy segments exist, formal script outputs absent | condition-level train/validation/test metrics |
| M8 | All-CFD Fossen baseline | Designed, not implemented | `Transition Stage\辨识方法\全CFD常系数Fossen水动力模型基准构建方案文档.md` | B2 baseline coefficients and trajectory comparison |
| M9 | Model comparison variants | Not started | v2 variants defined | B1/B2/B3 and selected M0-M6 metrics folded into Section 5.2 |
| M10 | Release envelope and CG--CB guidance | Not started | no envelope sweep outputs | maps, contours, robust envelope, guidance table |
| M11 | Final paper writing | Partly unblocked for Section 4.1--4.3 | `Transition Stage\01_论文主线\Section4_4.1-4.3_AI写作说明.md` and updated `section45证据矩阵.md` | final figures/tables locked; Section 5 evidence still missing |

## Immediate Action Items

1. Refactor or clearly separate the old permeability/`mu_theta` implementation from the v2 manuscript path.
2. Change free-decay identification outputs from `mu_theta` to direct `I_theta_eff`, `d_q`, and `d_q_abs`.
3. Create a free-decay setup config where `K_cable` is allowed, and a release config where `K_cable = 0` or `cable.enabled = false`.
4. Package forced-acceleration CFD results for `m_x,m_z`: generate the missing `m_z` force/residual figures if needed, and either add acceleration-amplitude sensitivity or state the current single-acceleration boundary in Section 4.3/4.5.
5. Implement or document the all-CFD constant-coefficient Fossen baseline using the numerical captive-test scheme.
6. Convert the three legacy segment CSVs into the manifest-driven `derived/segments` workflow or regenerate canonical segments.
7. Produce script-generated validation outputs, not notebook-only evidence.
8. Build Section 5 in this order: release experiments and validation, model comparison, release-envelope and layout analysis.
9. Do not finalize abstract/results claims until the corresponding figures and tables are reproducibly generated.

## Terminology Guardrails

Use:

- inflatable flexible corner reflector;
- open-mouth fairing / fairing-attached transition stage;
- passive transition / passive reorientation;
- transition-stage dynamics;
- Fossen-based 3DOF hybrid model;
- CFD-derived angle-of-attack coefficient maps;
- CFD virtual captive / forced-acceleration tests;
- pitch free-decay test;
- effective pitch inertia;
- feasible release envelope;
- robust release envelope;
- CG--CB layout guidance.

Avoid in the final manuscript:

- buoyancy airbag;
- full hydrodynamic model;
- full 6-DOF model;
- full transient/free-running CFD as the baseline claim;
- hydrostatic CFD forces;
- permeability correction;
- internal-water participation factor;
- `mu_x`, `mu_z`, `mu_theta`;
- unverified numerical conclusions.

## 2026-06-09 Section 3 Framework-Figure Writing Decisions

This section supersedes the 2026-06-07 Section 3.3 wording package. The previous public-facing term `source-partitioned hydrodynamic closure` is now historical/internal only; do not use it as a title, abstract term, contribution term, or highlight term.

Current Section 3 decisions:

1. Use `hybrid CFD-experimental hydrodynamic parameter identification` as the public-facing contribution concept.
2. Section 3 should formulate the transition-stage 3DOF model, define the body-frame convention, define `V`, `\alpha`, and `Q`, and explain the source-specific identification logic at the model level.
3. Section 3 should end with one framework figure showing how each evidence source enters the final 3DOF model.
4. Do not include a source-allocation table or `Table~\ref{tab:term_allocation}` in Section 3.
5. The framework figure should communicate:
   - steady angle-of-attack CFD simulations provide `C_X(\alpha), C_Z(\alpha), C_m(\alpha)`;
   - forced-acceleration CFD simulations identify `m_x,m_z`;
   - pitch free-decay tests identify `I_theta_eff,d_q,d_q_abs`;
   - physical/CAD measurements provide geometry, mass, buoyancy, and CG--CB quantities.
6. Section 4 should not repeat this as a numbered overview. It should start with a short unnumbered bridge paragraph and then move directly to `4.1 CFD setup`.
7. Prefer `steady angle-of-attack CFD simulations`, `angle-of-attack-dependent coefficient maps`, and `coefficient maps`. Avoid `static oblique-flow simulations` as a current section title because the sweep includes axial, normal, and reverse-axial cases.
8. The quasi-steady paragraph should define the approximation and defer the validity check to results:
   - introduce `k_q=|q|L/(2V)` methodologically in Section 4.2;
   - report the numerical distribution and interpretation in Section 5.1 or 5.2.
9. The Munk-type discussion should be mechanistic:
   - the analytical `(m_z-m_x)uw` term is an inviscid constant-coefficient approximation;
   - the final model uses `QALC_m(\alpha)` to represent the actual-geometry, angle-of-attack-dependent quasi-steady pitching moment;
   - do not include both terms in the final pitch equation.

Status:

- These are current writing decisions and manuscript-text boundaries.
- Final manuscript prose should still be checked by PaperSpine after Section 3/4 text, figures, and equation/table consistency checks are updated.
