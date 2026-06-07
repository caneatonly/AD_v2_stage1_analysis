# Project Memory: Ocean Engineering Transition-Stage Study

Last updated: 2026-05-27

## Primary Working Documents

The project's main notes and paper-planning documents live in:

```text
C:\Users\22296\iCloudDrive\iCloud~md~obsidian\Obsidian_git\Thsis
```

When project background, current writing direction, or progress needs to be understood, read this Obsidian folder first. Highest-priority documents:

1. `00_总控\主线总控-latest.md`
   - Current writing controller and single source of truth.
   - Supersedes older `docs/主线总控.md` and historical permeability-model notes.
2. `00_总控\论文idea梳理 5.24.md`
   - 2026-05-24 pivot note.
   - Locks the change away from revolution-body empirical formulas and permeability correction.
3. `02_研究内容1_姿态翻转\02_辨识与实验\all_cfd_fossen_baseline_scheme.md`
   - Detailed all-CFD constant-coefficient Fossen baseline scheme.
   - Defines the fair comparison baseline against the proposed hybrid model.

Repo files under `sim_flip/`, `docs/`, `drafts/`, and `figures/` are implementation/artifact context, but the current manuscript strategy is controlled by the Obsidian documents above.

## Current Position

The paper is now a **Fossen-based 3DOF hybrid modeling, CFD/free-decay parameter-closure, validation, release-envelope, and CG--CB layout-guidance paper** for a miniature underwater platform that deploys an inflatable flexible corner reflector.

The model boundary is the fairing-attached, reflector-uninflated transition stage: the platform is released near-horizontal and passively reorients toward a near-vertical working attitude before fairing separation and reflector inflation.

Current technical spine:

```text
mission-oriented miniature deployment platform
-> passive transition-stage problem
-> Fossen-based underactuated surge--heave--pitch 3DOF hybrid model
-> static CFD AoA maps for large-angle quasi-static loads
-> CFD virtual captive / forced-acceleration tests for m_x and m_z
-> pitch free-decay tests for I_theta_eff, d_q, and d_q_abs
-> tank release validation + baseline comparison + ablation
-> feasible/robust release envelope and CG--CB layout guidance
```

The paper should not be framed as a full 6-DOF AUV paper, a full transient/free-running CFD paper, a control paper, an acoustic scattering paper, or a hardware-only engineering note.

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
| `C_X(alpha), C_Z(alpha), C_m(alpha)` | static AoA CFD maps | large-angle quasi-static hydrodynamic loads |
| `m_x, m_z` | CFD virtual captive / forced-acceleration tests | effective translational inertia |
| `I_theta_eff` | pitch free-decay tests | directly identified effective pitch inertia |
| `d_q, d_q_abs` | pitch free-decay tests | residual pitch damping |
| `K_cable` | free-decay setup only | compensation term, excluded from real release simulation |

Static AoA CFD must not be described as identifying added mass or damping. Free-decay must not be described as identifying surge/heave parameters.

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
- The static AoA CFD moment coefficient already contains the attitude-dependent pitching moment associated with large-angle translational motion.

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
| B2 | all-CFD low-order parameter closure | check fully CFD-closed low-order baseline |
| B3 | proposed CFD/free-decay hybrid model | main model |

## Ablation Plan

Delete old dry / fully entrapped / permeability-correction ablations.

Current ablation variants:

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
2. `sim_flip/data/coefficients_final_lookup.csv` is now the runtime default static AoA lookup table.
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
| M0 | v2 storyline locked | Done | Obsidian `00_总控\主线总控-latest.md`, `论文idea梳理 5.24.md` | old permeability route no longer used in manuscript |
| M1 | Platform/problem narrative | Partly done | `drafts/main5.8.pdf`, Section 2 notes | source images and final captions organized |
| M2 | Hybrid model equation | Partly done | `sim_flip/src/dynamics.py`, v2 controller notes | code, symbols, and Section 3 wording aligned |
| M3 | Static CFD AoA maps | Mostly closed | `sim_flip/data/coefficients_final_lookup.csv`; `sim_flip/results/static_aoa_review/coefficients_final_lookup_metadata.json` | complete Fig. 8/Table 4 packaging; resolve or document missing `Residuals_15.csv` |
| M4 | Translational effective inertia `m_x,m_z` | Not closed | requires CFD virtual captive / forced-acceleration tests | `m_x,m_z` values with uncertainty or sensitivity justification |
| M5 | Pitch free-decay identification | Needs refactor | old scripts still identify `mu_theta` | direct `I_theta_eff,d_q,d_q_abs` output |
| M6 | Manifest-driven pipeline | Not populated | manifest and derived/results folders empty | reproducible outputs under `sim_flip/results/` |
| M7 | Tank release validation | Notebook/legacy only | legacy segments exist, formal script outputs absent | condition-level train/validation/test metrics |
| M8 | All-CFD Fossen baseline | Designed, not implemented | `all_cfd_fossen_baseline_scheme.md` | B2 baseline coefficients and trajectory comparison |
| M9 | Ablation study | Not started | v2 ablation variants defined | M0-M6 metrics and figures |
| M10 | Release envelope and CG--CB guidance | Not started | no envelope sweep outputs | maps, contours, robust envelope, guidance table |
| M11 | Final paper writing | Blocked by evidence | Section 4-5 outputs missing | final figures/tables locked |

## Immediate Action Items

1. Refactor or clearly separate the old permeability/`mu_theta` implementation from the v2 manuscript path.
2. Change free-decay identification outputs from `mu_theta` to direct `I_theta_eff`, `d_q`, and `d_q_abs`.
3. Create a free-decay setup config where `K_cable` is allowed, and a release config where `K_cable = 0` or `cable.enabled = false`.
4. Design or run minimum CFD virtual captive / forced-acceleration cases for `m_x` and `m_z`.
5. Implement or document the all-CFD constant-coefficient Fossen baseline using the numerical captive-test scheme.
6. Convert the three legacy segment CSVs into the manifest-driven `derived/segments` workflow or regenerate canonical segments.
7. Produce script-generated validation outputs, not notebook-only evidence.
8. Build Section 5 in this order: validation, B1/B2/B3 baseline comparison, M0-M6 ablation, release envelope, CG--CB layout guidance.
9. Do not finalize abstract/results claims until the corresponding figures and tables are reproducibly generated.

## Terminology Guardrails

Use:

- inflatable flexible corner reflector;
- open-mouth fairing / fairing-attached transition stage;
- passive transition / passive reorientation;
- transition-stage dynamics;
- Fossen-based 3DOF hybrid model;
- CFD-derived AoA maps;
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

## 2026-06-07 Section 3.3 Writing Decisions

Current writing focus: Section 3.3, now framed as `Source-partitioned hydrodynamic closure for the transition-stage 3DOF model`.

Key decisions from the writing discussion:

1. Use `Source-partitioned hydrodynamic closure for the transition-stage 3DOF model` rather than `Source-partitioned 3DOF model for transition stage`.
   - The model itself is still a Fossen-based 3DOF transition model.
   - What is source-partitioned is the hydrodynamic / parameter closure, not the full governing model.
2. Avoid using `AoA` before definition.
   - In main prose, first write `angle of attack` or `angle-of-attack-dependent`.
   - Define `\alpha` in the Section 3 coordinate/state definition area, after `u,w,q,\theta` and the body-fixed frame are introduced:
     `V=\sqrt{u^2+w^2}`, `\alpha=\operatorname{atan2}(w,u)`, and `Q=\tfrac{1}{2}\rho V^2`.
   - Section 4 should later state that the static CFD sweep uses the same body-frame angle-of-attack convention.
3. The Section 3.3 opening should keep the constant-coefficient model as a fair low-order reference.
   - Do not make the paragraph sound like an attack on the baseline.
   - Do not introduce the Munk-type term too early or too heavily in the opening paragraph.
   - The opening should motivate closure mismatch through the wide variation of instantaneous angle of attack and speed during passive reorientation.
4. Candidate closing sentence for the opening paragraph:
   - `Accordingly, we propose a source-partitioned hydrodynamic closure within the 3DOF Fossen framework, in which the conventional constant hydrodynamic derivatives are replaced by angle-of-attack-dependent load coefficients and separately identified effective-inertia terms.`
5. In the source-allocation paragraph, use first-person method language where appropriate, but keep the paragraph about modeling logic rather than Section 4 execution.
   - Do not write as if the AoA maps are already being constructed in Section 3.
   - Prefer: `we use ... to represent`, `we identify ... from`, `we estimate ... from`.
   - Keep CFD map construction and interpolation details for Section 4.
6. Prefer `CFD-derived coefficient maps`, `angle-of-attack-dependent coefficient functions`, or `coefficient maps`.
   - Avoid the ambiguous shorthand `CFD maps` alone in formal manuscript prose.
7. The explanation after the equation/table should be concise and mechanistic.
   - Its purpose is to explain why the closure uses `QALC_m(\alpha)` in place of the analytical Munk-type term.
   - It should not drift into baseline/ablation logistics unless the surrounding paragraph specifically discusses model comparison.
   - Candidate text:
     `Among these allocations, the pitching-moment closure is the key modification from the constant-coefficient model. The Munk-type term (m_z-m_x)uw is an inviscid, constant-coefficient approximation of the moment induced by translational motion, whereas the present transition involves large-incidence flow around the actual fairing-attached geometry. We therefore represent this quasi-steady translational pitching moment by QALC_m(\alpha), so that the pitch moment is closed with the same angle-of-attack-dependent coefficient description as the surge and heave loads. Rate-dependent residual pitch effects are treated separately through the free-decay damping terms.`
8. The quasi-steady paragraph should define the approximation and defer the validity check to results.
   - Do not claim that `k_q` is already sufficiently small unless the result artifact exists.
   - Candidate text:
     `The use of C_X(\alpha), C_Z(\alpha), and C_m(\alpha) also defines the quasi-steady nature of the load closure. These coefficients represent the steady oblique-flow response at prescribed angles of attack, while the ODE model evaluates them at the instantaneous \alpha and V along the transition trajectory. The approximation is therefore appropriate only when the platform rotation is slow relative to the convective time scale of the incoming flow; wake-memory and phase-lag effects associated with rapid separated-flow evolution are not resolved by this representation. We quantify this condition using the reduced pitch rate k_q=|q|L/(2V), whose distribution along the transition trajectories is reported in Section~\ref{sec:results}.`
9. The current `Table~\ref{tab:term_allocation}` draft has a structural problem.
   - The header has four columns: `Model term`, `Reference closure`, `Proposed closure`, and `Evidence source`.
   - The first three rows must also have four cells.
   - Example row structure:
     `Surge force & Constant-coefficient damping, -(d_u+d_{u|u|}|u|)u & AoA map, QAC_X(\alpha) & Static AoA CFD \\`

Status:

- These are writing decisions and manuscript-text candidates; `draft/section3.tex` has not yet been edited in this memory update.
- Final manuscript prose should still be checked by PaperSpine after Section 3.3 is edited and after equation/table consistency checks are run.
