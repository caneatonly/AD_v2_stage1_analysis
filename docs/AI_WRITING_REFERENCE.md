# AI Writing and Manuscript Governance Reference

> Authority: This is the project-wide writing governance document for AI agents working on the manuscript.
>
> Scope: frozen claims, narrative logic, notation/contracts, evidence rules, and source-of-truth workflow.
>
> Non-scope: detailed pipeline operations, figure specifications, and LaTeX build procedures (those live in their own canonical docs).

## 0. Quick Context

- Project: mission-oriented self-suspending underwater platform (horizontal launch -> vertical stabilization phase)
- Target journal: Ocean Engineering (Elsevier)
- Manuscript content source of truth: `paper/sections/*.md`
- LaTeX assembly workspace: `paper/latex/`
- Simulation/pipeline implementation: `sim_flip/`
- Parameter source of truth: `sim_flip/configs/params_nominal.yaml`
- Convention source of truth: `sim_flip/src/conventions.py`

## 1. Canonical Document Boundaries

Use this table to avoid duplicate documentation.

| Domain | Canonical document |
|---|---|
| Writing policy, frozen claims, notation, evidence contracts | `docs/AI_WRITING_REFERENCE.md` |
| Documentation hierarchy and routing | `docs/README.md` |
| Paper workspace layout and section entrypoints | `paper/README.md` |
| Figure IDs, placeholders, figure content requirements | `paper/figures/FIGURE_REQUIREMENTS.md` |
| LaTeX workspace and build instructions | `paper/latex/README.md` |
| Fixed pipeline operation, commands, outputs, troubleshooting | `sim_flip/PIPELINE_OPERATION_GUIDE.md` |
| `sim_flip` module overview and migration summary | `sim_flip/README.md` |

## 2. Frozen Manuscript Positioning

### 2.1 Working Title (Locked for drafting)

Design and Characterization of a Mission-Oriented Self-Suspending Underwater Platform with Anisotropic Permeability-Corrected Dynamics and Experimental Identification

### 2.2 Scope Boundaries (Locked)

| Boundary | Decision |
|---|---|
| Paper scope | Horizontal launch -> vertical stabilization phase only |
| Depth control | Future work only |
| Mission context | Underwater corner-reflector deployment may be stated explicitly |
| Sensitive details | Abstract mission-specific tactical details; keep dynamics and identification assumptions explicit |
| Conclusion scope | Limited to current geometry, mass distribution, and actuation strategy |
| Generalization | Only within the studied operating-condition family and current platform configuration |

### 2.3 Three Innovation Claims (for Abstract/Introduction)

1. Platform/system innovation: design, implementation, and experimental characterization of a mission-oriented self-suspending underwater platform.
2. Dynamics innovation: 3-DOF anisotropic permeability-corrected model with interpretable parameters `mu_x`, `mu_z`, `mu_theta`.
3. Engineering verification innovation: closed evidence chain from free-decay identification to out-of-sample validation to sensitivity/design guidance.

### 2.4 CFD Positioning (Locked)

CFD is a model-evidence bridge, not an independent novelty claim.

CFD provides:
- static hydrodynamic mappings `C_X(alpha)`, `C_Z(alpha)`, `C_m(alpha)`, `X_cp(alpha)`
- mechanism support and coefficient evidence for model-term replacement logic

CFD does not provide rotational damping; rotational damping is identified from experiments.

## 3. Narrative and Writing Rules

### 3.1 Core Narrative Logic (Locked)

Correct narrative:
- A specific deployment mission requires a dedicated platform.
- This platform must complete a horizontal-launch to vertical-stabilization transition.
- Therefore, transition dynamics must be modeled and experimentally validated to ensure deployment success.

Do not frame the paper as a generic claim that traditional platforms broadly fail in such transitions.

### 3.1.1 Platform-First Spine (Frozen)

The paper must read as an Ocean Engineering-style engineering story with **platform realization and method/evidence chain co-equal**, while **the platform is the narrative entry point**:

- Mission scenario and success criterion (corner-reflector deployment; horizontal release with initial velocity; vertical working posture).
- Platform engineering solution (design + fabrication/implementation + deployment workflow).
- Experimental evidence program and data governance (free-decay protocol, instrumentation, and anti-leakage splits).
- Interpretable reduced-order dynamics and closure logic (permeability-corrected inertial coupling; CFD as static closure evidence; damping from experiments).
- Identification, validation, and bounded design guidance (no-retuning validation; discrepancy-aware operating guidance).

Avoid an over-model-centric framing (e.g., “we propose a corrected model” as the main hook). The model and identification are essential, but they are presented as the **analysis and decision-support backbone for a purpose-built platform in a specific mission phase**.

### 3.2 Abstract Rules

- Mention underwater corner-reflector deployment explicitly.
- Explicitly state that a mission-oriented platform was **designed and fabricated/implemented** (high-level only).
- Present platform engineering and modeling/identification as co-equal contributions, with platform as the entry point.
- Keep abstract high-level: qualitative performance wording only (avoid dense metrics and pipeline micro-details).
- Keep abstract <= 250 words.
- Avoid references in abstract unless unavoidable.

### 3.3 Introduction Structure (5-paragraph logic)

1. Mission context and operational requirement
2. Why mission constraints demand a purpose-built platform (engineering gap)
3. Platform concept and realized system (what was built; what phase is targeted)
4. Why modeling/identification is required for this platform and phase (evidence chain; CFD role bounded)
5. Contributions, scope boundaries, and roadmap (paper spine)

### 3.4 Claim-to-Evidence Contract

Every high-level claim must map to a manuscript evidence object.

| Claim | Required evidence |
|---|---|
| Platform novelty | system architecture / fabrication figure |
| Dynamics interpretability | governing equations + parameter table |
| CFD-to-model logic | coefficient curves + pressure-center evidence |
| Identification reliability | diagnostics and uncertainty summaries |
| Practical validity | sim-real overlays + metric tables |
| Engineering implication | sensitivity maps and operating envelopes |

### 3.5 Writing Constraints

- Avoid unverified ``literature gap'' absolutes (e.g., ``no prior work''). If a gap must be stated, use careful, evidence-consistent phrasing such as ``to our knowledge'' / ``limited dedicated reports'' and ensure the Introduction provides verifiable citations.
- Do not imply universal generalization across all underwater platforms.
- Do not expose mission-sensitive details beyond approved abstraction.
- Keep symbols and sign conventions consistent with `sim_flip/src/conventions.py`.
- Use English labels in figures and SI units everywhere.
- Write as a journal paper, not a software manual (avoid code-path prose in manuscript body).

## 4. Manuscript Source-of-Truth Workflow

```text
paper/sections/*.md            <- AUTHORITATIVE manuscript content (edit here)
        -> paper/latex/sections/*.tex   <- LaTeX mirror/assembly inputs
        -> paper/latex/out/main.pdf      <- compiled manuscript output
```

Rules:
- Edit Markdown first.
- Keep LaTeX aligned with Markdown content.
- Do not silently diverge `.tex` content from the Markdown source.

## 5. Section Architecture and Current Status

| # | Section | Source file | Status |
|---|---|---|---|
| - | Nomenclature | `paper/sections/00a_nomenclature.md` | Draft complete |
| 0 | Abstract | `paper/sections/00_abstract.md` | Draft complete |
| 1 | Introduction | `paper/sections/01_introduction.md` | Draft complete |
| 2 | Platform design and implementation | `paper/sections/02_platform_design_and_implementation.md` | Draft (needs detail expansion) |
| 3 | Experimental setup and data governance | `paper/sections/03_experimental_setup_and_data_governance.md` | Draft (needs setup detail) |
| 4 | Dynamic model and hydrodynamic closure | `paper/sections/04_dynamic_model_and_hydrodynamic_closure.md` | Draft complete |
| 5 | Parameter identification and diagnostics | `paper/sections/05_parameter_identification_pipeline.md` | Draft (needs results/diagnostics) |
| 6 | Model validation against free-decay experiments | `paper/sections/06_model_validation_against_free_decay_experiments.md` | Skeleton |
| 7 | Parametric analysis and design guidance | `paper/sections/07_parametric_analysis_and_design_guidance.md` | Skeleton |
| 8 | Conclusions | `paper/sections/08_conclusions.md` | Draft |

Nomenclature placement policy:
- In the final manuscript, place nomenclature before Introduction.
- Markdown source remains `paper/sections/00a_nomenclature.md`.

## 6. Frozen Symbol and Parameter Contracts

### 6.1 Coordinate and State Conventions

Canonical source: `sim_flip/src/conventions.py`

- Body axes: `x_b` forward, `z_b` downward
- Pitch: nose-up positive
- Pitch rate: `q = dtheta/dt` about `+y_b`
- Angle of attack: `alpha = atan2(w, u)`
- Velocity state: `nu = [u, w, q]^T`
- Configuration state: `eta = [theta]`
- Simulation state vector: `[u, w, q, theta]^T`

### 6.2 Key Physical Parameters (baseline)

Canonical numerical source: `sim_flip/configs/params_nominal.yaml`

| Parameter | Symbol | Value | Unit |
|---|---|---:|---|
| Dry mass | `m_dry` | 2.55 | kg |
| Wet mass | `m_wet` | 2.76 | kg |
| Inner-water mass | `m_water,inner` | 0.21 | kg |
| Pitch inertia | `I_yy` | 0.05741 | kg*m^2 |
| Inner-water inertia | `I_water,inner` | 0.01119 | kg*m^2 |
| Buoyancy equivalent mass | `B_mass` | 2.55 | kg |
| Buoyancy offset | `x_b` | 0.02535 | m |

### 6.3 Permeability Parameters (Frozen interpretation)

- `mu_x`, `mu_z`, `mu_theta` in `[0, 1]`
- Near 0: weak coupling (high internal permeability)
- Near 1: strong coupling (inner water moves with rigid body)

Added-mass closure (conceptual contract):
- `X_udot_total = X_udot_outer - mu_x * m_water,inner`
- `Z_wdot_total = Z_wdot_outer - mu_z * m_water,inner`
- `M_qdot_total = M_qdot_outer - mu_theta * I_water,inner`

Important denominator rule:
- Translational effective inertia denominators use `m_dry` (not `m_wet`).

## 7. Equation Governance (Frozen)

These rules prevent double counting and code-manuscript mismatch.

- Retain required rigid-body / inertia / restoring terms.
- CFD supplies static hydrodynamic mappings, not rotational damping.
- Keep rotational damping as identified/theory term.
- If `C_m(alpha)` already captures static Munk-like effects, do not add an extra explicit Munk term.
- Keep symbol usage frozen via `conventions.py` across CFD, identification, and simulation.

## 8. Data, Validation, and Evidence Policy

### 8.1 Dataset and Split Rules (Frozen)

- Current baseline: 3 free-decay segments; expansion target: 12-condition matrix
- Split by condition blocks, not random samples
- Same condition must not appear in both train and validation/test
- Leave-one-theta-level-out validation is the preferred CV design
- Present in-sample evidence before out-of-sample evidence

### 8.2 Pipeline Operation Ownership

Do not duplicate commands or troubleshooting here.

Use:
- `sim_flip/README.md` for module entrypoints and migration summary
- `sim_flip/PIPELINE_OPERATION_GUIDE.md` for detailed commands, outputs, and troubleshooting

## 9. Figure, LaTeX, and Journal Compliance Ownership

### 9.1 Figure Planning

- Canonical figure list, placeholder rules, visual constraints: `paper/figures/FIGURE_REQUIREMENTS.md`
- Manuscript sections should only contain placeholders, not full figure requirements

### 9.2 LaTeX Workspace and Build

- Canonical LaTeX workspace/build instructions: `paper/latex/README.md`
- This document should not duplicate package lists or full build walkthroughs

### 9.3 Ocean Engineering Submission Compliance (Quick checklist)

Use this checklist as a writing gate; do not duplicate detailed journal text snapshots.

- Abstract <= 250 words, factual and standalone
- Keywords: 1-7
- Highlights: 3-5 bullets, each <= 85 characters (separate file)
- SI units and editable equations
- CRediT statement, competing interest, data availability, acknowledgements
- Bibliography complete and consistent with in-text citations (author-year style)

## 10. AI Agent Collaboration Rules (Repo-specific)

- Before editing manuscript prose, read this file and the target section file.
- Before editing figure placeholders or figure IDs, read `paper/figures/FIGURE_REQUIREMENTS.md`.
- Before editing LaTeX build behavior, read `paper/latex/README.md`.
- Before editing pipeline commands/docs, read `sim_flip/README.md` and `sim_flip/PIPELINE_OPERATION_GUIDE.md`.
- If a rule belongs to another doc, add a link instead of copying the rule here.
- When restructuring docs, update `docs/README.md` to keep the hierarchy current.

## 11. Near-Term Writing Priorities

1. Expand Section 2 with platform-design/fabrication detail and Table 1.
2. Expand Section 3 with experimental setup detail and Fig. 3 governance.
3. Complete Section 6 validation narrative, figures, and Table 2.
4. Complete Section 7 sensitivity/design guidance with bounded claims.
5. Finalize end-matter and verified bibliography (CRediT, data availability, declarations).

## 12. Governance Change Log

- 2026-02-24: Deduplicated this document. Moved detailed operational, figure, and LaTeX procedures to their domain-specific canonical docs and kept this file focused on writing governance.
