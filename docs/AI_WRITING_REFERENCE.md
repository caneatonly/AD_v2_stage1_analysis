# AI Writing & Formatting Reference Document

> **Authority**: This is the single authoritative reference for any AI agent working on this project's manuscript writing, LaTeX formatting, and content governance. It supersedes and consolidates all scattered documentation.
>
> **Last updated**: 2026-02-15
>
> **Canonical location**: `docs/AI_WRITING_REFERENCE.md`

---

## 0. Quick Context

- **Project**: Design and characterization of a mission-oriented self-suspending underwater platform
- **Target journal**: **Ocean Engineering** (Elsevier, IF 5.5, CiteScore 8.4, acceptance rate ~31%)
- **Template**: `elsarticle.cls` with `[final,5p,times,twocolumn]` (two-column final format)
- **Citation style**: author-year (`elsarticle-harv.bst`)
- **Source of truth for content**: `paper/sections/*.md` (Markdown); `paper/latex/` mirrors Markdown
- **Source of truth for parameters**: `sim_flip/configs/params_nominal.yaml`
- **Source of truth for conventions**: `sim_flip/src/conventions.py`

---

## 1. Title, Scope & Innovation Claims (FROZEN)

### 1.1 Title (Locked)

**Design and Characterization of a Mission-Oriented Self-Suspending Underwater Platform with Anisotropic Permeability-Corrected Dynamics and Experimental Identification**

> Note: This title is 24 words — consider shortening to ~15 words before submission, but the current version is locked for drafting.

### 1.2 Scope Boundaries (Locked)

| Boundary | Decision |
|----------|----------|
| Paper scope | Horizontal launch → vertical stabilization phase ONLY |
| Depth control | Future work / outlook only |
| Mission context | "Underwater corner-reflector deployment" — explicit mention allowed |
| Sensitive details | Abstract mission-object tactical details; retain all dynamic/identification assumptions |
| Conclusion scope | Limited to current geometry, mass distribution, and actuation strategy |
| Generalization | Only within the defined operating-condition family and platform configuration |

### 1.3 Three Innovation Claims (for Abstract & Introduction)

1. **Platform/System Innovation**: Design, implementation, and experimental characterization of a novel mission-oriented self-suspending underwater platform — this is a first-class contribution, not a modeling appendage.
2. **Dynamics Modeling Innovation**: 3-DOF anisotropic permeability-corrected model with physically interpretable parameters $\mu_x, \mu_z, \mu_\theta$ representing directional internal-water coupling strength.
3. **Engineering Verification Innovation**: Closed evidence chain: free-decay identification → out-of-sample sim-real evaluation → sensitivity/risk envelope → engineering design implications.

### 1.4 CFD Positioning (Locked)

CFD serves as a **"model-evidence bridge"** — not an independent novelty claim. It provides:
- Static hydrodynamic mappings: $C_X(\alpha), C_Z(\alpha), C_m(\alpha), X_{cp}(\alpha)$
- Mechanism and coefficient evidence for model-term replacement logic
- CFD does NOT provide rotational damping (that comes from identification)

---

## 2. Narrative Architecture & Writing Rules

### 2.1 Core Narrative Logic

The correct narrative logic (user-corrected, locked):

> Because **underwater corner-reflector deployment** requires a dedicated platform → we designed this platform → this platform's mission workflow inherently includes a horizontal-launch to vertical-stabilization transition → therefore we study this transition's dynamics to ensure deployment success.

**DO NOT** write: "Traditional platforms generally fail in horizontal-to-vertical transitions." This is incorrect. The transition is important because THIS specific mission-designed platform must pass through it.

### 2.2 Abstract Rules

- Open with mission object explicitly: underwater corner reflector deployment
- Emphasize platform design/manufacturing as co-equal with modeling
- Use **qualitative performance wording only** — no numeric metrics in Abstract
- Dense quantitative reporting goes in Results sections
- Maximum 250 words (current draft: ~220 words ✓)
- Avoid references/citations in Abstract unless absolutely required

### 2.3 Introduction Structure (5 paragraphs)

1. Mission context and operational requirement
2. Platform design rationale and engineering innovation
3. Modeling gap and anisotropic permeability-corrected dynamics
4. Validation chain and empirical evidence
5. Three contributions + explicit scope boundaries + section roadmap

### 2.4 Claim-to-Evidence Contract

Every high-level claim MUST map to at least one evidence object in the manuscript:

| Claim | Required Evidence |
|-------|-------------------|
| Platform novelty | System diagram + fabrication/architecture figure |
| Dynamics interpretability | Equation set + parameter table |
| CFD-to-model logic | Coefficient curves and CP migration evidence |
| Identification reliability | CV/bootstrap diagnostics |
| Practical validity | In-sample/out-of-sample sim-real plots + metric tables |
| Engineering implication | Sensitivity maps and design envelopes |

### 2.5 Writing Constraints

- Do NOT imply universal generalization to all underwater platforms
- Do NOT expose mission-sensitive details beyond allowed abstraction
- Keep notation consistent with pipeline contracts (Section 4)
- Avoid premature hard quantitative claims until expanded experiments integrated
- Target audience: Ocean Engineering — engineering relevance + reproducibility + mechanism support
- Use English labels in all figures
- All parameters with units in SI

---

## 3. Manuscript Structure & Section Status

### 3.1 Section Architecture (Final — 7-section structure, updated 2026-02-15)

> **Change log**: Merged former Sec 5 (Identification) and Sec 6 (Validation) into a single
> "Parameter Identification and Model Validation" section; renumbered former Sec 7→6, Sec 8→7;
> added Nomenclature section before Introduction; moved Fig. 12 from Sensitivity to Validation.
>
> **Change log (title update)**: All section/subsection titles renamed to OE concise-descriptive
> style. Methodology names removed from headings; details expand on first use in body text.

| # | Section | Source file | Status | Notes |
|---|---------|------------|--------|-------|
| — | Nomenclature | `paper/sections/00a_nomenclature.md` | ✅ Draft complete | Latin/Greek/subscript/abbreviation tables |
| 0 | Abstract | `paper/sections/00_abstract.md` | ✅ Draft complete | Qualitative results, ~220 words |
| 1 | Introduction | `paper/sections/01_introduction.md` | ✅ Draft complete | 5-paragraph structure, 3 contributions |
| 2 | Platform description and deployment scenario | `paper/sections/02_system_and_mission_scenario.md` | ✅ Draft complete | Platform + conventions + experiment plan |
| 3 | Dynamic model formulation | `paper/sections/03_anisotropic_permeability_corrected_dynamics.md` | ✅ Draft complete | Full equations + parameter mapping |
| 4 | CFD-based hydrodynamic characterization | `paper/sections/04_cfd_to_model_mapping_and_validation.md` | ⚠️ Skeleton | Needs: credibility, coefficients, CP, flow snapshots |
| 5 | Experimental identification and model validation | `paper/sections/05_parameter_identification_and_model_validation.md` | ⚠️ Skeleton | Merged: protocol + diagnostics + sim-real + metrics |
| 6 | Parametric study and design implications | `paper/sections/06_sensitivity_analysis_and_design_implications.md` | ⚠️ Skeleton | Needs: envelopes, BG, damping, strategy |
| 7 | Conclusions | `paper/sections/07_conclusions.md` | ⚠️ Skeleton | Needs: summary + limitations + future work |

### 3.2 Required End-Matter Sections (in main.tex)

- Data availability statement (TODO)
- CRediT authorship contribution statement (TODO)
- Declaration of competing interest (template done)
- Acknowledgements (TODO)
- References (placeholder BibTeX entries only — need 30-50 real citations)

### 3.3 Nomenclature Placement (OE Convention)

Following Ocean Engineering standard practice (observed in ~90% of published papers),
Nomenclature is placed as a standalone section **before Introduction** in the final manuscript.
In the Markdown source tree, it is stored as `paper/sections/00a_nomenclature.md`.
In LaTeX, it will be rendered via `\section*{Nomenclature}` immediately after `\end{frontmatter}`,
before `\input{sections/01_introduction}`.

### 3.3 Source-of-Truth Workflow

```
paper/sections/*.md  ←── AUTHORITATIVE (edit here)
       ↓ (propagate)
paper/latex/sections/*.tex  ←── MIRROR (must reflect Markdown)
       ↓ (compile)
paper/latex/out/main.pdf
```

**Rule**: NEVER edit `.tex` files independently. Always edit `.md` first, then update `.tex`.

---

## 4. Symbol Conventions & Parameter Contracts (FROZEN)

### 4.1 Coordinate System

Defined in `sim_flip/src/conventions.py` (canonical source):

| Symbol | Convention |
|--------|-----------|
| Body axes | $x_b$ forward, $z_b$ downward |
| Pitch | Nose-up positive |
| Pitch rate | $q = d\theta/dt$ about $+y_b$ |
| Surge velocity | $u$ along $+x_b$ |
| Heave velocity | $w$ along $+z_b$ |
| Angle of attack | $\alpha = \text{atan2}(w, u)$ (degrees) |

### 4.2 State Interface

- Velocity state: $\nu = [u, w, q]^T$
- Configuration state: $\eta = [\theta]$
- Simulation state vector: $\mathbf{x} = [u, w, q, \theta]^T$
- Kinematic closure: $\dot{\theta} = q$

### 4.3 CFD Coefficient Convention

| Coefficient | Direction | Sign |
|-------------|-----------|------|
| $C_X$ | Along $+x_b$ | Standard |
| $C_Z$ | Along $+z_b$ | Standard |
| $C_m$ | About $+y_b$ | $C_m > 0$ = nose-up |

### 4.4 Key Physical Parameters (from `params_nominal.yaml`)

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Dry mass | $m_{dry}$ | 2.55 | kg |
| Wet mass | $m_{wet}$ | 2.76 | kg |
| Inner-water mass | $m_{water,inner}$ | 0.21 | kg |
| Pitch inertia | $I_{yy}$ | 0.05741 | kg·m² |
| Inner-water inertia | $I_{water,inner}$ | 0.01119 | kg·m² |
| Buoyancy equivalent mass | $B_{mass}$ | 2.55 | kg |
| Buoyancy offset | $x_b$ | 0.02535 | m |

### 4.5 Permeability Parameters

$$\mu_x, \mu_z, \mu_\theta \in [0, 1]$$

Physical interpretation:
- $\mu \approx 0$: weak coupling (high internal permeability, inner water moves freely)
- $\mu \approx 1$: strong coupling (inner water moves as lumped with rigid body)

Total added-mass formulation:
$$X_{\dot{u},total} = X_{\dot{u},outer} - \mu_x \cdot m_{water,inner}$$
$$Z_{\dot{w},total} = Z_{\dot{w},outer} - \mu_z \cdot m_{water,inner}$$
$$M_{\dot{q},total} = M_{\dot{q},outer} - \mu_\theta \cdot I_{water,inner}$$

Effective inertias (ODE denominators):
$$m_x = m_{dry} - X_{\dot{u},total}, \quad m_z = m_{dry} - Z_{\dot{w},total}, \quad I_y = I_{yy} - M_{\dot{q},total}$$

**Important**: Translational denominators use $m_{dry}$ (NOT $m_{wet}$).

---

## 5. Equation Governance Rules (FROZEN)

These rules prevent double-counting and ensure equation-code consistency.

| Rule | Description |
|------|-------------|
| **Conservation terms** | Retain Fossen-style rigid-body/inertia/Coriolis/restoring terms where physically required |
| **CFD provides** | Static hydrodynamic mappings via lookup tables: $C_X(\alpha), C_Z(\alpha), C_m(\alpha), X_{cp}(\alpha)$ |
| **Rotational damping** | Retained as identification/theory term: $(d_q + d_{q,abs}|q|)q$ — NOT from static CFD |
| **Non-duplication principle** | When $C_m(\alpha)$ already contains Munk moment effects, do NOT add an explicit theoretical Munk term |
| **Symbol freezing** | All symbols/coordinates are frozen via `conventions.py` — prevents CFD/identification/simulation inconsistency |

### 5.1 Governing Equations

Compact planar Fossen-style 3-DOF:

$$m_x \dot{u} - m_z wq = X_{cfd}(\alpha) - (W - B)\sin\theta + T$$
$$m_z \dot{w} + m_x uq = Z_{cfd}(\alpha) + (W - B)\cos\theta$$
$$I_y \dot{q} = M_{cfd}(\alpha) + M_{damp}(q) + M_{bg}(\theta) + M_{cable}(\theta,q) + M_{thruster}$$
$$\dot{\theta} = q$$

Where:
- $W = m_{dry} g$, $B = B_{mass} g$
- $V = \sqrt{u^2 + w^2}$, $Q = \frac{1}{2}\rho V^2$ (safeguard: $Q=0$ when $V < V_{eps}$)
- $X_{cfd} = Q A_{ref} C_X(\alpha)$, $Z_{cfd} = Q A_{ref} C_Z(\alpha)$, $M_{cfd} = Q A_{ref} L_{ref} C_m(\alpha)$
- $M_{bg}(\theta) = B(z_b \sin\theta + x_b \cos\theta)$
- $M_{damp}(q) = -(d_q + d_{q,abs}|q|)q$
- $M_{cable}(\theta,q) = -K_{cable}(\theta - \theta_{eq}) - C_{cable,q} q$ (when cable enabled)
- $T$ and $M_{thruster}$: set to zero for free-decay identification

### 5.2 Code-to-Equation Mapping

| Equation Term | Code Module | Config Source |
|---------------|-------------|---------------|
| State vector, ODE RHS | `sim_flip/src/dynamics.py` | — |
| $V, Q, \alpha$ computation | `sim_flip/src/kinematics.py` | — |
| $C_X, C_Z, C_m$ interpolation | `sim_flip/src/cfd_table.py` | `sim_flip/data/cfd_table_clean.csv` |
| Added-mass totals, effective inertias | `sim_flip/src/added_mass.py` | `params_nominal.yaml` |
| All physical parameters | — | `sim_flip/configs/params_nominal.yaml` |

---

## 6. Experimental Program & Data Governance

### 6.1 Current Data Status

- **Baseline**: 3 free-decay segments available
- **Expansion target**: 12 conditions (9 new needed)
- **Condition matrix**: $\theta_0 \in \{5°, 30°, 55°\}$ × $q_0 \in \{-0.5, 0, +0.5\}$ rad/s
- **Repeats**: ≥ 2 per condition

### 6.2 Anti-Leakage Rules (FROZEN)

| Rule | Detail |
|------|--------|
| Split method | Condition-block split (NOT random sample points) |
| Cross-set contamination | Same condition MUST NOT appear in both train and validation/test |
| Validation design | Leave-one-$\theta$-level-out cross-validation |
| Figure ordering | In-sample first, then out-of-sample |

### 6.3 Experimental Protocol

- One raw TXT file = one excitation → one free-decay → stabilization
- NEVER record multiple excitations in a single file
- File naming: `RYYYYMMDD_##.txt` in `sim_flip/data/raw/runs/`
- Required columns: `angleX, angleY, gyroX, gyroY, time` (millisecond ticks, strictly monotonic)

---

## 7. Identification & Evaluation Protocol

### 7.1 Pipeline Stages

```
Stage A: Signal preprocessing (raw_preprocess.py)
  → PCA projection → symbol anchoring → resampling → filtering → differentiation
  → Output: sim_flip/data/derived/run_csv/{run_id}.csv

Stage B: Automatic segmentation (segment_lock.py)
  → Peak-only start detection → stability endpoint → max 1 segment per run
  → Output: sim_flip/data/derived/segments/{segment_id}.csv + manifest update

Stage C: Sample selection (run_identification_cv.py)
  → Filter by split_tag + cv_fold from manifest
  → Output: selected_segments.csv, preprocess_status.csv

Stage D: Frequency estimation (frequency.py)
  → peak / zero-cross / acf / psd methods + consistency flag

Stage E: Joint parameter identification
  → Step3 (energy method, NNLS): d_q, d_qq
  → Step4 (ODE minimization, multi-start): mu_theta, d_q, d_qq, K_cable
  → Optional bootstrap for confidence intervals
```

### 7.2 Pipeline Commands

```powershell
conda activate pytorchlearning
python sim_flip/scripts/run_identification_cv.py --prepare-only
python sim_flip/scripts/run_identification_cv.py --fit-only --fit-split-tag train --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags val --cv-fold holdout_v1
python sim_flip/scripts/run_sim_real_eval.py --eval-split-tags test --cv-fold holdout_v1
python sim_flip/scripts/run_sensitivity_suite.py
python sim_flip/scripts/build_paper_figures.py
```

### 7.3 Mandatory Metrics (must report in paper)

**Absolute metrics** (per split/condition):
- $TAAE_\theta$, $TASE_\theta$
- $RMSE_\theta$, $MAE_\theta$, $MaxAbs_\theta$
- $dt_{90}$ (settling time error)
- Overshoot error
- Steady-state mean/std (last 20% window)
- $RMSE_q$, $MAE_q$

**Composite normalized gap score**:
$$GapScore = 0.35 N_{RMSE_\theta} + 0.20 N_{MAE_\theta} + 0.20 N_{RMSE_q} + 0.15 N_{t90} + 0.10 N_{over}$$

Where normalization bases are:
- $N_{RMSE_\theta} = RMSE_\theta / |90 - \theta_0|$
- $N_{MAE_\theta} = MAE_\theta / |90 - \theta_0|$
- $N_{t90} = |dt_{90}| / t_{90,exp}$ (fallback to $t_{80}$)
- $N_{over} = |overshoot\_error| / |90 - \theta_0|$
- $N_{RMSE_q} = RMSE_q / \max(P95(|q_{exp}|), \epsilon)$

If $q$ data unavailable, renormalize remaining weights.

**Dual reporting**: Report both raw metrics AND phase-aligned metrics to separate phase error from amplitude/model mismatch.

### 7.4 Acceptance Gates (recommended thresholds)

- Median $TAAE_\theta \leq 5°$
- Median $|TASE_\theta| \leq 2°$
- Median $|dt_{90}| \leq 0.3$ s

---

## 8. Figure Architecture (FROZEN)

### 8.1 Master Figure List

| ID | Title | Section | Status |
|----|-------|---------|--------|
| Fig. 1 | Mission-oriented platform architecture and deployment workflow | Sec. 1 | TODO |
| Fig. 2 | Body-axis definition, sign convention, and state interface | Sec. 2 | TODO |
| Fig. 3 | Experimental condition matrix, repeat policy, and anti-leakage split strategy | Sec. 2 | TODO |
| Fig. 4 | Model-term map: physical mechanisms → equation blocks → code interfaces | Sec. 3 | TODO |
| Fig. 5 | CFD credibility summary: mesh refinement and residual convergence | Sec. 4 | TODO |
| Fig. 6 | Full-AoA $C_X/C_Z/C_m$ curves with high-AoA local enlargement | Sec. 4 | TODO |
| Fig. 7 | Pressure-center migration $X_{cp}(\alpha)$ with stability-region annotations | Sec. 4 | TODO |
| Fig. 8 | Representative flow and pressure snapshots at 0°/30°/60°/90° AoA | Sec. 4 | TODO |
| Fig. 9 | Identification diagnostics: residuals, multi-start convergence, bootstrap CIs | Sec. 5 | TODO |
| Fig. 10 | Multi-run in-sample and out-of-sample sim-real trajectories ($\theta$, $q$) | Sec. 5 | TODO |
| Fig. 11 | Time-domain error traces with zero-reference and RMSE bands | Sec. 5 | TODO |
| Fig. 12 | Condition-normalized metric heatmap and GapScore ranking | Sec. 5 | TODO |
| Fig. 13 | Sensitivity envelopes for initial energy, BG, and damping uncertainty | Sec. 6 | TODO |
| Fig. 14 | Strategy surface $J(u_0, \theta_{off})$ with feasible valley regions | Sec. 6 | TODO |

Detailed requirements for each figure: see `paper/figures/FIGURE_REQUIREMENTS.md`.

### 8.2 In-Text Placeholder Format

Use exactly: `[Fig. X. <Title>. Insert here.]` in Markdown sections.
Use `\begin{figure}[htbp]...\end{figure}` with `% TODO: Insert Fig. X` in LaTeX.

### 8.3 Visual Design Rules (FROZEN)

- Serif font family, consistent font size across all figures
- Colorblind-safe palette
- Thin major grid lines only
- Remove top/right spines for 2D plots
- Subpanel tags: `(a)`, `(b)`, `(c)` ...
- Downsampling allowed ONLY for rendering speed, NEVER for metric computation
- All labels in English, SI units

---

## 9. LaTeX Formatting Rules

### 9.1 Document Class Configuration

```latex
% CURRENT (two-column final format):
\documentclass[final,5p,times,twocolumn]{elsarticle}

% For review with line numbers, use:
% \documentclass[review,5p,times,twocolumn]{elsarticle}

% For draft single-column editing, use:
% \documentclass[authoryear,preprint,review,12pt]{elsarticle}
```

### 9.2 Citation & Reference Style

- Style: author-year (authoryear option)
- BST file: `paper/elsarticle/elsarticle-harv.bst`
- Command: `\citep{key}` for parenthetical, `\citet{key}` for textual
- BibTeX file: `paper/latex/references.bib`
- All placeholder entries (R2-R6) must be replaced with real bibliographic data before submission

### 9.3 Required Packages

```latex
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{lineno}      % for review mode
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage{url}
```

### 9.4 Text Formatting Rules

| ❌ DO NOT | ✅ DO |
|-----------|-------|
| `\texttt{mu\_x}` | `$\mu_x$` |
| `\texttt{mu\_z}` | `$\mu_z$` |
| `\texttt{mu\_theta}` | `$\mu_\theta$` |
| `\path{sim_flip/src/dynamics.py}` | "the simulation repository" or omit |
| `\texttt{rigid\_body}` | omit or use prose |
| Backtick-quoted code in prose | Mathematical notation or plain English |
| Code-style parameter names in text | LaTeX math-mode symbols |

**Rationale**: This is a journal submission, not a software manual. Code paths and variable names in `\texttt{}` are inappropriate for Ocean Engineering publication. Parameter names must use standard mathematical notation. Implementation details belong in a Data Availability statement or supplementary material.

### 9.5 Equation Formatting

- Use `equation` for standalone equations
- Use `align` for multi-line equation systems
- Number all equations (default in elsarticle)
- Cross-reference with `\eqref{}`
- Keep subscripts consistent: use comma-separated qualifiers (e.g., $m_{water,inner}$)

### 9.6 Figure Handling in Two-Column Mode

- Normal figures: `\begin{figure}[htbp]`
- Wide figures spanning both columns: `\begin{figure*}[htbp]`
- Normal tables: `\begin{table}[htbp]`
- Wide tables: `\begin{table*}[htbp]`
- Consider using `figure*` for Fig. 1 (platform architecture), Fig. 3 (condition matrix), Fig. 8 (flow snapshots 2×2), Fig. 10 (multi-run trajectories)

### 9.7 Highlights Requirements

- Separate file: `paper/latex/highlights.txt`
- 3-5 bullet points
- Each bullet ≤ 85 characters
- Capture key novelty and results

### 9.8 Build Command

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper/latex/out paper/latex/main.tex
```

Build output: `paper/latex/out/`

---

## 10. Journal-Specific Requirements (Ocean Engineering)

### 10.1 Submission Checklist

- [ ] Manuscript ≤ 40 double-spaced pages (incl. tables and illustrations)
- [ ] Abstract ≤ 250 words, no references
- [ ] Keywords: 1-7 keywords
- [ ] Highlights: 3-5 bullets, each ≤ 85 characters
- [ ] All figures with English labels
- [ ] CRediT authorship contribution statement
- [ ] Data availability statement
- [ ] Declaration of competing interest
- [ ] Acknowledgements section (before references)
- [ ] Complete bibliography (no placeholder entries)
- [ ] Graphical abstract (optional but recommended)

### 10.2 OE Style Notes

- Engineering relevance must be clear — not a pure methods paper
- Reproducibility is valued — provide enough detail for replication
- CFD papers should include mesh independence and residual convergence evidence
- Experimental validation is expected for modeling papers
- Sensitivity analysis adds engineering value (design implications)

### 10.3 Backup Journal Strategy

| Priority | Journal | IF | Fit |
|----------|---------|-----|-----|
| 1 | **Ocean Engineering** | 5.5 | Best balance of impact and topic match |
| 2 | Applied Ocean Research | 4.4 | Strong match, stricter novelty bar |
| 3 | J. Marine Sci. & Tech. (Springer) | 2.0 | Good match, higher acceptance |
| 4 | JMSE | 2.8 | Broader scope, pragmatic backup |

---

## 11. Sensitivity & Design Implication Package

### 11.1 Required Analyses

| Analysis | What to sweep | What to report |
|----------|---------------|----------------|
| Initial-energy envelope | $\theta_0$-$q_0$ grid | Success/failure map, phase portraits |
| Thrust timing | Shutdown angle $\theta_{off}$ vs always-on | Settling time, overshoot, terminal $q$, vertical displacement |
| Strategy surface | $J(u_0, \theta_{off})$ | 3D/contour with feasible valley regions |
| BG sensitivity | $BG \pm 30\%$ | $\omega_n$, $\zeta$, settling, oscillation envelope |
| Damping robustness | $k_{damp} \in [0.5, 2.0]$ | Stability boundary, performance degradation trend |

### 11.2 Practical Guidance Rule

If experimental budget is limited:
- Use calibrated simulation for dense sweeps
- Anchor trends with selected physical experiments

---

## 12. Repository Structure Reference

```
C:\AD_v2_stage1_analysis\
├── paper/
│   ├── sections/*.md          # [AUTHORITATIVE] Manuscript drafts
│   │   ├── 00_abstract.md
│   │   ├── 00a_nomenclature.md    # NEW: symbol table (OE convention)
│   │   ├── 01_introduction.md
│   │   ├── 02_system_and_mission_scenario.md
│   │   ├── 03_anisotropic_permeability_corrected_dynamics.md
│   │   ├── 04_cfd_to_model_mapping_and_validation.md
│   │   ├── 05_parameter_identification_and_model_validation.md  # Merged 05+06
│   │   ├── 06_sensitivity_analysis_and_design_implications.md   # Renumbered
│   │   └── 07_conclusions.md                                     # Renumbered
│   ├── latex/
│   │   ├── main.tex           # LaTeX main file
│   │   ├── sections/*.tex     # LaTeX mirrors of Markdown
│   │   ├── highlights.txt     # Highlights plain text
│   │   ├── highlights_items.tex
│   │   ├── references.bib     # BibTeX (needs real entries)
│   │   └── out/               # Build output
│   ├── elsarticle/            # Official Elsevier template files
│   │   ├── elsarticle-harv.bst
│   │   └── elsarticle-num.bst
│   ├── figures/
│   │   └── FIGURE_REQUIREMENTS.md  # [FROZEN] Figure planning
│   ├── 论文发表计划.md        # [FROZEN] Master plan (directly under paper/)
│   └── savedpromots/
│       └── 2026-02-13-工作缓存.md  # [DEPRECATED]
├── sim_flip/
│   ├── src/                   # Core simulation modules
│   │   ├── dynamics.py        # ODE right-hand side
│   │   ├── kinematics.py      # V, Q, alpha computation
│   │   ├── cfd_table.py       # Coefficient interpolation
│   │   ├── added_mass.py      # Permeability-corrected masses
│   │   └── conventions.py     # [FROZEN] Coordinate conventions
│   ├── analysis/              # Pipeline analysis modules
│   ├── configs/
│   │   ├── params_nominal.yaml    # [FROZEN] Physical parameters
│   │   ├── id_protocol.yaml       # Identification protocol config
│   │   └── experiment_manifest.csv # Run/segment tracking
│   ├── data/
│   │   ├── raw/runs/          # Raw experimental TXT files
│   │   └── derived/           # Processed CSV files
│   ├── results/               # Pipeline outputs
│   ├── scripts/               # Entry-point scripts
│   ├── tests/                 # Test suite
│   ├── PIPELINE_OPERATION_GUIDE.md  # [AUTHORITATIVE] Pipeline handbook
│   └── README_MIGRATION.md   # Legacy→Fixed pipeline mapping
├── docs/
│   ├── AI_WRITING_REFERENCE.md    # ← THIS FILE (unified reference)
│   └── plans/                     # Historical planning documents
├── figures/                   # Pipeline-generated figures (fig01-fig08)
└── .vscode/                   # Editor configuration
```

---

## 13. DO / DO NOT Quick Reference

### ✅ ALWAYS DO

- Use $\mu_x, \mu_z, \mu_\theta$ in math mode for parameters
- Follow the Markdown → LaTeX workflow (edit `.md` first)
- Use condition-block data splits (never random sample splits)
- Report both raw and phase-aligned metrics
- Include CFD credibility evidence (mesh independence + residuals)
- Limit conclusions to current geometry/mass/actuation
- Use `\citep{}` for author-year parenthetical citations
- Keep figure ordering monotonic (Fig. 1 → Fig. 14)
- Use SI units throughout
- Run `lsp_diagnostics` / build check after LaTeX edits

### ❌ NEVER DO

- Use `\texttt{}` for parameter names in paper text
- Use `\path{}` for repository file paths in paper text
- Add explicit Munk-moment term when $C_m(\alpha)$ already contains it
- Mix train/validation samples from the same condition
- Put dense numeric metrics in the Abstract
- Claim universality beyond the tested configuration family
- Record multiple excitation events in a single data file
- Use downsampled data for metric computation
- Suppress LaTeX errors with workarounds
- Edit `.tex` files without updating corresponding `.md` first

---

## 14. Outstanding Tasks (Writing Roadmap)

### Priority 1: Complete skeleton sections to submittable quality
1. Section 4 (CFD): Write credibility summary, coefficient analysis, CP migration, flow snapshots
2. Section 5 (Identification + Validation): Write protocol details, diagnostic analysis, trajectory comparison, error analysis, metric interpretation
3. Section 6 (Sensitivity): Write envelope analysis, BG/damping study, strategy surface discussion
4. Section 7 (Conclusions): Write validated findings summary, limitations restatement, future work

### Priority 2: Data & evidence expansion
6. Expand experiments from 3 to 12 conditions
7. Generate all 14 figures via pipeline scripts
8. Replace placeholder BibTeX entries with 30-50 real references
9. Backfill quantitative claims in Abstract/Introduction after data expansion

### Priority 3: Submission preparation
9. Add Nomenclature to LaTeX main.tex as front-matter section
10. Add Appendix structure for supplementary CFD details
11. Shorten title to ≤ 15 words
12. Complete CRediT, Data availability, and Acknowledgements sections
13. Fill or remove Graphical Abstract
14. Final language polishing pass

---

## 15. Document Governance History

This document consolidates and supersedes:

| Former Document | Status | Disposition |
|-----------------|--------|-------------|
| `paper/README.md` | Active | Retained as project README; frozen rules absorbed here |
| `paper/latex/README.md` | Active | Retained as build instructions; formatting rules absorbed here |
| `paper/论文发表计划.md` | **Frozen master plan** | Retained; all content absorbed here |
| `paper/savedpromots/2026-02-13-工作缓存.md` | **Deprecated** | Superseded by this document; marked for archival |
| `docs/plans/2026-02-12-mission-oriented-paper-writing-design.md` | **Historical** | Archived; narrative rules absorbed here |
| `paper/figures/FIGURE_REQUIREMENTS.md` | Active | Retained as figure-specific reference; summary absorbed here |
| `sim_flip/README.md` | **Merged** | Content merged into PIPELINE_OPERATION_GUIDE.md |
| `sim_flip/PIPELINE_OPERATION_GUIDE.md` | **Authoritative** | Retained as sole pipeline handbook |

**Rule**: When this document and a former document conflict, **this document wins**.
