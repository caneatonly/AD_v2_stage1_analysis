# Mission-Oriented Paper Writing Design (ARCHIVED)

> **⚠️ HISTORICAL**: This document has been archived. Its content has been absorbed into `docs/AI_WRITING_REFERENCE.md`.
> Do NOT use this file for active guidance. Refer to the unified reference document instead.

Date: 2026-02-12 (archived: 2026-02-14)
Scope: Abstract + Introduction drafting strategy for the manuscript:
`Design and Characterization of a Mission-Oriented Self-Suspending Underwater Platform with Anisotropic Permeability-Corrected Dynamics and Experimental Identification`

## 1. Confirmed Decisions

- Writing objective now: start manuscript drafting before full experimental expansion is finished.
- Narrative mode: mission-driven, aggressive claim style.
- Introduction backbone: task-first, then platform-first, then model and validation chain.
- Task mention policy: explicitly mention underwater corner-reflector deployment in Abstract opening sentence.
- Result style in Abstract: qualitative only (no numeric metrics at this stage).
- Metrics exposure policy: key metrics are still mandatory in Results, but not densely exposed in Abstract.

## 2. Core Narrative Architecture

The paper opening will follow:

1. Mission requirement: underwater corner-reflector deployment motivates a dedicated platform.
2. Platform innovation: system design and fabrication are first-class contributions, not background.
3. Critical transition process: horizontal launch to vertical stabilization is a mission-critical phase for this platform.
4. Model innovation: anisotropic permeability-corrected 3-DOF dynamics with physically interpretable coupling parameters (`mu_x`, `mu_z`, `mu_theta`).
5. Evidence chain: free-decay identification -> out-of-sample sim-real comparison -> sensitivity and design envelope analysis.
6. Boundary statement: conclusions are limited to current geometry, mass distribution, and actuation strategy.

This removes the weak claim that “traditional platforms generally fail in this phase” and replaces it with a stronger, correct logic:
the phase is important because this new mission-specific platform must pass through it.

## 3. Abstract Design Rules (Locked)

- Open with mission object explicitly: underwater corner reflector deployment.
- Emphasize platform design/manufacturing contribution in parallel with modeling.
- State dynamics and identification contributions as integrated with platform engineering.
- Keep CFD positioned as model-evidence bridge, not independent novelty centerpiece.
- Use qualitative performance wording only in Abstract for now.
- Defer dense quantitative reporting to Results section.

## 4. Introduction Design Rules (Locked)

Introduction will be drafted in five paragraphs:

1. Mission context and operational requirement.
2. Platform design rationale and engineering innovation.
3. Modeling gap and anisotropic permeability-corrected dynamics.
4. Validation chain and what is empirically verified.
5. Three contributions and explicit scope boundaries.

## 5. Claim-to-Evidence Contract

Each high-level claim must map to at least one explicit evidence object later in the manuscript:

- Platform novelty -> system diagram + fabrication/architecture figure.
- Dynamics interpretability -> equation set + parameter table.
- CFD-to-model logic -> coefficient curves and CP migration evidence.
- Identification reliability -> CV/bootstrap diagnostics.
- Practical validity -> in-sample/out-of-sample sim-real plots and metric tables.
- Engineering implication -> sensitivity maps and design envelopes.

## 6. Writing Constraints and Risk Control

- Do not imply universal generalization to all underwater platforms.
- Do not expose mission-sensitive details beyond allowed abstraction.
- Keep notation consistent with existing pipeline contracts.
- Avoid premature hard quantitative claims in Abstract until expanded experiments are integrated.
- Keep wording aligned with Ocean Engineering audience: engineering relevance + reproducibility + mechanism support.

## 7. Immediate Execution Plan

1. Draft paper-ready English Abstract using the locked qualitative rule.
2. Draft full English Introduction (5-paragraph structure).
3. Keep placeholders only where factual values still depend on upcoming additional runs.
4. Backfill values and tighten claims after new experiment batch is integrated.

## 8. Open Items (for next turn)

- Final abstract length target (e.g., 180-220 words vs 220-260 words).
- Whether to create a dedicated `paper/sections/00_abstract.md`.
- Citation insertion style in Introduction (temporary placeholders vs immediate BibTeX keys).
