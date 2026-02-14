# Paper Scaffold

This directory is the lightweight manuscript scaffold for the fixed pipeline.

- `sections/` keeps manuscript section drafts (including `00_abstract.md`).
- `figures/` stores exported paper-ready figures from scripts.
- `tables/` stores exported tables.
- `latex/` stores LaTeX publish artifacts (Markdown remains the source of truth).

Current section files:

- `00_abstract.md`
- `00a_nomenclature.md`
- `01_introduction.md`
- `02_system_and_mission_scenario.md`
- `03_anisotropic_permeability_corrected_dynamics.md`
- `04_cfd_to_model_mapping_and_validation.md`
- `05_parameter_identification_and_model_validation.md` (merged from former 05+06)
- `06_sensitivity_analysis_and_design_implications.md` (renumbered from former 07)
- `07_conclusions.md` (renumbered from former 08)

The scaffold is intentionally minimal and data-driven. Update figures/tables by rerunning pipeline scripts.

## Ocean Engineering guide-for-authors principles (frozen)

This repo treats the Ocean Engineering Guide for Authors as the root policy for manuscript structure and submission artifacts.

Source (accessed 2026-02-13): `https://www.sciencedirect.com/journal/ocean-engineering/publish/guide-for-authors`

Frozen rules for this project:

- Source of truth: `paper/sections/*.md` is authoritative; `paper/latex/` is publish-only and must mirror Markdown.
- Highlights: prepare highlights as a separate editable file (see `paper/latex/highlights.txt`); keep bullets short and scannable.
- Abstract: avoid references/citations unless absolutely required; keep results phrasing appropriate for current evidence maturity.
- Structure: use clearly numbered sections and subsections; keep notation and terminology consistent across the manuscript.
- References: use the journal's author-year style and ensure every citation has a complete bibliographic entry.
- Acknowledgements: keep as a separate section placed directly before the reference list in the LaTeX manuscript.
- Author contributions: include a CRediT authorship contribution statement section (fill before submission).
- Data statement: include a Data availability statement section (fill before submission).
