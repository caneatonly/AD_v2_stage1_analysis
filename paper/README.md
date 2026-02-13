# Paper Scaffold

This directory is the lightweight manuscript scaffold for the fixed pipeline.

- `sections/` keeps manuscript section drafts (including `00_abstract.md`).
- `figures/` stores exported paper-ready figures from scripts.
- `tables/` stores exported tables.
- `latex/` stores LaTeX publish artifacts (Markdown remains the source of truth).

Current section files:

- `00_abstract.md`
- `01_introduction.md`
- `02_system_and_mission_scenario.md`
- `03_anisotropic_permeability_corrected_dynamics.md`
- `04_cfd_to_model_mapping_and_validation.md`
- `05_parameter_identification_free_decay.md`
- `06_sim_real_validation.md`
- `07_sensitivity_and_design_implications.md`
- `08_conclusion_and_outlook.md`

The scaffold is intentionally minimal and data-driven. Update figures/tables by rerunning pipeline scripts.

## Ocean Engineering guide-for-authors principles (frozen)

This repo treats the Ocean Engineering Guide for Authors as the root policy for manuscript structure and submission artifacts.

Source (accessed 2026-02-13): `https://www.sciencedirect.com/journal/ocean-engineering/publish/guide-for-authors`

Frozen rules for this project:

- Source of truth: `paper/sections/*.md` is authoritative; `paper/latex/` is publish-only and must mirror Markdown.
- Manuscript class/style: default repo baseline is Elsevier `elsarticle` in `preprint` + `authoryear`; LaTeX double-column is permitted when explicitly needed by submission stage/editor request.
- Highlights: provide exactly 3--5 bullets in a separate editable file (`paper/latex/highlights.txt`), each concise (target <=85 characters).
- Abstract: provide a factual stand-alone abstract; avoid citations and undefined abbreviations.
- Keywords: provide up to 6 keywords in the `keyword` block and keep terminology consistent with section text.
- Structure: use clearly numbered sections/subsections; define symbols/units at first use and keep SI notation consistent.
- Declarations: keep dedicated sections for Data availability, CRediT authorship contribution statement, Funding, Declaration of generative AI use, and Declaration of competing interest.
- Acknowledgements: keep as a separate section placed directly before the reference list in the LaTeX manuscript.
- References: enforce the official local BST under `paper/elsarticle/` and ensure each citation has complete bibliographic metadata.
- Figures/tables: captions must be self-contained, abbreviations explained in caption or text, and numbering must match manuscript order.

- Layout models: `1p/3p/5p` are layout emulation options; keep the repository master in `preprint` unless a stage-specific format switch is required.

- OE guide quick reference for this repo: `paper/OE_AUTHOR_GUIDE_QUICK_REFERENCE.md`.
