# Paper Workspace

This directory contains manuscript sources and publication assets.

## Workspace Layout

- `sections/`: authoritative manuscript drafts in Markdown (`paper/sections/*.md`)
- `figures/`: figure requirements and generated manuscript figures
- `latex/`: LaTeX assembly and build artifacts (must stay aligned with Markdown)
- `elsarticle/`: local Elsevier template/class assets

## Source-of-Truth Rule (important)

- Manuscript content authority: `paper/sections/*.md`
- LaTeX files in `paper/latex/sections/*.tex` are mirror/assembly targets
- Do not make long-term content edits in LaTeX only

## Active Section Files

- `00_abstract.md`
- `00a_nomenclature.md`
- `01_introduction.md`
- `02_platform_design_and_implementation.md`
- `03_experimental_setup_and_data_governance.md`
- `04_dynamic_model_and_hydrodynamic_closure.md`
- `05_parameter_identification_pipeline.md`
- `06_model_validation_against_free_decay_experiments.md`
- `07_parametric_analysis_and_design_guidance.md`
- `08_conclusions.md`

## Collaboration Entry Points (canonical docs)

- Project-wide documentation map: `docs/README.md`
- Writing governance / frozen claims / notation rules: `docs/AI_WRITING_REFERENCE.md`
- Figure-specific requirements and placeholder policy: `paper/figures/FIGURE_REQUIREMENTS.md`
- LaTeX workspace/build instructions: `paper/latex/README.md`

## Common Tasks -> Open These Docs First

- Revise claims, scope, or manuscript wording policy -> `docs/AI_WRITING_REFERENCE.md`
- Add or reorder figure placeholders -> `paper/figures/FIGURE_REQUIREMENTS.md`
- Build or debug PDF compilation -> `paper/latex/README.md`
- Draft or revise section prose -> target file in `paper/sections/`
