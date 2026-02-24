# Documentation Architecture

Start here before editing code, pipeline settings, or manuscript content.

This file defines the documentation hierarchy and the ownership boundary of each document so AI agents do not duplicate rules across files.

## Hierarchy (authoritative order)

### Level 1: Project governance

- `docs/README.md` (this file): documentation map and ownership rules
- `docs/AI_WRITING_REFERENCE.md`: manuscript writing governance, frozen claims, notation, and evidence rules

### Level 2: Domain entry docs

- `sim_flip/README.md`: module overview, quick entrypoints, legacy migration mapping (summary)
- `paper/README.md`: paper workspace layout, source-of-truth workflow, collaboration entrypoints

### Level 3: Domain-specific operating docs

- `sim_flip/PIPELINE_OPERATION_GUIDE.md`: detailed pipeline operation, commands, outputs, troubleshooting
- `paper/figures/FIGURE_REQUIREMENTS.md`: figure plan, placeholders, visual constraints, figure ownership
- `paper/latex/README.md`: LaTeX workspace and build instructions

### Level 4: Manuscript content sources (not governance docs)

- `paper/sections/*.md`: manuscript section drafts (authoritative content source)

## Ownership Rules (prevent duplication)

- Put project-wide writing policy, frozen claims, notation, and evidence contracts only in `docs/AI_WRITING_REFERENCE.md`.
- Put pipeline commands, data contracts, and troubleshooting only in `sim_flip/PIPELINE_OPERATION_GUIDE.md`.
- Put figure IDs, placeholder policy, and figure content requirements only in `paper/figures/FIGURE_REQUIREMENTS.md`.
- Put LaTeX build commands and workspace usage only in `paper/latex/README.md`.
- Keep `sim_flip/README.md` and `paper/README.md` as index pages; they should summarize and link, not re-document detailed procedures.

## AI Agent Routing (what to open first)

- Pipeline run / data preprocessing / identification issue: `sim_flip/README.md` -> `sim_flip/PIPELINE_OPERATION_GUIDE.md`
- Manuscript structure / claims / notation / writing constraints: `docs/AI_WRITING_REFERENCE.md`
- Figure planning / placeholder mismatch / figure asset ownership: `paper/figures/FIGURE_REQUIREMENTS.md`
- LaTeX compile / formatting / output path: `paper/latex/README.md`
- Section drafting or revision: target file in `paper/sections/`

## Maintenance Rules

- If two documents conflict, follow the document that owns that domain above.
- When adding a new doc, add it to this hierarchy and state its ownership boundary.
- Do not copy large blocks of rules between docs; link to the canonical file instead.
