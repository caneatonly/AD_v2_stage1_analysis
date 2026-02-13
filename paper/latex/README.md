# LaTeX Workspace

This folder stores LaTeX sources for the manuscript.

Workflow (frozen intent):
1. Draft and iterate content in `paper/sections/*.md`.
2. After content is confirmed, edit/assemble the LaTeX version here (and build PDF via LaTeX Workshop).

Build output directory is configured to: `paper/latex/out/` (see `.vscode/settings.json`).

Quick build from repo root:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper/latex/out paper/latex/main.tex`

Submission-side artifacts (placeholders for now):
- `paper/latex/highlights.txt` (each bullet <= 85 characters)

Reference style policy (Ocean Engineering):
- Use author-year citations in LaTeX (`authoryear` option).
- Enforce the official local BST from `paper/elsarticle/` via `\\bibliographystyle{paper/elsarticle/elsarticle-harv}` in `paper/latex/main.tex`.

Guide-for-authors compliance checklist (frozen):
- Use `elsarticle` `preprint` + `authoryear` as baseline; switch to LaTeX double-column only when explicitly needed by submission stage/editor requirement.
- Keep highlights synchronized between `highlights_items.tex` and `highlights.txt` (3--5 bullets, concise).
- Keep declaration sections in `main.tex`: Data availability, CRediT, Funding, AI declaration, Competing interest, Acknowledgements.
- Keep bibliography style fixed to `paper/elsarticle/elsarticle-harv` (do not switch to external BSTs).
- Ensure every `\cite` key resolves to a complete entry in `references.bib` before submission export.
- `1p/3p/5p` are available layout modes; keep this repo's canonical draft in `preprint` to reduce churn unless a format switch is requested.
