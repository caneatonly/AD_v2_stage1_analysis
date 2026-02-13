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
- Use numeric citation style in LaTeX for this project.
- Strictly use a local official BST from `paper/elsarticle/` (current setting: `\\bibliographystyle{paper/elsarticle/elsarticle-num}`).
