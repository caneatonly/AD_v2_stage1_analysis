# LaTeX Workspace

This folder stores LaTeX sources for the manuscript.

Workflow (frozen intent):
1. Draft and iterate content in `paper/sections/*.md`.
2. After content is confirmed, edit/assemble the LaTeX version here (and build PDF via LaTeX Workshop).

Build output directory is configured to: `paper/latex/out/` (see `.vscode/settings.json`).

Quick build from repo root:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper/latex/out paper/latex/main.tex`

Policy and formatting governance are maintained in `docs/AI_WRITING_REFERENCE.md`.
