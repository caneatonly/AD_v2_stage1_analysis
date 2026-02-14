# Ocean Engineering (OE) Author Guide Quick Reference (Frozen for this repo)

Source basis: OE Guide for Authors text snapshot provided by project owner (updated to Feb 2026 in the snapshot).

## 1) Scope fit (submit only if in-scope)
- Focus on ocean/marine engineering applications (offshore, naval architecture, coastal/harbour, underwater tech, marine control, etc.).
- Avoid out-of-scope themes (e.g., pure theory without ocean-engineering application, military-only applications, unrelated broad material/corrosion studies).

## 2) Article type and review
- Typical type: Original research paper.
- Peer review model: **single anonymized** (reviewers anonymous; author info is shown to editors/reviewers).

## 3) Writing and formatting essentials
- Provide editable source files (`.tex` for LaTeX).
- For Word: single-column required.
- For LaTeX: double-column is **permitted**, but not mandatory.
- Use SI units; equations as editable math text; clearly numbered sections/subsections.

## 4) Required front matter and statements
- Concise title; complete author/affiliation/corresponding-author details.
- Abstract: factual, stand-alone, <= 250 words; avoid unnecessary references and undefined abbreviations.
- Keywords: 1--7.
- Highlights: 3--5 bullets, each <= 85 characters, separate editable file.
- Funding statement required (or explicit no-funding sentence).
- Competing interest declaration required.
- CRediT author contribution statement required.
- Data availability statement required.
- AI declaration: add only if generative AI was used in manuscript preparation (section placed before references).
- Acknowledgements in separate section directly before references.

## 5) Data policy (Option C in snapshot)
- Prefer depositing research data in an appropriate repository and citing/linking dataset in article.
- If data cannot be shared, provide explicit reason in data statement.

## 6) References
- Author-year citation style.
- Ensure one-to-one consistency between in-text citations and reference list.
- Prefer DOI links when available.
- Mark preprints clearly; use formal published version if available.

## 7) Artwork and tables
- Figures as separate files (logical naming, cited in text, numbered by appearance).
- Captions required for all figures/tables; explain symbols/abbreviations.
- Tables must be editable (not image screenshots).

## 8) Repo-level implementation policy in this project
- Manuscript source of truth: `paper/sections/*.md`.
- LaTeX publication assembly: `paper/latex/main.tex` + `paper/latex/sections/*.tex`.
- Bibliography style must use local OE-compatible BST under `paper/elsarticle/`.
