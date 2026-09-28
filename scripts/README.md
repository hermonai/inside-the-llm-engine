# Repository scripts

## Checks for the second edition

- `check-structure.sh` — required files; `AUTHORING.md` under its 3,000-word
  cap; `docs/` holding only the plan and status; the chapter plan numbered in
  sequence, 59 chapters, with identical titles in `docs/STRUCTURE.md`
  and `BOOK.md`; first-edition artifacts still present; and a guard that fails
  on IP addresses, credential paths, private keys or tokens in tracked files.
- `check-textbook.py` — 17 construction and 42 systems chapters under their
  respective authoring contracts; native figures, worked problems, the
  executable request trace (`ch14`, printed Chapter 31), independent worked
  calculations and the standard-library Python foundation tests.
- `check-foundations.py` — all 42 prerequisite bridges and 13 test cases
  covering illustrative arithmetic, the leading Part I learning path and the
  corrected systems forward-pass softmax figure; called by `check-textbook.py`,
  so CI runs it too.
- `check-links.py` — repository-relative Markdown links. `archive/` mirrors
  original paths; links to an archived file resolve through that mirror, and an
  archived document's links resolve from its original directory.
- `build-textbook.py` — the PDF (XeLaTeX via latexmk); fails on overfull boxes,
  missing glyphs and unresolved references.
- `review-textbook-pdf.py` — renders pages for visual review.

## Checks that guard first-edition material

These still run in CI; the material they check is frozen, not edited.

- `check-diagram-style.py`, `check-diagram-width.py` — the Unicode text
  diagrams in `diagrams/`.
- `check-native-figures.py`, `figures/build.py --check`,
  `check-industrial-visuals.py`, `check-figure-browser.cjs` — the SVG atlas,
  its TikZ conversion and the HTML exporter.
- `check-tensor-visual-parity.py`, `check-linear-visual-parity.py`,
  `check-normalization-visual-parity.py`, `check-qkv-visual-parity.py`,
  `check-rope-visual-parity.py` — run the real `mini-engine` examples and
  compare them with independent Python oracles (Part I's construction evidence).

Retired scripts are in `archive/scripts/`; see `archive/README.md`.
