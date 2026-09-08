# Repository Scripts

`check-structure.sh` verifies the book skeleton, 94 chapter specifications,
completed chapter artifact sets, and chapter word-count gates.

`check-links.py` validates repository-relative Markdown targets using only the
Python standard library. `check-diagram-style.py` reconciles the canonical
inventory and rejects legacy connectors; `check-diagram-width.py` enforces the
100-column display bound; `check-math-style.py` checks display-math structure,
shape declarations, and required equation IDs. Rust formatting, build, tests,
Clippy, and all repository checks run in the lightweight CI workflow.

`check-linear-visual-parity.py` runs the real Chapter 6 Rust example and compares
its full output and address metadata with an independent F32 Python oracle.
It additionally checks contribution/tail enumeration, pinned annotation values,
four actual Rust source excerpts and fourteen embedded canonical figures.
`check-figure-browser.cjs` exercises all five step sequences, offline Chapters
5/6 images and MathML, keyboard/reduced-motion controls, three viewport widths,
and Chapter 6 SVG text bounds. Publication structure and visual review helpers
are documented in `docs/FIGURE_BUILD.md`.
