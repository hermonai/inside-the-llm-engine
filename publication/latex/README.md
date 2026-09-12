# Native LaTeX edition

Build from this directory with `make`. `main.tex` contains all nine written
chapters and uses native TikZ `.tex` illustrations: no rasterized labels,
no embedded PDF plates and no verbatim character-box graphs.

The 65 semantic SVG plates have a deterministic, fail-closed TikZ translation.
Legacy diagrams preserve their reviewed topology using vector strokes and
proportional typeset labels. This mechanical migration is not a claim that each
legacy plate has received the detailed Chapter 5–9 editorial redesign.

To synchronize prose and figures, run `publication/build.py`, then
`publication/build-tex.py` from the repository root. Commit generated `.tex`
alongside the source changes. `publication/build-tex.py --check` detects drift.
The Markdown source remains the web authoring surface; the portable `.tex`
edition and all its figure sources are shipped, not hidden in ignored build output.

Requires Pandoc 3.9 for regeneration and TeX Live 2025/XeLaTeX with DejaVu,
Latin Modern Math, TikZ, adjustbox, fvextra, xeCJK, Fandol and twemojis to build.
The PDF is `output/pdf/inside-the-llm-engine-tex.pdf` at repository root.
