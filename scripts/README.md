# Repository Scripts

`check-structure.sh` verifies the book skeleton, 94 chapter specifications,
completed chapter artifact sets, and chapter word-count gates.

`check-links.py` validates repository-relative Markdown targets using only the
Python standard library. `check-diagram-style.py` reconciles the canonical
inventory and rejects legacy connectors; `check-diagram-width.py` enforces the
100-column display bound; `check-math-style.py` checks display-math structure,
shape declarations, and required equation IDs. Rust formatting, build, tests,
Clippy, and all repository checks run in the lightweight CI workflow.
