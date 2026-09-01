# Repository Scripts

`check-structure.sh` verifies the book skeleton, 94 chapter specifications,
Chapter 1 artifact set, and Chapter 1 word-count gate.

`check-links.py` validates repository-relative Markdown targets using only the
Python standard library. Rust formatting, build, tests, Clippy, both repository
checks, and canonical diagram width run in the lightweight CI workflow.
