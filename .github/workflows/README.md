# Workflow Policy

The lightweight `book-and-engine` workflow checks repository structure,
relative Markdown links, canonical diagram width, Rust formatting, compilation,
tests, and Clippy. It does not download model weights or install a documentation
framework.

Later real-model gates must name required fixtures explicitly. CI must never
silently convert a required release gate into a skipped success.
