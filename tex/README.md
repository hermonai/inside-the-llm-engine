# Authored textbook

This directory is the active manuscript, not a generated export. Edit
`chapters/chNN.tex`, `figures/*.tex`, and `worked/chNN.tex` directly.
`inside-the-llm-engine.tex` assembles nine written chapters, a preface,
glossary, evidence notes and selective index.

## Build

From the repository root:

```sh
make textbook-check
make textbook
python3 scripts/review-textbook-pdf.py output/pdf/inside-the-llm-engine-textbook.pdf build/textbook/review
```

The build uses XeLaTeX/latexmk, TeX Live 2025, TeX Gyre Pagella/Heros and
Pagella Math, DejaVu Sans Mono, TikZ, fvextra, xeCJK/Fandol and twemojis.
The source/CLI checker uses Python 3.9+ and Rust/Cargo. Page review additionally
uses Poppler and Pillow. Use a fresh review directory when pagination changes.
No Pandoc or Markdown input is needed to build this edition.

Generated PDFs and logs remain ignored under `output/pdf/` and `build/`.
The builder rejects overfull boxes, missing glyphs and unresolved references.
The checker binds the early request trace to the actual CLI, checks independent
worked calculations and counts the expected chapter/figure/problem inventory.
Visual review is still required; a clean log does not prove good layout.

## Editorial boundary

Chapter 1 is a new explanation built around a reproducible request. Chapters
2–9 retain substantial reviewed material imported once from the historical
edition, then revised in LaTeX. The opening chapters correct obsolete
fake-model examples, tokenizer/model vocabulary assumptions and API snippets.
Each chapter adds three original worked synthesis problems. This is a partial
editorial rewrite, not a claim that all old prose has been re-authored.

Fifty-two reviewed semantic plates were imported as editable TikZ and cropped
to their mechanism content. Thirteen new figures teach request events,
ownership, token boundaries, BPE, UTF-8, model arithmetic, categorical sampling
and temperature. The active manuscript does not include the 28 mechanically
rendered legacy text diagrams. Historical figure IDs and Rust/Python parity
fixtures remain available outside this directory.

`scripts/migrate-textbook-once.py` and `scripts/finish-textbook-import.py`
document the one-time migration; they are not build dependencies and must not
be used to regenerate authored chapters. Future web publication should derive
from the LaTeX source or an explicitly maintained semantic representation.

Code listings distinguish literal APIs and reference algorithms. Always run
the linked full implementation and tests: a short explanatory loop is not a
substitute for its shape checks, errors and ownership contracts.

## Next chapter

Chapter 10 derives causal self-attention from the checked positioned Q/K and
unchanged V. Its dense reference, independent oracle, failure tests and
native illustrations must advance together. Persistent KV caching, a complete
decoder and industrial serving remain later milestones.
