# Unicode Text Diagram Library

Core explanations use polished Unicode box-drawing text so they remain readable
on GitHub, in modern terminals and editors, in AI context, and in print without
an external renderer. Store reusable diagrams as `.txt` files under the topic
directories; chapters may embed them but should link to the canonical source.

Define a legend when more than one arrow/state convention appears. Recommended:
`──▶` control flow, `══▶` bulk data, `┄┄▶` optional/fallback, `[X]` mutable
state, and `(X)` immutable data. Use `│`, `─`, and box-drawing junctions instead
of legacy `|`, `-`, and `+` borders. Show ownership, mutation, concurrency
boundaries, residency, and terminal/failure paths. Keep lines within 100 display
columns and test in a monospaced view. Diagrams are technical artifacts, not
decoration. `scripts/upgrade-text-diagrams.py` upgrades legacy connectors;
`scripts/check-diagram-width.py` performs Unicode-aware width validation.

Chapter 2's text/token/UTF-8 and chat-contract diagrams are indexed under
[`tokenizer/`](tokenizer/README.md).
Chapter 3's numerical model diagrams are indexed under
[`model/`](model/README.md).
Chapter 4's sampling and autoregressive ownership diagrams are indexed under
[`sampling/`](sampling/README.md).
Chapter 5's tensor layout and lifetime diagrams are indexed under
[`tensor/`](tensor/README.md).
Chapter 6's dot-product, GEMV, GEMM, locality, and kernel-boundary diagrams are
indexed under [`linear/`](linear/README.md).
