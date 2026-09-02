# Plain-Text Diagram Library

Core explanations use ASCII/plain text so they survive GitHub, terminals,
editors, AI context, and print. Store reusable diagrams as `.txt` files under
the topic directories; chapters may embed them but should link to the canonical
source.

Define a legend when more than one arrow/state convention appears. Recommended:
`-->` control flow, `==>` bulk data, `- - >` optional/fallback, `[X]` mutable
state, `(X)` immutable data. Show ownership, mutation, concurrency boundaries,
residency, and terminal/failure paths. Keep lines at a reviewable width and test
in a monospaced view. Diagrams are technical artifacts, not decoration.

Chapter 2's text/token/UTF-8 and chat-contract diagrams are indexed under
[`tokenizer/`](tokenizer/README.md).
Chapter 3's numerical model diagrams are indexed under
[`model/`](model/README.md).
Chapter 4's sampling and autoregressive ownership diagrams are indexed under
[`sampling/`](sampling/README.md).
