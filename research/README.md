# Research

Evidence behind the book. Nothing here is manuscript prose.

## Second edition

- [`FRONTIER.md`](FRONTIER.md) — the dated, sourced ledger of the state of the
  art: hardware specifications, model configurations and techniques, each with
  its primary source, the date it was checked, and its maturity. No frontier
  claim enters a chapter without an entry here (`AUTHORING.md` §6).
- `measurements/` — one record per measured number printed in the book:
  date, hardware, software versions, model file and quantization, command,
  workload, repetitions, statistic and raw output (`AUTHORING.md` §5).

## First edition (preserved)

The per-chapter research notes and benchmark records below were written for
the first edition's Chapters 1–9, now Appendix A and Chapter 14. They remain
valid evidence for those texts; their process rules were superseded by
`AUTHORING.md` (the old rules are in `archive/docs/SOURCE_POLICY.md`).

- [Part I notes](part-01/README.md), from
  [Chapter 1 — The Missing Half of AI](part-01/chapter-01-the-missing-half-of-ai.md).
- [Part II notes](part-02/README.md), through Chapter 9 (RoPE).
- `benchmarks/` — loop order, blocked GEMM, GEMV versus GEMM, traversal order,
  sampling cost and projection scaling, with their losses.
- `hermon/README.md` — a dated reconnaissance of Hermon; re-check its claims
  against source before use.
- [Diagram and math retrofit](editorial/diagram-math-retrofit.md), `astra/`,
  `textbook/` — records of the first edition's visual and LaTeX passes.
