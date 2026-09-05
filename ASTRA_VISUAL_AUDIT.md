# Visual audit

Baseline: 78 canonical Unicode sources at book commit `8588ff8`, inspected
2026-09-05. All sources remain in place. The [per-diagram ledger](research/astra/diagram-audit.md)
assigns a disposition to every registered ID. Disposition describes the next
editorial pass; it does not silently modify or remove the source.

## Findings

- D050 has a correct scalar equation but crossing connectors visually pair
  different reduction indices. REDRAW with parallel same-index contributions.
- D049 and D054 draw A into B, suggesting transformation between operands.
  REDRAW with independent operand edges entering an explicit multiply operator.
- Ownership figures need UML composition only for actual stored fields, and
  labeled dependencies for borrowed views. A borrow is not object inheritance.
- Existing text grammar already distinguishes progression, bulk data, borrowing,
  and optional paths. Retain it; do not redefine arrows to incompatible meanings.
- Timing and roofline illustrations are analytical. New timelines must carry
  iteration labels rather than imply uniform elapsed time or measured speedups.
- Sequences can make token emission, normalization and transpose much clearer.
  Keep complete static panels even when a playable counterpart exists.

No diagram is removed. Three are rejected as direct vector-conversion sources
until corrected; their historical files are retained. The remaining sources
are retained or assigned a more suitable sequence/UML/memory/dataflow form in
the ledger. New prototypes demonstrate a visual language rather than replace
all 78 figures.

## Prototype review criteria

Ten semantic scene sources drive TXT, SVG and atlas pages. RoPE, cache growth,
and batching have explicit frame state plus keyboard-operated HTML playback.
QKV/RoPE/attention/cache prototypes are EDUCATIONAL MODEL, with a visible
implementation boundary. Runtime figures distinguish CURRENT default from
PREVIEW and LIBRARY components.

Review geometry at print width, label contrast in grayscale, Unicode glyph
coverage, caption completeness, and laptop/tablet scaling. Validate semantic
fixtures independently of geometry. Manifest/build checks cannot determine
whether an unlabeled arrow tells the truth: that remains a review obligation.
Results and actual limitations are recorded in [validation](research/astra/validation.md).
