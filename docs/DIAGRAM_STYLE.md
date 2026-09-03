# Diagram Style

Canonical diagrams are executable editorial specifications: they must answer a
technical question without depending on color, an image renderer, or nearby
prose. Store reusable diagrams as UTF-8 `.txt` files under `diagrams/`, register
them in `diagrams/INDEX.md`, and link to that source from the chapter.

## Visual grammar

Use one small, stable vocabulary throughout the book:

```text
┌────────────┐  component or owner       (weights) immutable data
│ [request]  │  mutable state            [cache]   mutable state
└────────────┘

──▶ control or ordered progression       ══▶ bulk data movement
┄┄▶ optional path or fallback             ──▷ reference or borrowed view
──✕ failure or rejected transition       ◀──▶ bidirectional relationship
```

Prefer `┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼` for structure. Do not use `+---+`, `|`,
ASCII arrows such as `->`, or backticks as connectors. Use labels on arrows
when the payload or event is not obvious. If a diagram needs a nonstandard
symbol, add a local legend.

## Semantic rules

- A box is a component, state owner, or bounded storage region—not decoration.
- Parentheses identify immutable data; square brackets identify mutable state.
- Direction must match causality or movement. Avoid arrows that merely mean
  “related to.”
- State machines name events on transitions and show terminal states explicitly.
- Ownership diagrams distinguish owner, borrower/view, mutation, and lifetime.
- Memory diagrams show logical indices separately from physical offsets.
- Tensor diagrams include shapes and identify the contracted or transformed axis.
- Performance diagrams separate measured values, analytical bounds, and
  conceptual trends. Never draw a predicted speedup as measured evidence.
- Current-system diagrams attach truth labels: `CURRENT`, `PREVIEW`, `LIBRARY`,
  `TARGET`, `HISTORICAL`, `EXTERNAL`, or `INFERENCE`.

## Diagram classes

| Class | Required content | Canonical example |
| --- | --- | --- |
| Architecture | boundaries, responsibilities, external interfaces | `runtime/inference-stack.txt` |
| Control flow | ordered actions, branches, terminal/failure paths | `runtime/request-to-token.txt` |
| Data flow | payload labels and transformations | `model/token-id-to-logits.txt` |
| State machine | named states, transition events, terminal states | `sampling/generation-terminal-state-machine.txt` |
| Ownership | owner, borrowers, mutations, release point | `tensor/tensor-ownership.txt` |
| Memory layout | logical coordinate, stride, physical offset | `tensor/row-major-offsets.txt` |
| Tensor shape | every input/output shape and contracted axis | `linear/gemm-shape-contract.txt` |
| Performance | metric/units and measured-versus-model status | `runtime/latency-decomposition.txt` |

## Layout and accessibility

- Keep every line at or below 100 display columns; narrower is preferred.
- Optimize for a monospaced terminal and grayscale print.
- Read top-to-bottom or left-to-right with few crossings. Split a crowded
  diagram instead of shrinking labels into ambiguity.
- Put the question the artifact answers in `diagrams/INDEX.md` and its topic
  `README.md`. A diagram that answers no question should not be canonical.
- Preserve meaningful whitespace, but do not rely on tabs.
- Use words as well as symbols for failure, mutation, and truth status.

## Authoring and review

1. State the technical question.
2. Choose one diagram class and the smallest sufficient visual grammar.
3. Draw the canonical `.txt` artifact, then embed or link it from prose.
4. Verify semantics against code, mathematics, and source status.
5. Run `python3 scripts/check-diagram-style.py` and
   `python3 scripts/check-diagram-width.py`.
6. Register additions or replacements in `diagrams/INDEX.md`.

Reviewers check meaning before polish: missing owners, false sequencing, hidden
copies, ambiguous axes, and unlabeled status are correctness defects.
