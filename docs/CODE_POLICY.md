# Code Policy

The book is an executable engineering curriculum, not prose with decorative
fragments.

## Roles by language

- **Rust** is the main teaching runtime: model structures, GGUF parsing,
  scheduling, ownership, request lifecycle, server, metrics, and safe wrappers.
- **C** teaches deliberate native boundaries: stable ABI, arenas, dtype
  conversion, SIMD, low-level attention, and provider contracts.
- **Python** is reserved for clarity-first tensor demonstrations, scalar
  oracles, tiny numerical examples, and result visualization.

These choices express boundaries and ecosystem needs; they do not imply that a
language is inherently fast or slow.

## Repository roles

- `code/reference/`: independent, readable oracles and hand-computable cases.
- `code/mini-engine/`: staged, executable ENGINE-0 through ENGINE-10 runtime.
- `code/experiments/`: reproducible probes whose assumptions and outputs are
  recorded; experiments are not production APIs.

## Requirements

Instructional code must be executable at its milestone, understandable,
formatted, tested, tied to a chapter, and free of unnecessary framework weight.
Label pseudocode. Do not omit difficult lifetime, bounds, error, or numerical
logic while presenting a fragment as complete. State intentional limitations,
for example: “greedy sampling only, to isolate forward-pass correctness.”

Optimizations require a clarity-first oracle and differential tests before a
benchmark. Cover boundary sizes, MHA/GQA/MQA geometry where relevant, short and
long contexts, aligned and partial blocks, cancellation, and allocation
failure. Unsafe code requires a narrow boundary and a documented invariant for
every unsafe operation.

## Milestone discipline

Each ENGINE-N state must remain reconstructible by tagged examples, chapter
history, or an explicit compatibility layer. Later performance work must not
make earlier conceptual tests unreadable. Avoid copying Hermon; compare against
it only after the teaching design is understood.

## Dependencies and artifacts

Prefer small, pinned dependencies with a clear teaching purpose. Do not commit
model weights, secrets, machine paths, build outputs, or unexplained generated
files. Real-model tests must accept an external model path and skip clearly
when the fixture is unavailable.
