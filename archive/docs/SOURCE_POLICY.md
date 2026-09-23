# Source Policy

The book treats evidence as part of the implementation.

## Priority hierarchy

1. **Current source code** for claims about what Hermon or another engine
   executes. Record repository, commit, paths, and inspection date.
2. **Current canonical architecture documents** for intent, contracts, and
   system-wide context. In Hermon these presently include
   `CORE_ENGINE_ARCHITECTURE.md`, `DESIGN.md`, `ENGINE_STRATEGY.md`,
   `INTERNALS.md`, `STORAGE_ARCHITECTURE.md`, `KERNEL_DESIGN.md`,
   `PERFORMANCE.md`, `BENCHMARKING.md`, `INNOVATIONS.md`, and `ROADMAP.md`.
3. **Measured experiments** with the complete metadata required by
   `BENCHMARK_POLICY.md`.
4. **Primary external sources:** papers, specifications, official project
   documentation/repositories, and vendor documentation.
5. **Secondary explanations** used only for context and never to overrule
   primary evidence.

Source code can contain stale comments. Resolve disagreements by tracing the
actual call path and tests, then record the discrepancy rather than hiding it.

## Research-note format

Each substantial chapter has a note containing: question, sources, verified
facts, status classifications, open questions, terminology, code locations,
planned diagrams, experiments, and claims still requiring verification.
Include stable links or repository-relative paths plus a commit identifier.

## Hermon claim workflow

Before publication:

1. Inspect the current default request path and runtime selection.
2. Inspect relevant feature/env gates and fallback behavior.
3. Locate the owning crate/files and tests.
4. Check canonical docs for intent and limitations.
5. Classify as CURRENT, PREVIEW, LIBRARY, TARGET, or HISTORICAL.
6. Use EXTERNAL for other systems and INFERENCE for deductions.
7. Re-verify before final review; do not rely on an old inventory.

“The crate exists” does not prove that an API request reaches it. “A test
passes” does not prove supported model coverage. “A roadmap says shipped” does
not outrank a gate in source. Preserve these distinctions in prose.

## Citation discipline

Cite claims near the claim. Prefer the most direct source. Quote sparingly and
paraphrase with technical precision. Record access dates for changing web
sources and commit hashes for repositories. Separate a source's claim from the
book's inference.

## Performance and historical evidence

Never restate a target as measured performance. Never multiply isolated ratios
and describe the product as measured end to end. Historical measurements must
retain their original hardware, software, workload, and commit context.
Negative findings remain useful when their setup and limits are reproducible.
