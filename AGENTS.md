# Repository Operating Guide

Visual regeneration is now a separate bounded workstream. Read
`ASTRA_REGENERATION_PLAN.md` and `figures/VISUAL_LANGUAGE.md` before visual
changes. The first milestone establishes prototypes and builds; it does not
complete Chapter 8 or regenerate all chapters. Preserve the pre-existing
Chapter 8 research and working-tree status changes. Next regeneration pilot:
Chapter 5, then 6 and 7, before completing Chapter 8.

Read this file, `docs/STATUS.md`, `docs/ROADMAP.md`, and `git status` before
working. This repository is the open book **Inside the LLM Engine: From First
Token to Production-Grade Inference**. Its mission is to take a programmer from
the first token through the design, implementation, verification, measurement,
and operation of a production-grade inference engine.

## Source of truth

The operational hierarchy is:

1. `docs/STATUS.md` for current project state and next task.
2. `docs/BOOK_CONSTITUTION.md` for non-negotiable editorial rules.
3. `docs/OUTLINE.md` for the 94 chapter specifications.
4. `docs/CHAPTER_CONTRACT.md` for a finished chapter's obligations.
5. `docs/SOURCE_POLICY.md` for evidence and Hermon status classification.
6. `docs/CODE_POLICY.md`, `docs/MATH_STYLE.md`, `docs/DIAGRAM_STYLE.md`,
   `docs/STYLE_GUIDE.md`, and `docs/BENCHMARK_POLICY.md` for domain-specific
   rules.

`BOOK.md` is the public table of contents; it must remain consistent with the
detailed outline. `README.md` describes the project but does not override the
documents above.

## Repository architecture

- `manuscript/part-NN/`: polished chapter prose and part indexes.
- `code/reference/`: independent, clarity-first oracles.
- `code/mini-engine/`: the staged Rust teaching engine.
- `code/experiments/`: disposable but reproducible measurements.
- `research/`: evidence logs; one substantial note per chapter.
- `diagrams/`: canonical `.txt` diagrams, grouped by system area.
- `docs/`: editorial contracts, curriculum, status, and policies.
- `scripts/`: repository checks and reproducibility helpers.

Do not put prose in `code/`, polished claims in raw research notes, or unverified
Hermon statements in the manuscript.

## Chapter workflow

1. Claim one bounded chapter or task in `docs/STATUS.md`.
2. Read its `docs/OUTLINE.md` specification and prerequisites.
3. Create or update its research note with questions, sources, verified facts,
   open questions, terminology, code locations, diagrams, and experiments.
4. Classify Hermon claims as CURRENT, PREVIEW, LIBRARY, TARGET, HISTORICAL,
   EXTERNAL, or INFERENCE before drafting.
5. Draft the mental model and derivation before optimization detail.
6. Implement the chapter milestone and independent oracle where required.
7. Add correctness tests before performance measurements.
8. Add or revise canonical Unicode text diagrams.
9. Run technical, editorial, cross-link, and terminology passes.
10. Update `docs/STATUS.md`. Use only: PLANNED, RESEARCHING, OUTLINED,
    DRAFTING, CODE-COMPLETE, TECH-REVIEW, EDIT-REVIEW, COMPLETE.

No chapter is COMPLETE merely because prose exists.

## Factual verification

For current Hermon behavior, inspect the current source at a recorded commit.
Code outranks documentation. Current canonical architecture documents rank next;
reproducible measurements follow; primary external sources follow those. Do not
promote a file's existence to an end-to-end claim, a target to current behavior,
or an estimate to a measurement. See `docs/SOURCE_POLICY.md`.

The Hermon repository is a case study, not a dependency of `mini-engine` and not
the book's subject. Re-verify its source and tests before every “Inside Hermon”
section; `research/hermon/README.md` is a dated reconnaissance map, not eternal
truth.

## Diagrams, code, math, and benchmarks

- Follow `docs/DIAGRAM_STYLE.md`: every canonical Unicode diagram answers a
  named question, uses the shared arrow/state grammar, fits 100 display
  columns, and is registered in `diagrams/INDEX.md` with a one-line purpose.
- Main teaching code is Rust. Use Python for independent numerical clarity and C
  for explicit kernel/ABI lessons. Label pseudocode. Each milestone must run,
  test, and remain understandable.
- Follow `docs/MATH_STYLE.md`: every central equation defines symbols and
  shapes, distinguishes vectors/matrices from scalars, states units and
  approximation status, and maps to an oracle or test when numerical.
- Every benchmark records commit, build, hardware, software, model,
  quantization, workload, concurrency, mode/provider, cache state, repetitions,
  statistic, and control. Incorrect output invalidates the benchmark.

## Naming and writing

Use `part-NN` and `chapter-NN-topic.md`. Use `ENGINE-N` for curriculum
milestones, uppercase status labels, and canonical terms from
`docs/TERMINOLOGY.md`. Use the full book title on first public reference.
Avoid mystical or dismissive explanations. Ask: what data exists, what shape is
it, where does it live, who owns and mutates it, what does it cost, what can run
concurrently, how can it fail, and how is it proved?

## Git hygiene

Inspect the working tree before editing and preserve unrelated work. Prefer
atomic conventional commits. Never rewrite history or use destructive cleanup
commands. Run `git diff --check` and relevant tests before committing. Do not
commit models, benchmark blobs, generated build output, secrets, or machine
paths. Never push unless the user explicitly authorizes it.

## Current state and next task

Phase 0 repository architecture and Phase 1 are complete. Phase 2 is in
progress: Chapters 5–7, Tensor Substrate v1, ENGINE-2's checked reference and
blocked scalar kernels, Transformer Primitives v1, Labs 16–38, 78 canonical
Unicode text diagrams, and the Chapter 1–6 diagram/math retrofit are complete.
The full suite contains 163 tests. The authoritative state and next task are in
`docs/STATUS.md`.

The next bounded task is Chapter 8 — Queries, Keys, and Values. Start from the
ENGINE-2 GEMM/GEMV layer and Chapter 7's residual-width activations; derive and
build checked Q/K/V projections and head-shape transformations without
beginning attention scores, masking, attention softmax, value aggregation,
RoPE, KV caching, GGUF, quantization, SIMD, BLAS, or accelerator execution.
