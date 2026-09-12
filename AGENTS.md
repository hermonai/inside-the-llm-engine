# Repository Operating Guide

Active authoring decision: read `TEXTBOOK_STANDARD.md` and `tex/README.md`.
Write new and revised chapters directly in `tex/chapters/` and illustrations
in `tex/figures/`. Build with `make textbook-check` and `make textbook`.
Markdown and the old publication exporter are historical evidence, not the
source of the new textbook. Preserve their regression gates.

Historical visual regeneration is a separate bounded workstream. Read
`ASTRA_REGENERATION_PLAN.md` and `figures/VISUAL_LANGUAGE.md` before visual
changes. The first milestone established prototypes and builds; the visual
pilots for Chapters 5, 6 and 7 are complete. Chapter 8 is now complete with
checked QKV/head semantics, ten native-vector figures, full-component parity
and Labs 39–48. This does not regenerate all chapters. Next bounded new
curriculum task: Chapter 10, causal self-attention with native TikZ illustrations.

Read this file, `docs/STATUS.md`, `docs/ROADMAP.md`, and `git status` before
working. This repository is the open book **Inside the LLM Engine: From model
weights and KV memory to industrial inference serving**. Its mission is to take a programmer from
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

- `tex/chapters/`: active authored LaTeX chapter prose.
- `tex/figures/`, `tex/worked/`: native illustrations and worked problems.
- `manuscript/part-NN/`: preserved historical Markdown chapters and part indexes.
- `code/reference/`: independent, clarity-first oracles.
- `code/mini-engine/`: the staged Rust teaching engine.
- `code/experiments/`: disposable but reproducible measurements.
- `research/`: evidence logs; one substantial note per chapter.
- `diagrams/`: historical `.txt` semantic references; not textbook artwork.
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
8. Author native TikZ mechanism diagrams; retain useful historical semantics.
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

- Follow `TEXTBOOK_STANDARD.md` for active artwork: native geometric edges,
  typeset labels, captions and color/grayscale review. Every figure answers a
  named question. Historical Unicode diagrams retain `docs/DIAGRAM_STYLE.md`
  and their inventory/width checks; do not use them as print illustrations.
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

Use `tex/chapters/chNN.tex` for active prose and `part-NN` for research/labs.
Use `ENGINE-N` for curriculum
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
progress: Chapters 5–9, Tensor Substrate v1, ENGINE-2's checked reference and
blocked scalar kernels, Transformer Primitives v1, checked QKV, standard RoPE and Labs 16–58
are complete. The 65-plate historical atlas has native TikZ counterparts.
The active LaTeX-first textbook uses 65 native figures and 27 new worked
synthesis problems. Chapter 1 is rewritten; Chapters 2–9 retain substantial
reviewed material with corrections and new pedagogy. Do not call this a full rewrite.
The full suite contains 218 unit/integration tests and two compile-fail doctests.
The authoritative state and next task are in
`docs/STATUS.md`.

Chapter 7 visual regeneration is complete: ten canonical plates connect token
identity, table storage, activation ownership and two-pass RMSNorm, with an
independent oracle and checked fixture. Pass D in `ASTRA_REGENERATION_PLAN.md`
is complete. Chapter 6
now has fourteen canonical vector plates, two step sequences and executable
Rust/Python/figure/source parity; its three historical benchmark records remain
unchanged. Chapter 8 adds QKV; Chapter 9 adds positioned Q/K and unchanged V,
with explicit pairing, partial prefix, F64 phases and failure-atomic in-place mutation.

The next new curriculum chapter is Chapter 10 — Causal Self-Attention. Start
from Chapter 9 positioned Q/K and unchanged V; derive scaled scores, per-query
causal visibility, stable softmax and value mixing with explicit MHA/GQA/MQA
geometry. Build a checked dense scalar reference, independent oracle and native
TikZ mechanism plates. Preserve all earlier tests. Do not begin persistent KV
caching, FFN, a complete decoder, GGUF, quantization, SIMD, BLAS or accelerators.
