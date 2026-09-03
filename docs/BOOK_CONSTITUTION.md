# Book Constitution

This document defines the editorial invariants of **Inside the LLM Engine**.
Process details may change; these rules require an explicit, recorded decision
to change.

## Mission and reader promise

The book teaches a motivated programmer to explain, implement, test, profile,
optimize, operate, and extend a production-grade LLM inference engine. The
progression is the product: concepts, executable milestones, deliberate
failures, production evidence, and verification must advance together.

The book connects language-model theory, numerical computing, model
representation, systems engineering, hardware execution, serving, and
operations. It must not become an isolated collection of tutorials or product
documentation for Hermon.

## Truth categories

Every claim about an implementation has a known status:

- **CURRENT** — confirmed behavior on the default current execution path.
- **PREVIEW** — implemented and runnable, but explicitly gated or non-default.
- **LIBRARY** — a usable component exists, but it is not integrated end to end.
- **TARGET** — intended or designed future architecture.
- **HISTORICAL** — a previous implementation or result used as evidence.
- **EXTERNAL** — behavior of another project, paper, vendor, or system.
- **INFERENCE** — an interpretation derived from evidence rather than an
  implementation claim.

Literal labels are optional in polished prose when the status is otherwise
unambiguous. The research note must record it. Never turn TARGET into CURRENT,
source existence into release validation, or an estimate into a measurement.

Part XIV adds a second reader-facing vocabulary: **TODAY**, **NEAR TERM**,
**FRONTIER**, and **RESEARCH QUESTION**. Map TODAY to verified current systems;
use the others to keep invention visibly separate from implementation.

## Teaching sequence

Major concepts should normally progress through:

```text
question ──▶ why the problem exists ──▶ mental model ──▶ mathematical model
         ──▶ reference implementation ──▶ failure at scale ──▶ systems design
         ──▶ optimized implementation ──▶ Hermon evidence ──▶ correctness proof
         ──▶ benchmark ──▶ limits and next generation
```

Show the naive design before the optimization. Do not start with an equation
dump or use jargon before establishing its data, shape, ownership, and purpose.
Central topics may require 6,000–12,000 words or more; conceptual chapters may
be shorter. Completeness controls length, not a quota.

## Three recurring journeys

- **Follow the token:** text → tokenizer → ID → embedding → layers → logits →
  sampler → new ID → bytes → stream.
- **Follow the byte:** GGUF → mapped/loaded tensor → packed execution → KV
  write → host/device residency → attention read → output.
- **Follow the owner:** creation → mutation rights → references → pinning →
  eviction → release → cancellation/failure behavior.

These journeys keep logical semantics, physical representation, and lifetime
rules connected.

## Hermon case-study rules

Hermon is the primary industrial reference, not the teaching implementation.
Before an “Inside Hermon” section, inspect current source, relevant tests,
canonical architecture documents, runtime gating, and benchmark evidence. Record
the commit and status category. Preserve verified failed experiments; negative
results often explain architecture better than success stories.

## Correctness before speed

The proof ladder is: shape validation, hand-computable scalar tests,
independent reference, optimized-path differential, order/thread differential,
real-model differential, sanitizers/fuzzing, then performance comparison. Cover
boundary shapes and cache/prefix alignment cases. A fast incorrect result has
no benchmark value.

## Model support discipline

Do not claim support from a family name. Verify metadata, tensor names,
normalization, positional encoding, attention/mask/head geometry, activation,
MoE or recurrent semantics, output head, quantization, tokenizer, and chat
template, then pass equivalence tests.

## Diagrams and prose

Core architecture must remain understandable in plain Markdown, terminals,
editors, AI context windows, and print. Prefer polished Unicode box-drawing
text with defined arrow meanings; show state, ownership, concurrency, and
residency. Keep a monospaced, color-free representation that needs no external
renderer. `DIAGRAM_STYLE.md` defines the grammar and inventory contract;
`MATH_STYLE.md` defines notation, shapes, units, and numerical evidence. The
voice is technical, direct, concrete, curious, and never
mystical or condescending.

## Completion

A chapter is not complete because prose exists. It must satisfy
`CHAPTER_CONTRACT.md`, including evidence, diagrams, code where required,
correctness, review, cross-links, terminology, exercises, and references. The
authoritative completion ledger is `STATUS.md`.
