# Visual regeneration plan

This plan is the active regeneration workstream. `docs/STATUS.md` records
Chapter 8's completed QKV milestone (2026-09-12). Its original research was
preserved, refreshed and implemented. Existing COMPLETE milestones remain historical achievements,
not assertions that they have passed the new visual gates.

## Milestone A: audit and prototype system

Audit seven chapters, all existing diagrams, displayed equations and executable
examples. Refresh Hermon and its pinned llama.cpp source map. Establish one
semantic scene source, ten representative static plates, three animations,
manifest validation, an atlas, and PDF/HTML builds. Commit on
`astra-visual-rewrite` and push that branch. No mass manuscript rewrite.

## Chapter regeneration order

Architecture recalibration (2026-09-10): the
[industrial audit](docs/INDUSTRIAL_CURRICULUM_AUDIT.md) and
[migration proposal](docs/INDUSTRIAL_ARCHITECTURE_MIGRATION.md) add a
source-backed industrial coverage map and four reference plates. They do not
change canonical chapter states or the bounded order below. The proposed
Chapter 84-before-83 reading order requires a later reviewed outline migration.

Pass B is COMPLETE (2026-09-07): eight embedded canonical plates, executable
Rust/Python/fixture parity, actual ownership UML, six added Rust checks,
updated Labs 16-21 and vector PDF/offline HTML publication. See the
[Chapter 5 review](research/astra/chapter05-regeneration.md).

Pass C is COMPLETE (2026-09-08): Chapter 6 has fourteen embedded canonical
plates, two accessible step sequences, one executable numerical/address fixture,
six added tests, repaired operand edges, fresh source-status/orientation evidence,
and a 33-page standalone visual edition. Historical performance records retain
their losses and original pins. See the
[Chapter 6 review](research/astra/chapter06-regeneration.md).

Pass D completes the Chapter 7 visual edition: ten canonical plates, one
four-step numerical sequence, five added tests, Rust/Python/address parity and
standalone PDF/offline publication. See the [review](research/astra/chapter07-regeneration.md).
Pass E, the new QKV chapter, is COMPLETE: 15 Rust tests, an independent full
component oracle, ten canonical SVG/TikZ plates, Labs 39–48, a 19-page chapter
and a 169-page native ebook. No position or attention work is bundled into it.
The next new curriculum chapter is Chapter 9 per `docs/STATUS.md`; detailed
Chapters 1–4 visual regeneration remains a separate pending pass, not a reason
to claim those older plates have already received the same editorial redesign.

| Pass | Work | Exit gate |
| --- | --- | --- |
| B | Chapter 5 tensor pilot — COMPLETE | One matrix survives logical/physical/transpose/copy sequence; actual Rust UML; five parity reviews; static PDF complete |
| C | Chapter 6 matrix pilot — COMPLETE | Independent input edges; executable numerical/address sequence; measured locality evidence; source-backed CPU/GPU distinction; vector/offline publication |
| D | Chapter 7 normalization — COMPLETE | Ten vector plates, two-pass sequence, epsilon curve and overflow gates; unchanged operators and historical stress tests |
| E | Chapter 8 — COMPLETE | Preserved research implemented; checked QKV/head API, independent oracle, tests, labs, prose and ten native-vector figures; no attention yet |
| F | Chapters 1–4 | Progressive architecture, bytes, historical tiny model, sampling and lifecycle; retain all regressions |
| G | Chapters 9–13 | Position then attention then FFN then block then stack; prototype designs become canonical only after reference/oracle parity |
| H | Parts III–VI | File bytes and packed weights, profiling, dense KV then paging, request state then batching; measured claims gated |
| I | Parts VII–XII | Kernel/backend boundary, hardware, speculation, MoE, correctness and operations |
| J | Parts XIII–XV | Fresh production tours, explicitly future architecture, final integrated engine |

Keep all 94 specifications and chapter numbering for now. No evidence justifies
renumbering the curriculum during an infrastructure pass. Introduce source and
memory zooms within chapters rather than move production abstractions ahead of
their mathematical prerequisites.

## Per-chapter storyboard

Before prose changes, record hero/where-we-are, components, mechanism, changed
data, equation, physical layout, actual software, experiment, production
comparison, and synthesis. The [structured storyboard](figures/storyboards.md)
covers all seven existing chapters and the coming QKV/position/attention lessons.
Use D=4, H=2, head width 2 for new Transformer visuals; retain D=3 for ENGINE-1
regressions and chapter-specific stress fixtures. No silent fixture migration.

Each rewritten chapter must pass prose/math, math/code, code/oracle,
figure/semantics, and production/source parity, plus the visual gates in
[VISUAL_PEDAGOGY](docs/VISUAL_PEDAGOGY.md). Record old strengths, corrected
misconceptions, new artifacts, evidence and residual limitations in its research
note. A figure-count target is not an acceptance criterion.
