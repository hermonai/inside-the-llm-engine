# Visual regeneration plan

This plan is the active regeneration workstream. `docs/STATUS.md` retains the
in-progress Chapter 8 work exactly as it arrived; that chapter is still
RESEARCHING. Existing COMPLETE milestones remain historical achievements,
not assertions that they have passed the new visual gates.

## Milestone A: audit and prototype system

Audit seven chapters, all existing diagrams, displayed equations and executable
examples. Refresh Hermon and its pinned llama.cpp source map. Establish one
semantic scene source, ten representative static plates, three animations,
manifest validation, an atlas, and PDF/HTML builds. Commit on
`astra-visual-rewrite` and push that branch. No mass manuscript rewrite.

## Chapter regeneration order

| Pass | Work | Exit gate |
| --- | --- | --- |
| B | Chapter 5 tensor pilot | One matrix survives logical/physical/transpose/copy sequence; actual Rust UML; five parity tests; static PDF complete |
| C | Chapter 6 matrix pilot | Fix three misleading diagrams; numerical row sequence; locality comparison with source-backed costs |
| D | Chapter 7 normalization | Two-pass numerical sequence, epsilon/overflow plates; unchanged oracle and stress tests |
| E | Complete Chapter 8 | Reuse preserved research; checked QKV/head API, independent oracle, tests, labs, prose and figures; no attention yet |
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
