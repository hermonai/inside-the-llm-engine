# Industrial curriculum recalibration

2026-09-10 · bounded architecture pass · **proposal, not a manuscript reset**.

The book already has the right broad progression: first build a correct token
generator, then understand tensors and Transformer operators, then load real
weights, manage state and schedule users, and finally optimize and operate the
system. The attached photograph covers mostly the first of these machines. It
does not justify replacing this progression with a long list of vendor features.

## State established before editing

The inspected branch was `astra-visual-rewrite`, at
`463518dd12b05a0fe71af689a00181512a6ad0ff`. Seven chapters were already written
and had completed their historical implementation gates. Chapter 5's eight and
Chapter 6's fourteen canonical vector plates were complete. Chapter 6 was **not**
the active unfinished task: Git identifies Chapter 7 visual regeneration next.
Chapter 8 remains the next new curriculum chapter, with pre-existing research.

Four pre-existing paths were kept out of this pass: `docs/STATUS.md`, the
Chapter 1 manuscript, `diagrams/runtime/model-vs-engine.txt`, and Chapter 8's
research note. Their contents are not staging material for this change. No
numerical API, test fixture, benchmark result or completed prose was rewritten.
The DNA Computing and Evolutor repositories are outside this prompt's scope.

## What changes, and what does not

Retain **15 parts, 94 chapters, 14 appendices** and ENGINE-0 through ENGINE-10.
The proposed [architecture](INDUSTRIAL_BOOK_ARCHITECTURE.md) preserves all eight
contract fields for every chapter and adds an explicit industrial amendment.
The [migration table](INDUSTRIAL_ARCHITECTURE_MIGRATION.md) records every chapter,
including those whose position stays unchanged. No additional chapter is needed
merely to name another engine or optimization.

The main structural proposal is to study Chapter 84 before Chapter 83: first
partition computation across workers, then transfer prompt-produced state to a
different decode pool. Stable chapter IDs avoid breaking links. The canonical
outline is deliberately not reordered during this audit. Rename Part XIV to
“Distributed Inference and Emerging Architectures” only in a later reviewed
migration, separating contemporary external capabilities from research targets.

## Gap analysis

“Present” below means represented in the outline, **not** already implemented.

| Area | Existing coverage | Gap and precise repair | Gate |
| --- | --- | --- | --- |
| Model semantics | Strong 2–13 progression | Keep model parameters distinct from activations; no invented proprietary dimensions. | Existing scalar/oracle contracts; Chapter 7 visual parity next. |
| Files and quantization | Strong GGUF/packing sequence 14–18 | Add safetensors comparison to 14; weight/activation/KV precision distinctions to 16, 22, 34. | Pinned format specification; malformed-input tests; numerical error budget. |
| Prefill/decode | Present 19–22 | Treat phases as different work shapes, not unconditional hardware bottlenecks. | One-clock timestamps; measured workload and shapes. |
| Scheduling | Strong lifecycle/batching 23–27 | Give chunked prefill, fairness and backpressure concrete traces. | Bounded queues, cancellation and tail-latency counterexample. |
| KV memory | Extensive 21–22, 28–34, 49–50 | Explicit logical/physical/reserved accounting, cache identity and sharing. | Refcount/lifetime proof; physical bytes counted once. |
| Attention kernels | Present 34, 40–41, 45 | Connect recurrence to tiling without confusing FlashAttention with paging. | Full algorithm reading and scalar equivalence. |
| Graphs / compilation | Too deferred toward 87 | Put the first execution-plan/capture case in 45/48; revisit protocol design in 87. | Dynamic-shape/address invalidation and replay test. |
| Speculation | Strong 51–53 foundation | Add verified modern proposal variants as comparisons, not replacements for exact acceptance logic. | Full-paper derivation, first rejection, correction and rollback. |
| MoE | Strong residency focus 54–58 | Add activation dispatch/combine and link expert communication to 84. | Routing equivalence and accounting for communication. |
| Structured outputs | Insufficient explicit ownership | Chapter 67 owns grammar state, masks and unsatisfiable continuation; 4 only previews it. | Constraint-state and cancellation tests; validity ≠ truth. |
| Adapters / multimodal | Routing home exists in 68 | Adapter identity/residency and typed encoder/projector inputs become named contracts. | No reuse across incompatible adapter/model context. |
| Distributed inference | 83–85 present, but too future/simulator-oriented | Add real multi-worker reference and fault-injected transfer; study 84 before 83. | CPU correctness reference possible; real GPU/node scaling requires hardware records. |
| Operations | Strong 59–73 | Make metrics, SLO-goodput, security and failure boundaries cross-cutting. | Reproducible traces and declared failure/censoring policy. |
| Database analogy | Present 82/86 | Teach allocation, indexing, admission and state transfer as concrete analogies, not proof of database-equivalent semantics. | Identify exactly where transaction/durability analogies break. |

The [industrial landscape](../research/industrial-engine-landscape.md) records
which external claims were documented, inspected or not assessed. It deliberately
does not turn documentation-index links into tested feature support. Current
external implementations motivate teaching material; they do not change the
teaching engine's implementation status or Hermon's default execution path.

## Two interacting machines and one service

The model specifies a function over typed values and retained state. The engine
chooses representations, memory placement, execution plans and kernels that
implement that function. Serving decides which requests receive work, how much
work may be admitted and how results or failures reach clients. A request can
wait while no model operation is running for it. A graph can execute several
requests together. A shared KV page can outlive the connection that first created
it. These interactions are why one serial “input to output” chain is insufficient.

The four [new reference plates](../figures/industrial-atlas.md) separate stack,
ownership, latency and memory. Every chapter now has a
[chapter-specific visual contract](../figures/CHAPTER_ILLUSTRATION_PLAN.md),
derived from its real question, experiments and correctness tests. These are
production specifications; only rendered, inspected figures count as delivered.

## Skeptical review and revisions

This is an author-performed review through ten lenses, **not** a claim that ten
independent experts or agents reviewed the book.

| Lens | Objection | Revision / remaining gate |
| --- | --- | --- |
| Beginner | A master chart may overwhelm before the first working program. | Keep it an overview with three focused zooms; preserve Chapters 1–4's executable loop. |
| ML semantics | A cache box can look like a required mathematical stage. | Remove sequential numbering; show KV read/write beside execution and name shape contracts. |
| Numerical analysis | Correct formulas can still imply universal layouts. | Dense uniform KV assumptions explicit; latency/byte examples marked synthetic and executable. |
| Kernel engineer | Compilation, fusion and graph replay may be conflated. | Distinct objects in Chapters 45/48, not a single optimization checkbox. |
| Memory engineer | Cancellation arrows can imply immediate free. | Route cancel to scheduler; separate retirement, in-flight completion and refcount release. |
| Serving engineer | Average token time hides stalls and queue pressure. | Distinct TTFT/ITL/TPOT/E2E and later SLO-qualified distributions. |
| Distributed engineer | A simulated network is not a distributed implementation. | Add multi-worker correctness and actual transfer/failure gates; no inferred scaling. |
| Security reviewer | Shared prefixes/adapters can cross tenant or model boundaries. | Explicit reuse identity, isolation policy and Chapter 72 threat model. |
| Researcher | A vendor feature survey dates quickly. | Teach durable contracts; keep discovery gaps and source pins visible. |
| Editor / visual reviewer | 94 new generic diagrams would dilute meaning. | Preserve mature assets; per-chapter questions and counterexamples; no mass-produced later manuscript artwork. |

## Completion boundary

This pass supplies an audit, source ledger, migration/roadmap proposal, dependency
graph, milestone map, four vector reference plates, all-chapter illustration plan
and a pipeline animation storyboard. It does not complete Chapters 8–94, implement
distributed inference, establish benchmark superiority or finalize Chapter 7's
new visual edition. See [execution report](INDUSTRIAL_RECALIBRATION_STATUS.md)
for the checks actually run and the exact next bounded chapter task.
