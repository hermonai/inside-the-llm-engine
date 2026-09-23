# Industrial recalibration execution report

Integration note, 2026-09-11: this historical architecture-draft report's pending
Chapter 7/publication items are superseded by `research/astra/chapter07-regeneration.md`.
Atlas order follows the current manifest, not historical page numbers below.
Native LaTeX/TikZ is an additional publication format, not 94 completed chapters.

2026-09-10. This is the attachment's requested 50-item report in a durable,
reviewable form. Scope is architecture and illustration infrastructure, not
mass generation of later chapters. Publication QA is recorded below when complete.

| # | Requested result | Recorded result |
| --- | --- | --- |
| 1 | Starting commit | `463518dd12b05a0fe71af689a00181512a6ad0ff` |
| 2 | Branch | `astra-visual-rewrite` |
| 3 | Architecture-audit commit | The commit introducing this report; resolve with `git log --format=%H --diff-filter=A -- docs/INDUSTRIAL_RECALIBRATION_STATUS.md`. The hash is also reported in the user handoff. |
| 4 | Unrelated work preserved | Existing STATUS, Chapter 1 manuscript, model-vs-engine text diagram and Chapter 8 research note were not edited or staged; hashes below. |
| 5 | Completed chapters | Seven historical completed chapters; no new completion state claimed. |
| 6 | Current engine milestone | ENGINE-2 checked tensor/linear substrate plus Chapter 7 Transformer Primitives v1; ENGINE-1 regression preserved. Full Transformer decoder still planned. |
| 7 | Tests | 173 Rust unit/integration tests and 2 compile-fail doctests passed. Six Python oracles and six release examples passed. New industrial QA rejects 15 fixture mutations. |
| 8 | Figure / atlas state | 35 SVG/TXT scenes: 22 existing Chapter 5/6 canonical, 9 existing prototypes and 4 new industrial reference plates. Five existing animations retained. 79 inventoried Unicode diagrams including the new curriculum spine. |
| 9 | Engines inspected | Hermon local source; llama.cpp/GGML pinned server documentation; vLLM, SGLang and TensorRT-LLM official documentation. Not five built/tested runtimes. |
| 10 | Versions / commits | Exact observed revisions and inspection depth in the [source map](../research/industrial-source-map.md). Moving documentation is not falsely assigned to those pins. |
| 11 | Image audit | Nine stages and five side annotations evaluated in the [pipeline audit](../research/pipeline-diagram-audit.md). |
| 12 | Image strengths | Clear orientation from input to output, attention/FFN grouping and introduction to state reuse. |
| 13 | Image corrections | Weight/activation shapes, unsupported architecture numbers, optional speculation, sampling composition, streaming overlap and workload-dependent bottlenecks. |
| 14 | Missing explicit ownership | Graph/capture, adapters, structured decoding and distributed transfer gates needed clear chapter homes. |
| 15 | Already covered in outline | Tensor/model sequence, GGUF, packed weights, KV, paging, batching, native boundaries, correctness and production operations. Present in outline does not mean implemented. |
| 16 | Deeper treatment | Cache identity, activation/expert communication, precision categories, feature combinations and actual multi-worker evidence. |
| 17 | Part count | Keep 15; proposed Part XIV retitle only. |
| 18 | Chapter count | Keep 94; no chapter inflation. |
| 19 | Retained positions | All physical chapter files and current canonical reading order remain unchanged. |
| 20 | Moved chapters | None physically. Proposed reading order 84 before 83 in the review map. |
| 21 | Splits | None proposed in this pass. |
| 22 | New chapters | None; named sections and acceptance gates fill the identified gaps. |
| 23 | Removal / merge | None; 14 existing appendices preserved. |
| 24 | Engine milestones | ENGINE-0 through ENGINE-10 retained; DIST-REFERENCE, TRANSFER-REFERENCE and MULTINODE-EVIDENCE are proposed later gates, not implemented engine versions. |
| 25 | Dependency graph | Generated 94-node acyclic spine with stable IDs, JSON, editable DOT and Unicode TXT. Exact chapter prerequisites preserved in prose. |
| 26 | Master architecture | New original SVG/TXT distinguishes serving, model semantics and execution, with persistent KV beside computation. |
| 27 | Performance metrics | TTFT, ITL, TPOT, E2E have a common-clock worked trace; later chapters own distributions, arrival process and SLO-qualified goodput. |
| 28 | Memory model | Disjoint physical categories, dense uniform formula, logical versus reserved bytes and shared-prefix fixture. |
| 29 | Prefill / decode | Chapters 20/26: different work shapes, chunking and interference; no universal compute-bound/memory-bound claim. |
| 30 | KV architecture | Chapters 21–22/28–34/49–50: semantics, physical pages, sharing, identity, eviction and pressure, with later architecture exceptions. |
| 31 | Scheduling | Chapters 23–27 own state transitions, token budgets, fairness, bounded queues and cancellation. |
| 32 | Attention kernels | Chapters 34/40–41/45 separate paging, online normalization, tiling and deterministic partition merge. |
| 33 | Quantization | Chapters 16–17/22/34 separate weight, activation and KV precision, scales, packing and accumulation. |
| 34 | Speculation | Chapters 51–53 derive verification before modern proposal comparisons and account for rejection/rollback costs. |
| 35 | MoE | Chapters 54–58 add activation dispatch/combine to expert residency; communication linked to 84. |
| 36 | Distributed inference | Chapters 84–85 require actual worker/node execution records where claimed, not simulator-derived scaling. |
| 37 | Disaggregation | Chapter 83 owns typed KV transfer, version/layout handshake, acknowledgment and failure injection after worker partitioning. |
| 38 | Graph / compiler | Chapters 45/48 own the first concrete case; 87 revisits abstraction/interchange without claiming an adopted standard. |
| 39 | Serving | Chapters 67–69 add constraint state, adapters, typed multimodal inputs and streaming contracts. |
| 40 | Observability / correctness | Chapters 59–66 and 70–73 retain oracles, equivalence, lifetime and measurement boundaries. |
| 41 | Security / failure | Chapter 72 covers untrusted artifacts, tenant/cache isolation and resource exhaustion; failure gates recur across the ladder. |
| 42 | Database research path | Chapters 82/86 connect paging/indexing/admission to concrete state policies while explicitly limiting durability/transaction analogies. |
| 43 | Hermon role | Longitudinal pinned case study; CURRENT/PREVIEW/LIBRARY/TARGET kept distinct, not vendor features imported into the default route. |
| 44 | Other-engine policy | Primary sources, dated docs, exact pins where code inspected, compatibility gates; no rankings or adopted vendor benchmark headlines. |
| 45 | Publication / visuals | Four editable reference plates, numerical mutation guards, all-chapter illustration contracts and a six-frame animation storyboard; later artwork remains planned. |
| 46 | Migration | Review-only proposals in separate documents; canonical outline, completed manuscripts and numerical APIs preserved. |
| 47 | Checks | Structural, diagram, math, figure, plan, numerical parity and full Rust checks passed. Publication/browser checks: pending final run. |
| 48 | Commit list | One bounded architecture commit planned; exact ID supplied in the final handoff. |
| 49 | Push status | Not pushed. This turn does not publish repository changes. |
| 50 | Next bounded task | Chapter 7 visual regeneration: embedding lookup/sequence ownership, two-pass RMSNorm, learned gain, epsilon, overflow/underflow and LayerNorm comparison; parity with existing Rust/Python, then standalone PDF/offline review. No Chapter 8 implementation in that pass. |

## Preservation fingerprints

SHA-256, captured before editing and checked again before commit:

| Pre-existing path | SHA-256 |
| --- | --- |
| `diagrams/runtime/model-vs-engine.txt` | `19551659f5bda42be43c6d0bd67eb98d552c55aec2098ed63a42b6e11676022a` |
| `docs/STATUS.md` | `f4f8f7721b52f10248b97b7694a48eeda6f02f87da9e3cdc9984c0d2c4f39438` |
| `manuscript/part-01/chapter-01-the-missing-half-of-ai.md` | `e210661b7e168d4c2e409075380f6e877c5c428780d3b21ff4993de6633a02e4` |
| `research/part-02/chapter-08-queries-keys-and-values.md` | `17e363d7d238b6fce996ab8578a59f4c2571727b84be4c0edd588153cf96e7df` |

## Reproducible validation

Run the existing CI workflow, which now also checks the industrial plan and
fixture mutations. Publication uses `publication/build.py`; visual review is
not replaced by `publication/check-page-bounds.py`. The browser script validates
the existing accessible animations/offline chapters and new industrial SVG bounds.
The atlas's first four pages are the new reference plates. Existing manuscript
exports are regression builds, not newly authored chapter editions.

Six release examples were executed as smoke tests. Their timing output was not
promoted into a new benchmark record and does not replace historical results.
One missing publication dependency (`svglib`) was installed in isolated scratch
storage; no project-wide or global dependency environment was modified.
