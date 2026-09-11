# Industrial roadmap and engine milestones

REVIEW PROPOSAL. The canonical [roadmap](ROADMAP.md), [outline](OUTLINE.md),
chapter states and numerical APIs remain unchanged. The detailed proposed
[chapter map](INDUSTRIAL_BOOK_ARCHITECTURE.md) is an extension of the outline,
not a second manuscript. Its generated [dependency graph](industrial-dependencies.json)
has 94 nodes and an acyclic prerequisite spine; the
[Unicode reading map](../diagrams/runtime/industrial-curriculum.txt) and
[editable graph source](../diagrams/runtime/industrial-curriculum.dot) expose it.
Exact prose prerequisites remain beside each chapter; the graph is a course
spine, not a claim to encode every concept dependency.

## Preserve the executable ladder

| Existing milestone | Retained result / gate | Industrial clarification |
| --- | --- | --- |
| ENGINE-0 | Request-to-terminal lifecycle | Identity, cancellation and exactly one outcome remain foundational. |
| ENGINE-1 | Tiny complete autoregressive model | Historical fixture remains a regression, not a Transformer claim. |
| ENGINE-2 | Checked tensors and linear algebra progressing to the decoder in Chapter 13 | Existing Chapter 5–7 APIs stay stable; later position/attention/FFN added incrementally. |
| ENGINE-3 | Real GGUF runner, Chapter 18 | File metadata and precision are checked contracts. |
| ENGINE-4 | Cached generation, Chapter 21 | Cache semantics before physical paging; validate cached/uncached equivalence. |
| ENGINE-5 | Single-user server, Chapter 23 | Protocol behavior separate from numerical operations. |
| ENGINE-6 | Multi-user batching, Chapters 26–27 | Chunking, fairness and bounded queue/cancel tests. |
| ENGINE-7 | Paged inference, Chapter 34 | Logical handle, physical page, sharing and accounting invariants. |
| ENGINE-8 | Native kernels, Chapter 41 | Scalar oracle, stable ABI and lifetime checks before speed claims. |
| ENGINE-9 | Provider execution, Chapter 48 | Add graph/capture invalidation case; record real dispatch and transfer costs. |
| ENGINE-10 | Production-shaped engine, Chapter 73 | Supported combination matrix, SLO evidence, observability and threat model. |

“Production-shaped” describes learning scope, not an assurance that an unaudited
teaching engine is ready to serve untrusted production traffic.

## Named research gates after the existing ladder

Do not renumber ENGINE milestones to make research look already implemented.

| Proposed gate | Chapter | Bounded artifact | What it cannot prove |
| --- | --- | --- | --- |
| DIST-REFERENCE | 84, read first | Two actual workers implement a small partitioned operation; compare with one-worker oracle and inject a worker failure. CPU processes are an acceptable reference. | GPU efficiency, network scale or collective-library performance. |
| TRANSFER-REFERENCE | 83, read second | Serialize a small typed KV fixture; negotiate layout/version; transfer between processes; acknowledge ownership; reject stale/duplicate/incomplete messages. | Production disaggregation throughput or reliability. |
| MULTINODE-EVIDENCE | 85 | Run on distinct nodes, record transport/placement/timeout, inject failure and compare semantics. | General scaling from one test setup. Hardware absence leaves this gate incomplete. |
| REPLACEMENT-PROOF | 94 | Swap one component behind a stable boundary; preserve correctness and rollback; measure on the declared workload. | A claim that every surveyed optimization has been implemented. |

## Production sequence

1. Finish Chapter 7 visual regeneration against existing embedding/RMSNorm code.
2. Finish Chapter 8 QKV/head primitives with the preserved research note; no attention implementation yet.
3. Continue the tested Transformer sequence through Chapter 13, then the existing ENGINE ladder.
4. Apply each industrial amendment at its accountable chapter, with fresh claim-level source checks.
5. Review the proposed Part XIV retitle and 84-before-83 reading order when that part is scheduled; do not move files now.

Each chapter requires a mechanism illustration and a meaningful counterexample
where visual explanation helps; diagram count alone is never an exit gate.
