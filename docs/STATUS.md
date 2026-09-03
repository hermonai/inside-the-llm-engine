# Project Status

Last updated: 2026-09-03.

## Phase 0 ledger

| Area | Status | Evidence / next gate |
| --- | --- | --- |
| Repository bootstrap | COMPLETE | Empty public repository cloned and structured |
| Book constitution and policies | COMPLETE | Core editorial/source/code/math/style/benchmark contracts created |
| Master outline | OUTLINED | 15 parts, 94 chapter authoring specifications; review again before each phase |
| Public README and BOOK | COMPLETE | Launch-facing overview and table of contents agree |
| Glossary and terminology | IN PROGRESS | Chapters 1–6 system, token, numerical model, sampling, tensor-memory, linear-algebra, and streaming terms added; expand with each chapter |
| Hermon reconnaissance | COMPLETE | Initial map plus Chapters 1–6 request/tokenizer/logit/sampling/tensor/kernel boundaries verified at `hermon` commit `472a44c` |
| Manuscript part indexes | COMPLETE | 15 part contracts plus appendices scaffolded |
| Diagram system | COMPLETE | Sixty-three inventoried canonical Unicode diagrams; shared grammar plus automated style and display-width gates |
| Diagram/math retrofit | COMPLETE | Chapters 1–6 audited; 11 diagrams added, 2 redesigned, 47 equation blocks standardized, and 20 explicit shape declarations added |
| Research system | COMPLETE | Inventories and note templates established |
| Code project | COMPLETE | ENGINE-2 adds dependency-free checked reference and blocked scalar kernels over Tensor Substrate v1; 133 tests pass |
| Initial CI | COMPLETE | Structure, links, diagram style/width, math structure, Rust format/check/test/Clippy workflow added |
| License | PLANNED | Maintainers must choose prose and code licensing; no license inferred from Hermon |

Phase 0 remains complete as repository architecture. Phase 1 completion is
tracked separately below.

## Phase 1 ledger

| Scope | Status | Evidence / next gate |
| --- | --- | --- |
| Part I (Ch. 1–4) | COMPLETE | Four reviewed chapters, complete ENGINE-1 generation loop, Labs 1–15, and independent numerical oracles |
| Chapter 1 — The Missing Half of AI | COMPLETE | 6,913-word reviewed chapter, primary-source research, seven canonical diagrams |
| Chapter 2 — From Text to Tokens | COMPLETE | 7,373-word reviewed chapter, primary-source research, nine canonical diagrams, two-tokenizer comparison |
| Chapter 3 — The Smallest Possible Language Model | COMPLETE | 6,373-word reviewed chapter, primary-source research, eight canonical diagrams, full-vector Python oracle |
| Chapter 4 — Logits, Sampling, and the Autoregressive Loop | COMPLETE | 6,306-word reviewed chapter, primary-source research, nine canonical diagrams, fixed-draw Python oracle |
| ENGINE-0 | COMPLETE | Dependency-free tokenized request/runtime/stream lifecycle; byte oracle, BPE, chat/template contract, strict UTF-8 framing; 37 tests and full Rust gate pass |
| ENGINE-1 | COMPLETE | Immutable model logits; separate greedy and stochastic selection; stable softmax, temperature, top-k/top-p, categorical sampling, request-owned seeded RNG, feedback, and single terminal owner; 83 tests at the Phase 1 boundary, 133 in the current full suite |
| Lab 1 — Generate One Token Manually | COMPLETE | Independent candidate oracle plus CHECK/BUILD/BREAK/EXTEND exercise |
| Labs 2–4 — Tokenization / UTF-8 / chat template | COMPLETE | Hand BPE, split-byte streaming, malformed terminal policy, and wrong-template experiments |
| Labs 5–8 — Numerical forward / causality / context / shape | COMPLETE | Full hand logits, one-weight intervention, same-last-token proof, and typed malformed-shape failures |
| Labs 9–15 — Sampling / feedback / failure | COMPLETE | Stable softmax, temperature, fixed-draw categorical selection, top-k/top-p, full-loop tracing, seeded reproduction, and typed sampler failures |

## Phase 2 ledger

| Scope | Status | Evidence / next gate |
| --- | --- | --- |
| Part II (Ch. 5–13) | IN PROGRESS | Chapters 5–6 complete; Chapter 7 is next |
| Chapter 5 — Tensors Without Magic | COMPLETE | 6,083-word reviewed chapter, primary-source research, thirteen canonical diagrams, traversal record, and independent offset oracle |
| Tensor Substrate v1 | COMPLETE | Owned canonical `f32` tensors, immutable strided views, exclusive canonical mutation, checked indexing/extent arithmetic, explicit materialization, and ENGINE-1 parameter migration |
| Labs 16–21 — Tensor memory | COMPLETE | Hand offsets, metadata transpose, reshape gate, non-contiguous copy, overflow failures, and aliasing/mutation exercises |
| Chapter 6 — Matrix Multiplication: The Engine Room | COMPLETE | 7,480-word reviewed chapter, primary-source research, seventeen canonical diagrams, three performance records, and independent numerical oracle |
| ENGINE-2 / Linear Algebra Kernel Layer v1 | COMPLETE | Strided dot/GEMV/GEMM reference kernels, canonical-only blocked scalar GEMM, explicit layout/ownership/error contracts, and ENGINE-1 projection migration |
| Labs 22–29 — Linear algebra kernels | COMPLETE | Hand dot/GEMV/GEMM, loop-order offsets, tile tails, typed failures, deterministic equivalence, and GEMV/GEMM measurement |

## Curriculum status

| Scope | Status | Milestone |
| --- | --- | --- |
| Part I (Ch. 1–4) | COMPLETE | ENGINE-1 is the smallest complete autoregressive inference engine |
| Part II (Ch. 5–13) | IN PROGRESS | Chapters 5–6, Tensor Substrate v1, and ENGINE-2 complete; Chapter 7 is next |
| Part III (Ch. 14–18) | PLANNED | ENGINE-3 |
| Part IV (Ch. 19–22) | PLANNED | ENGINE-4 |
| Part V (Ch. 23–27) | PLANNED | ENGINE-5 / ENGINE-6 |
| Part VI (Ch. 28–34) | PLANNED | ENGINE-7 |
| Part VII (Ch. 35–41) | PLANNED | ENGINE-8 |
| Part VIII (Ch. 42–48) | PLANNED | ENGINE-9 |
| Part IX (Ch. 49–53) | PLANNED | Decode optimization |
| Part X (Ch. 54–58) | PLANNED | MoE / inference memory |
| Part XI (Ch. 59–66) | PLANNED | Correctness regime |
| Part XII (Ch. 67–73) | PLANNED | ENGINE-10 |
| Part XIII (Ch. 74–79) | PLANNED | Hermon case study |
| Part XIV (Ch. 80–88) | PLANNED | Frontier architecture |
| Part XV (Ch. 89–94) | PLANNED | Graduation project |
| Appendices A–N | PLANNED | Reference material |

## Open decisions

1. Select licenses for prose, diagrams, and code; decide whether one or
   separate licenses are appropriate.
2. Choose the small, redistributable model fixtures for later equivalence labs.
3. Decide the publication toolchain only after Markdown-first manuscript needs
   are demonstrated.

## Next recommended task

Execute only Chapter 7 — Embeddings and RMSNorm. Preserve the explicit
embedding row lookup and ENGINE-2 kernel boundary, derive RMSNorm with an
independent oracle, and keep storage, accumulation, epsilon, ownership, and
failure semantics visible. Do not begin attention, Q/K/V, RoPE, KV caching,
GGUF, quantization, SIMD intrinsics, GPU execution, or autograd.
