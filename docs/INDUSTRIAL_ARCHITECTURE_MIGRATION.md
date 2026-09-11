# Industrial architecture migration proposal

REVIEW ONLY. No manuscript moves or renumbering have been applied.

| Chapter | Action | Rationale / amendment |
| --- | --- | --- |
| 1. The Missing Half of AI | KEEP NUMBER | Use the four industrial atlas plates as optional overview/zoom references. Separate model semantics, execution and serving; revisit rather than overload the opening chapter. |
| 2. From Text to Tokens | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 3. The Smallest Possible Language Model | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 4. Logits, Sampling, and the Autoregressive Loop | KEEP NUMBER | Add a forward reference to constraint-state ownership in Chapter 67; preserve the current tested sampler API and distinguish token choice from output validity. |
| 5. Tensors Without Magic | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 6. Matrix Multiplication: The Engine Room | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 7. Embeddings and RMSNorm | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 8. Queries, Keys, and Values | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 9. Position: RoPE From First Principles | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 10. Causal Self-Attention | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 11. The Feed-Forward Network | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 12. One Complete Transformer Layer | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 13. The Decoder Stack and Next-Token Generation | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 14. What Is Actually Inside a Model File? | KEEP NUMBER | Compare a pinned safetensors specification with GGUF: metadata, tensor naming, layout, integrity and loading obligations. Do not imply container conversion guarantees model equivalence. |
| 15. GGUF From the Bytes Up | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 16. Quantization From F32 to Packed Weights | KEEP NUMBER | Separate weight-only, activation and KV quantization. Show scale groups, storage bits versus effective bytes and calibration/error budgets; keep the first packed fixture small. |
| 17. Packed Matrix Multiplication | KEEP NUMBER | Trace packed load, scale application, accumulation dtype and output tolerance. Compare optimized kernels only against the scalar oracle on identical logical values. |
| 18. Loading and Running a Real GGUF Model | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 19. Measure First: Profiling an Inference Engine | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 20. Prefill and Decode Are Different Workloads | KEEP NUMBER | Use explicit client/server timing boundaries and workload-dependent roofline reasoning. Add the latency plate; prefill/decode names do not determine the bottleneck. |
| 21. Why the KV Cache Exists | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 22. KV Cache Memory Mathematics | KEEP NUMBER | Count valid, allocated and reserved bytes separately. Add dense/GQA/sliding-window applicability limits and a quantized-KV metadata budget; defer MLA to Chapter 80. |
| 23. One User Is Easy | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 24. The Inference Request State Machine | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 25. Serving Multiple Users | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 26. Continuous Batching | KEEP NUMBER | Make chunked prefill a concrete scheduling policy with per-iteration token budget and decode latency tradeoff. Trace one long prompt beside short decodes. |
| 27. Fairness, Backpressure, Cancellation, and Streaming | KEEP NUMBER | Test fairness, bounded queues, cancellation and admission under a synthetic arrival trace; do not replace tail-latency analysis with average throughput. |
| 28. Why Flat and Slot-Bound KV Caches Fail | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 29. Paging Comes to AI | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 30. Building a Block Pool | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 31. Prefix Indexing and Radix Trees | KEEP NUMBER | State every cache-key assumption: model revision, tokenization/template, adapter and relevant position/execution context. Prefix equality alone is not universal cache validity. |
| 32. Shared Prefixes and Copy-on-Write | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 33. Eviction, Pressure, and Admission | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 34. Paged Attention | KEEP NUMBER | Explicitly distinguish paged KV addressing, prefix reuse and tiled attention. Add quantized-KV read/dequantization to the layout contract, without claiming every backend supports it. |
| 35. Where the Native Boundary Belongs | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 36. Designing a Stable Kernel ABI | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 37. Arena Allocation and Hot-Path Memory Discipline | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 38. Lock-Free Block Allocation and Refcounts | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 39. Bulk KV Writes | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 40. Online Softmax | KEEP NUMBER | Connect the stable online softmax recurrence to tiled attention with an exact small oracle. Read the full FlashAttention algorithm before asserting kernel-level equivalence. |
| 41. Split-K and Deterministic Attention Planning | KEEP NUMBER | Make partition merge order, partial normalization state and determinism visible. Explain when scheduling or reduction changes floating-point results. |
| 42. SIMD From First Principles | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 43. ARM NEON | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 44. x86 AVX2 and ISA Dispatch | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 45. Thinking Like a GPU | KEEP NUMBER | Separate model graph, compiler IR, fused kernel and captured launch graph. Show a dependency before a performance optimization. |
| 46. Metal and Unified Memory | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 47. CUDA and Device Mirrors | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 48. Why a GPU Can Be Slower Than a CPU | KEEP NUMBER | Own the first bounded graph lowering/capture case study: dynamic shapes, specialization, buffer addresses, warmup and replay invalidation. Measure transfer/launch overhead before explaining a CPU/GPU result. |
| 49. Prefix Caching | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 50. Sticky Slots as an Intermediate Design | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 51. Speculative Decoding | KEEP NUMBER | Read the full exact speculative-sampling paper; derive acceptance/correction and show first rejection plus KV rollback. Distinguish greedy verification from stochastic distribution preservation. |
| 52. Prompt-Lookup Decoding | KEEP NUMBER | Contrast prompt lookup/ngram proposals with draft models; add a source-verified comparison sidebar for EAGLE/MTP with their model/training assumptions. These remain optional proposal mechanisms. |
| 53. When Speculation Loses | KEEP NUMBER | Charge draft, verification, rollback and memory costs to useful accepted tokens; include a counterexample where speculation loses. |
| 54. Why MoE Changes the Inference Engine | KEEP NUMBER | Trace top-k expert routing, dispatch, expert compute and combine; separate activation communication from expert-weight residency. |
| 55. Models Larger Than Available VRAM | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 56. Expert Storage and Paging | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 57. Residency, Pinning, Eviction, and Queue Depth | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 58. Toward Unified Inference Memory | KEEP NUMBER | Use an allocation ownership ledger across weight/KV/expert pools. Do not equate a unified policy with a physically unified address space. |
| 59. Fast Wrong Answers Are Still Wrong | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 60. Scalar Oracles | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 61. Differential Testing | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 62. Numerical Determinism | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 63. Concurrency Bugs That Still Produce Plausible Text | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 64. Ownership and Lifetime Failures | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 65. Sanitizers, Fuzzing, and Boundary Testing | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 66. Real-Model Equivalence | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 67. Protocols and the AI Gateway | KEEP NUMBER | Own grammar/JSON-schema constrained decoding: compiler/automaton state, allowed-token mask, unsatisfiable continuation, stop behavior and cancellation. Syntactic validity is not semantic truth. |
| 68. Model Resolution and Routing | KEEP NUMBER | Own adapter identity, loading, residency and routing; introduce multimodal encoder/projector inputs and per-request state. Include reuse isolation across adapters and model revisions. |
| 69. Streaming as a Systems Contract | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 70. Metrics and Observability | KEEP NUMBER | Name timestamps and units for TTFT, ITL, TPOT, E2E, queue delay and goodput. Use trace IDs and censoring/failure policy; never infer token timing solely from transport chunks. |
| 71. Failure Containment | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 72. Security and Untrusted Models | KEEP NUMBER | Threat-model untrusted model files, templates, adapters, tenant cache reuse, prompt logging and resource exhaustion. Pin a parser boundary and test rejection. |
| 73. Benchmarking Without Lying to Yourself | KEEP NUMBER | Report SLO-qualified goodput, tail distributions, warm/cold phases, arrival process, concurrency, model, precision and hardware. No synthetic timing fixture becomes a speed claim. |
| 74. Hermon's System Architecture | KEEP NUMBER | Update the case-study ledger to the inspected Hermon pin without rewriting historical measurements. Keep CURRENT, PREVIEW, LIBRARY and TARGET distinct. |
| 75. Why Hermon Did Not Rewrite Everything | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 76. The Substitution Ladder | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 77. Anatomy of the Hermon Source Tree | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 78. Follow One Request Through Hermon | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 79. Follow One Token Through Hermon | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 80. Hybrid Transformer Architectures | KEEP NUMBER | Separate externally deployed architectural variants from research hypotheses. Add typed multimodal input and MLA/sliding-window/hybrid state-accounting exceptions, sourced per actual model. |
| 81. Recurrent State and STATE Pages | KEEP NUMBER | A recurrent STATE page stores a bounded recurrence state, not a dense KV history. Define rollback/checkpoint behavior before using speculation. |
| 82. Unified Memory Economics | KEEP NUMBER | Compare physical tiers with explicit transfer latency/bandwidth and ownership. A cost model is an analytical prediction, not a measured migration policy. |
| 83. Prefill/Decode Disaggregation | KEEP NUMBER; MOVE READING ORDER (proposal only) | Read after Chapter 84 in the proposed production order. Build a two-process KV-transfer reference with layout/version handshake, ownership acknowledgement, duplicate delivery and failure injection before any performance claim. |
| 84. Multi-GPU Execution | KEEP NUMBER; MOVE READING ORDER (proposal only) | Read before Chapter 83 in the proposed production order. Separate TP/PP/DP/context/expert parallelism; add real two-worker correctness execution where hardware permits, with a CPU reference path and no GPU-scaling claim from simulation. |
| 85. Multi-Node Inference | KEEP NUMBER | Extend the multi-worker contract to multi-node placement, collective failure, timeout and recovery. Require an actual distributed integration record; a simulator alone cannot establish production readiness. |
| 86. The Inference Engine as a Database | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 87. Toward a Universal Inference Execution Protocol | KEEP NUMBER | Revisit IR and executable-plan contracts introduced in Chapters 45/48. Compare a minimal interchange boundary against concrete consumers; do not present a proposed protocol as an adopted standard. |
| 88. What Comes After Today's Transformer Runtime? | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 89. Designing the Final Mini Engine | KEEP NUMBER | Select supported feature combinations and explicitly exclude unimplemented ones. Graduation scope is an evidence-backed engine, not a checklist of every surveyed vendor feature. |
| 90. End-to-End Implementation | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 91. Correctness Gate | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 92. Performance Gate | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 93. Production Gate | KEEP NUMBER | Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test. |
| 94. Replace One Hermon Component and Prove It | KEEP NUMBER | Replace one narrowly bounded component only after semantic, ownership, ABI, failure and performance gates. Require a reversible integration and an honest status label. |

Part XIV proposed retitle: **Distributed Inference and Emerging Architectures**.
This separates externally documented systems from speculative research.
No chapter is split, merged, removed or appended in this pass. The existing
14 appendices remain unchanged; no additional chapter count is justified yet.
