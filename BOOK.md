# Inside the LLM Engine

## From First Token to Production-Grade Inference

This is the reader-facing table of contents. The authoring specification for
each chapter lives in [docs/OUTLINE.md](docs/OUTLINE.md).

Companion: [visual prototype atlas](figures/ATLAS.md). Build the current seven
chapters as PDF/HTML using the [publication instructions](docs/FIGURE_BUILD.md).

### Part I — What Actually Happens When an LLM Answers?

1. [The Missing Half of AI](manuscript/part-01/chapter-01-the-missing-half-of-ai.md)
2. [From Text to Tokens](manuscript/part-01/chapter-02-from-text-to-tokens.md)
3. [The Smallest Possible Language Model](manuscript/part-01/chapter-03-the-smallest-possible-language-model.md)
4. [Logits, Sampling, and the Autoregressive Loop](manuscript/part-01/chapter-04-logits-sampling-autoregressive-loop.md)

Milestone: ENGINE-0 / ENGINE-1.

### Part II — Build a Transformer Inference Engine

5. [Tensors Without Magic](manuscript/part-02/chapter-05-tensors-without-magic.md)
6. [Matrix Multiplication: The Engine Room](manuscript/part-02/chapter-06-matrix-multiplication-the-engine-room.md)
7. [Embeddings and Normalization](manuscript/part-02/chapter-07-embeddings-and-normalization.md)
8. Queries, Keys, and Values
9. Position: RoPE From First Principles
10. Causal Self-Attention
11. The Feed-Forward Network
12. One Complete Transformer Layer
13. The Decoder Stack and Next-Token Generation

Milestone: ENGINE-2.

### Part III — Run a Real Model

14. What Is Actually Inside a Model File?
15. GGUF From the Bytes Up
16. Quantization From F32 to Packed Weights
17. Packed Matrix Multiplication
18. Loading and Running a Real GGUF Model

Milestone: ENGINE-3.

### Part IV — Why Naive Inference Is Slow

19. Measure First: Profiling an Inference Engine
20. Prefill and Decode Are Different Workloads
21. Why the KV Cache Exists
22. KV Cache Memory Mathematics

Milestone: ENGINE-4.

### Part V — From Model Runner to Inference Server

23. One User Is Easy
24. The Inference Request State Machine
25. Serving Multiple Users
26. Continuous Batching
27. Fairness, Backpressure, Cancellation, and Streaming

Milestone: ENGINE-5 / ENGINE-6.

### Part VI — Paged Inference Memory

28. Why Flat and Slot-Bound KV Caches Fail
29. Paging Comes to AI
30. Building a Block Pool
31. Prefix Indexing and Radix Trees
32. Shared Prefixes and Copy-on-Write
33. Eviction, Pressure, and Admission
34. Paged Attention

Milestone: ENGINE-7.

### Part VII — Native Kernel Engineering

35. Where the Native Boundary Belongs
36. Designing a Stable Kernel ABI
37. Arena Allocation and Hot-Path Memory Discipline
38. Lock-Free Block Allocation and Refcounts
39. Bulk KV Writes
40. Online Softmax
41. Split-K and Deterministic Attention Planning

Milestone: ENGINE-8.

### Part VIII — SIMD and Accelerator Providers

42. SIMD From First Principles
43. ARM NEON
44. x86 AVX2 and ISA Dispatch
45. Thinking Like a GPU
46. Metal and Unified Memory
47. CUDA and Device Mirrors
48. Why a GPU Can Be Slower Than a CPU

Milestone: ENGINE-9.

### Part IX — Modern Decode Optimization

49. Prefix Caching
50. Sticky Slots as an Intermediate Design
51. Speculative Decoding
52. Prompt-Lookup Decoding
53. When Speculation Loses

### Part X — Mixture-of-Experts Inference

54. Why MoE Changes the Inference Engine
55. Models Larger Than Available VRAM
56. Expert Storage and Paging
57. Residency, Pinning, Eviction, and Queue Depth
58. Toward Unified Inference Memory

### Part XI — Correctness Engineering

59. Fast Wrong Answers Are Still Wrong
60. Scalar Oracles
61. Differential Testing
62. Numerical Determinism
63. Concurrency Bugs That Still Produce Plausible Text
64. Ownership and Lifetime Failures
65. Sanitizers, Fuzzing, and Boundary Testing
66. Real-Model Equivalence

### Part XII — Production Inference Engineering

67. Protocols and the AI Gateway
68. Model Resolution and Routing
69. Streaming as a Systems Contract
70. Metrics and Observability
71. Failure Containment
72. Security and Untrusted Models
73. Benchmarking Without Lying to Yourself

Milestone: ENGINE-10.

### Part XIII — Inside Hermon

74. Hermon's System Architecture
75. Why Hermon Did Not Rewrite Everything
76. The Substitution Ladder
77. Anatomy of the Hermon Source Tree
78. Follow One Request Through Hermon
79. Follow One Token Through Hermon

### Part XIV — Beyond Today's Engine

80. Hybrid Transformer Architectures
81. Recurrent State and STATE Pages
82. Unified Memory Economics
83. Prefill/Decode Disaggregation
84. Multi-GPU Execution
85. Multi-Node Inference
86. The Inference Engine as a Database
87. Toward a Universal Inference Execution Protocol
88. What Comes After Today's Transformer Runtime?

### Part XV — Graduation Project

89. Designing the Final Mini Engine
90. End-to-End Implementation
91. Correctness Gate
92. Performance Gate
93. Production Gate
94. Replace One Hermon Component and Prove It

### Appendices

A. Mathematical Reference
B. Tensor Shape Reference
C. GGUF Reference
D. Quantization Reference
E. CPU Architecture Primer
F. GPU Architecture Primer
G. Rust for Inference Engineers
H. C for Kernel Boundaries
I. Benchmark Reproduction Guide
J. Glossary
K. Symbols and Notation
L. Recommended Papers
M. Source-Code Navigation Guide
N. Hardware Laboratory Guide
