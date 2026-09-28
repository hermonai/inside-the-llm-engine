# Structure of the book

This is the plan of record for **Inside the LLM Engine: The Systems Engineering
of Large Language Model Inference** (second edition). It replaces the 94-chapter
`docs/OUTLINE.md`, now in `archive/docs/`. Change it deliberately: record every
change to a chapter boundary in the change log at the end, with its reason.
Do not grow the book past about 45 chapters; merge instead.

How chapters are written — voice, anatomy, evidence rules, figures, labs — is in
[`AUTHORING.md`](../AUTHORING.md). What is done and what is next is in
[`STATUS.md`](STATUS.md).

## The organizing principle

Every inference engine is a negotiation between four scarce resources:

| Resource | Short | Where it binds |
| --- | --- | --- |
| Memory bandwidth — bytes moved per token | **BW** | decode |
| Memory capacity — bytes resident | **CAP** | batch size, context length, model size |
| Compute — floating-point operations | **FLOP** | prefill, large batches |
| Latency — time to first token, time per output token | **LAT** | the user |

and two outcomes: **correctness** (a fast wrong answer is worthless) and
**cost** (tokens per dollar at a latency target, that is, goodput).

Every technique enters the book as a response to a *measured* constraint and is
charged for what it spends of the others. The *Ledger* column below is the
chapter's one-line entry in that accounting: what the technique buys and what
it pays with. Part I chapters establish the accounting itself.

## Part I — The Physics of Inference

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 1 | Why Inference Is Its Own Discipline | Inference is sequential, stateful, latency-bound and priced per token — a different engineering problem from training. Introduces the four resources, the two outcomes and the constraint ledger. | Defines the ledger |
| 2 | The Forward Pass, Compressed | One decode step as a systems engineer sees it: every operator with its shapes, parameter bytes, FLOPs and state, for a dense and an MoE model. Appendix A builds it from scratch. | FLOP ≈ 2 × active params per token; BW ≈ weight bytes per step |
| 3 | The Roofline: Why Decode Leaves Your GPU Idle | Arithmetic intensity and the ridge point, measured: batch-one decode streams every weight once per token and runs near memory bandwidth while the arithmetic units idle. Batch size is the lever that moves work toward the compute roof. | Decode is BW-bound; prefill is FLOP-bound |
| 4 | Prefill and Decode: Two Workloads, One Model | Prompt processing is a compute-bound matrix–matrix workload and generation a bandwidth-bound matrix–vector one. Defines TTFT, TPOT/ITL and end-to-end latency, and shows why each phase wants its own batch size, schedule and even hardware. | Two cost models, one weight set |
| 5 | The KV Cache: Inference's Central Data Structure | Caching keys and values turns quadratic recomputation into linear reads and in exchange creates growing, per-request, mutable state whose bytes per token decide concurrency, context length and cost. | Buys FLOP; spends CAP and BW |
| 6 | Napkin Math: Latency, Throughput and Cost per Token | Combine the roofline, the two phases and the KV cache into a five-minute model of step time, batch size, throughput, latency and dollars per million tokens, then check it against measurements. | The accounting every later chapter uses |

## Part II — Making One Request Fast

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 7 | Matrix Multiplication on Real Hardware | How GEMM maps onto tensor cores, shared memory and caches, and why the M×N×K shape — not the FLOP count — decides which roof binds. Skinny decode GEMMs waste most of the machine. | Buys achieved FLOP/s and BW; spends on-chip capacity |
| 8 | Attention as a Memory Problem: From Online Softmax to FlashAttention | Attention's cost is moving the score matrix, not computing it. Online softmax lets tiles live in on-chip memory; each FlashAttention generation follows its GPU generation's asynchrony and precision. | Buys BW and CAP (no N² scores); spends on-chip SRAM |
| 9 | Decode Attention: FlashDecoding, Split-K and Paged Kernels | One query per sequence leaves the GPU starved. Splitting the KV sequence and merging partial softmaxes restores parallelism; paged layouts add an indirection the kernel must hide. | Buys LAT; spends a reduction pass and workspace |
| 10 | Quantizing Weights: INT8, INT4, GPTQ, AWQ and K-Quants | Fewer bits per weight means fewer bytes per token: decode speeds up almost in proportion and larger models fit. The price is measured accuracy and dequantization work. | Buys BW, CAP; spends correctness, FLOP |
| 11 | Below 16 Bits in Floating Point: FP8, MXFP4 and NVFP4 | Block-scaled floating-point formats that tensor cores compute natively: how E4M3/E5M2, microscaling and two-level NVFP4 scaling trade range, accuracy and throughput for weights, activations and KV. | Buys FLOP/s, BW, CAP; spends correctness |
| 12 | Quantizing the KV Cache | At long context the KV cache, not the weights, dominates bytes. Quantizing it doubles or quadruples capacity, and its errors compound across layers and positions differently from weight errors. | Buys CAP, BW; spends long-context correctness |
| 13 | How Kernels Are Written Now: CUDA, Triton, Tile DSLs and Graph Capture | The tools that turn these algorithms into fast kernels — CUDA/CUTLASS, Triton, CuTe DSL, TileLang, tile-level CUDA — and the launch overhead that makes graph capture and fusion mandatory at small batch. | Buys LAT; spends compile time, flexibility |

## Part III — Serving Many Requests

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 14 | The Request Lifecycle: Streaming, Cancellation and Terminal Ownership | A request is a state machine that streams bytes, honours cancellation and ends exactly once. Its timestamps define TTFT and ITL. (Absorbs the first edition's Chapter 1.) | Defines the correctness contract and latency terms |
| 15 | Continuous Batching | Batching at every decode step instead of every request turns the roofline's batch-size lever into throughput without making short requests wait for long ones. | Buys FLOP utilisation; spends per-request LAT, CAP |
| 16 | Paged KV Memory | Contiguous KV reservations waste most of the memory they hold. Fixed-size blocks and per-request block tables — virtual memory for the KV cache — raise usable capacity and enable sharing. | Buys CAP; spends an indirection in every attention kernel |
| 17 | Prefix Caching and RadixAttention | System prompts, documents, conversation history and agent loops repeat. Indexing KV blocks by content skips their prefill and turns retained memory into saved compute and lower TTFT. | Buys FLOP, TTFT; spends CAP |
| 18 | Chunked Prefill and Phase Interference | A long prompt admitted into a running batch stalls every decode behind it. Token-budgeted prefill chunks keep inter-token latency bounded while keeping the GPU busy. | Buys ITL; spends TTFT |
| 19 | Scheduling Under Pressure: Preemption, Fairness and SLO-Aware Admission | When demand exceeds KV capacity the scheduler must preempt (recompute or swap), share fairly among tenants and admit only work that can meet its objective. Goodput, not throughput, is the target. | Buys goodput, fairness; spends FLOP or BW |
| 20 | Structured Output at Engine Speed | Grammar-constrained decoding masks invalid tokens at every step. The engineering problem is making the mask cheaper than the step it gates. | Buys correctness (validity); spends CPU compute, LAT |
| 21 | One Server, Many Models: LoRA, Adapters and Routing | Serving thousands of fine-tuned variants from one base model with batched low-rank adapters, adapter paging and routing, instead of one GPU per model. | Buys CAP, cost; spends FLOP |

## Part IV — Beyond One Token per Step

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 22 | Speculative Decoding: The Draft–Verify Contract | Guess several tokens cheaply, verify them in one forward pass, and accept them by a rule that provably preserves the target distribution. Expected speedup follows from acceptance rate and draft cost. | Buys LAT; spends FLOP, CAP |
| 23 | Modern Speculation: The EAGLE Family, Medusa and Multi-Token Prediction | Drafting from the target model's own features and heads — feature-level drafts, dynamic draft trees, parallel drafting, multi-token-prediction modules trained with the model — raises acceptance at low draft cost. | Buys LAT; spends FLOP, training |
| 24 | When Speculation Loses | Speculation converts idle compute into latency. At high batch, low acceptance or on compute-bound hardware the ledger turns negative; measured cases and a break-even model. | Prices the FLOP-for-LAT trade |

## Part V — Architectures That Reshape the Engine

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 25 | Shrinking the KV Cache: MQA, GQA and Multi-Head Latent Attention | Sharing key/value heads and compressing them into a low-rank latent (with absorbed projections and decoupled RoPE) cuts KV bytes per token several-fold and changes what the attention kernel computes. | Buys CAP, BW; spends kernel FLOP |
| 26 | Mixture of Experts | Routing each token to a few experts decouples total parameters from active parameters, until batching activates many experts at once and load balance becomes the engine's problem. | Buys FLOP and BW per token; spends CAP |
| 27 | Hybrid Models: State Spaces and Linear Attention | Recurrent layers keep a fixed-size state instead of a growing cache. Hybrids interleave them with attention, and KV-centric engines must learn to cache, copy and branch that state. | Buys CAP, BW at length; spends engine complexity |
| 28 | Long Context: Windows, Sinks and Sparse Attention | Million-token contexts need more than a larger cache: windows, sinks, position scaling, KV eviction and trained sparse attention each buy memory or compute at a measurable risk to correctness. | Buys CAP, FLOP; spends correctness |
| 29 | Reasoning Models and Test-Time Compute | Thousands of hidden tokens per answer and many sampled candidates: decode-dominated, KV-growing, high-variance workloads change batching, speculation, caching and cost per answer. | Shifts load to BW, CAP |
| 30 | Multimodal Inference | Images, audio and video enter as encoder outputs that expand into hundreds or thousands of tokens; the encoder is a separate compute-bound stage to schedule, cache and sometimes disaggregate. | Adds FLOP, CAP, TTFT |

## Part VI — Scaling Out

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 31 | Tensor, Pipeline and Context Parallelism | Splitting one model across devices — within layers, across layers, along the sequence — and pricing each split in collective bytes against interconnect bandwidth. | Buys CAP, aggregate BW; spends interconnect |
| 32 | Expert Parallelism at Scale | Experts on different GPUs make every MoE layer an all-to-all exchange; dispatch/combine kernels, redundant experts and communication overlap decide whether wide expert parallelism pays. | Buys CAP, throughput; spends interconnect, LAT |
| 33 | Disaggregated Serving | Prefill and decode on separate pools let each use its best batch size and parallelism — if the KV cache moves between them faster than it could be recomputed. | Buys goodput; spends network BW |
| 34 | The KV Cache as a Distributed Storage Tier | KV that outlives its request becomes data to store, index, evict and ship across HBM, DRAM, SSD and the network. The load-versus-recompute break-even governs every tier. | Buys FLOP, TTFT; spends storage CAP and BW |

## Part VII — Inference on the Hardware You Have

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 35 | Offloading Across VRAM, RAM and NVMe | Models larger than accelerator memory run by placing layers, experts or KV in slower tiers — and then PCIe and NVMe, not the GPU, set the token rate. | Buys CAP; spends BW, LAT |
| 36 | Streaming Experts From Disk: A Measured Case Study | Hermon's expert pager on a real MoE checkpoint: hit rate as the whole lever, a falsified queue-depth hypothesis, a bottleneck that moves between I/O and compute, and a thread-count determinism bug. | CAP via storage BW, measured |
| 37 | Unified Inference Memory | When CPU and GPU share one memory, the tiers collapse and one allocator can manage weights, KV blocks and experts under a single residency policy. | Buys CAP; spends shared BW |

## Part VIII — Correctness and Measurement

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 38 | Fast Wrong Answers: Oracles, Differential Testing and Determinism | Every optimization is a chance to change the answer. Independent oracles, differential tests across kernels and batch shapes, and fixed reduction orders keep speed honest. | Protects correctness; spends test compute, sometimes throughput |
| 39 | Benchmarking Without Lying to Yourself | TTFT, ITL, throughput and goodput under realistic arrivals, with the controls, percentiles and records that make a number reproducible — and the usual ways benchmarks mislead. | Protects every cost claim |

## Part IX — Production and Capstone

| # | Chapter | Intent | Ledger |
| --- | --- | --- | --- |
| 40 | Operating Inference in Production | Autoscaling against cold starts, cache-aware routing, multi-tenant isolation, observability, failure handling and cost control for a fleet rather than one engine. | Trades cost for LAT headroom and reliability |
| 41 | Anatomy of Real Engines: vLLM, SGLang, llama.cpp and Hermon | Four engines read at pinned commits through the ledger: where each spends its complexity, what runs by default, and which trade-offs follow from its hardware and users. | Synthesis |
| 42 | Capstone: Build a Mini Engine End to End | Assemble a small engine with paged KV, continuous batching, streaming and a differential oracle, then place its measured performance on the roofline. | Synthesis |

## Appendices

The appendices form three reference groups, without renumbering existing
labels: **build the computation** (A), **execution and representation**
(B–E), and **notation and evidence** (F–G). A's opener supplies a first
prediction and explicit reading routes. B–G are concise reference drafts,
not full chapters or an exhaustive hardware/format catalogue. Planned A.9–A.11
remain unfinished; Chapters 2 and 42 provide explicit interim bridges.
Every main chapter has a targeted foundation bridge after its opening
phenomenon and before its napkin math. Existing chapter depth statuses stay
unchanged by these additions.

| | Appendix | Intent |
| --- | --- | --- |
| A | The Transformer From Scratch | The build-it-yourself path: the first edition's Chapters 2–9 as sections A.1–A.8, lightly edited, with the checked `mini-engine`. Planned A.9 Causal Self-Attention, A.10 The Feed-Forward Network and A.11 The Decoder Stack complete a runnable decoder for the later labs. |
| B | GPU Architecture for Inference Engineers | SMs, warps, tensor cores, the memory hierarchy, HBM and interconnects, with the numbers the chapters use. |
| C | CPU and SIMD | Cores, caches, NEON/AVX/AMX/SME and threads — the CPU as an inference target. |
| D | Model File Formats: GGUF and safetensors | What is inside a checkpoint, byte by byte, and how an engine maps it. |
| E | Numerics: Floating Point and Quantization Formats | Bit layouts, ranges and rounding of every format the book uses. |
| F | Notation | Symbols, shapes and units used across chapters. |
| G | Reproducing the Book's Measurements | Hardware, software and commands behind every measured number. |

## Where the first edition's work went

Nothing is thrown away. Chapters moved with `git mv`, so `git log --follow`
shows their history.

| First edition | Path before | Second-edition home |
| --- | --- | --- |
| Ch 1 The Missing Half of AI — lifecycle, terminal ownership, state machine, latency definitions | `tex/chapters/ch01.tex` | Chapter 14, `tex/chapters/ch14-request-lifecycle.tex` |
| Ch 2 From Text to Tokens | `tex/chapters/ch02.tex` | Appendix A.1, `tex/appendices/a1-text-to-tokens.tex` |
| Ch 3 The Smallest Possible Language Model | `tex/chapters/ch03.tex` | Appendix A.2, `tex/appendices/a2-smallest-model.tex` |
| Ch 4 Logits, Sampling and the Autoregressive Loop | `tex/chapters/ch04.tex` | Appendix A.3, `tex/appendices/a3-sampling.tex`; compressed into Chapter 2 |
| Ch 5 Tensors Without Magic | `tex/chapters/ch05.tex` | Appendix A.4, `tex/appendices/a4-tensors.tex` |
| Ch 6 Matrix Multiplication: The Engine Room | `tex/chapters/ch06.tex` | Appendix A.5, `tex/appendices/a5-matmul.tex`; its GEMV/GEMM, loop-order and tiling material and three benchmark records feed Chapters 3 and 7 |
| Ch 7 Embeddings and Normalization | `tex/chapters/ch07.tex` | Appendix A.6, `tex/appendices/a6-embeddings-norm.tex` |
| Ch 8 Queries, Keys and Values | `tex/chapters/ch08.tex` | Appendix A.7, `tex/appendices/a7-qkv.tex`; MHA/GQA/MQA geometry feeds Chapter 25 |
| Ch 9 Position: RoPE From First Principles | `tex/chapters/ch09.tex` | Appendix A.8, `tex/appendices/a8-rope.tex`; position scaling feeds Chapter 28 |
| Worked problems `tex/worked/ch01.tex`–`ch09.tex` | `tex/worked/` | Renamed to follow their chapter (`ch14-…`, `a1-…`–`a8-…`) |
| 65 TikZ figures | `tex/figures/` | Unchanged paths. Reuse candidates below |
| `code/mini-engine/` and its 220 tests | — | Appendix A's engine and the substrate for later labs; unchanged path |
| Python oracles, differential-testing discipline | `code/reference/` | Appendix A and Chapter 38; the method of every lab |
| Benchmark records (loop order, blocked GEMM, GEMV vs GEMM, traversal, sampling cost, projection scaling) | `research/benchmarks/` | Evidence for Chapters 3 and 7 and Appendix G; unchanged path |
| Labs 1–58 | `labs/lab-NN-*.md` | Appendix A labs; unchanged path |
| Markdown edition, text diagrams, SVG atlas, HTML/PDF exporter | `manuscript/`, `diagrams/`, `figures/`, `publication/` | Frozen history; their parity checks still run in CI |
| Governance, outline, audits, plans | `docs/*`, root `*.md` | `archive/`, mirroring original paths; see `archive/README.md` |

### Figure reuse candidates

| Figures | First used | Reuse in |
| --- | --- | --- |
| `request-trace`, `request-states`, `request-timeline`, `parameter-owners` | Ch 1 | Chapter 14 (moved with it) |
| `utf8-buffer` | Ch 2 | Chapter 14 (streaming bytes) |
| `model-path`, `sampling-feedback` | Ch 3, 4 | Chapter 2 |
| `ch06-gemv`, `ch06-gemm`, `ch06-crossover`, `ch06-throughput` | Ch 6 | Chapter 3 (GEMV vs GEMM reuse, the crossover) |
| `ch06-hierarchy`, `ch06-tiles`, `ch06-hardware`, `ch06-memory` | Ch 6 | Chapter 7 |
| `ch08-heads`, `ch08-cost` | Ch 8 | Chapters 5 and 25 (KV footprint, head sharing) |
| `ch09-scaling` | Ch 9 | Chapter 28 |

A figure keeps one source file. Where two chapters need it, the second inputs
the same file; a redrawn variant gets a new name.

## File layout

| What | Where |
| --- | --- |
| Chapters | `tex/chapters/chNN-<slug>.tex` |
| Appendices | `tex/appendices/<letter><n>-<slug>.tex` |
| Worked problems | `tex/worked/<chapter stem>.tex` |
| Figures | `tex/figures/<name>.tex` (native TikZ) |
| Labs | `code/labs/chNN-<slug>/` (self-contained) |
| Frontier ledger | `research/FRONTIER.md` |
| Measurement records | `research/measurements/` |

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-09-28 | Added contextual foundation bridges to all 42 chapters; grouped appendices by purpose, retaining A.1–A.8 and all file paths | Teach prerequisites at the point of use without another format migration, duplicated evidence, or a new numbering scheme |
| 2026-09-23 | Adopted the 42-chapter, nine-part plan; first edition's Chapters 2–9 become Appendix A, Chapter 1 moves toward Chapter 14 | The unique inference material arrived around Chapter 19 of 94; production techniques sat under "the future"; strictly linear chapters could not be written or read out of order |
| 2026-09-23 | Appendix A keeps its eight moved chapters as numbered sections A.1–A.8 rather than one merged chapter | Preserves their history, labels and worked problems with light edits; each remains readable alone |
| 2026-09-23 | Appendix A gains planned sections A.9–A.11 (attention, FFN, decoder stack) | Later labs need a complete small decoder; the first edition planned these as Chapters 10–13 |
