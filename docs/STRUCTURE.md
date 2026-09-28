# Structure of the book

The plan of record for **Inside the LLM Engine**. On 2026-09-28 the author
asked that the former appendices become the main learning path, and that
explanations precede performance accounting. This replaces the earlier
42-chapter, nine-part plan; its complete text is preserved in
`archive/docs/STRUCTURE-2026-09-28-before-foundations.md`.

## The reader's path

First build a prediction. Then build the decoder. Learn how its numbers live
in files and hardware. Only then ask how to make it fast, share it between
users and operate a service. **59 chapters in ten parts**: 17 construction
chapters followed by the 42 existing systems chapters. This deliberate increase
promotes existing substantial material; it does not create 59 new outlines.

Part I uses concrete examples, small equations, executable checks and worked
answers. Parts II–X retain the resource accounting and measured industrial
cases, but introduce their question and vocabulary before presenting the data.
The reading route is beginner-accessible; the later technical depth is retained.
No appendix is required to understand the main argument.

## Numbering and continuity

Printed chapter numbers follow reading order. Source IDs do **not** change
when a chapter moves. Thus `ch14-request-lifecycle`, its lab and historical
measurement records keep their names, but the printed chapter is now 31.
The source column below is authoritative. Use symbolic LaTeX references in
prose; never infer a printed number from a lab directory. The eight existing
construction chapters keep their equations, labels, tests and first-edition
lab IDs. Former `app:` labels remain compatibility aliases, not appendices.
All former Appendix B–G content is incorporated in Chapters 12–17.

## Part I — Build and Understand a Language Model

| # | Chapter | Source ID | Reader can… |
| --- | --- | --- | --- |
| 1 | From Text to Tokens | f01-text-to-tokens | Turn text into IDs, preserve byte boundaries and understand the tokenizer contract. |
| 2 | The Smallest Possible Language Model | f02-smallest-model | Compute one prediction with visible weights; distinguish IDs, vectors and logits. |
| 3 | Logits, Sampling, and the Autoregressive Loop | f03-sampling | Convert scores into a choice and build the autoregressive feedback loop. |
| 4 | Tensors Without Magic | f04-tensors | Trace a value through shapes, strides, ownership and views. |
| 5 | Matrix Multiplication: The Engine Room | f05-matmul | Build and test the matrix products used by a decoder. |
| 6 | Embeddings and Normalization | f06-embeddings-norm | Look up token vectors and normalize rows without losing shape or scale. |
| 7 | Queries, Keys, and Values | f07-qkv | Project queries, keys and values and split them into heads. |
| 8 | Position: RoPE From First Principles | f08-rope | Give token position a precise meaning through two-dimensional rotations. |
| 9 | Causal Attention, One Row at a Time | f09-causal-attention | Derive causal scores, stable probabilities and weighted values; test the mask. |
| 10 | The Feed-Forward Network and the Residual Stream | f10-feed-forward | Build ReLU and SwiGLU transformations and follow both residual additions. |
| 11 | Assembling a Decoder That Generates Tokens | f11-decoder | Join the operators into a tiny decoder, then compare full-prefix and cached execution. |
| 12 | Model File Formats: GGUF and safetensors | f12-model-formats | Read tensor metadata and offsets; separate file, architecture and tokenizer contracts. |
| 13 | Numerics: Floating Point and Quantization Formats | f13-numerics | Understand rounding, quantization, error bounds and what equivalence means. |
| 14 | CPU and SIMD | f14-cpu-simd | Follow loops through caches, vector lanes, tails and threads. |
| 15 | GPU Architecture for Inference Engineers | f15-gpu-architecture | Map work to threads, blocks and memory; distinguish submission from completion. |
| 16 | Reading Shapes, Bytes and Performance Bounds | f16-notation | Read tensor equations and convert parameters into bytes, work and time. |
| 17 | Reproducing the Book's Measurements | f17-measurements | Design a reproducible experiment and interpret its timestamps and output. |

## Part II — The Physics of Inference

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 18 | Why Inference Is Its Own Discipline | ch01-why-inference | Inference is sequential, stateful, latency-bound and priced per token — a different engineering problem from training. Introduces the four resources, the two outcomes and the constraint ledger. |
| 19 | The Forward Pass, Compressed | ch02-forward-pass | One decode step as a systems engineer sees it: every operator with its shapes, parameter bytes, FLOPs and state, for a dense and an MoE model. Part I builds it from scratch. |
| 20 | The Roofline: Why Decode Leaves Your GPU Idle | ch03-roofline | Count useful arithmetic per byte moved, predict the limiting resource, then test the prediction. Dense batch-one decode often streams weights with little reuse; batching, cache residency, sparse activation and context length can change that diagnosis. |
| 21 | Prefill and Decode: Two Workloads, One Model | ch04-prefill-decode | Explain why many prompt positions offer more weight reuse than one new position per sequence. Determine when compute, bandwidth or overhead dominates instead of assigning each phase a fixed bottleneck. Define TTFT, TPOT/ITL and end-to-end latency before comparing schedules. |
| 22 | The KV Cache: Inference's Central Data Structure | ch05-kv-cache | Reuse earlier keys and values instead of repeating their projections. Dense attention still reads a growing prefix per decode step. Derive the resulting per-request state and its consequences for concurrency, context length and cost. |
| 23 | Napkin Math: Latency, Throughput and Cost per Token | ch06-napkin-math | Combine the roofline, the two phases and the KV cache into a five-minute model of step time, batch size, throughput, latency and dollars per million tokens, then check it against measurements. |

## Part III — Making One Request Fast

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 24 | Matrix Multiplication on Real Hardware | ch07-matmul-hardware | Map GEMM onto tensor cores, shared memory and caches. Explain how shape, precision, layout and reuse govern utilization; a FLOP count alone cannot predict time, particularly for skinny decode matrices. |
| 25 | Attention as a Memory Problem: From Online Softmax to FlashAttention | ch08-flashattention | Derive online softmax and tiled attention that avoid materializing the full score matrix in device memory. Separate reduced memory traffic from unchanged dense pairwise arithmetic, then examine how newer kernels use hardware asynchrony and precision. |
| 26 | Decode Attention: FlashDecoding, Split-K and Paged Kernels | ch09-decode-attention | One query per sequence leaves the GPU starved. Splitting the KV sequence and merging partial softmaxes restores parallelism; paged layouts add an indirection the kernel must hide. |
| 27 | Quantizing Weights: INT8, INT4, GPTQ, AWQ and K-Quants | ch10-weight-quantization | Account for codes, scales and reconstruction work. Fewer stored bytes can improve capacity and weight-bandwidth-limited decode; kernel support and other bottlenecks decide the realized speedup. Test numerical and task-level error separately. |
| 28 | Below 16 Bits in Floating Point: FP8, MXFP4 and NVFP4 | ch11-low-precision-float | Block-scaled floating-point formats that tensor cores compute natively: how E4M3/E5M2, microscaling and two-level NVFP4 scaling trade range, accuracy and throughput for weights, activations and KV. |
| 29 | Quantizing the KV Cache | ch12-kv-quantization | Find the workload where KV capacity or traffic becomes limiting. Quantize keys and values with explicit scales and metadata, and measure the memory benefit against conversion cost and context-dependent error. |
| 30 | How Kernels Are Written Now: CUDA, Triton, Tile DSLs and Graph Capture | ch13-kernel-languages | Read the execution model behind CUDA/CUTLASS, Triton, CuTe DSL, TileLang and tile-level CUDA. Measure submission overhead before choosing graph replay or fusion; distinguish supported, released capabilities from proposed ones. |

## Part IV — Serving Many Requests

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 31 | The Request Lifecycle: Streaming, Cancellation and Terminal Ownership | ch14-request-lifecycle | A request is a state machine that streams bytes, honours cancellation and ends exactly once. Its timestamps define TTFT and ITL. (Absorbs the first edition's Chapter 1.) |
| 32 | Continuous Batching | ch15-continuous-batching | Batching at every decode step instead of every request turns the roofline's batch-size lever into throughput without making short requests wait for long ones. |
| 33 | Paged KV Memory | ch16-paged-kv | Measure waste from contiguous reservations, then derive fixed-size blocks and per-request block tables. Charge paging for tail waste, metadata and indirection as well as its capacity and sharing benefits. |
| 34 | Prefix Caching and RadixAttention | ch17-prefix-caching | System prompts, documents, conversation history and agent loops repeat. Indexing KV blocks by content skips their prefill and turns retained memory into saved compute and lower TTFT. |
| 35 | Chunked Prefill and Phase Interference | ch18-chunked-prefill | A long prompt admitted into a running batch stalls every decode behind it. Token-budgeted prefill chunks keep inter-token latency bounded while keeping the GPU busy. |
| 36 | Scheduling Under Pressure: Preemption, Fairness and SLO-Aware Admission | ch19-scheduling | When demand exceeds KV capacity the scheduler must preempt (recompute or swap), share fairly among tenants and admit only work that can meet its objective. Goodput, not throughput, is the target. |
| 37 | Structured Output at Engine Speed | ch20-structured-output | Grammar-constrained decoding masks invalid tokens at every step. The engineering problem is making the mask cheaper than the step it gates. |
| 38 | One Server, Many Models: LoRA, Adapters and Routing | ch21-multi-model-serving | Serving thousands of fine-tuned variants from one base model with batched low-rank adapters, adapter paging and routing, instead of one GPU per model. |

## Part V — Beyond One Token per Step

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 39 | Speculative Decoding: The Draft–Verify Contract | ch22-speculative-decoding | Guess several tokens cheaply, verify them in one forward pass, and accept them by a rule that provably preserves the target distribution. Expected speedup follows from acceptance rate and draft cost. |
| 40 | Modern Speculation: The EAGLE Family, Medusa and Multi-Token Prediction | ch23-modern-speculation | Drafting from the target model's own features and heads — feature-level drafts, dynamic draft trees, parallel drafting, multi-token-prediction modules trained with the model — raises acceptance at low draft cost. |
| 41 | When Speculation Loses | ch24-when-speculation-loses | Speculation converts idle compute into latency. At high batch, low acceptance or on compute-bound hardware the ledger turns negative; measured cases and a break-even model. |

## Part VI — Architectures That Reshape the Engine

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 42 | Shrinking the KV Cache: MQA, GQA and Multi-Head Latent Attention | ch25-shrinking-kv | Sharing key/value heads and compressing them into a low-rank latent (with absorbed projections and decoupled RoPE) cuts KV bytes per token several-fold and changes what the attention kernel computes. |
| 43 | Mixture of Experts | ch26-mixture-of-experts | Routing each token to a few experts decouples total parameters from active parameters, until batching activates many experts at once and load balance becomes the engine's problem. |
| 44 | Hybrid Models: State Spaces and Linear Attention | ch27-hybrid-models | Recurrent layers keep a fixed-size state instead of a growing cache. Hybrids interleave them with attention, and KV-centric engines must learn to cache, copy and branch that state. |
| 45 | Long Context: Windows, Sinks and Sparse Attention | ch28-long-context | Million-token contexts need more than a larger cache: windows, sinks, position scaling, KV eviction and trained sparse attention each buy memory or compute at a measurable risk to correctness. |
| 46 | Reasoning Models and Test-Time Compute | ch29-reasoning-workloads | Thousands of hidden tokens per answer and many sampled candidates: decode-dominated, KV-growing, high-variance workloads change batching, speculation, caching and cost per answer. |
| 47 | Multimodal Inference | ch30-multimodal | Images, audio and video enter as encoder outputs that expand into hundreds or thousands of tokens; the encoder is a separate compute-bound stage to schedule, cache and sometimes disaggregate. |

## Part VII — Scaling Out

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 48 | Tensor, Pipeline and Context Parallelism | ch31-model-parallelism | Splitting one model across devices — within layers, across layers, along the sequence — and pricing each split in collective bytes against interconnect bandwidth. |
| 49 | Expert Parallelism at Scale | ch32-expert-parallelism | Experts on different GPUs make every MoE layer an all-to-all exchange; dispatch/combine kernels, redundant experts and communication overlap decide whether wide expert parallelism pays. |
| 50 | Disaggregated Serving | ch33-disaggregated-serving | Compare colocated and separate prefill/decode pools under the same arrivals and latency target. Price KV handoff, queueing, network contention and duplicated resources against phase-specific scheduling and hardware benefits. |
| 51 | The KV Cache as a Distributed Storage Tier | ch34-kv-storage-tier | KV that outlives its request becomes data to store, index, evict and ship across HBM, DRAM, SSD and the network. The load-versus-recompute break-even governs every tier. |

## Part VIII — Inference on the Hardware You Have

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 52 | Offloading Across VRAM, RAM and NVMe | ch35-offloading | Models larger than accelerator memory run by placing layers, experts or KV in slower tiers — and then PCIe and NVMe, not the GPU, set the token rate. |
| 53 | Streaming Experts From Disk: A Measured Case Study | ch36-expert-streaming | Hermon's expert pager on a real MoE checkpoint: hit rate as the whole lever, a falsified queue-depth hypothesis, a bottleneck that moves between I/O and compute, and a thread-count determinism bug. |
| 54 | Unified Inference Memory | ch37-unified-memory | Distinguish a unified address space from physically shared memory. Explain what sharing removes and what remains: caches, bandwidth contention, synchronization, allocation and placement costs. |

## Part IX — Correctness and Measurement

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 55 | Fast Wrong Answers: Oracles, Differential Testing and Determinism | ch38-fast-wrong-answers | Every optimization is a chance to change the answer. Independent oracles, differential tests across kernels and batch shapes, and fixed reduction orders keep speed honest. |
| 56 | Benchmarking Without Lying to Yourself | ch39-benchmarking | TTFT, ITL, throughput and goodput under realistic arrivals, with the controls, percentiles and records that make a number reproducible — and the usual ways benchmarks mislead. |

## Part X — Production and Capstone

| # | Chapter | Source ID | Intent |
| --- | --- | --- | --- |
| 57 | Operating Inference in Production | ch40-production | Autoscaling against cold starts, cache-aware routing, multi-tenant isolation, observability, failure handling and cost control for a fleet rather than one engine. |
| 58 | Anatomy of Real Engines: vLLM, SGLang, llama.cpp and Hermon | ch41-real-engines | Four engines read at pinned commits through the ledger: where each spends its complexity, what runs by default, and which trade-offs follow from its hardware and users. |
| 59 | Capstone: Build a Mini Engine End to End | ch42-capstone | Assemble a small engine with paged KV, continuous batching, streaming and a differential oracle, then place its measured performance on the roofline. |

## Editorial priorities

1. Keep the existing long construction chapters intact; remove backward
   assumptions that the reader already knows the systems chapters.
2. Complete attention, feed-forward and decoder composition with runnable,
   deliberately tiny examples and independently checked arithmetic.
3. Expand the short hardware, formats, numerical and measurement references
   into teaching chapters: not lists of terminology.
4. Rewrite the entries into the existing systems chapters in plain language.
   A short first-reading route is a guide, not a substitute for the derivation.
5. Continue the in-flight lifecycle lab and the full-depth systems pass without
   inventing measurements or relabelling drafts as finished.

## Source map

| Material | Location |
| --- | --- |
| Part I chapters | `tex/foundations/fNN-*.tex` |
| Systems chapters (stable pre-migration IDs) | `tex/chapters/chNN-*.tex` |
| Native vector illustrations | `tex/figures/` |
| Worked answers | `tex/worked/`, named for source ID |
| Teaching engine and independent oracles | `code/mini-engine/`, `code/reference/` |
| New Part I checks | `code/foundations/` |
| Existing self-contained systems labs | `code/labs/chNN-*/` |
| Historical construction labs | `labs/` |
| Measured evidence and dated primary sources | `research/measurements/`, `research/FRONTIER.md` |

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-09-28 | Promoted all appendices to Part I, added the missing attention/FFN/decoder construction, kept stable systems source IDs | Author requests a continuous, easy-to-follow build-from-scratch textbook before advanced engineering |
| 2026-09-28 | Replaced the 45-chapter cap and one-anatomy rule with distinct construction and systems chapter contracts | Completeness is a learning outcome; introductory chapters must not start by assuming their own prerequisites |
| 2026-09-23 | Adopted the earlier 42-chapter plan | Preserved in the dated archived plan with all first-edition mappings |
