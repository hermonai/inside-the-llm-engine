# Status

## Current reading path — 2026-10-03

The author requested that all former appendices become the main learning path.
The book now has **59 chapters in ten parts**, with no appendix detour:

- Part I, printed Chapters 1–17: build the model, then understand its bytes,
  numerics, hardware and measurements.
- Parts II–X, printed Chapters 18–59: the existing 42 systems chapters.

Printed numbers follow reading order. Stable source and lab IDs do not.
For example, `ch14-request-lifecycle` is now printed Chapter 31.
[STRUCTURE.md](STRUCTURE.md) gives the complete mapping.

## What this revision changes

The eight substantial construction chapters moved into `tex/foundations/`
with their figures, worked problems and compatibility labels intact. New
chapters complete causal attention, the FFN/residual block and a runnable
cached decoder. The former B–G references are developed teaching chapters,
not outline shells. Part I has 17 FULL construction chapters under the
construction contract in [AUTHORING.md](../AUTHORING.md); FULL is not VERIFIED.

New small examples use `code/foundations/` with only the Python standard
library. Its decoder uses illustrative, untrained weights. Scalar hand
fixtures and full-prefix/cached comparisons test different levels of the
computation; shared scalar primitives are explicitly not independent oracles
for themselves. Hardware lesson outputs are derived arithmetic, not benchmarks.

The first 14 systems chapter introductions now explain their question and
offer a first-reading route before the measurements. This is an entry-point
clarity pass, not a claim that every paragraph of the systems manuscript has
been rewritten. Existing measured results and dated frontier claims remain.
The book contains 178 native figures and 117 worked problems. Every one of
the 42 systems chapters has a native mechanism diagram. The author merged
the second-edition work into main at `2c0474d`; this pass follows that checkout.

Chapter 12, `f12-model-formats`, now has about 5,100 prose words and a
concrete byte-to-computation route: a 256-byte GGUF fixture, typed metadata,
descriptor coordinates, alignment, F32 strides, actual Q4_0 bit packing,
graph construction and CPU execution. Three new native figures show the
byte regions, nibble interpretation and loader/graph/executor interaction.
Source listings come directly from the checked Python and C++ labs.
Worked problems now distinguish structural, numerical and model contracts.

The optional reference check uses pinned upstream ggml `353b63b`. Direct
weight decoding agrees exactly, and the executed dense projection gives
`[49,19]`. The initial Q4_0 dot-product expectation failed; the chapter
derives the CPU's Q8_0 activation rounding and predicts the observed
`-7.99951171875`. The failed expectation and two matching finished runs are
preserved in `research/measurements/2026-10-03-gguf-byte-lab.md`.
These are numerical observations, not speed measurements.

The diagram pass fixes crowded nodes, headings, message labels and
paged-memory arrows. The preface explains the notation. The authoring
policy now requires concrete representations, implementations and
distinguishing failure tests when deepening a lesson; minimum word counts
alone do not establish depth.

The preceding pass deepened Chapters 13–15 to about 3,500 prose words each,
preserving the earlier lessons while reaching actual representations and
execution paths. Numerics derives FP16/BF16 bits, subnormals, ties,
accumulation, fused arithmetic and projection-error bounds. CPU/SIMD
connects addresses to lines, then a native four-lane loop to its integer
oracle, assembly, tails and output ownership. GPU architecture traces
odd-sized output tiles, publication/retirement, global sectors, shared banks
and competing residency ceilings. Seven new native diagrams and nine worked
problems support these mechanisms. Repeated CPU introductory prose was
consolidated without removing its technical points.

Two standard-library Python lessons and a portable C fixture make the new
derivations runnable. The checker now treats three worked problems as a
minimum, not a ceiling, and requires a solution for every construction
problem; negative regression tests preserve both requirements. See
`research/measurements/2026-10-03-foundation-machines.md` for raw outputs,
oracle coverage and the deliberately broken tail variant.

The accounting pass deepens Chapter 16 to about 3,300 prose words and Chapter 17
to about 3,600. Chapter 16 follows a projection from axes through packed
metadata, traffic, dimensions, cache reservation and conditional bounds.
Chapter 17 turns a request cohort into separate wire, correct-completion
and latency-qualified rates, then implements an auditable experiment.
Four new native diagrams, six additional solved problems and three source
listings make the reasoning inspectable. Repeated measurement guidance
was consolidated without removing its technical points.

Two standard-library lessons add eighteen tests. The optional timed mode
compares small synchronous Python integer products with an exact independent
oracle. Two nine-pair invocations, including preparation costs, are preserved
in `research/measurements/2026-10-03-foundation-experiments.md` with raw
times and source/input hashes. These are Python observations, not a native
GEMM, GPU, model or inference-service speed claim.

The latest pass deepens Chapters 9--11 to about 3,100, 2,900 and 3,300
prose words respectively. Attention derives the shifted prediction boundary,
offset chunk masks, GQA axes, operand counts and independent row properties.
The FFN follows both normalized branch inputs and unnormalized residuals,
rectangular matrix coordinates, matched parameter budgets and hidden-tile
lifetimes. Decoder assembly inventories all 280 matrix entries, proves causal
cache reuse by induction and separates selection, feedback and stopping.
Six new native diagrams, nine additional worked problems and three actual
source listings support the explanations. Existing lessons remain intact.

The new standard-library inspection lesson adds seventeen tests. A separate
70-digit Decimal row oracle and analytic residual/zero-update fixtures
complement, rather than replace, shared-primitive parity checks.
The sweep compares 375 three-token histories across three head/layer
configurations. Instrumented calls confirm 24 versus ten projected rows
and 124 versus 60 attention pairs for the stated three-prediction cohort.
Offset-mask and position-reset negative controls expose plausible wrong
answers. The trace retains raw logits and evaluated frontiers in
`research/measurements/2026-10-03-foundation-decoder.md`.
These are correctness observations and derived counts, not speed measurements.

## Validation of this revision

The 702-page PDF builds with XeLaTeX/biber: no overfull boxes, missing glyphs
or unresolved references. All 702 pages rendered and passed the visible
word-bounds audit. The 29 pages of the latest three chapters were reviewed
in colour and grayscale, including all nine figures and three source listings.
Visual review corrected arrows crossing output boxes, a touching-node
connector, key-label spacing and a split worked answer. The previous
Chapters 12--17, preface, Part II transition and 33 systems mechanism-plate
reviews remain in their prior pass records.
This is author QA, not independent technical review or a claim that every
unchanged page received a new visual inspection.

Relevant local checks pass: structure/preservation/secrets, relative links,
textbook contracts and six CLI trace tests, the 14 foundation arithmetic/structure
tests, all 80 foundation tests, frozen diagrams,
deterministic figures, native-figure parity and industrial mutation guards.
The accounting tests independently enumerate work and padding, distinguish
percentile populations, retain failed/incorrect output, reject invalid traces,
abort a corrupted measured route and recompute timing medians from raw rows.
No performance threshold is asserted.
The previous native C fixture's 516-case NEON, UBSan and forced-scalar
results, static analysis, assembly inspection and failed tail mutation remain
in the machine lesson's record; they were not rerun in this pass.
AddressSanitizer remains unverified on this Mac. The combined Linux CI
sanitizer check is committed, but its remote execution has not been observed.
The prior independent ggml CPU result (direct decode, tensor orientation and
32 packed-coordinate probes), five lifecycle tests and full Rust/parity CI
results remain historical evidence; they were not rerun in this pass.
This pass does not promote the lifecycle chapter to FULL or claim new timings.
The previous accounting pass's small Python experiment remains in its dated
record; no accelerator or model performance measurement is claimed.

## Next

1. Complete the existing lifecycle lab's full chapter, now Chapter 31
   (`ch14-request-lifecycle`), preserving the measured cancellation failure.
   Chapters 9--17 now have concrete depth passes; keep their runnable reading
   route when developing the later system mechanisms.
2. Continue the straightforward, concrete-first edit through the bodies of
   systems chapters, then deepen the remaining 29 ZERO chapters. Keep
   the original measurements and failure cases; explain them more gradually.
3. Resolve the five remaining hardware-measurement TODOs when the required
   machines are available. Do not replace missing measurements with predictions.
4. Obtain independent technical review before assigning VERIFIED.

## Historical evidence

The complete pre-restructure status ledger, including measurement sources,
outstanding hardware issues and editorial decisions, is preserved in
`archive/docs/STATUS-2026-09-28-before-foundations.md`. Its numbers refer
to the former chapter order. The earlier first-edition ledger remains at
`archive/docs/STATUS.md`. Research records and lab directory names remain
unchanged so their provenance is not rewritten.

## Chapter status

Generated from the manuscript. Source IDs distinguish preserved lab names
from the current printed chapter numbers. Prose word counts are approximate.

<!-- chapter-status:begin (generated by scripts/check-textbook.py --write-status) -->
| # | Chapter | Source ID | Status | Words | TODOs |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | From Text to Tokens | f01-text-to-tokens | FULL | 7,523 | 0 |
| 2 | The Smallest Possible Language Model | f02-smallest-model | FULL | 6,381 | 0 |
| 3 | Logits, Sampling, and the Autoregressive Loop | f03-sampling | FULL | 6,389 | 0 |
| 4 | Tensors Without Magic | f04-tensors | FULL | 8,079 | 0 |
| 5 | Matrix Multiplication: The Engine Room | f05-matmul | FULL | 10,829 | 0 |
| 6 | Embeddings and Normalization | f06-embeddings-norm | FULL | 7,398 | 0 |
| 7 | Queries, Keys, and Values | f07-qkv | FULL | 5,819 | 0 |
| 8 | Position: RoPE From First Principles | f08-rope | FULL | 5,314 | 0 |
| 9 | Causal Attention, One Row at a Time | f09-causal-attention | FULL | 3,079 | 0 |
| 10 | The Feed-Forward Network and the Residual Stream | f10-feed-forward | FULL | 2,908 | 0 |
| 11 | Assembling a Decoder That Generates Tokens | f11-decoder | FULL | 3,336 | 0 |
| 12 | Model File Formats: GGUF and safetensors | f12-model-formats | FULL | 5,130 | 0 |
| 13 | Numerics: Floating Point and Quantization Formats | f13-numerics | FULL | 3,498 | 0 |
| 14 | CPU and SIMD | f14-cpu-simd | FULL | 3,465 | 0 |
| 15 | GPU Architecture for Inference Engineers | f15-gpu-architecture | FULL | 3,567 | 0 |
| 16 | Reading Shapes, Bytes and Performance Bounds | f16-notation | FULL | 3,279 | 0 |
| 17 | Reproducing the Book's Measurements | f17-measurements | FULL | 3,579 | 0 |
| 18 | Why Inference Is Its Own Discipline | ch01-why-inference | FULL | 6,878 | 0 |
| 19 | The Forward Pass, Compressed | ch02-forward-pass | FULL | 7,004 | 0 |
| 20 | The Roofline: Why Decode Leaves Your GPU Idle | ch03-roofline | FULL | 5,243 | 0 |
| 21 | Prefill and Decode: Two Workloads, One Model | ch04-prefill-decode | FULL | 5,187 | 0 |
| 22 | The KV Cache: Inference's Central Data Structure | ch05-kv-cache | FULL | 5,172 | 0 |
| 23 | Napkin Math: Latency, Throughput and Cost per Token | ch06-napkin-math | FULL | 5,309 | 0 |
| 24 | Matrix Multiplication on Real Hardware | ch07-matmul-hardware | FULL | 5,094 | 0 |
| 25 | Attention as a Memory Problem: From Online Softmax to FlashAttention | ch08-flashattention | FULL | 6,289 | 0 |
| 26 | Decode Attention: FlashDecoding, Split-K and Paged Kernels | ch09-decode-attention | FULL | 5,190 | 0 |
| 27 | Quantizing Weights: INT8, INT4, GPTQ, AWQ and K-Quants | ch10-weight-quantization | FULL | 5,244 | 0 |
| 28 | Below 16 Bits in Floating Point: FP8, MXFP4 and NVFP4 | ch11-low-precision-float | FULL | 5,242 | 0 |
| 29 | Quantizing the KV Cache | ch12-kv-quantization | FULL | 5,185 | 0 |
| 30 | How Kernels Are Written Now: CUDA, Triton, Tile DSLs and Graph Capture | ch13-kernel-languages | FULL | 6,323 | 0 |
| 31 | The Request Lifecycle: Streaming, Cancellation and Terminal Ownership | ch14-request-lifecycle | ZERO | 3,438 | 0 |
| 32 | Continuous Batching | ch15-continuous-batching | ZERO | 2,124 | 0 |
| 33 | Paged KV Memory | ch16-paged-kv | ZERO | 1,946 | 0 |
| 34 | Prefix Caching and RadixAttention | ch17-prefix-caching | ZERO | 2,173 | 0 |
| 35 | Chunked Prefill and Phase Interference | ch18-chunked-prefill | ZERO | 1,958 | 0 |
| 36 | Scheduling Under Pressure: Preemption, Fairness and SLO-Aware Admission | ch19-scheduling | ZERO | 2,004 | 0 |
| 37 | Structured Output at Engine Speed | ch20-structured-output | ZERO | 1,893 | 0 |
| 38 | One Server, Many Models: LoRA, Adapters and Routing | ch21-multi-model-serving | ZERO | 1,941 | 0 |
| 39 | Speculative Decoding: The Draft–Verify Contract | ch22-speculative-decoding | ZERO | 2,384 | 0 |
| 40 | Modern Speculation: The EAGLE Family, Medusa and Multi-Token Prediction | ch23-modern-speculation | ZERO | 2,100 | 0 |
| 41 | When Speculation Loses | ch24-when-speculation-loses | ZERO | 1,928 | 0 |
| 42 | Shrinking the KV Cache: MQA, GQA and Multi-Head Latent Attention | ch25-shrinking-kv | ZERO | 3,323 | 0 |
| 43 | Mixture of Experts | ch26-mixture-of-experts | ZERO | 3,064 | 0 |
| 44 | Hybrid Models: State Spaces and Linear Attention | ch27-hybrid-models | ZERO | 2,837 | 0 |
| 45 | Long Context: Windows, Sinks and Sparse Attention | ch28-long-context | ZERO | 2,711 | 0 |
| 46 | Reasoning Models and Test-Time Compute | ch29-reasoning-workloads | ZERO | 2,223 | 0 |
| 47 | Multimodal Inference | ch30-multimodal | ZERO | 2,107 | 0 |
| 48 | Tensor, Pipeline and Context Parallelism | ch31-model-parallelism | ZERO | 2,524 | 1 |
| 49 | Expert Parallelism at Scale | ch32-expert-parallelism | ZERO | 2,262 | 1 |
| 50 | Disaggregated Serving | ch33-disaggregated-serving | ZERO | 2,103 | 0 |
| 51 | The KV Cache as a Distributed Storage Tier | ch34-kv-storage-tier | ZERO | 2,378 | 1 |
| 52 | Offloading Across VRAM, RAM and NVMe | ch35-offloading | ZERO | 3,574 | 2 |
| 53 | Streaming Experts From Disk: A Measured Case Study | ch36-expert-streaming | ZERO | 4,326 | 0 |
| 54 | Unified Inference Memory | ch37-unified-memory | ZERO | 3,203 | 0 |
| 55 | Fast Wrong Answers: Oracles, Differential Testing and Determinism | ch38-fast-wrong-answers | ZERO | 3,003 | 0 |
| 56 | Benchmarking Without Lying to Yourself | ch39-benchmarking | ZERO | 3,276 | 0 |
| 57 | Operating Inference in Production | ch40-production | ZERO | 2,905 | 0 |
| 58 | Anatomy of Real Engines: vLLM, SGLang, llama.cpp and Hermon | ch41-real-engines | ZERO | 2,629 | 0 |
| 59 | Capstone: Build a Mini Engine End to End | ch42-capstone | ZERO | 3,372 | 0 |
<!-- chapter-status:end -->
