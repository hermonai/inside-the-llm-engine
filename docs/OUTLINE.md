# Detailed Authoring Outline

This is the master specification for **15 parts and 94 chapters**. `BOOK.md` is
the readable table of contents; this file tells authors what each chapter must
teach, build, prove, and hand to the next chapter.

Every entry explicitly covers: purpose, key question, prerequisites, concepts,
mathematics, systems concepts, hardware concepts, implementation work, Hermon
connection, external systems connection, planned ASCII diagrams, experiments,
correctness tests, benchmark where applicable, misconceptions, failure cases,
deliverable, and the next chapter's assumptions. “None” means intentionally out
of scope, not forgotten.

## Part I — What Actually Happens When an LLM Answers?

### Chapter 1 — The Missing Half of AI

- **Purpose / key question:** Define inference engineering and answer what lies between an API request and the first streamed token.
- **Prerequisites:** Programming, processes, files, and basic algebra; no Transformer knowledge.
- **Concepts:** Training versus inference, model artifact, runtime, request, forward pass, state, provider, stream, latency, throughput.
- **Mathematics / systems / hardware:** A latency decomposition and throughput/queueing vocabulary; process boundaries, immutable weights versus mutable request state; CPU/GPU named only as execution locations.
- **Implementation / Hermon / external:** Sketch ENGINE-0 interfaces; classify Hermon's API, dispatcher, default runtime, and providers at the inspected commit; contrast library, server, and managed-service boundaries using primary project docs.
- **Diagrams / experiments:** Whole-stack request map; “follow the token,” “follow the byte,” and “follow the owner” previews; trace a streamed API response without claiming model internals from wire timing.
- **Correctness / benchmark:** Check that one request has exactly one terminal outcome; measure TTFT versus full latency only as a vocabulary exercise with environment recorded.
- **Misconceptions / failures / deliverable / next:** Inference is not “training without gradients,” an API is not the engine, and GPU presence does not prove execution; deliver a stack map and ENGINE-0 boundary; Chapter 2 assumes text must become model identifiers.

### Chapter 2 — From Text to Tokens

- **Purpose / key question:** Explain exactly how bytes become vocabulary identifiers and back, and why tokenization is model semantics.
- **Prerequisites:** Chapter 1's request map; UTF-8 familiarity is introduced as needed.
- **Concepts:** Bytes, Unicode, normalization, pre-tokenization, vocabulary, merge/model rules, special tokens, chat templates, encode/decode, streaming byte fragments.
- **Mathematics / systems / hardware:** Sequence-length and vocabulary notation; deterministic transforms and bounded buffers; cache locality mentioned, no provider optimization yet.
- **Implementation / Hermon / external:** Build a toy byte tokenizer plus interface for a real tokenizer; verify Hermon's real path uses pinned llama.cpp while `hermon-tokenizer` is research/library; consult tokenizer specifications and model metadata.
- **Diagrams / experiments:** Text-to-bytes-to-ids pipeline and split UTF-8 token stream; compare token counts for small inputs under two documented tokenizers.
- **Correctness / benchmark:** Round-trip byte fixtures, special-token rules, malformed UTF-8 policy, deterministic IDs; no speed headline, only tokens/input and allocation observations.
- **Misconceptions / failures / deliverable / next:** Tokens are not words, decode pieces need not be valid UTF-8 independently, and wrong templates change outputs; deliver tokenizer contract and fixtures; Chapter 3 assumes stable token IDs.

### Chapter 3 — The Smallest Possible Language Model

- **Purpose / key question:** Show how a token ID can produce scores for a next token before introducing a Transformer.
- **Prerequisites:** Token IDs and basic vectors.
- **Concepts:** Parameters, embedding lookup, hidden state, output projection, vocabulary score, model shape, learned versus runtime data.
- **Mathematics / systems / hardware:** Vector lookup and `h W_out` with explicit `[hidden] × [hidden,vocab]`; immutable weight storage, temporary activations, scalar CPU execution.
- **Implementation / Hermon / external:** Build ENGINE-1 as a tiny table/projection model with fixed weights; connect only to Hermon's tensor/model facade concept, not its Transformer details; compare with a primary neural language-model reference.
- **Diagrams / experiments:** One-ID forward path and ownership map; hand-compute logits for a three-token vocabulary, then alter one weight.
- **Correctness / benchmark:** Shape validation, exact scalar expected logits, invalid ID rejection; count operations and bytes without a performance claim.
- **Misconceptions / failures / deliverable / next:** A score is not a probability, hidden state is not hidden reasoning, and dimensions are contracts; deliver executable tiny model and oracle; Chapter 4 assumes logits exist.

### Chapter 4 — Logits, Sampling, and the Autoregressive Loop

- **Purpose / key question:** Turn one forward result into repeated token generation with explicit state and stopping.
- **Prerequisites:** Chapter 3 logits and Chapter 2 decoding.
- **Concepts:** Softmax, argmax, temperature, RNG/seed, top-k/top-p overview, end-of-generation token, length/stop limits, autoregressive feedback.
- **Mathematics / systems / hardware:** Stable softmax and categorical sampling; sampler/request ownership and termination; scalar implementation before vector optimization.
- **Implementation / Hermon / external:** Complete ENGINE-0/1 generation loop and Lab 1; map Hermon's sampler facade and stream terminal contract as CURRENT; consult primary sampling definitions.
- **Diagrams / experiments:** Logits-to-token loop and sampler-state ownership; manually generate one token, then compare greedy and seeded stochastic traces.
- **Correctness / benchmark:** Probability sum/tolerance, deterministic greedy output, reproducible seeded sequence, exactly-once terminal state; measure per-stage time only as instrumentation.
- **Misconceptions / failures / deliverable / next:** Softmax is unnecessary for argmax but necessary for probabilities, temperature zero needs defined behavior, and feedback grows state; deliver ENGINE-1 with tests; Part II assumes the loop but not the source of logits.

## Part II — Build a Transformer Inference Engine

### Chapter 5 — Tensors Without Magic

- **Purpose / key question:** Make shape, stride, dtype, layout, and ownership explicit enough to implement every later operation.
- **Prerequisites:** Arrays and Chapter 3 vectors.
- **Concepts:** Rank, dimension, element count, contiguous/strided layout, row-major convention, view versus copy, dtype, broadcast as an explicit rule.
- **Mathematics / systems / hardware:** Index-to-offset formulas and checked shape products; allocation/lifetime and aliasing; cache-line/coalescing preview.
- **Implementation / Hermon / external:** Add a small checked tensor/view layer to `mini-engine`; relate to Hermon's safe tensor facade and packed bridge without copying it; compare NumPy/PyTorch semantics from official docs.
- **Diagrams / experiments:** Shape/stride memory map and view ownership; transpose/view/copy a tiny matrix and inspect offsets.
- **Correctness / benchmark:** Overflow, bounds, zero dimensions, non-contiguous rejection/support; compare iteration orders for locality without generalizing.
- **Misconceptions / failures / deliverable / next:** Shape does not define physical layout, views do not own bytes, and dtype affects storage and arithmetic; deliver tensor contract; Chapter 6 assumes reliable 2-D access.

### Chapter 6 — Matrix Multiplication: The Engine Room

- **Purpose / key question:** Derive matmul as both linear algebra and a memory-traffic problem.
- **Prerequisites:** Chapter 5 layouts and dot products introduced here.
- **Concepts:** Dot product, matrix-vector, matrix-matrix, inner dimension, accumulation, tiling, arithmetic intensity.
- **Mathematics / systems / hardware:** `[M,K]×[K,N]->[M,N]`, FLOPs and byte lower bounds; workspace and loop ordering; cache/register/SIMD hierarchy preview.
- **Implementation / Hermon / external:** Implement scalar matvec/matmul and a blocked variant; connect to Hermon's packed GGML projections and native-kernel substitution policy; consult BLAS contracts.
- **Diagrams / experiments:** Row/column contraction and cache-tile movement; sweep loop order/tile size on fixed hardware.
- **Correctness / benchmark:** Hand-computable matrices, odd/tail sizes, accumulation tolerance, reference differential; record compiler/build/hardware before timing.
- **Misconceptions / failures / deliverable / next:** FLOP count alone does not predict speed, matvec and matmul have different reuse, and low precision still needs accumulation rules; deliver tested primitive; Chapter 7 assumes linear projections.

### Chapter 7 — Embeddings and RMSNorm

- **Purpose / key question:** Build the first real decoder operations and show how normalization stabilizes scale.
- **Prerequisites:** Tensor indexing, matmul, token IDs.
- **Concepts:** Embedding table, hidden vector, RMSNorm, epsilon, learned scale, residual-stream preview.
- **Mathematics / systems / hardware:** Row gather and `x / sqrt(mean(x^2)+eps) * weight` with `[tokens,hidden]`; immutable weights versus activation buffers; reduction/vectorization considerations.
- **Implementation / Hermon / external:** Implement embedding and scalar RMSNorm; verify relevant Hermon paged forward ordering before case-study text; compare primary model architecture definitions.
- **Diagrams / experiments:** Token-row gather and RMSNorm reduction; scale input and observe normalized output, vary epsilon near zero.
- **Correctness / benchmark:** Exact small-vector oracle, zero vector, large magnitude, aliasing rules; benchmark reduction separately only after equivalence.
- **Misconceptions / failures / deliverable / next:** RMSNorm is not LayerNorm, epsilon is semantic metadata, and embeddings are weights; deliver normalized hidden states; Chapter 8 assumes projection-ready activations.

### Chapter 8 — Queries, Keys, and Values

- **Purpose / key question:** Explain what Q, K, and V represent, which token owns each, and how head geometry changes storage.
- **Prerequisites:** Matmul, hidden states, shape notation.
- **Concepts:** Q/K/V projections, query heads, KV heads, head dimension, MHA/GQA/MQA, per-token ownership.
- **Mathematics / systems / hardware:** Projection shapes and query-head-to-KV-head mapping; separate activation buffers and future cache state; bundled projection and memory-reuse preview.
- **Implementation / Hermon / external:** Implement split Q/K/V projections and head views; connect to Hermon's bundled packed projections as verified PREVIEW behavior; use primary model configs for geometry examples.
- **Diagrams / experiments:** Hidden-to-QKV fan-out and GQA head map; compare KV bytes for MHA/GQA/MQA at fixed query heads.
- **Correctness / benchmark:** Shape divisibility, mapping boundaries, independent projection oracle; measure bundled versus separate only later with controls.
- **Misconceptions / failures / deliverable / next:** Q/K/V are token-derived vectors, not database records; GQA shares KV heads, not queries; deliver explicit tensors; Chapter 9 assumes Q/K positions.

### Chapter 9 — Position: RoPE From First Principles

- **Purpose / key question:** Derive how a position-dependent rotation lets attention distinguish token order.
- **Prerequisites:** Q/K head vectors, trigonometry refreshed locally.
- **Concepts:** Paired dimensions, angular frequency, absolute rotation, relative dot-product effect, theta/scaling metadata.
- **Mathematics / systems / hardware:** 2-D rotations across head pairs and position indices; in-place mutation ownership; sin/cos computation/table/vector tradeoffs.
- **Implementation / Hermon / external:** Implement scalar RoPE with explicit layout; verify the selected model's dimension ordering before Hermon comparison; use RoPE paper/model docs as primary sources.
- **Diagrams / experiments:** Rotating coordinate pairs and relative-position dot product; compute positions 0/1 by hand and test extrapolation settings.
- **Correctness / benchmark:** Norm preservation, position zero, odd/partial rotary dimension policy, reference tolerance; compare table versus compute only under defined sequence range.
- **Misconceptions / failures / deliverable / next:** RoPE is not an added positional vector, layout/scaling are model semantics, and extrapolation is not automatic; deliver positioned Q/K; Chapter 10 assumes causal scores.

### Chapter 10 — Causal Self-Attention

- **Purpose / key question:** Derive how each query reads only permitted prior positions and produces a contextual value.
- **Prerequisites:** Q/K/V, RoPE, stable softmax basics.
- **Concepts:** Score, scaling, causal mask, visible positions, softmax weights, weighted value sum, head concatenation.
- **Mathematics / systems / hardware:** Full tensor shapes for MHA/GQA/MQA, `1/sqrt(head_dim)`, stable row softmax, `O(n^2)` prefill work; score scratch and memory access.
- **Implementation / Hermon / external:** Build naive dense scalar attention and Lab 2; relate to Hermon's `CpuPagedAttention` only after semantics; compare original Transformer and current model definitions.
- **Diagrams / experiments:** Causal visibility triangle and one query scanning K/V; disable mask or scaling to expose behavior.
- **Correctness / benchmark:** Hand-computable masked attention, no future influence, GQA mapping, finite extreme scores; benchmark dense reference only as baseline.
- **Misconceptions / failures / deliverable / next:** Attention weights are not explanations, causal masking is per query, and softmax stability is required; deliver verified attention output; Chapter 11 assumes contextual hidden vectors.

### Chapter 11 — The Feed-Forward Network

- **Purpose / key question:** Explain the token-wise parameter-heavy transform that follows attention.
- **Prerequisites:** Matmul, normalization, residual concept.
- **Concepts:** Up/gate/down projections, SiLU, gated FFN, intermediate width, token independence within a layer.
- **Mathematics / systems / hardware:** `down(SiLU(gate(x)) ⊙ up(x))` shapes and parameter/byte counts; activation workspace reuse; projection fusion/bundling preview.
- **Implementation / Hermon / external:** Implement scalar dense FFN; connect to Hermon's packed gate/up bundle and fused-block experiment with correct PREVIEW/HISTORICAL labels; inspect model architecture specs.
- **Diagrams / experiments:** Gated branch fan-out/rejoin; change activation or ordering and observe equivalence failure.
- **Correctness / benchmark:** Hand-sized vectors, activation edge values, bundled versus separate equality; later benchmark matmul grouping with exact controls.
- **Misconceptions / failures / deliverable / next:** FFN is not attention, “MLP” architecture varies, and fusion must preserve ordering; deliver residual-ready FFN; Chapter 12 assembles a layer.

### Chapter 12 — One Complete Transformer Layer

- **Purpose / key question:** Account for every operation, buffer, residual, and owner in one decoder layer.
- **Prerequisites:** Chapters 7–11.
- **Concepts:** Pre-norm layer, attention residual, FFN residual, buffer reuse, layer-local versus persistent state.
- **Mathematics / systems / hardware:** Compose equations with `[tokens,hidden]` invariants; liveness and scratch allocation; kernel boundaries and synchronization points.
- **Implementation / Hermon / external:** Assemble one tested layer with trace hooks; compare ordering with Hermon's verified GGUF Llama preview and selected external architecture, avoiding family-general claims.
- **Diagrams / experiments:** Full layer data/ownership flow and activation liveness timeline; trace tiny inputs through each checkpoint.
- **Correctness / benchmark:** Stage-by-stage oracle, residual aliasing tests, wrong-order negative fixture; profile stage shares without optimizing yet.
- **Misconceptions / failures / deliverable / next:** Residuals do not remove need for correct normalization, buffers cannot be reused while live, and architecture ordering is semantic; deliver one layer; Chapter 13 stacks it.

### Chapter 13 — The Decoder Stack and Next-Token Generation

- **Purpose / key question:** Turn one layer into an end-to-end tiny Transformer that returns next-token logits.
- **Prerequisites:** Complete layer, embeddings, sampler loop.
- **Concepts:** Layer stack, per-layer parameters/state, final normalization, tied/untied output head, selected logit position.
- **Mathematics / systems / hardware:** Repeated shape invariants and total parameter/work formulas; model/context ownership; layer-by-layer execution and potential parallel limits.
- **Implementation / Hermon / external:** Complete ENGINE-2 with a tiny deterministic checkpoint; connect to Hermon's model/context facade and preview forward while preserving status; use a primary tiny-model definition.
- **Diagrams / experiments:** End-to-end decoder stack and layer-state table; generate several tokens from fixed weights and inspect traces.
- **Correctness / benchmark:** Full Python/scalar differential, final-position selection, tied-head equality, deterministic greedy sequence; baseline tokens/s labeled educational.
- **Misconceptions / failures / deliverable / next:** Stacking correct-looking layers can still violate model config, output head semantics matter, and plausible text is not proof; deliver ENGINE-2; Part III assumes correct semantics and asks how bytes encode them.

## Part III — Run a Real Model

### Chapter 14 — What Is Actually Inside a Model File?

- **Purpose / key question:** Inventory the metadata, tensors, tokenizer, templates, and provenance needed to reproduce model semantics.
- **Prerequisites:** ENGINE-2 and binary-file basics.
- **Concepts:** Container, metadata key, tensor descriptor, offset/alignment, architecture/config, tokenizer data, quantization tag, artifact provenance.
- **Mathematics / systems / hardware:** Shape/byte accounting and alignment; file mapping/read ownership and untrusted-input boundary; storage page/cache preview.
- **Implementation / Hermon / external:** Design a format-neutral model manifest inspector; relate to `hermon-gguf` scope and its refusal to infer support; consult GGUF/model-card primary sources.
- **Diagrams / experiments:** File regions and semantic dependency map; inspect metadata/tensor names of an explicitly licensed fixture.
- **Correctness / benchmark:** Bounds/overflow/truncation threat cases, tensor byte reconciliation; measure scan I/O separately from load/compute.
- **Misconceptions / failures / deliverable / next:** A file extension does not establish architecture support, names alone are insufficient, and metadata is untrusted; deliver manifest schema; Chapter 15 specializes it to GGUF.

### Chapter 15 — GGUF From the Bytes Up

- **Purpose / key question:** Parse a GGUF header, typed metadata, tensor directory, alignment, and bounded tensor ranges safely.
- **Prerequisites:** Chapter 14 manifest and checked arithmetic.
- **Concepts:** Magic/version, typed values/arrays, dimensions, GGML type code, relative tensor offset, data alignment.
- **Mathematics / systems / hardware:** Little-endian decoding, align-up, quantized block byte formulas; seek/read/mmap choices and parser budgets; OS page behavior only as consequence.
- **Implementation / Hermon / external:** Implement Lab 4 bounds-checked parser/index; compare behavior and threat checks with `hermon-gguf` CURRENT library; use official GGUF spec/source.
- **Diagrams / experiments:** Byte-level container layout and tensor range resolution; mutate a minimal fixture across valid/invalid cases.
- **Correctness / benchmark:** Truncation, huge counts, overlapping/non-contiguous offsets, invalid alignment/type/block width, exact bounded reader; benchmark metadata scan only if size/control stated.
- **Misconceptions / failures / deliverable / next:** Parsing metadata is not executing the model, offsets require alignment context, and quantized element counts have block constraints; deliver indexed file; Chapter 16 interprets packed types.

### Chapter 16 — Quantization From F32 to Packed Weights

- **Purpose / key question:** Explain how lower-bit block representations preserve usable weight values and alter execution.
- **Prerequisites:** GGUF tensor bytes, floating-point basics refreshed.
- **Concepts:** Group/block quantization, scale, zero/min metadata, symmetric/asymmetric layouts, F16/BF16, Q8/Q4 families, error and calibration.
- **Mathematics / systems / hardware:** Encode/decode formulas, block byte rate and error metrics; packed ownership versus expanded scratch; bandwidth, SIMD unpack, and accumulator dtype.
- **Implementation / Hermon / external:** Implement one scoped quant format plus F32 oracle; inspect pinned GGML format source before Hermon claims; consult primary quantization papers/specs.
- **Diagrams / experiments:** Packed block byte layout and widening path; quantize hand values and compare error/distribution.
- **Correctness / benchmark:** Golden bytes, tails/block divisibility, NaN/range policy, decode differential; measure effective bytes and decode cost, not a generic “4-bit speedup.”
- **Misconceptions / failures / deliverable / next:** Bit count does not identify a format, dequantizing the whole model defeats memory goals, and quality is workload-dependent; deliver tested codec; Chapter 17 computes directly from it.

### Chapter 17 — Packed Matrix Multiplication

- **Purpose / key question:** Multiply packed weights without materializing a full F32 copy.
- **Prerequisites:** Scalar matmul and selected quant block format.
- **Concepts:** Fused unpack/dot, block iteration, scale application, tiling, accumulator, row geometry, packed-kernel contract.
- **Mathematics / systems / hardware:** Dot products grouped by quant blocks; logical versus physical bytes and arithmetic intensity; register/L1 working set, SIMD opportunity.
- **Implementation / Hermon / external:** Implement scalar packed matvec then an optimized version; connect to Hermon's packed GGML bridge and native MoE matvec only within verified format scope; compare official kernel source.
- **Diagrams / experiments:** Packed row to accumulator flow; Lab 5 bandwidth/compute decomposition over sizes and formats.
- **Correctness / benchmark:** Golden packed rows, partial/invalid groups, double/F32 oracle tolerances, accumulation semantics; report build, CPU, bytes, and control.
- **Misconceptions / failures / deliverable / next:** Dequantization is computation and traffic, memory bandwidth may dominate, and equivalent formats need exact layout; deliver packed projection primitive; Chapter 18 integrates a model.

### Chapter 18 — Loading and Running a Real GGUF Model

- **Purpose / key question:** Prove scoped real-model support by mapping metadata/tensors to the exact Transformer semantics.
- **Prerequisites:** ENGINE-2, parser, quant codec, packed matmul.
- **Concepts:** Model-family adapter, tensor-name mapping, tokenizer/template binding, lazy/eager loading, tied head, supported-format matrix.
- **Mathematics / systems / hardware:** Validate every model dimension and byte range; model/context/session ownership; mmap/page faults, CPU/provider placement.
- **Implementation / Hermon / external:** Complete ENGINE-3 for one explicitly named model configuration; use Hermon's real GGUF differential pattern as evidence, not code; compare a trusted implementation at fixed revision.
- **Diagrams / experiments:** Metadata-to-runtime object graph and “follow the byte” real path; run a small prompt under teaching engine and oracle.
- **Correctness / benchmark:** Tokenizer/template, stage logits, greedy sequence, multiple contexts, unsupported metadata hard errors; report load, prefill, decode separately.
- **Misconceptions / failures / deliverable / next:** Architecture-name similarity is not support, plausible output is not equivalence, and source parsing does not cover all quants; deliver verified ENGINE-3; Part IV profiles it.

## Part IV — Why Naive Inference Is Slow

### Chapter 19 — Measure First: Profiling an Inference Engine

- **Purpose / key question:** Identify where time and bytes go before choosing an optimization.
- **Prerequisites:** Working ENGINE-3 and basic statistics.
- **Concepts:** Wall/CPU time, sampling profiler, instrumentation, flame graph, counters, warmup, variance, synchronization, observer effect.
- **Mathematics / systems / hardware:** Latency decomposition and percent-of-total; request/model/cache state controls; CPU samples, GPU timelines, memory bandwidth counters.
- **Implementation / Hermon / external:** Add spans/counters to `mini-engine`; study Hermon's profiling profile and benchmark discipline; use official profiler docs.
- **Diagrams / experiments:** Timeline from tokenize/load/prefill/decode/stream and measurement boundary map; profile a fixed workload before modifying code.
- **Correctness / benchmark:** Ensure instrumentation does not alter outputs; repeat/control cold versus warm; produce a complete benchmark manifest.
- **Misconceptions / failures / deliverable / next:** Microbenchmarks do not identify end-to-end bottlenecks, CPU time differs from wall time, and one run is not evidence; deliver baseline profile; Chapter 20 interprets two workload phases.

### Chapter 20 — Prefill and Decode Are Different Workloads

- **Purpose / key question:** Explain why prompt processing and token-at-a-time generation favor different execution shapes.
- **Prerequisites:** Profile, full forward pass, autoregressive loop.
- **Concepts:** Prefill, decode, query-token count, history length, arithmetic intensity, TTFT, inter-token latency.
- **Mathematics / systems / hardware:** Work/bytes by phase, matmul versus matvec tendency, attention score geometry; phase state transitions; CPU/GPU occupancy implications.
- **Implementation / Hermon / external:** Split metrics and code paths while preserving semantics; inspect Hermon's shared batched-loop treatment; compare primary engine scheduler docs.
- **Diagrams / experiments:** Prefill/decode shape comparison and request timeline; sweep prompt/output length and record phase profiles.
- **Correctness / benchmark:** Same logits across chunked/unchunked prefill, position continuity, phase-specific measurements with fixed model/cache.
- **Misconceptions / failures / deliverable / next:** Decode is not a smaller copy of prefill, TTFT and throughput can trade off, and chunking must preserve positions; deliver phase model; Chapter 21 isolates repeated work.

### Chapter 21 — Why the KV Cache Exists

- **Purpose / key question:** Demonstrate why decoder inference should retain prior keys/values instead of recomputing past token layers.
- **Prerequisites:** Causal attention, full decoder, prefill/decode distinction.
- **Concepts:** Immutable past activations, per-layer K/V reuse, append position, cached versus uncached decode.
- **Mathematics / systems / hardware:** Compare repeated full-prefix work with cached incremental work; persistent cache ownership; memory-capacity/bandwidth tradeoff.
- **Implementation / Hermon / external:** Lab 3 first runs uncached, profiles, then adds KV; contrast Hermon's default llama.cpp-managed KV with PREVIEW Hermon-owned pool; cite primary cache descriptions.
- **Diagrams / experiments:** Repeated no-cache layers versus append-only cached decode and per-layer layout; generate N tokens both ways.
- **Correctness / benchmark:** Temperature-zero logits/token sequence agree, cache positions reset correctly, multiple sequence isolation; report prefill plus N-token decode and bytes.
- **Misconceptions / failures / deliverable / next:** KV does not cache model output or remove attention over history, and stale cache can yield plausible text; deliver ENGINE-4 mechanism; Chapter 22 sizes it.

### Chapter 22 — KV Cache Memory Mathematics

- **Purpose / key question:** Predict cache capacity and traffic from model geometry, dtype, sequence length, and concurrency.
- **Prerequisites:** KV cache and head geometry.
- **Concepts:** Per-token/per-layer bytes, KV dtype, sequence capacity, fragmentation, batch/concurrency budget, GQA effect.
- **Mathematics / systems / hardware:** `layers × tokens × kv_heads × head_dim × 2 × bytes/element`, plus allocation overhead; budget/admission reasoning; RAM/VRAM bandwidth and residency.
- **Implementation / Hermon / external:** Build a checked capacity calculator and compare observed allocations; verify Hermon's reference-pool formula and configured dtypes before citation; compare other engines' official calculators.
- **Diagrams / experiments:** Geometry-to-bytes expansion and capacity budget; sweep context/concurrency/dtype and validate against instrumentation.
- **Correctness / benchmark:** Overflow/unit tests, MHA/GQA cases, allocator overhead disclosure; no tok/s claim—capacity and physical bytes only.
- **Misconceptions / failures / deliverable / next:** Model weight size does not include KV, quantized weights do not imply quantized KV, and logical bytes differ from reserved bytes; deliver memory model; Part V uses it for admission.

## Part V — From Model Runner to Inference Server

### Chapter 23 — One User Is Easy

- **Purpose / key question:** Expose assumptions hidden by a synchronous single-request loop before concurrency invalidates them.
- **Prerequisites:** ENGINE-4 and basic networking/async concepts.
- **Concepts:** Request boundary, model instance, context, session, stream, stop/cancel signal, resource lifetime.
- **Mathematics / systems / hardware:** Single-user latency path; exclusive ownership and blocking I/O; one provider queue without contention.
- **Implementation / Hermon / external:** Wrap ENGINE-4 in a minimal local streaming server; map Hermon's normalized local request surface; compare protocol basics from official specs.
- **Diagrams / experiments:** Single request owner/lifetime and synchronous call timeline; disconnect a client at three phases.
- **Correctness / benchmark:** Exactly one terminal result, bounded output, resource release on disconnect/error; baseline latency only.
- **Misconceptions / failures / deliverable / next:** A working demo is not multi-tenant, async syntax does not create safe sharing, and disconnect is a state transition; deliver ENGINE-5 shell; Chapter 24 formalizes states.

### Chapter 24 — The Inference Request State Machine

- **Purpose / key question:** Define legal request transitions and who owns every resource in each state.
- **Prerequisites:** Server shell and generation lifecycle.
- **Concepts:** Validate, queue, admit, prefill, decode, drain, complete, cancel, fail; sequence/request distinction; terminal idempotence.
- **Mathematics / systems / hardware:** State-transition invariants and deadlines; leases/channels/tasks; provider work that may outlive caller cancellation.
- **Implementation / Hermon / external:** Implement explicit enum/transitions and RAII cleanup; compare Hermon's stream contract and worker ownership; use state-machine/concurrency literature.
- **Diagrams / experiments:** Full state graph with failure/cancel edges and resource table by state; inject error at every transition.
- **Correctness / benchmark:** Property tests for legal transitions, exactly-once terminal event, no resource leak/double release; measure queue versus execution time separately.
- **Misconceptions / failures / deliverable / next:** Cancellation is not deletion, request and sequence need not be one-to-one, and success cleanup is insufficient; deliver tested lifecycle; Chapter 25 adds contention.

### Chapter 25 — Serving Multiple Users

- **Purpose / key question:** Show why independent serial loops underutilize or oversubscribe shared model/hardware resources.
- **Prerequisites:** State machine, context/KV ownership, concurrency primitives.
- **Concepts:** Admission queue, worker/context pool, model sharing, isolation, concurrency limit, head-of-line blocking.
- **Mathematics / systems / hardware:** Little's-law intuition and throughput/latency tradeoff; shared immutable weights versus mutable contexts; CPU thread/GPU queue oversubscription.
- **Implementation / Hermon / external:** Add bounded multi-request execution and pool baseline; compare Hermon's compatibility pool versus default batched runtime; consult primary serving-system designs.
- **Diagrams / experiments:** Multiple requests sharing weights but not state; sweep concurrency and observe saturation/queueing.
- **Correctness / benchmark:** Cross-request token/KV isolation, ordering-independent outputs, failure containment; report per-request tails and aggregate throughput.
- **Misconceptions / failures / deliverable / next:** More contexts may lower throughput, shared weights do not permit shared mutation, and concurrency is not batching; deliver controlled baseline; Chapter 26 batches iterations.

### Chapter 26 — Continuous Batching

- **Purpose / key question:** Rebuild the physical work each iteration so active sequences share execution without waiting for request-batch completion.
- **Prerequisites:** Multi-user baseline, prefill/decode shapes, request state.
- **Concepts:** Iteration scheduling, physical token batch, logical slot/sequence ID, admission, prefill chunking, decode step, dynamic membership.
- **Mathematics / systems / hardware:** Token-budget and utilization model; sole-mutator worker plus bounded channels; matvec-to-matmul and occupancy implications.
- **Implementation / Hermon / external:** Build ENGINE-6 and Lab 6; source-map Hermon's CURRENT `BatchedWorker` invariants; compare vLLM/SGLang official scheduler designs without claiming identity.
- **Diagrams / experiments:** One mixed prefill/decode iteration and sequence-slot timeline; replay staggered arrivals against serial/pool/batched baselines.
- **Correctness / benchmark:** Batched versus isolated greedy differential, positions, stop/cancel, failed-batch behavior; throughput, TTFT, ITL, tail latency, fairness with full manifest.
- **Misconceptions / failures / deliverable / next:** Continuous batching is not static batching, one failed shared call affects represented sequences, and bigger batches can hurt latency; deliver ENGINE-6; Chapter 27 adds policy and flow control.

### Chapter 27 — Fairness, Backpressure, Cancellation, and Streaming

- **Purpose / key question:** Keep a busy server bounded and responsive when clients, prompts, outputs, and failures differ.
- **Prerequisites:** Continuous-batched state machine and queues.
- **Concepts:** Fairness, priority, starvation, token budget, bounded buffer, backpressure, pacing, cancellation propagation, UTF-8-safe streaming.
- **Mathematics / systems / hardware:** Queue wait/service accounting and fairness metrics; channel capacity and ownership release; device work cancellation granularity.
- **Implementation / Hermon / external:** Add bounded streams, fair admission/token budget, cancellation, Lab 12 hook; compare Hermon's bounded stream and current worker policy versus TARGET scheduler; consult networking/queueing sources.
- **Diagrams / experiments:** Backpressure propagation and cancellation across API/runtime/provider; slow consumer, long prompt, and mixed-priority replay.
- **Correctness / benchmark:** No starvation under defined policy, memory bound, valid byte stream, exactly-once release; measure tails/fairness at saturation.
- **Misconceptions / failures / deliverable / next:** Dropping a socket does not automatically stop model work, unbounded channels hide overload, and fairness differs from equal tokens; deliver production-shaped ENGINE-6; Part VI replaces slot-bound storage.

## Part VI — Paged Inference Memory

### Chapter 28 — Why Flat and Slot-Bound KV Caches Fail

- **Purpose / key question:** Make fragmentation, fixed-slot capacity, duplicated prefixes, and affinity limits measurable.
- **Prerequisites:** KV memory math, multi-user scheduler, sticky context model.
- **Concepts:** Contiguous reservation, internal/external fragmentation, slot-bound lifetime, duplicated prefix, compaction, affinity.
- **Mathematics / systems / hardware:** Reserved versus used bytes under variable lengths; allocation policy and stranded capacity; large RAM/VRAM regions and copy cost.
- **Implementation / Hermon / external:** Instrument slot utilization and prefix duplication; use Hermon's CURRENT sticky slots versus PREVIEW page sharing as case study; compare PagedAttention motivation.
- **Diagrams / experiments:** Slot layouts under variable requests and duplicated system prompt; replay length distribution and calculate waste.
- **Correctness / benchmark:** Allocation accounting and isolation remain correct; report utilization/copies rather than conflating with throughput.
- **Misconceptions / failures / deliverable / next:** Sticky prefix reuse is valuable but not arbitrary page sharing, and fragmentation is physical policy not model semantics; deliver failure evidence; Chapter 29 derives paging.

### Chapter 29 — Paging Comes to AI

- **Purpose / key question:** Separate logical sequence order from physical KV placement while preserving attention semantics.
- **Prerequisites:** Slot failures, OS virtual-memory analogy introduced carefully.
- **Concepts:** Fixed-capacity block, logical block, physical block, block table, translation, non-contiguous allocation, block size tradeoff.
- **Mathematics / systems / hardware:** Logical-position-to-block/offset formulas and metadata/waste costs; allocator ownership; gather/scatter memory access.
- **Implementation / Hermon / external:** Introduce block-table interfaces without sharing yet; compare Hermon's `BlockTable` and vLLM's primary paper/docs while stating analogy limits.
- **Diagrams / experiments:** Logical sequence mapped to scattered physical blocks and OS analogy/breakpoints; vary block size on synthetic lengths.
- **Correctness / benchmark:** Translation boundaries `B-1/B/B+1`, order preservation, invalid IDs; measure fragmentation versus metadata/lookups.
- **Misconceptions / failures / deliverable / next:** AI paging need not page-fault to disk, physical non-contiguity must not alter position order, and smaller blocks are not free; deliver mapping model; Chapter 30 owns blocks.

### Chapter 30 — Building a Block Pool

- **Purpose / key question:** Define allocation, reference, mutation, and reuse rules for physical KV storage.
- **Prerequisites:** Block tables and KV geometry.
- **Concepts:** Preallocated slab, free list, BlockId, allocate, incref, decref, refcount, layer layout, exhaustion.
- **Mathematics / systems / hardware:** Pool-size checked products and per-block strides; allocator critical sections/atomics and RAII leases; alignment/cache-line concerns.
- **Implementation / Hermon / external:** Build Lab 7 safe CPU pool; compare Hermon's `CpuBlockPool` and native typed pool as separate PREVIEW/LIBRARY layers; draw on allocator literature.
- **Diagrams / experiments:** Pool metadata/data split and reference state machine; concurrent allocate/release stress.
- **Correctness / benchmark:** Zero/over-allocation, double-decref defense, refcount reuse iff zero, layer/offset bounds, sanitizer; benchmark allocator ops separately from attention.
- **Misconceptions / failures / deliverable / next:** Free-list membership and zero refcount must agree, an ID is not a pointer, and atomicity does not define ownership; deliver pool; Chapter 31 adds cache owners.

### Chapter 31 — Prefix Indexing and Radix Trees

- **Purpose / key question:** Find the longest reusable token prefix and make cache ownership explicit.
- **Prerequisites:** Block pool, token identity, trie familiarity introduced.
- **Concepts:** Prefix, radix compression, longest match, cache entry, independent references, insert, eviction, token/model/config cache key.
- **Mathematics / systems / hardware:** Lookup complexity and block-aligned reusable depth; synchronization/entry lifetime; pointer chasing and locality.
- **Implementation / Hermon / external:** Build prefix radix over token IDs; compare Hermon's `CpuPrefixRadix` PREVIEW and SGLang primary RadixAttention sources; do not treat text equality as key equality.
- **Diagrams / experiments:** Compressed radix branches with shared blocks and ownership graph; insert overlapping prompts and observe reference counts.
- **Correctness / benchmark:** Exact/partial/no match, replacement, eviction decref, concurrent readers, model/config separation; report hit rate and tokens saved with workload.
- **Misconceptions / failures / deliverable / next:** Cache entry ownership outlives requests, longest token prefix differs from string prefix, and a hit is not automatically useful; deliver index; Chapter 32 handles writable tails.

### Chapter 32 — Shared Prefixes and Copy-on-Write

- **Purpose / key question:** Explain why a shared partial tail is safe to read but unsafe to extend in place.
- **Prerequisites:** Prefix references, block capacity, continuation writes.
- **Concepts:** Full immutable block, partial tail, alias, private continuation, COW, publish boundary.
- **Mathematics / systems / hardware:** Valid-token range and copy bytes; mutation rights and lease transfer; block-copy bandwidth.
- **Implementation / Hermon / external:** Implement COW and Lab 8 corruption reproducer; study Hermon's cross-backend `copy_block` integration failure as HISTORICAL evidence; compare OS/filesystem COW analogy and limits.
- **Diagrams / experiments:** Two continuations before/after COW and mutation routing surface; disable COW to produce deterministic cross-request corruption.
- **Correctness / benchmark:** Aligned and partial prefixes, source unchanged, destination valid region equal, backend copy differential; measure copy overhead by tail length.
- **Misconceptions / failures / deliverable / next:** Refcounting alone does not prevent writes, full blocks need not be copied, and backend abstractions must cover every mutation; deliver safe sharing; Chapter 33 decides what stays resident.

### Chapter 33 — Eviction, Pressure, and Admission

- **Purpose / key question:** Decide which cached state to retain and whether a new request can safely enter under finite capacity.
- **Prerequisites:** Pool/refcounts, prefix index, request scheduler.
- **Concepts:** Resident cache owner, evictable versus pinned, LRU/cost policy, pressure signal, reserve, admission/retry, working set.
- **Mathematics / systems / hardware:** Capacity, reuse-value/cost heuristics, overcommit bounds; lock ordering and eviction races; RAM/VRAM pressure and copy cost.
- **Implementation / Hermon / external:** Add explicit eviction/admission policy and metrics; compare Hermon's PREVIEW radix eviction and storage pin-state rules; use buffer-pool/OS sources with analogy limits.
- **Diagrams / experiments:** Ownership/pin/eviction state machine and admission under pressure; replay skewed/uniform prefixes and forced exhaustion.
- **Correctness / benchmark:** Never evict live/pinned references, atomic retry, no leaks/starvation, deterministic policy where promised; hit rate, evictions, wait, and physical bytes.
- **Misconceptions / failures / deliverable / next:** Unpinned is not unowned, LRU is not universally optimal, and admission is part of correctness; deliver bounded cache manager; Chapter 34 reads scattered state.

### Chapter 34 — Paged Attention

- **Purpose / key question:** Compute causal attention directly through a block table without reconstructing dense KV.
- **Prerequisites:** Dense attention, block translation, COW-safe tables.
- **Concepts:** Paged gather, query-to-KV mapping, visible position, online scan preview, block-table lease during forward.
- **Mathematics / systems / hardware:** Same attention equation over translated positions; stable score/value accumulation; irregular memory access, locality, vector/GPU implications.
- **Implementation / Hermon / external:** Implement clarity-first CPU paged attention and finish ENGINE-7; compare Hermon's `CpuPagedAttention` oracle and vLLM designs; no native optimization yet.
- **Diagrams / experiments:** Query scan across scattered blocks and dense/paged semantic equivalence; randomize physical placement and block size.
- **Correctness / benchmark:** Dense differential across B-1/B/B+1, MHA/GQA, causal offsets, scrambled tables, shared prefixes; measure overhead/memory without claiming production speed.
- **Misconceptions / failures / deliverable / next:** Paging changes addressing, not math; gathering dense scratch defeats purpose; locks/leases must cover reads; deliver ENGINE-7 oracle; Part VII moves the hot path native.

## Part VII — Native Kernel Engineering

### Chapter 35 — Where the Native Boundary Belongs

- **Purpose / key question:** Decide which mechanism benefits from a native boundary without moving scheduling and ownership policy into opaque code.
- **Prerequisites:** ENGINE-7 profile and Rust/C literacy.
- **Concepts:** Hot path, host policy, kernel mechanism, FFI cost, safety boundary, substitution unit, bulk call.
- **Mathematics / systems / hardware:** Call overhead versus work size; ownership transfer and error containment; compiler/ISA access and provider portability.
- **Implementation / Hermon / external:** Profile then select paged attention/block operations as the first boundary; study Hermon's “own policy, borrow mechanism” rule and C kernel scope; compare stable native library boundaries.
- **Diagrams / experiments:** Host/runtime versus native mechanism map and too-fine/too-broad boundary examples; measure empty-call and realistic bulk-call overhead.
- **Correctness / benchmark:** Same oracle before/after boundary, panic/error translation, no hidden thread ownership; report crossover rather than “FFI is fast.”
- **Misconceptions / failures / deliverable / next:** C is not automatically fast, rewriting policy reduces reviewability, and per-element FFI erases gains; deliver boundary decision record; Chapter 36 specifies it.

### Chapter 36 — Designing a Stable Kernel ABI

- **Purpose / key question:** Make shapes, buffers, ownership, versioning, errors, and workspace explicit across compiled components.
- **Prerequisites:** Boundary decision, C representation, paged attention inputs.
- **Concepts:** Opaque handle, fixed-width type, `struct_size`, ABI version, status code, slice length, plan/workspace, capability query.
- **Mathematics / systems / hardware:** Checked byte/element formulas; allocator/handle lifetime and unwind behavior; alignment/device-pointer representation.
- **Implementation / Hermon / external:** Define C header plus safe Rust wrapper; compare Hermon's ABI handshake and opaque handles as LIBRARY evidence; consult platform ABI and versioning guidance.
- **Diagrams / experiments:** Rust-to-C ownership/lifetime diagram and version-size negotiation; compile mismatched client fixture and invalid shapes.
- **Correctness / benchmark:** Layout/size assertions, null/overflow/short-buffer rejection, error detail lifetime, fuzz boundary; measure bulk-call overhead with workspace declared.
- **Misconceptions / failures / deliverable / next:** Header compatibility needs runtime guards, raw pointers do not express length, and error codes need ownership semantics; deliver stable v1 contract; Chapter 37 supplies memory.

### Chapter 37 — Arena Allocation and Hot-Path Memory Discipline

- **Purpose / key question:** Eliminate unpredictable general allocation from kernel iterations while retaining auditable lifetime and alignment.
- **Prerequisites:** ABI, pool byte formulas, allocator basics.
- **Concepts:** Arena reserve/commit, alignment, bump/region allocation, reset point, high-water mark, workspace versus persistent state.
- **Mathematics / systems / hardware:** Align-up and capacity accounting; arena owns pool/workspaces with explicit destruction order; page size, huge pages, cache/TLB considerations.
- **Implementation / Hermon / external:** Implement native arena and stats; compare Hermon's C11 arena/native pool lifetime; consult allocator and OS VM sources.
- **Diagrams / experiments:** Arena regions/lifetimes and allocation timeline; stress alignment, exhaustion, reset, and repeated iterations.
- **Correctness / benchmark:** Overflow, zero size, destruction order, sanitizer, no hot-loop allocator calls; report allocation latency/high-water under controlled workload.
- **Misconceptions / failures / deliverable / next:** Arenas do not solve bounds, reset is unsafe while references live, and reserved address space differs from resident bytes; deliver arena; Chapter 38 makes allocation concurrent.

### Chapter 38 — Lock-Free Block Allocation and Refcounts

- **Purpose / key question:** Move frequent block lifecycle operations native and concurrent without losing the refcount invariant.
- **Prerequisites:** Safe pool semantics, atomics, C ABI, arena.
- **Concepts:** Atomic free stack/bitmap, compare-exchange, memory order, ABA risk, reference transition, bulk allocate/release.
- **Mathematics / systems / hardware:** Linearization points and invariant proof; host leases versus atomic metadata; cache-line contention and false sharing.
- **Implementation / Hermon / external:** Implement a scoped native allocator/refcounter with bulk API; compare Hermon's typed native pool and safe Rust wrapper; use primary lock-free references.
- **Diagrams / experiments:** Allocation state transition with linearization and contended cache lines; randomized multi-thread stress and forced exhaustion.
- **Correctness / benchmark:** No duplicate live ID, no lost block, reuse iff zero, ABA mitigation, thread-count differential, sanitizers; throughput versus safe locked oracle by contention.
- **Misconceptions / failures / deliverable / next:** Lock-free does not mean wait-free or automatically faster, relaxed ordering needs proof, and refcount does not grant mutation rights; deliver native pool; Chapter 39 moves KV data in bulk.

### Chapter 39 — Bulk KV Writes

- **Purpose / key question:** Replace per-element host/native crossings with one validated write over token/head ranges.
- **Prerequisites:** Native pool layout, Q/K/V tensors, block tables.
- **Concepts:** Bulk descriptor, source stride, destination block/offset, dtype conversion, partial range, transactional validation.
- **Mathematics / systems / hardware:** Source/destination byte ranges and layout transform; validate-before-mutate error semantics; vector stores, alignment, bandwidth.
- **Implementation / Hermon / external:** Add bulk write and route every KV mutation through backend surface; connect to Hermon's kernel ABI and historical missed `copy_block` lesson; compare DMA/copy API contracts.
- **Diagrams / experiments:** Q/K/V range scattered across block tails and backend mutation surface; write across B-1/B/B+1 and convert dtypes.
- **Correctness / benchmark:** Whole-range and scrambled tables, no partial mutation on invalid input, source alias policy, dense readback differential; bytes/s and call-count reduction.
- **Misconceptions / failures / deliverable / next:** A bulk API needs stronger validation, copy and append are both lifecycle mutations, and bandwidth alone is not model speed; deliver native KV ingress; Chapter 40 optimizes reads/reduction.

### Chapter 40 — Online Softmax

- **Purpose / key question:** Produce exact softmax-weighted values without allocating a score vector for the full history.
- **Prerequisites:** Stable dense/paged attention and reduction math.
- **Concepts:** Running maximum, running normalizer, rescaled accumulator, block/stream update, numerical invariant.
- **Mathematics / systems / hardware:** Derive update equations and mergeability with shapes/precision; workspace reduction state; register footprint and memory-traffic savings.
- **Implementation / Hermon / external:** Implement scalar online attention oracle then native version; compare Hermon's reference/native attention contract; consult FlashAttention/online-normalizer primary sources.
- **Diagrams / experiments:** Streaming score/value update and two-chunk merge; adversarial score ranges and chunk orders.
- **Correctness / benchmark:** Dense softmax differential, overflow/underflow, empty/one position, GQA, tolerance by dtype; scratch bytes and runtime across sequence lengths.
- **Misconceptions / failures / deliverable / next:** Online softmax does not approximate by definition, reduction order affects bits, and max rescaling must update both sum and value; deliver bounded-scratch attention; Chapter 41 parallelizes it deterministically.

### Chapter 41 — Split-K and Deterministic Attention Planning

- **Purpose / key question:** Parallelize long-history attention while making task completion order irrelevant to numerical reduction order.
- **Prerequisites:** Online softmax, host scheduler, ABI/workspace.
- **Concepts:** Plan/task/combine, query tile, KV split, immutable task grid, indexed workspace, deterministic combine, split threshold.
- **Mathematics / systems / hardware:** Merge partial max/sum/value states in fixed split order; host owns scheduling/workspace lifetime; cores/GPU workgroups and load balance.
- **Implementation / Hermon / external:** Build plan/execute/combine and finish ENGINE-8; compare Hermon's LIBRARY kernel contract and threshold discipline; consult flash-decoding primary sources.
- **Diagrams / experiments:** Deterministic task grid and out-of-order completion into indexed slots; randomize task order/thread count around thresholds.
- **Correctness / benchmark:** Serial/parallel equality or stated tolerance, split T-1/T/T+1, workspace bounds, task uniqueness, failure fallback; speedup/tail by sequence and threads.
- **Misconceptions / failures / deliverable / next:** Deterministic task creation alone is insufficient, completion order must not choose reduction order, and split-K can lose on short contexts; deliver ENGINE-8; Part VIII specializes execution.

## Part VIII — SIMD and Accelerator Providers

### Chapter 42 — SIMD From First Principles

- **Purpose / key question:** Map scalar loops to vector lanes while preserving tails, layout, and reduction semantics.
- **Prerequisites:** Native scalar kernel, CPU/memory basics.
- **Concepts:** Lane, vector width, load/store, horizontal reduction, mask/tail, alignment, intrinsic versus auto-vectorization, dispatch.
- **Mathematics / systems / hardware:** Lane-wise dot/exp/value update; same buffer ownership; registers, cache lines, throughput/latency.
- **Implementation / Hermon / external:** Add portable scalar plus one abstract vector microkernel and Lab 9 harness; compare Hermon's scalar oracle discipline; use vendor intrinsic docs.
- **Diagrams / experiments:** Scalar iterations packed into lanes and tail handling; inspect compiler output and sweep aligned/odd dimensions.
- **Correctness / benchmark:** All tails, alignment, NaNs/infinities policy, scalar differential, sanitizer; cycles/element and bandwidth with compiler/ISA recorded.
- **Misconceptions / failures / deliverable / next:** SIMD is not threading, width without locality may not help, and horizontal reductions alter order; deliver ISA-neutral microkernel contract; Chapter 43 maps NEON.

### Chapter 43 — ARM NEON

- **Purpose / key question:** Implement and validate the vector contract on ARM's baseline 128-bit SIMD.
- **Prerequisites:** Chapter 42 contract and ARM build access/emulation limits understood.
- **Concepts:** NEON vector types/intrinsics, FMA, lane reduction, feature detection, AArch64 assumptions.
- **Mathematics / systems / hardware:** Four-lane F32 accumulation and approximation choices; dispatch ownership; Apple/ARM cache and instruction considerations.
- **Implementation / Hermon / external:** Implement NEON attention inner loop with scalar fallback; compare Hermon's shipped NEON kernel/source tests; consult Arm official docs.
- **Diagrams / experiments:** NEON tile/register map; sweep head dimensions, contexts, and compiler flags on named hardware.
- **Correctness / benchmark:** Scalar differential, non-multiple tails, exact dispatch, cross-compile CI plus real-hardware record; report crossover and confidence.
- **Misconceptions / failures / deliverable / next:** Apple silicon is not synonymous with Metal, NEON support does not prove faster for every shape, and approximate exp affects tolerance; deliver NEON provider; Chapter 44 adds x86 dispatch.

### Chapter 44 — x86 AVX2 and ISA Dispatch

- **Purpose / key question:** Add a wider x86 path and select it safely at runtime without executing unsupported instructions.
- **Prerequisites:** SIMD contract, x86 feature-detection basics.
- **Concepts:** AVX2/FMA, 256-bit lanes, CPUID/runtime detection, function multiversioning, fallback, transition/tail concerns.
- **Mathematics / systems / hardware:** Eight-lane F32 grouping and fixed reduction; initialization/dispatch table ownership; x86 vector/cache behavior.
- **Implementation / Hermon / external:** Implement AVX2 specialization and dispatch tests; classify Hermon's AVX2 implementation at inspected commit; consult Intel/AMD compiler docs.
- **Diagrams / experiments:** Dispatch tree and AVX2 tile; run scalar-forced versus auto versus AVX2 on supported hardware.
- **Correctness / benchmark:** Unsupported-host safety, feature spoof/forced fallback, tails, scalar differential, thread counts; cycles/element plus frequency effects.
- **Misconceptions / failures / deliverable / next:** Build-machine ISA cannot be assumed on deploy host, zero-duration readings are measurement bugs, and wider vectors may lower frequency; deliver portable CPU binary; Chapter 45 generalizes providers.

### Chapter 45 — Thinking Like a GPU

- **Purpose / key question:** Reframe a kernel as many work items with explicit memory spaces, synchronization, launch, and transfer costs.
- **Prerequisites:** Plan/task/combine, SIMD, memory hierarchy.
- **Concepts:** Grid, block/threadgroup, warp/SIMD group, global/shared/thread memory, occupancy, divergence, launch, asynchronous queue.
- **Mathematics / systems / hardware:** Map attention task grid and work/byte ratios; host/device ownership and completion events; coalescing, occupancy, synchronization.
- **Implementation / Hermon / external:** Define provider capability/execute/fallback interface before GPU code; compare Hermon's backend-agnostic plan and host fallback; use CUDA/Metal official programming guides.
- **Diagrams / experiments:** CPU tasks versus GPU grid and memory hierarchy; paper-plan a shape and estimate launch/work threshold.
- **Correctness / benchmark:** Provider conformance requirements and asynchronous lifetime tests; no speed claim without implementation.
- **Misconceptions / failures / deliverable / next:** Thousands of threads do not remove serial reductions, device memory is not a Rust borrow, and launch overhead is real; deliver provider contract; Chapter 46 implements unified-memory Metal.

### Chapter 46 — Metal and Unified Memory

- **Purpose / key question:** Execute planned attention on Apple GPU while reasoning precisely about shared physical memory and synchronization.
- **Prerequisites:** GPU model, Objective-C/C bridge basics as needed.
- **Concepts:** Metal device/queue/pipeline/buffer, threadgroup, command buffer, unified memory, storage mode, completion, shape gate.
- **Mathematics / systems / hardware:** Map `(query,head,split)` tasks; buffer lifetime across command completion; Apple unified-memory bandwidth and fixed launch cost.
- **Implementation / Hermon / external:** Build a scoped Metal provider with CPU fallback; examine Hermon's gated Metal native attention and negative short-shape result; use Apple docs.
- **Diagrams / experiments:** Command lifecycle and shared-memory residency; Lab 10 Metal/CPU crossover sweep.
- **Correctness / benchmark:** Scalar differential by shape/dtype, unsupported fallback, command failure, buffer lifetime; warm/cold pipeline and synchronization included.
- **Misconceptions / failures / deliverable / next:** Unified memory does not mean zero synchronization or zero copies in every API, GPU can be slower, and compiled pipeline is not selected execution; deliver Metal provider; Chapter 47 tackles discrete devices.

### Chapter 47 — CUDA and Device Mirrors

- **Purpose / key question:** Maintain coherent device-resident/mirrored state and run the same plan on a discrete NVIDIA GPU.
- **Prerequisites:** Provider contract, GPU execution, KV ownership.
- **Concepts:** Device allocation, stream/event, host/device mirror, dirty state, transfer, kernel launch, capability, fallback after partial work.
- **Mathematics / systems / hardware:** PCIe/transfer plus kernel cost and mirror-byte accounting; lease until event completion; CUDA hierarchy/VRAM constraints.
- **Implementation / Hermon / external:** Implement or scaffold a scoped CUDA provider subject to available hardware, never fabricate results; inspect Hermon's shape-gated CUDA source and embedded/build approach; use NVIDIA docs.
- **Diagrams / experiments:** Host home/device mirror coherence and stream timeline; sweep resident versus transferred inputs where hardware exists.
- **Correctness / benchmark:** CPU differential, stale mirror injection, unsupported/error fallback overwrites workspace, multi-stream lifetime; include driver/device/clock/build.
- **Misconceptions / failures / deliverable / next:** VRAM mirror is not ownership home by default, async return is not completion, and a GPU build does not prove GPU execution; deliver CUDA provider or verified scaffold; Chapter 48 derives dispatch economics.

### Chapter 48 — Why a GPU Can Be Slower Than a CPU

- **Purpose / key question:** Build an evidence-based shape gate from fixed overhead, work, transfer, and contention rather than brand assumptions.
- **Prerequisites:** CPU/Metal/CUDA providers and benchmark policy.
- **Concepts:** Crossover, fixed/variable cost, residency, launch, synchronization, batch shape, hysteresis/load-aware limits, reproducibility.
- **Mathematics / systems / hardware:** `T_gpu = launch+transfer+work` versus `T_cpu`; provider selection ownership; integrated versus discrete memory economics.
- **Implementation / Hermon / external:** Implement a pure shape-based initial gate and Lab 10; analyze Hermon's documented Metal/CUDA gates as HISTORICAL/CURRENT-LIBRARY evidence at commit; compare vendor guidance.
- **Diagrams / experiments:** Cost curves/crossover table and deterministic fallback tree; matched CPU/GPU sweep across context/query shapes.
- **Correctness / benchmark:** Same outputs on both branches, T-1/T/T+1, forced modes, failure fallback; report losing cases and variance.
- **Misconceptions / failures / deliverable / next:** Peak FLOPs do not predict decode latency, gate changes can alter floating reduction, and adaptive load gates harm reproducibility unless recorded; deliver ENGINE-9; Part IX optimizes decode policy.

## Part IX — Modern Decode Optimization

### Chapter 49 — Prefix Caching

- **Purpose / key question:** Reuse computed KV for repeated token prefixes across requests without violating model/config/lifetime semantics.
- **Prerequisites:** Paged KV radix/COW, request identity, benchmark controls.
- **Concepts:** Cache key, computed prefix, reusable depth, cache hit, prefill tokens saved, invalidation, tenancy/security.
- **Mathematics / systems / hardware:** Saved prefill work versus lookup/retention cost; independent cache-owner references; capacity/residency effect.
- **Implementation / Hermon / external:** Integrate prefix cache with ENGINE-9 scheduler; distinguish Hermon's CURRENT sticky slots from PREVIEW radix cache; compare SGLang/vLLM primary sources.
- **Diagrams / experiments:** Cross-request prefix ownership and cold/warm timelines; shared-system-prompt workload with cache on/off.
- **Correctness / benchmark:** Cached/uncached logits, config/model/tokenizer keying, partial COW, eviction, tenant boundaries; TTFT/tokens saved/hit rate with cold control.
- **Misconceptions / failures / deliverable / next:** Text prefix is insufficient, cache hits do not skip decode attention, and retained state has security/capacity cost; deliver general cache; Chapter 50 studies an intermediate design.

### Chapter 50 — Sticky Slots as an Intermediate Design

- **Purpose / key question:** Understand when context-bound prefix reuse offers high value with lower complexity and where it stops scaling.
- **Prerequisites:** Prefix caching and multi-sequence contexts.
- **Concepts:** Warm slot, affinity, overlap, retained KV, slot eviction, conversation hint, fixed capacity.
- **Mathematics / systems / hardware:** Reuse probability versus slot count and stranded-capacity model; one-context ownership; existing provider context behavior.
- **Implementation / Hermon / external:** Implement/compare a sticky-slot mode; source-walk Hermon's CURRENT warm-slot/affinity policy; compare cache-affinity ideas in databases/servers with limits.
- **Diagrams / experiments:** Warm-slot routing and contrast with page-sharing radix; alternating conversations beyond slot capacity.
- **Correctness / benchmark:** Slot metadata never outlives KV, overlap leaves a fresh logit position, cross-conversation isolation; shared-prefix TTFT and churn workload.
- **Misconceptions / failures / deliverable / next:** Sticky slots are not paged prefix sharing, affinity is a hint not ownership, and favorable benchmarks can hide churn; deliver comparative design note/mode; Chapter 51 changes decode steps.

### Chapter 51 — Speculative Decoding

- **Purpose / key question:** Reduce target-model decode iterations while preserving the target distribution exactly under the chosen algorithm.
- **Prerequisites:** Sampling, KV rollback, batched scheduling.
- **Concepts:** Draft model/proposal, verification, acceptance prefix, rejection sample, rollback, lookahead, target distribution.
- **Mathematics / systems / hardware:** Acceptance probability and expected tokens/target pass; draft/target state ownership; verification batch utilization and dual-model placement.
- **Implementation / Hermon / external:** Implement a tiny draft-target reference algorithm before optimization; classify Hermon's prompt-lookup specialization separately; consult original speculative decoding papers.
- **Diagrams / experiments:** Draft/verify/accept/rollback timeline; deterministic toy distributions with full accept/partial/reject.
- **Correctness / benchmark:** Distributional test or exact greedy special case, rollback positions/KV, RNG consumption, EOG/stop; tokens per target pass plus end-to-end latency.
- **Misconceptions / failures / deliverable / next:** Accepted drafts are not trusted without target verification, rejection handling is semantic, and high acceptance may still lose; deliver correct speculation core; Chapter 52 removes the draft model.

### Chapter 52 — Prompt-Lookup Decoding

- **Purpose / key question:** Propose tokens from repeated prompt n-grams without running a second model.
- **Prerequisites:** Speculative verification and tokenized prompt.
- **Concepts:** N-gram index, prompt continuation proposal, predecessor logits, longest accepted prefix, adaptive statistics.
- **Mathematics / systems / hardware:** Lookup cost, acceptance and verification-batch shape; per-sequence proposal/rolling-window state; CPU lookup versus target provider work.
- **Implementation / Hermon / external:** Add prompt-lookup mode; source-walk Hermon's CURRENT opt-in PLD and adaptive cutoff; consult primary prompt-lookup work.
- **Diagrams / experiments:** N-gram match to proposed run and target-logit predecessor mapping; RAG quote, code, and creative prompts.
- **Correctness / benchmark:** Greedy equivalence, mismatch token handling, rollback suffix, duplicate matches/tie policy, mixed sequences; acceptance, target calls, batch size, end-to-end latency.
- **Misconceptions / failures / deliverable / next:** Prompt lookup is not retrieval generation, proposals do not change target semantics, and one workload's acceptance is not general; deliver PLD extension; Chapter 53 derives loss cases.

### Chapter 53 — When Speculation Loses

- **Purpose / key question:** Determine when proposal, verification batch inflation, rollback, and contention cost more than saved target steps.
- **Prerequisites:** Speculative and prompt-lookup implementations, benchmark policy.
- **Concepts:** Break-even acceptance, draft cost, verification cost, batch fattening, concurrency interaction, adaptive disable/reenable, window lag.
- **Mathematics / systems / hardware:** Expected-cost inequality and sensitivity; per-sequence versus global policy ownership; provider shape-dependent economics.
- **Implementation / Hermon / external:** Implement transparent metrics and adaptive gate; analyze Hermon's measured CUDA loss/adaptive rule without universalizing; compare primary studies.
- **Diagrams / experiments:** Break-even surface and adaptive state timeline; sweep acceptance, lookahead, concurrency, model/provider.
- **Correctness / benchmark:** Gate changes performance only, never output; threshold/window boundaries and re-enable; end-to-end matched static-off/static-on/adaptive.
- **Misconceptions / failures / deliverable / next:** Tokens/target pass is not throughput, acceptance alone omits batch cost, and adaptation needs observability; deliver evidence-driven decode policy; Part X changes weight residency.

## Part X — Mixture-of-Experts Inference

### Chapter 54 — Why MoE Changes the Inference Engine

- **Purpose / key question:** Explain how conditional expert activation changes weight demand, scheduling, batching, and correctness.
- **Prerequisites:** Dense FFN, packed weights, scheduler and providers.
- **Concepts:** Router/gate, expert, top-k, shared experts, expert parallelism, active versus total parameters, routing distribution.
- **Mathematics / systems / hardware:** Gate logits/top-k and per-token active parameter bytes; routing state and batching by expert; scattered weight access/accelerator occupancy.
- **Implementation / Hermon / external:** Build a tiny scalar MoE layer/oracle; classify Hermon's expert pager as LIBRARY not request path; inspect primary MoE model definitions.
- **Diagrams / experiments:** Token-router-expert-combine flow and dense/MoE byte demand; route synthetic tokens under uniform/skewed gates.
- **Correctness / benchmark:** Top-k/ties, weight ordering, combine weights, capacity/unsupported semantics; measure routing and expert compute separately.
- **Misconceptions / failures / deliverable / next:** Fewer active parameters do not make storage disappear, top-k semantics vary, and pager bytes are not generation; deliver MoE oracle; Chapter 55 confronts VRAM capacity.

### Chapter 55 — Models Larger Than Available VRAM

- **Purpose / key question:** Quantify when weights, KV, scratch, and other processes cannot simultaneously reside in fast memory.
- **Prerequisites:** Model/KV byte math and MoE active demand.
- **Concepts:** Working set, oversubscription, offload, placement, transfer bottleneck, static partition, model/KV competition.
- **Mathematics / systems / hardware:** Tier capacity/bandwidth/latency and per-token expert demand; placement ownership; VRAM/RAM/NVMe/PCIe or unified-memory distinctions.
- **Implementation / Hermon / external:** Add a memory-demand planner/simulator; relate to Hermon's storage architecture and current “VRAM mirror, host home” target carefully; consult vendor memory docs.
- **Diagrams / experiments:** Full inference memory budget and tier hierarchy; vary model/cache/context on hypothetical and measured machines.
- **Correctness / benchmark:** Units/overflow, requested versus effective placement, OOM/fallback behavior; capacity/transfer model labeled estimate until measured.
- **Misconceptions / failures / deliverable / next:** Advertised memory is not wholly available, unified memory does not erase bandwidth, and active weights still must arrive; deliver placement model; Chapter 56 pages experts.

### Chapter 56 — Expert Storage and Paging

- **Purpose / key question:** Store experts so their offsets are verifiable and fetch only routed weights without full expansion.
- **Prerequisites:** GGUF/quantization, block pool, storage I/O.
- **Concepts:** Expert container, aligned record, per-layer stride, packer, cache slot, acquire/prefetch, stored-form matvec.
- **Mathematics / systems / hardware:** Arithmetic addressing and padding/stride cost; store/cache ownership; buffered/direct I/O, page alignment, NVMe transfer.
- **Implementation / Hermon / external:** Build Lab 11 scoped expert packer/pager; study Hermon's LIBRARY v2 per-layer-stride container and K6 lessons; compare primary MoE offload work.
- **Diagrams / experiments:** Container layout and miss-to-cache-fill path; pack mixed-size layer records and replay routes.
- **Correctness / benchmark:** Checksums/ranges, short reads, layer stride, duplicate IDs, quant block boundaries, oracle matvec; storage bytes/s and ceiling explicitly not inference.
- **Misconceptions / failures / deliverable / next:** Fixed record size may fail real mixed quantization, page cache can fake NVMe speed, and prefetch hint may spawn no work; deliver pager; Chapter 57 defines residency lifetime.

### Chapter 57 — Residency, Pinning, Eviction, and Queue Depth

- **Purpose / key question:** Keep expert bytes valid while used, evictable when safe, and honestly report effective I/O concurrency.
- **Prerequisites:** Expert pager, refcounts/leases, benchmark controls.
- **Concepts:** Residency reference, user pin, evictable state, requested/effective direct I/O, queue depth, batched misses, prefetch hint, RAII batch lease.
- **Mathematics / systems / hardware:** Cache-hit/token-byte ceiling and queue-depth/record-size latency model; lease state machine; NVMe queue and aligned destination constraints.
- **Implementation / Hermon / external:** Implement pin state and optional batched reads; reproduce Hermon's LIBRARY stale-residency/lifetime lessons and falsified 2–3× queue-depth hypothesis; consult OS I/O docs.
- **Diagrams / experiments:** Three-state residency machine and synchronous/batched I/O timelines; uniform/skewed routes, QD sweep, forced read error.
- **Correctness / benchmark:** Never evict pinned, unwind all-or-nothing, duplicate expert handling, requested/effective metrics, cache clear; hit/read bytes/bandwidth and no tok/s without model compute.
- **Misconceptions / failures / deliverable / next:** Unpinned differs from absent, large reads may already saturate a device, and direct-I/O flags can be ineffective; deliver robust pager; Chapter 58 unifies budgets.

### Chapter 58 — Toward Unified Inference Memory

- **Purpose / key question:** Decide whether KV, recurrent state, expert weights, and scratch can share one placement/eviction abstraction without hiding their semantic differences.
- **Prerequisites:** Paged KV, expert residency, provider placement.
- **Concepts:** Page kind, home versus mirror, pin/lease, dirty/immutable, tier manager, cost model, catalog, shared budget.
- **Mathematics / systems / hardware:** Benefit/cost/transfer/next-use policy and mixed-workload capacity; owner/mutability per page kind; VRAM/RAM/NVMe movement.
- **Implementation / Hermon / external:** Design a simulation and typed page interface, labeled TARGET; analyze Hermon's unified-tier proposal and database analogy limits; compare buffer pools/OS memory.
- **Diagrams / experiments:** Unified logical page kinds over tiers and mixed-budget eviction; compare static KV/expert partitions with dynamic policy in simulation.
- **Correctness / benchmark:** Dirty mutable state never discarded, immutable experts need no writeback, pin/failure/cancel rules; simulated hit/transfer/wait labeled model, not measured speed.
- **Misconceptions / failures / deliverable / next:** A shared allocator does not make page semantics identical, KV and experts have different recomputation/dirty costs, and a catalog is not execution; deliver FRONTIER design requirements; Part XI hardens all paths.

## Part XI — Correctness Engineering

### Chapter 59 — Fast Wrong Answers Are Still Wrong

- **Purpose / key question:** Explain why fluent output, determinism, and unit tests can coexist with serious inference defects.
- **Prerequisites:** Full optimized runtime and failure examples.
- **Concepts:** Numerical correctness, semantic correctness, plausible-text failure, integration gap, undefined support, benchmark invalidation.
- **Mathematics / systems / hardware:** Error propagation from logits to token choices; cross-layer contracts; backend-specific defects.
- **Implementation / Hermon / external:** Create a wrong-but-plausible fault catalog and gate benchmarks on checks; use Hermon's missed backend copy mutation as verified HISTORICAL case; consult numerical testing sources.
- **Diagrams / experiments:** Proof ladder and defect propagation chain; inject mask, position, stale-KV, and tensor-layout faults.
- **Correctness / benchmark:** Define what each subsequent level proves; no performance result survives failed equivalence.
- **Misconceptions / failures / deliverable / next:** Readable text is not correctness, deterministic wrong is still wrong, and backend unit tests cannot prove runtime routing; deliver test strategy; Chapter 60 builds the base oracle.

### Chapter 60 — Scalar Oracles

- **Purpose / key question:** Build references simple and independent enough to expose optimized-path bugs.
- **Prerequisites:** Tensor math and code-policy understanding.
- **Concepts:** Hand-computable fixture, scalar implementation, high-precision accumulator, independence, golden value, scope.
- **Mathematics / systems / hardware:** Exact/specified calculations and tolerance derivation; minimal state; intentionally hardware-agnostic scalar path.
- **Implementation / Hermon / external:** Consolidate Python/Rust scalar token, attention, quant, and MoE oracles; compare Hermon's scalar C/dense reference roles while noting independence risks.
- **Diagrams / experiments:** Oracle-to-implementations comparison; mutate each optimized path and verify oracle catches it.
- **Correctness / benchmark:** Boundary shapes, extreme values, expected failures, oracle cross-check with hand math; oracles are not performance baselines.
- **Misconceptions / failures / deliverable / next:** Slow does not guarantee correct, shared helper code can correlate bugs, and golden outputs need provenance; deliver oracle suite; Chapter 61 scales comparisons.

### Chapter 61 — Differential Testing

- **Purpose / key question:** Systematically compare implementations across shapes, dtypes, layouts, orders, and providers.
- **Prerequisites:** Independent scalar oracles and optimized paths.
- **Concepts:** Differential generator, tolerance, absolute/relative/ULP error, metamorphic property, seeded case, shrink/reproducer.
- **Mathematics / systems / hardware:** Error metrics tied to accumulation; isolate state/config; provider matrix.
- **Implementation / Hermon / external:** Build reusable differential harness; map Hermon's paged/dense, native/scalar, provider, and real-model tests; consult property-testing guidance.
- **Diagrams / experiments:** Test matrix and failure-shrinking flow; randomized boundary-focused cases.
- **Correctness / benchmark:** Coverage includes B/T thresholds, MHA/GQA, short/long, aligned/partial prefix, task orders; record failures, do not time as benchmark.
- **Misconceptions / failures / deliverable / next:** One tolerance does not fit all outputs, matching tokens can hide logit error, and shared RNG/order matters; deliver harness; Chapter 62 specifies determinism.

### Chapter 62 — Numerical Determinism

- **Purpose / key question:** Decide which outputs must be bitwise stable, tolerance-stable, or distributionally equivalent across order and hardware.
- **Prerequisites:** Floating reduction, differential tests, providers.
- **Concepts:** Reduction order, race, FMA, approximation, seed, bitwise/numerical/token determinism, reproducibility contract.
- **Mathematics / systems / hardware:** Non-associativity and fixed combine order; sampler/RNG ownership; ISA/GPU variation.
- **Implementation / Hermon / external:** Define per-operation determinism requirements and task-index combine; compare Hermon's host-scheduled deterministic plan; consult IEEE/vendor docs.
- **Diagrams / experiments:** Completion-order versus combine-order; vary thread/task/provider order repeatedly.
- **Correctness / benchmark:** Thread-count/order reproducibility, documented cross-provider tolerances, RNG sequence, gate thresholds; record determinism settings in benchmarks.
- **Misconceptions / failures / deliverable / next:** Determinism does not imply correctness, floating addition is not associative, and GPU race-free can still differ by plan; deliver contract/tests; Chapter 63 targets concurrency bugs.

### Chapter 63 — Concurrency Bugs That Still Produce Plausible Text

- **Purpose / key question:** Find state races, cross-request contamination, stale metadata, and ordering errors that avoid crashes.
- **Prerequisites:** Request/worker ownership, block lifecycle, determinism.
- **Concepts:** Sole mutator, race, logical-slot reuse, stale prefix, lost cancellation, batch-wide failure, linearizability.
- **Mathematics / systems / hardware:** Interleaving/state invariants; channels/locks/atomics; asynchronous device completion.
- **Implementation / Hermon / external:** Add schedule perturbation and multi-request stress; examine Hermon's dedicated worker and lifecycle invariants; use concurrency testing literature.
- **Diagrams / experiments:** Two-request interleaving and stale-slot timeline; inject yields, reorder completions, cancel during shared batch.
- **Correctness / benchmark:** Isolated-output differential under concurrency, no cross-request state, worker survives failure, terminal exactly once; performance secondary.
- **Misconceptions / failures / deliverable / next:** Race-free fields do not prove protocol correctness, plausible tokens can be contaminated, and global locks can hide rather than solve ownership; deliver concurrency suite; Chapter 64 audits lifetimes.

### Chapter 64 — Ownership and Lifetime Failures

- **Purpose / key question:** Prove every weight, context, page, mirror, expert lease, stream, and sampler has one coherent lifetime.
- **Prerequisites:** Follow-the-owner method and all memory systems.
- **Concepts:** Owner, borrower/reference, lease, pin, RAII, cycle, double release, leak, use-after-free, cancellation unwind.
- **Mathematics / systems / hardware:** Reference-state invariants; destruction/lock order; host/device event lifetime and external handles.
- **Implementation / Hermon / external:** Create lifetime tables and fault-injection guards; use Hermon's partial-tail backend and expert-residency bugs as case studies; compare resource-management patterns.
- **Diagrams / experiments:** Ownership graphs for KV/expert/device stream and success/error/cancel unwind; inject every acquisition failure.
- **Correctness / benchmark:** Zero leaked blocks/pins/tasks, no reuse while referenced, duplicate acquisition, idempotent cleanup, sanitizer; benchmark only after guard overhead understood.
- **Misconceptions / failures / deliverable / next:** Refcount zero is a result of complete ownership accounting, RAII cannot cover foreign async work without a completion lease, and cache ownership differs from request ownership; deliver lifetime proof matrix; Chapter 65 attacks boundaries.

### Chapter 65 — Sanitizers, Fuzzing, and Boundary Testing

- **Purpose / key question:** Exercise memory, parser, ABI, allocator, and numerical boundaries beyond hand-selected cases.
- **Prerequisites:** Native code, parser, differential harness.
- **Concepts:** ASan/UBSan/MSan/TSan scope, fuzzer, corpus, mutation, property, timeout/oom limit, boundary matrix.
- **Mathematics / systems / hardware:** Shape-product/offset overflow space; isolate processes and reproducible seeds; sanitizer/provider compatibility.
- **Implementation / Hermon / external:** Add fuzz targets for GGUF/ABI/block tables and sanitizer scripts for C; compare Hermon's kernel sanitize and parser tests; use tool official docs.
- **Diagrams / experiments:** Input-to-parser/kernel containment and corpus lifecycle; seed with valid minimal cases then mutate.
- **Correctness / benchmark:** Crashes, leaks, undefined behavior, hangs, invariant errors; fuzz executions are coverage evidence, not performance results.
- **Misconceptions / failures / deliverable / next:** No sanitizer proves semantic logits, fuzzing invalid bytes does not replace valid-model equivalence, and TSan has false/unsupported regions; deliver hardening suite; Chapter 66 proves real models.

### Chapter 66 — Real-Model Equivalence

- **Purpose / key question:** Demonstrate actual supported model semantics across prompt positions and decode, not merely component arithmetic.
- **Prerequisites:** ENGINE-9, oracles/differentials, licensed external fixture.
- **Concepts:** Trusted baseline, logit checkpoint, greedy sequence, prompt corpus, model artifact hash, coverage matrix, release gate.
- **Mathematics / systems / hardware:** Layer/final logit tolerances and error accumulation; reset/cache isolation; same provider or disclosed cross-provider comparison.
- **Implementation / Hermon / external:** Build Lab 13 with fixed model revision and baseline; mirror Hermon's ignored real-model differential pattern while closing fixture automation for mini-engine; inspect baseline source.
- **Diagrams / experiments:** Component-to-model proof ladder and corpus matrix; prompts crossing blocks/context shapes and repeated prefixes.
- **Correctness / benchmark:** Tokenizer/template, logits, greedy outputs, cached/uncached, batched/isolated, providers, repeated runs; only equivalent configurations advance performance gates.
- **Misconceptions / failures / deliverable / next:** One prompt is not support, token equality may hide margin changes, and ignored fixture tests are not ordinary-CI proof; deliver real-model gate; Part XII productionizes only passed paths.

## Part XII — Production Inference Engineering

### Chapter 67 — Protocols and the AI Gateway

- **Purpose / key question:** Normalize multiple wire protocols into one truthful engine request without leaking protocol quirks into execution policy.
- **Prerequisites:** ENGINE-9 request lifecycle and HTTP/JSON basics.
- **Concepts:** Schema, compatibility surface, sync/stream response, normalized request, usage, error mapping, protocol version.
- **Mathematics / systems / hardware:** Token/usage accounting; edge/runtime boundary and validation; hardware intentionally hidden behind normalized capability.
- **Implementation / Hermon / external:** Add one native API then scoped OpenAI-compatible adapter; inspect Hermon's CURRENT OpenAI/Ollama/Anthropic normalization; use official protocol docs.
- **Diagrams / experiments:** Many protocols to one request object and response/error mapping; replay equivalent requests across adapters.
- **Correctness / benchmark:** Schema validation, stop/sampling mapping, stream/sync semantic equivalence, error codes, usage; protocol overhead measured separately from model.
- **Misconceptions / failures / deliverable / next:** Wire compatibility does not imply feature equivalence, unknown fields need policy, and HTTP success is not model success; deliver gateway; Chapter 68 chooses models/providers.

### Chapter 68 — Model Resolution and Routing

- **Purpose / key question:** Resolve a user model identifier to an exact local artifact or provider with explicit policy and provenance.
- **Prerequisites:** Model manifest, gateway, provider interfaces.
- **Concepts:** Canonical path, alias, registry, provider prefix, revision/hash, local/cloud route, fallback policy, capability.
- **Mathematics / systems / hardware:** Routing-cost inputs without fake optimizer; cache key and model-instance ownership; local device capability versus remote service.
- **Implementation / Hermon / external:** Build deterministic resolver and explicit router; map Hermon's CURRENT literal/home/cloud/Ollama resolution with source verification; consult provider/model-hub APIs.
- **Diagrams / experiments:** Model-name decision tree and artifact identity flow; ambiguous aliases, missing artifact, offline provider.
- **Correctness / benchmark:** No silent provider/model substitution, canonical cache identity, capability errors, secret boundary; measure resolution separately, no inference claim.
- **Misconceptions / failures / deliverable / next:** Same display name may identify different weights, fallback can violate privacy/cost intent, and provider support is not model support; deliver resolver; Chapter 69 treats stream behavior.

### Chapter 69 — Streaming as a Systems Contract

- **Purpose / key question:** Define ordering, UTF-8, backpressure, cancellation, terminal events, and usage for a reliable token stream.
- **Prerequisites:** Gateway, request state, bounded channels.
- **Concepts:** Piece/delta, framing, heartbeat, bounded receiver, slow consumer, half-close, terminal usage/error, idempotent finish.
- **Mathematics / systems / hardware:** Buffer/queue bounds and latency; producer-consumer ownership; provider completion can precede/lag network writes.
- **Implementation / Hermon / external:** Implement SSE/NDJSON-safe stream and Lab 12; inspect Hermon's CURRENT Piece/Done/error contract and UTF-8 buffer; use protocol/networking specs.
- **Diagrams / experiments:** Runtime-channel-wire pipeline and slow-client backpressure; disconnect at prefill/decode/final event.
- **Correctness / benchmark:** Valid framing/UTF-8, ordered pieces, exactly one terminal outcome, no Done after error, cleanup; time first byte/token and blocked-producer behavior.
- **Misconceptions / failures / deliverable / next:** Token boundaries need not be text boundaries, socket close is not complete cancellation, and buffering changes latency/memory; deliver stream contract; Chapter 70 observes it.

### Chapter 70 — Metrics and Observability

- **Purpose / key question:** Expose enough state to explain latency, cache behavior, saturation, failures, and provider selection without perturbing hot paths.
- **Prerequisites:** Runtime phases, scheduler/cache/provider concepts.
- **Concepts:** Counter, gauge, histogram, trace/span, cardinality, queue/execution split, cache tokens saved, active/warm/pinned state.
- **Mathematics / systems / hardware:** Rate/percentile/hit-rate definitions; lock-free snapshot/aggregation ownership; CPU/GPU/I/O signals.
- **Implementation / Hermon / external:** Add structured metrics/traces; map Hermon's CURRENT metrics/Prometheus/TUI claims to source; consult OpenTelemetry/Prometheus specs.
- **Diagrams / experiments:** Causal observability map from symptom to subsystem; generate slow client, cache churn, provider fallback, and verify signals.
- **Correctness / benchmark:** Monotonic counters, label bounds, terminal accounting, snapshot consistency, low overhead; benchmark instrumentation on/off.
- **Misconceptions / failures / deliverable / next:** Metrics do not prove causality, high-cardinality model/path labels can harm systems, and averages hide tails; deliver observability contract; Chapter 71 contains faults.

### Chapter 71 — Failure Containment

- **Purpose / key question:** Prevent malformed input, one request, one model, one shared batch, or one provider failure from corrupting unrelated work.
- **Prerequisites:** State machine, gateway, ownership, observability.
- **Concepts:** Failure domain, batch-global error, worker recovery, circuit breaker, timeout, panic/process isolation, degraded fallback, rollback.
- **Mathematics / systems / hardware:** Retry/timeout budget; resource unwind and state invalidation; device reset/async error boundaries.
- **Implementation / Hermon / external:** Add injection points and recovery rules; inspect Hermon's CURRENT failed-batch invalidation/worker survival and explicit preview errors; use resilient service patterns.
- **Diagrams / experiments:** Failure domains and recovery transitions; inject parser, model-load, decode, device, stream, and disk errors.
- **Correctness / benchmark:** Unaffected requests remain isolated where contract permits, failed cache metadata invalidated, capacity recovered, errors observable; measure recovery and blast radius.
- **Misconceptions / failures / deliverable / next:** Retrying generation can duplicate output or RNG effects, CPU fallback needs clean workspace/state, and catching panic is not memory safety; deliver containment suite; Chapter 72 treats hostile inputs.

### Chapter 72 — Security and Untrusted Models

- **Purpose / key question:** Treat model files, templates, prompts, API inputs, local paths, native code, and secrets as distinct trust boundaries.
- **Prerequisites:** Parser fuzzing, gateway, resolver, native ABI.
- **Concepts:** Supply chain, checksum/signature, path traversal, parser budget, denial of service, secret storage, tenancy/cache leakage, native sandbox boundary.
- **Mathematics / systems / hardware:** Size/compute admission limits; least-privilege ownership; GPU/driver/native attack surface.
- **Implementation / Hermon / external:** Harden resolver/parser/API limits and produce threat model; inspect Hermon's supply-chain/security docs and safe/unsafe crate boundaries; consult format/OS security guidance.
- **Diagrams / experiments:** Trust-boundary/data-flow diagram and attack-to-control table; fuzz huge metadata, malicious paths, prompt/cache tenant collisions.
- **Correctness / benchmark:** Reject over-budget/invalid artifacts safely, secret redaction, tenant cache partitioning, dependency audit; record security-control overhead where relevant.
- **Misconceptions / failures / deliverable / next:** Local does not mean trusted, Rust cannot sanitize C/drivers or hostile resource demand, and model metadata is executable configuration; deliver threat model/controls; Chapter 73 measures truthfully.

### Chapter 73 — Benchmarking Without Lying to Yourself

- **Purpose / key question:** Produce reproducible performance evidence that separates correctness, workload, state, hardware, and control.
- **Prerequisites:** ENGINE-10 candidate, profiling, observability, benchmark policy.
- **Concepts:** Workload manifest, warm/cold cache, concurrency/arrival, TTFT/ITL/throughput/tail, control, repetitions/statistic, contamination, estimate.
- **Mathematics / systems / hardware:** Distributions/confidence and latency-throughput tradeoffs; reset/isolation between runs; clocks, thermal state, driver/storage caches.
- **Implementation / Hermon / external:** Build Lab 14 harness and artifact schema; critique/reproduce scoped Hermon methodology rather than repeat headlines; compare benchmark standards.
- **Diagrams / experiments:** Benchmark boundary/state-reset flow and metric decision table; deliberately contaminate prefix/model/OS caches and detect it.
- **Correctness / benchmark:** Equivalence gate first, full manifest/raw output, matched stop/model/quantization, repeated order-randomized controls; this chapter's deliverable is the benchmark suite.
- **Misconceptions / failures / deliverable / next:** Independent ratios do not multiply into a measurement, storage ceiling is not tok/s, and favorable workload is not universal; deliver ENGINE-10 evidence framework; Part XIII audits Hermon with it.

## Part XIII — Inside Hermon

### Chapter 74 — Hermon's System Architecture

- **Purpose / key question:** Establish the current system boundary and exact default/compatibility/preview/library/target topology at a recorded commit.
- **Prerequisites:** Parts I–XII and source policy.
- **Concepts:** API/core/runtime/engine/llama bridge/GGUF/paged-KV/kernels/bench crate roles and status categories.
- **Mathematics / systems / hardware:** No new math; model-runtime ownership and concurrency; production providers versus Hermon-owned native providers.
- **Implementation / Hermon / external:** Refresh `research/hermon` by fetching/inspecting source and tests; this chapter is the canonical case study, not a product overview; compare only where primary evidence supports it.
- **Diagrams / experiments:** Source-verified system topology and three runtime modes; build/test normal and preview feature configurations where available.
- **Correctness / benchmark:** Check doc claims against actual dispatch gates/call graph; no reused benchmark number without reproducer/metadata.
- **Misconceptions / failures / deliverable / next:** Crate existence does not mean request-path integration, root README can lag source, and pinned llama.cpp remains production mechanism; deliver dated architecture map; Chapter 75 explains the strategy.

### Chapter 75 — Why Hermon Did Not Rewrite Everything

- **Purpose / key question:** Analyze how wrapping a proven engine enables product/runtime progress while native components mature behind gates.
- **Prerequisites:** Hermon topology, boundary/substitution concepts.
- **Concepts:** Pinned upstream, narrow safe wrapper, policy/mechanism split, compatibility, rollback, rewrite risk, learning loop.
- **Mathematics / systems / hardware:** Migration risk/cost rather than new math; ownership boundary; inherited provider coverage versus native capability.
- **Implementation / Hermon / external:** Trace `hermon-llamacpp` feature/build/shim and `hermon-engine` facade; verify unsafe boundaries; compare incremental replacement patterns.
- **Diagrams / experiments:** Proven mechanism inside owned policy shell and rewrite/substitution risk map; inspect dependency/build modes.
- **Correctness / benchmark:** Wrapper equivalence/real-model tests and explicit stub behavior; do not claim native superiority from architecture.
- **Misconceptions / failures / deliverable / next:** Reuse is not lack of ambition, a wrapper still owns contracts, and vendoring creates update/security obligations; deliver strategy analysis; Chapter 76 formalizes gates.

### Chapter 76 — The Substitution Ladder

- **Purpose / key question:** Show how reference structures become a default native path through evidence rather than a flag flip.
- **Prerequisites:** Hermon wrapper strategy and correctness/benchmark gates.
- **Concepts:** Reference data structure, bulk ABI, scalar oracle, specialization, shadow execution, explicit preview, default with rollback.
- **Mathematics / systems / hardware:** Acceptance tolerances and performance control; state migration/ownership completeness; per-provider conformance.
- **Implementation / Hermon / external:** Re-verify each stage in `ENGINE_STRATEGY.md` against current code/status; identify gaps without converting plans into facts; compare safe migration practices.
- **Diagrams / experiments:** Seven-rung ladder with evidence gates and rollback; place current paged/native/MoE components on it.
- **Correctness / benchmark:** Each rung names tests and measurement required; highlight model fixture/1,000-prompt/open integration gates.
- **Misconceptions / failures / deliverable / next:** Unit equivalence is not integration, preview is not release, and default flip needs operational rollback; deliver live gate matrix; Chapter 77 navigates source owners.

### Chapter 77 — Anatomy of the Hermon Source Tree

- **Purpose / key question:** Teach a contributor where contracts live and which changes are local versus cross-layer.
- **Prerequisites:** Architecture and substitution ladder.
- **Concepts:** Workspace crate responsibility, feature graph, unsafe boundary, vendor subtree, docs/tests/bench locations, canonical versus historical docs.
- **Mathematics / systems / hardware:** No new math; module ownership/dependency constraints; build features for CPU/Metal/CUDA/Vulkan/ROCm/SYCL versus native kernel implementations.
- **Implementation / Hermon / external:** Produce a current annotated tree from Cargo/source; verify manifests and call sites; no external comparison required beyond Cargo conventions.
- **Diagrams / experiments:** Crate dependency/ownership graph and “change X, inspect Y” map; follow a protocol field, GGUF key, radix change, and ISA kernel.
- **Correctness / benchmark:** Confirm unsafe lint boundaries, feature compilation, test fixture gates; no performance claim.
- **Misconceptions / failures / deliverable / next:** File names are not current status, comments can be stale, and backend build features do not mean native-kernel parity; deliver navigation guide; Chapter 78 follows a request.

### Chapter 78 — Follow One Request Through Hermon

- **Purpose / key question:** Trace validation, model resolution, dispatch, admission, batched execution, stream backpressure, and terminal accounting on the default path.
- **Prerequisites:** Source tree and production serving concepts.
- **Concepts:** Protocol handler, normalized message/options, canonical model path, Dispatcher cache, `SequenceRequest`, worker slot, `StreamItem`.
- **Mathematics / systems / hardware:** Queue/execution metrics; API task versus sole-mutator worker ownership; llama.cpp provider selection beneath facade.
- **Implementation / Hermon / external:** Trace concrete functions from one protocol handler through `engine_route`, Dispatcher, BatchedRuntime, and response framing at current commit; compare preview as side branch only.
- **Diagrams / experiments:** Source-linked sequence diagram and resource ownership by phase; run a local fixture if available with logs/metrics.
- **Correctness / benchmark:** Piece/Done/error contract, bounded channels, batch failure invalidation, worker recovery; timings only if full setup recorded.
- **Misconceptions / failures / deliverable / next:** Global map lock does not cover execution, API async task does not mutate context, and default is not paged; deliver request walkthrough; Chapter 79 follows numerical state.

### Chapter 79 — Follow One Token Through Hermon

- **Purpose / key question:** Trace tokenization, chat template, prefill/decode batching, logits, sampling, UTF-8 buffering, KV mutation, and stream output.
- **Prerequisites:** Request path and full model semantics.
- **Concepts:** Token position, logical sequence ID, prompt chunk, pending token, logit index, sampler state, KV rollback, token piece.
- **Mathematics / systems / hardware:** Position/logit predecessor mapping for normal and PLD paths; worker/context ownership; pinned llama.cpp execution and gated alternative.
- **Implementation / Hermon / external:** Trace current batched source and optional PLD; separately trace paged preview only with PREVIEW labels; no unsupported family claims.
- **Diagrams / experiments:** Token lifecycle with positions and PLD verification branch; fixed greedy prompt trace if model fixture exists.
- **Correctness / benchmark:** Token bytes/UTF-8, logit-position selection, accepted/rejected draft KV, stop/EOG, isolated baseline; no performance beyond reproducible run.
- **Misconceptions / failures / deliverable / next:** One output token may span/merge bytes, draft position logic is easy to reverse, and Hermon-owned paged forward is not default; deliver token walkthrough; Part XIV derives future systems from verified invariants.

## Part XIV — Beyond Today's Engine

Every chapter in this part must explicitly label **TODAY**, **NEAR TERM**,
**FRONTIER**, and **RESEARCH QUESTION** where applicable.

### Chapter 80 — Hybrid Transformer Architectures

- **Purpose / key question:** Adapt the inference mental model when layers mix full attention, sliding/local attention, linear/recurrent mechanisms, and MoE.
- **Prerequisites:** Transformer, KV, MoE, model-support discipline.
- **Concepts:** Hybrid layer schedule, local window, stateful mixer, sinks, per-layer state kind, heterogeneous kernel plan.
- **Mathematics / systems / hardware:** Layer-specific visible history/state update; typed ownership; provider capability per operator.
- **Implementation / Hermon / external:** Build a metadata-driven toy hybrid plan, not a family claim; treat Hermon's STATE block reservation/hybrid design as TARGET unless current source proves otherwise; inspect primary model papers.
- **Diagrams / experiments:** Mixed layer stack and state-kind table; simulate memory/work under layer mixtures.
- **Correctness / benchmark:** Per-layer semantics, position/mask, state reset, unsupported hard error; estimates labeled, real measurements only on implemented model.
- **Misconceptions / failures / deliverable / next:** “Transformer” no longer implies uniform KV at every layer, similar names hide semantics, and generic metadata needs adapters; deliver requirements; Chapter 81 focuses STATE.

### Chapter 81 — Recurrent State and STATE Pages

- **Purpose / key question:** Manage mutable fixed-size recurrent/linear-attention state alongside append-oriented KV.
- **Prerequisites:** Hybrid models and unified-memory vocabulary.
- **Concepts:** Recurrent state, state page, update-in-place, checkpoint/fork, conversation branch, dirty state, rollback.
- **Mathematics / systems / hardware:** State transition `s_t=f(s_{t-1},x_t)` and byte scaling independent of full context; exclusive mutation/COW snapshots; provider residency.
- **Implementation / Hermon / external:** Add toy recurrent operator and typed STATE page prototype; classify Hermon's native block kind as LIBRARY design, not integrated model support; use primary recurrent-model specs.
- **Diagrams / experiments:** KV append versus STATE mutation and branch/COW; fork two continuations and reproduce alias corruption without COW.
- **Correctness / benchmark:** Sequential oracle, reset/fork/rollback, cancel/error, provider mirror coherence; state bytes/update latency separately.
- **Misconceptions / failures / deliverable / next:** Fixed-size state is not immutable, KV page rules cannot be copied blindly, and checkpointing has semantic cost; deliver STATE contract; Chapter 82 combines budgets.

### Chapter 82 — Unified Memory Economics

- **Purpose / key question:** Derive placement decisions across weights, KV, STATE, experts, scratch, and mirrors from value/cost rather than one cache policy.
- **Prerequisites:** Chapter 58 design and STATE pages.
- **Concepts:** Page catalog, recomputation cost, dirty/writeback cost, reuse horizon, transfer cost, shared budget, admission reserve.
- **Mathematics / systems / hardware:** Candidate cost function and sensitivity; global manager versus subsystem leases; tier bandwidth/capacity/topology.
- **Implementation / Hermon / external:** Extend simulator with typed economics; compare Hermon's TARGET storage design and explicitly state analogy limits; draw on database/OS caching research.
- **Diagrams / experiments:** Typed page catalog and eviction decision matrix; mixed chat/MoE/recurrent workloads under static/dynamic budgets.
- **Correctness / benchmark:** Never lose dirty state, reserve forward progress, reproducible simulation inputs; results are simulation until implemented/measured.
- **Misconceptions / failures / deliverable / next:** One “temperature” metric cannot capture all page types, global optimization can increase tail latency, and future reuse is uncertain; deliver cost-model requirements; Chapter 83 separates phases physically.

### Chapter 83 — Prefill/Decode Disaggregation

- **Purpose / key question:** Decide when distinct workers/devices for compute-heavy prefill and memory-heavy decode outweigh KV transfer and coordination.
- **Prerequisites:** Phase economics, distributed ownership, network basics.
- **Concepts:** Phase worker, KV handoff, transfer protocol, admission/routing, failure domain, recompute versus move, locality.
- **Mathematics / systems / hardware:** Prefill compute, KV transfer bytes/time, decode savings, break-even; ownership transfer/lease; network/RDMA/device topology.
- **Implementation / Hermon / external:** Build a trace/simulator and handoff schema, labeled FRONTIER for mini-engine/Hermon unless verified; study primary disaggregated serving systems.
- **Diagrams / experiments:** Prefill worker -> KV transfer -> decode worker with failure edges; sweep prompt/context/network bandwidth.
- **Correctness / benchmark:** Exact model/revision/positions/layout, complete/atomic handoff, retry/recompute semantics; simulation estimates versus real two-node measurements clearly separated.
- **Misconceptions / failures / deliverable / next:** Compute separation does not eliminate state movement, retry can duplicate stream output, and topology dominates; deliver break-even/protocol requirements; Chapter 84 partitions one execution.

### Chapter 84 — Multi-GPU Execution

- **Purpose / key question:** Partition weights, operators, or experts across devices while accounting for collectives, memory, and failure.
- **Prerequisites:** Provider contract, model layers, networking/collectives basics.
- **Concepts:** Tensor parallel, pipeline parallel, expert parallel, replica, collective, shard, topology, bubble.
- **Mathematics / systems / hardware:** Partitioned matmul/attention and communication volumes; shard ownership; PCIe/NVLink/device topology.
- **Implementation / Hermon / external:** Create small simulator/reference split, no unsupported production claim; classify Hermon's multi-GPU roadmap as TARGET unless source changes; inspect Megatron/vLLM/vendor primary docs.
- **Diagrams / experiments:** TP all-reduce, pipeline stages, expert routing across GPUs; simulate latency by topology/batch.
- **Correctness / benchmark:** Sharded versus single-device differential, collective order, partial failure, model/config identity; real measurements require exact topology.
- **Misconceptions / failures / deliverable / next:** Aggregate VRAM does not behave as one uniform pool, communication can dominate decode, and parallel modes compose nontrivially; deliver plan matrix; Chapter 85 crosses hosts.

### Chapter 85 — Multi-Node Inference

- **Purpose / key question:** Extend execution and serving across machines with explicit partitioning, replication, state transfer, and failure domains.
- **Prerequisites:** Multi-GPU, disaggregation, distributed systems basics.
- **Concepts:** Data/model parallel serving, coordinator, lease, shard/replica, consistency, backpressure, network partition, retry/idempotency.
- **Mathematics / systems / hardware:** Network communication/queue models and availability tradeoffs; request/KV ownership across nodes; NIC/RDMA/accelerator topology.
- **Implementation / Hermon / external:** Build a failure-aware simulator/protocol sketch labeled FRONTIER; do not infer Hermon support from roadmap; inspect primary distributed inference systems.
- **Diagrams / experiments:** Multi-node ownership/failure map and state transfer; inject loss, delay, node failure, duplicate terminal response.
- **Correctness / benchmark:** Shard identity, exactly-once-visible stream semantics, lease expiry, recovery/recompute; simulation versus cluster results explicit.
- **Misconceptions / failures / deliverable / next:** Networking adds partial failure, replication does not make mutable KV interchangeable, and retries are observable; deliver distributed requirements; Chapter 86 develops an analogy.

### Chapter 86 — The Inference Engine as a Database

- **Purpose / key question:** Use database concepts to clarify planning, buffer management, indexes, operators, and distributed execution—and identify where the analogy fails.
- **Prerequisites:** Full engine architecture and database vocabulary introduced locally.
- **Concepts:** Query/request, logical/physical plan, buffer pool/page, prefix index, operator/kernel, cost optimizer/scheduler, catalog, distributed query/inference.
- **Mathematics / systems / hardware:** Cost/cardinality uncertainty versus token/shape demand; ownership/transactions differences; storage/compute tiers.
- **Implementation / Hermon / external:** Build an exact comparison table and one planner simulation; relate to Hermon's storage architecture as INFERENCE/TARGET, not product claim; use database primary texts.
- **Diagrams / experiments:** Side-by-side pipelines and buffer/page mapping; apply a buffer-pool policy then identify failure on mutable KV/state.
- **Correctness / benchmark:** Analogies checked against counterexamples; simulation inputs/results labeled.
- **Misconceptions / failures / deliverable / next:** Requests are not declarative SQL, output generation is stateful/iterative, and transaction semantics do not transfer wholesale; deliver bounded analogy; Chapter 87 derives requirements, not fake syntax.

### Chapter 87 — Toward a Universal Inference Execution Protocol

- **Purpose / key question:** Derive what a portable plan/provider/state interface would need before proposing any syntax or standard.
- **Prerequisites:** Provider ABI, heterogeneous models, distributed ownership, database analogy.
- **Concepts:** Logical operator, physical plan, shape/dtype/layout capability, state handle, placement, async event, cancellation, versioning, conformance.
- **Mathematics / systems / hardware:** Cost/shape descriptors and semantic invariants; transferable versus local ownership; CPU/GPU/remote capability negotiation.
- **Implementation / Hermon / external:** Draft requirements and minimal typed IR experiment, explicitly FRONTIER; compare MLIR/XLA/vendor/runtime interfaces using primary specs; do not invent “SQL for AI.”
- **Diagrams / experiments:** Model semantics -> planner -> provider graph and state-handle lifecycle; lower one attention step to two providers.
- **Correctness / benchmark:** Conformance oracle, unsupported capability, version mismatch, cancellation/failure, deterministic plan metadata; no performance claim from interface alone.
- **Misconceptions / failures / deliverable / next:** Lowest-common-denominator APIs lose optimization, opaque state impedes safe transfer, and syntax before requirements is premature; deliver requirements/IR prototype; Chapter 88 synthesizes directions.

### Chapter 88 — What Comes After Today's Transformer Runtime?

- **Purpose / key question:** Synthesize durable engine invariants for new architectures without predicting specific winners.
- **Prerequisites:** Hybrid/state/memory/distributed/protocol chapters.
- **Concepts:** Specialized AI operating system, admission, logical inference memory, placement, planning, providers, autonomous adaptation, open research questions.
- **Mathematics / systems / hardware:** Resource/control-loop formulation; global ownership and failure boundaries; heterogeneous/local/remote hardware.
- **Implementation / Hermon / external:** Produce a research agenda and small simulations, clearly TODAY/NEAR TERM/FRONTIER/QUESTION; relate Hermon only through verified current lessons.
- **Diagrams / experiments:** Inference-as-OS stack and research dependency tree; test one policy hypothesis in simulation rather than claim architecture.
- **Correctness / benchmark:** State what evidence would falsify each proposal and which semantics must remain invariant; no speculative benchmark headlines.
- **Misconceptions / failures / deliverable / next:** “Operating system” is an analogy, autonomy needs safe bounds/observability, and future models may invalidate today's page types; deliver frontier agenda; Part XV integrates proven curriculum.

## Part XV — Graduation Project

### Chapter 89 — Designing the Final Mini Engine

- **Purpose / key question:** Freeze a coherent scope, supported model contract, architecture, ownership rules, and acceptance gates for the graduation runtime.
- **Prerequisites:** Parts I–XIV and completed core labs.
- **Concepts:** Architecture decision record, supported matrix, subsystem boundary, invariant, non-goal, release gate, rollback.
- **Mathematics / systems / hardware:** Capacity/performance budgets and tensor semantics; complete owner/resource table; target CPU and optional providers.
- **Implementation / Hermon / external:** Produce final design and component map; identify but do not yet modify one Hermon substitution candidate; compare verified reference requirements.
- **Diagrams / experiments:** End-to-end architecture and owner/lifetime matrix; design review with failure walkthroughs.
- **Correctness / benchmark:** Acceptance matrix includes model equivalence, concurrency, security, performance, and production behavior; no code-complete claim yet.
- **Misconceptions / failures / deliverable / next:** Graduation is not feature maximalism, unsupported behavior must fail explicitly, and architecture diagrams need executable contracts; deliver reviewed design; Chapter 90 integrates.

### Chapter 90 — End-to-End Implementation

- **Purpose / key question:** Assemble tokenizer, model loading, packed math, KV paging, batching, providers, sampling, and streaming without bypassing ownership surfaces.
- **Prerequisites:** Approved Chapter 89 design and component tests.
- **Concepts:** Composition root, configuration, dependency injection, lifecycle, capability, feature gate, migration path.
- **Mathematics / systems / hardware:** Validate end-to-end shapes/capacity; create/destroy order; provider plan/placement.
- **Implementation / Hermon / external:** Finish the scoped mini-engine executable and server; keep Hermon separate except optional comparison harness; pin all external references.
- **Diagrams / experiments:** Concrete module/call graph and one request/token/byte/owner trace; smoke workloads across supported modes.
- **Correctness / benchmark:** Build/format/unit/integration, explicit unsupported errors, leak/cancel/failure tests; collect baseline only after smoke equivalence.
- **Misconceptions / failures / deliverable / next:** Components can pass while integration routing is wrong, shortcuts around backend mutations recreate stale state, and feature flags need matrices; deliver integrated candidate; Chapter 91 gates correctness.

### Chapter 91 — Correctness Gate

- **Purpose / key question:** Decide whether the integrated engine is semantically trustworthy enough to measure and expose.
- **Prerequisites:** End-to-end candidate and Part XI suite.
- **Concepts:** Release corpus, conformance matrix, blocker, waiver, regression artifact, reproducibility.
- **Mathematics / systems / hardware:** Tolerance and token-margin analysis; clean-state isolation; provider/ISA coverage.
- **Implementation / Hermon / external:** Run scalar/component/concurrency/fuzz/sanitizer/real-model gates; compare with pinned baseline and record artifacts; Hermon replacement remains untouched.
- **Diagrams / experiments:** Gate funnel and failure triage path; run boundary prompts/configs and injected errors.
- **Correctness / benchmark:** This entire chapter is the correctness record; any blocker prevents Chapter 92 performance conclusions.
- **Misconceptions / failures / deliverable / next:** Passing one token sequence is not a corpus, waivers must narrow support, and test skips must be counted; deliver signed gate report; Chapter 92 benchmarks passed configurations only.

### Chapter 92 — Performance Gate

- **Purpose / key question:** Determine where the correct engine meets, exceeds, or misses explicit workload/hardware targets.
- **Prerequisites:** Passed correctness gate and Lab 14 harness.
- **Concepts:** Target workload, baseline, regression budget, profile, crossover, bottleneck, negative result.
- **Mathematics / systems / hardware:** Latency/throughput/tail/memory analysis with confidence; reset/ownership of cache state; provider-specific profiles.
- **Implementation / Hermon / external:** Run controlled CPU/provider, prompt/output, concurrency, and cache experiments; compare to Hermon only with matched semantics/setup.
- **Diagrams / experiments:** Performance envelope and bottleneck timeline; required warm/cold, serial/batched, dense/paged, provider sweeps.
- **Correctness / benchmark:** Recheck outputs during benchmarks, publish manifests/raw results, estimates separated, unfavorable results retained.
- **Misconceptions / failures / deliverable / next:** One headline ratio cannot define the engine, faster wrong runs are discarded, and cache contamination invalidates comparison; deliver performance gate report; Chapter 93 tests operations.

### Chapter 93 — Production Gate

- **Purpose / key question:** Prove the engine behaves safely under load, cancellation, failures, hostile inputs, observability, and deployment lifecycle.
- **Prerequisites:** Correct and characterized engine.
- **Concepts:** Soak, saturation, SLO, graceful shutdown, readiness, reload, resource cap, security control, rollback.
- **Mathematics / systems / hardware:** Queue/capacity/SLO budgets; drain and ownership release; provider/device failure/recovery.
- **Implementation / Hermon / external:** Run load/soak/fault/security/deploy checks and document operator runbook; compare production surfaces only where matched.
- **Diagrams / experiments:** Deploy/readiness/drain lifecycle and fault-injection matrix; slow clients, OOM pressure, corrupt models, provider failure, restart/cache state.
- **Correctness / benchmark:** No leaks/deadlocks/starvation, bounded memory/queues, truthful metrics, graceful terminal behavior, recoverable rollback; publish operational evidence.
- **Misconceptions / failures / deliverable / next:** A benchmark server is not production, health endpoint alone does not prove readiness, and graceful shutdown must account for device work; deliver ENGINE-10 production gate; Chapter 94 performs bounded substitution.

### Chapter 94 — Replace One Hermon Component and Prove It

- **Purpose / key question:** Demonstrate contribution-level mastery by substituting one bounded Hermon component through its real contracts and proving correctness/performance/rollback.
- **Prerequisites:** ENGINE-10 gates, refreshed Hermon source map, explicit maintainer scope and safety review.
- **Concepts:** Integration seam, shadow/A-B execution, migration, feature gate, rollback, upstream contribution, evidence package.
- **Mathematics / systems / hardware:** Component-specific equivalence/cost model; ownership completeness across seam; relevant provider matrix.
- **Implementation / Hermon / external:** Choose a non-destructive candidate only after current-source review, implement on a branch, preserve default until gates pass, and avoid broad rewrite; use Hermon's substitution ladder.
- **Diagrams / experiments:** Before/after call and ownership path plus release/rollback ladder; component unit, runtime integration, real-model, concurrency, and matched performance A/B.
- **Correctness / benchmark:** All Hermon gates relevant to the component, model fixture, failure/rollback, full benchmark metadata; do not claim merge/default unless repository state confirms it.
- **Misconceptions / failures / deliverable / next:** Source compilation is not integration, a preview flag is not release, and contribution success includes clear negative findings; deliver a reviewable proof package; next is full-book technical and editorial review, not an invented Chapter 95.

## Appendix plan

- **A — Mathematical Reference:** Linear algebra, probability, floating point, asymptotics, and dimensional notation used by the book.
- **B — Tensor Shape Reference:** Canonical shapes for MHA/GQA/MQA, layers, logits, batches, KV, and MoE.
- **C — GGUF Reference:** Versioned field/type/layout notes with primary-source links and parser limits.
- **D — Quantization Reference:** Supported teaching formats, exact block layouts, formulas, and kernel coverage.
- **E — CPU Architecture Primer:** Caches, TLBs, SIMD, threads, NUMA, and measurement.
- **F — GPU Architecture Primer:** Grids, memory spaces, occupancy, transfers, synchronization, and topology.
- **G — Rust for Inference Engineers:** Ownership, concurrency, FFI wrappers, async streams, and testing patterns used here.
- **H — C for Kernel Boundaries:** ABI, pointer/size discipline, alignment, atomics, sanitizers, and build integration.
- **I — Benchmark Reproduction Guide:** Manifests, commands, controls, raw artifacts, and result interpretation.
- **J — Glossary:** Published form of `GLOSSARY.md` with chapter cross-links.
- **K — Symbols and Notation:** Stable symbols, dimensions, units, and layout conventions.
- **L — Recommended Papers:** Annotated primary literature by chapter and claim.
- **M — Source-Code Navigation Guide:** Mini-engine, Hermon, and external-reference navigation at recorded versions.
- **N — Hardware Laboratory Guide:** Reproducible CPU/Metal/CUDA/storage lab setup, safety, and fixture requirements.
