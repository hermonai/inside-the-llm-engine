# Chapter-by-chapter illustration production plan

This is a production plan, not a claim of 94 finished illustrated chapters.
Four new reference-atlas SVG/TXT plates are implemented in this pass.
Chapter 5/6/7 canonical plates are complete; Chapter 8 is the next bounded curriculum task.

Every chapter needs a teaching question, a mechanism figure and a failure
or comparison figure where these clarify its actual logic. Use editable SVG
with Unicode TXT companion, semantic labels, a precise caption and evidence.
Never turn prose into decorative boxes or label planned figures as published.

Acceptance: at most six outer panels per plate; minimum 15 px body labels;
named tensor shapes/units/owners; distinguish parameters, activations,
operators and persistent state; check numeric fixtures, SVG bounds, grayscale
and print legibility. More detail belongs in a linked zoom, not smaller type.

## Chapter 1 — The Missing Half of AI

- **Reader question:** Define inference engineering and answer what lies between an API request and the first streamed token.
- **Planned visual content / experiment:** Whole-stack request map; “follow the token,” “follow the byte,” and “follow the owner” previews; trace a streamed API response without claiming model internals from wire timing.
- **Visual grammar:** Request and token flow. Owners: request / sampler / decoder.
- **Required counterexample / boundary:** invalid input, stop or disconnect.
- **Test anchor:** Check that one request has exactly one terminal outcome; measure TTFT versus full latency only as a vocabulary exercise with environment recorded.
- **Industrial refinement:** Use the four industrial atlas plates as optional overview/zoom references. Separate model semantics, execution and serving; revisit rather than overload the opening chapter.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-INDUSTRIAL-STACK-001.
- **Next scene IDs:** FIG-CH01-MECHANISM-001 and FIG-CH01-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 2 — From Text to Tokens

- **Reader question:** Explain exactly how bytes become vocabulary identifiers and back, and why tokenization is model semantics.
- **Planned visual content / experiment:** Text-to-bytes-to-ids pipeline, split UTF-8 token stream, and wrong-template experiment; compare token counts for small inputs under two documented tokenizers.
- **Visual grammar:** Request and token flow. Owners: request / sampler / decoder.
- **Required counterexample / boundary:** invalid input, stop or disconnect.
- **Test anchor:** Round-trip byte fixtures, special-token rules, malformed UTF-8 policy, deterministic IDs; no speed headline, only tokens/input and allocation observations.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH02-MECHANISM-001 and FIG-CH02-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 3 — The Smallest Possible Language Model

- **Reader question:** Show how a token ID can produce scores for a next token before introducing a Transformer.
- **Planned visual content / experiment:** One-ID forward path and ownership map; hand-compute logits for a three-token vocabulary, then alter one weight.
- **Visual grammar:** Request and token flow. Owners: request / sampler / decoder.
- **Required counterexample / boundary:** invalid input, stop or disconnect.
- **Test anchor:** Shape validation, exact scalar expected logits, invalid ID rejection; count operations and bytes without a performance claim.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH03-MECHANISM-001 and FIG-CH03-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 4 — Logits, Sampling, and the Autoregressive Loop

- **Reader question:** Turn one forward result into repeated token generation with explicit state and stopping.
- **Planned visual content / experiment:** Logits-to-token loop and sampler-state ownership; manually generate one token, then compare greedy and seeded stochastic traces.
- **Visual grammar:** Request and token flow. Owners: request / sampler / decoder.
- **Required counterexample / boundary:** invalid input, stop or disconnect.
- **Test anchor:** Probability sum/tolerance, deterministic greedy output, reproducible seeded sequence, exactly-once terminal state; measure per-stage time only as instrumentation.
- **Industrial refinement:** Add a forward reference to constraint-state ownership in Chapter 67; preserve the current tested sampler API and distinguish token choice from output validity.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH04-MECHANISM-001 and FIG-CH04-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 5 — Tensors Without Magic

- **Reader question:** Make shape, stride, dtype, layout, and ownership explicit enough to implement every later operation.
- **Planned visual content / experiment:** Eight canonical SVG/Unicode text plates carry one matrix through storage, ownership, copy, reshape, transpose, slice, borrowing and production representation; Rust/Python/fixture parity proves the values and offsets. Retain the original higher-rank oracle and traversal-order measurement.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Overflow, bounds, zero dimensions, non-contiguous rejection/support; compare iteration orders for locality without generalizing.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-TENSOR-STRIDE-001, FIG-TENSOR-TRANSPOSE-001, FIG-TENSOR-COPY-001, FIG-TENSOR-RESHAPE-001, FIG-TENSOR-SLICE-001, FIG-TENSOR-OWNERSHIP-001, FIG-TENSOR-LIFETIME-001, FIG-TENSOR-PRODUCTION-001.
- **Next scene IDs:** FIG-CH05-MECHANISM-001 and FIG-CH05-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 6 — Matrix Multiplication: The Engine Room

- **Reader question:** Derive matmul as both linear algebra and a memory-traffic problem.
- **Planned visual content / experiment:** Row/column contraction, access order, cache-tile movement, FLOP/byte/reuse traces, and kernel stack; sweep loop order/tile size and GEMV/GEMM shapes on fixed hardware.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Hand-computable matrices, odd/tail sizes, accumulation tolerance, reference differential; record compiler/build/hardware before timing.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-GEMV-ROW-001, FIG-CH06-ENGINE-MAP, FIG-CH06-DOT, FIG-CH06-GEMV-ONE-ROW, FIG-CH06-GEMV-FULL, FIG-CH06-GEMV-MEMORY, FIG-CH06-GEMM-THREE-VIEWS, FIG-CH06-LOOP-ORDER, FIG-CH06-TILING-TAILS, FIG-CH06-MEMORY-HIERARCHY, FIG-CH06-SIMD-GPU, FIG-CH06-CROSSOVER, FIG-CH06-THROUGHPUT, FIG-CH06-TEACHING-PRODUCTION, FIG-CH06-SOURCE-PATH.
- **Next scene IDs:** FIG-CH06-MECHANISM-001 and FIG-CH06-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 7 — Embeddings and RMSNorm

- **Reader question:** Build the first real decoder operations and show how normalization stabilizes scale.
- **Planned visual content / experiment:** Fifteen canonical token/table/ownership/residual/RMS/precision/source-path diagrams; scale sweep and `f32` magnitude stress without a timing benchmark.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Independent Python oracle and 30 deterministic Rust tests over IDs, shapes, layouts, ownership, epsilon, zero, underflow, and overflow; no performance claim without a candidate and controlled workload.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-CH07-JOURNEY, FIG-CH07-LOOKUP, FIG-CH07-SEQUENCE, FIG-CH07-OWNERS, FIG-CH07-PASSES, FIG-CH07-EPSILON, FIG-CH07-RANGE, FIG-CH07-COST, FIG-CH07-CONTRAST, FIG-CH07-SOURCE, FIG-TRANSFORMER-PIPELINE-001.
- **Next scene IDs:** FIG-CH07-MECHANISM-001 and FIG-CH07-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 8 — Queries, Keys, and Values

- **Reader question:** Explain what Q, K, and V represent, which token owns each, and how head geometry changes storage.
- **Planned visual content / experiment:** Hidden-to-QKV fan-out and GQA head map; compare KV bytes for MHA/GQA/MQA at fixed query heads.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Shape divisibility, mapping boundaries, independent projection oracle; measure bundled versus separate only later with controls.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-CH08-JOURNEY, FIG-CH08-TRACE, FIG-CH08-HEADS, FIG-CH08-OWNERS, FIG-CH08-GEOMETRY, FIG-CH08-SEQUENCE, FIG-CH08-PACKING, FIG-CH08-COST, FIG-CH08-BOUNDARY, FIG-CH08-SOURCE, FIG-QKV-HEADS-001.
- **Next scene IDs:** FIG-CH08-MECHANISM-001 and FIG-CH08-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 9 — Position: RoPE From First Principles

- **Reader question:** Derive how a position-dependent rotation lets attention distinguish token order.
- **Planned visual content / experiment:** Rotating coordinate pairs and relative-position dot product; compute positions 0/1 by hand; compare analytical interpolation/base slopes without claiming model quality.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Norm preservation, position zero, odd/partial rotary dimension policy, reference tolerance, position precision and late-overflow atomicity; table-versus-compute remains a specified future experiment, not a measured result.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-CH09-JOURNEY, FIG-CH09-PLANE, FIG-CH09-FREQUENCIES, FIG-CH09-RELATIVE, FIG-CH09-PAIRING, FIG-CH09-PARTIAL, FIG-CH09-TRANSACTION, FIG-CH09-POSITIONS, FIG-CH09-SCALING, FIG-CH09-SOURCE, FIG-ROPE-ROTATION-001.
- **Next scene IDs:** FIG-CH09-MECHANISM-001 and FIG-CH09-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 10 — Causal Self-Attention

- **Reader question:** Derive how each query reads only permitted prior positions and produces a contextual value.
- **Planned visual content / experiment:** Causal visibility triangle and one query scanning K/V; disable mask or scaling to expose behavior.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Hand-computable masked attention, no future influence, GQA mapping, finite extreme scores; benchmark dense reference only as baseline.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-ATTENTION-STEPS-001.
- **Next scene IDs:** FIG-CH10-MECHANISM-001 and FIG-CH10-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 11 — The Feed-Forward Network

- **Reader question:** Explain the token-wise parameter-heavy transform that follows attention.
- **Planned visual content / experiment:** Gated branch fan-out/rejoin; change activation or ordering and observe equivalence failure.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Hand-sized vectors, activation edge values, bundled versus separate equality; later benchmark matmul grouping with exact controls.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH11-MECHANISM-001 and FIG-CH11-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 12 — One Complete Transformer Layer

- **Reader question:** Account for every operation, buffer, residual, and owner in one decoder layer.
- **Planned visual content / experiment:** Full layer data/ownership flow and activation liveness timeline; trace tiny inputs through each checkpoint.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Stage-by-stage oracle, residual aliasing tests, wrong-order negative fixture; profile stage shares without optimizing yet.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH12-MECHANISM-001 and FIG-CH12-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 13 — The Decoder Stack and Next-Token Generation

- **Reader question:** Turn one layer into an end-to-end tiny Transformer that returns next-token logits.
- **Planned visual content / experiment:** End-to-end decoder stack and layer-state table; generate several tokens from fixed weights and inspect traces.
- **Visual grammar:** Typed tensor and algorithm trace. Owners: weight owner / activation buffer.
- **Required counterexample / boundary:** shape, mask or numerical mismatch.
- **Test anchor:** Full Python/scalar differential, final-position selection, tied-head equality, deterministic greedy sequence; baseline tokens/s labeled educational.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH13-MECHANISM-001 and FIG-CH13-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 14 — What Is Actually Inside a Model File?

- **Reader question:** Inventory the metadata, tensors, tokenizer, templates, and provenance needed to reproduce model semantics.
- **Planned visual content / experiment:** File regions and semantic dependency map; inspect metadata/tensor names of an explicitly licensed fixture.
- **Visual grammar:** Byte layout and decoding trace. Owners: mapped artifact / decoded tensor.
- **Required counterexample / boundary:** truncation, alignment or unsupported encoding.
- **Test anchor:** Bounds/overflow/truncation threat cases, tensor byte reconciliation; measure scan I/O separately from load/compute.
- **Industrial refinement:** Compare a pinned safetensors specification with GGUF: metadata, tensor naming, layout, integrity and loading obligations. Do not imply container conversion guarantees model equivalence.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH14-MECHANISM-001 and FIG-CH14-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 15 — GGUF From the Bytes Up

- **Reader question:** Parse a GGUF header, typed metadata, tensor directory, alignment, and bounded tensor ranges safely.
- **Planned visual content / experiment:** Byte-level container layout and tensor range resolution; mutate a minimal fixture across valid/invalid cases.
- **Visual grammar:** Byte layout and decoding trace. Owners: mapped artifact / decoded tensor.
- **Required counterexample / boundary:** truncation, alignment or unsupported encoding.
- **Test anchor:** Truncation, huge counts, overlapping/non-contiguous offsets, invalid alignment/type/block width, exact bounded reader; benchmark metadata scan only if size/control stated.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH15-MECHANISM-001 and FIG-CH15-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 16 — Quantization From F32 to Packed Weights

- **Reader question:** Explain how lower-bit block representations preserve usable weight values and alter execution.
- **Planned visual content / experiment:** Packed block byte layout and widening path; quantize hand values and compare error/distribution.
- **Visual grammar:** Byte layout and decoding trace. Owners: mapped artifact / decoded tensor.
- **Required counterexample / boundary:** truncation, alignment or unsupported encoding.
- **Test anchor:** Golden bytes, tails/block divisibility, NaN/range policy, decode differential; measure effective bytes and decode cost, not a generic “4-bit speedup.”
- **Industrial refinement:** Separate weight-only, activation and KV quantization. Show scale groups, storage bits versus effective bytes and calibration/error budgets; keep the first packed fixture small.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH16-MECHANISM-001 and FIG-CH16-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 17 — Packed Matrix Multiplication

- **Reader question:** Multiply packed weights without materializing a full F32 copy.
- **Planned visual content / experiment:** Packed row to accumulator flow; Lab 62 bandwidth/compute decomposition over sizes and formats.
- **Visual grammar:** Byte layout and decoding trace. Owners: mapped artifact / decoded tensor.
- **Required counterexample / boundary:** truncation, alignment or unsupported encoding.
- **Test anchor:** Golden packed rows, partial/invalid groups, double/F32 oracle tolerances, accumulation semantics; report build, CPU, bytes, and control.
- **Industrial refinement:** Trace packed load, scale application, accumulation dtype and output tolerance. Compare optimized kernels only against the scalar oracle on identical logical values.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH17-MECHANISM-001 and FIG-CH17-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 18 — Loading and Running a Real GGUF Model

- **Reader question:** Prove scoped real-model support by mapping metadata/tensors to the exact Transformer semantics.
- **Planned visual content / experiment:** Metadata-to-runtime object graph and “follow the byte” real path; run a small prompt under teaching engine and oracle.
- **Visual grammar:** Byte layout and decoding trace. Owners: mapped artifact / decoded tensor.
- **Required counterexample / boundary:** truncation, alignment or unsupported encoding.
- **Test anchor:** Tokenizer/template, stage logits, greedy sequence, multiple contexts, unsupported metadata hard errors; report load, prefill, decode separately.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH18-MECHANISM-001 and FIG-CH18-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 19 — Measure First: Profiling an Inference Engine

- **Reader question:** Identify where time and bytes go before choosing an optimization.
- **Planned visual content / experiment:** Timeline from tokenize/load/prefill/decode/stream and measurement boundary map; profile a fixed workload before modifying code.
- **Visual grammar:** Timeline and physical byte accounting. Owners: profiler clock / retained state.
- **Required counterexample / boundary:** wrong boundary or double-counted memory.
- **Test anchor:** Ensure instrumentation does not alter outputs; repeat/control cold versus warm; produce a complete benchmark manifest.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH19-MECHANISM-001 and FIG-CH19-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 20 — Prefill and Decode Are Different Workloads

- **Reader question:** Explain why prompt processing and token-at-a-time generation favor different execution shapes.
- **Planned visual content / experiment:** Prefill/decode shape comparison and request timeline; sweep prompt/output length and record phase profiles.
- **Visual grammar:** Timeline and physical byte accounting. Owners: profiler clock / retained state.
- **Required counterexample / boundary:** wrong boundary or double-counted memory.
- **Test anchor:** Same logits across chunked/unchunked prefill, position continuity, phase-specific measurements with fixed model/cache.
- **Industrial refinement:** Use explicit client/server timing boundaries and workload-dependent roofline reasoning. Add the latency plate; prefill/decode names do not determine the bottleneck.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-INDUSTRIAL-LATENCY-001.
- **Next scene IDs:** FIG-CH20-MECHANISM-001 and FIG-CH20-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 21 — Why the KV Cache Exists

- **Reader question:** Demonstrate why decoder inference should retain prior keys/values instead of recomputing past token layers.
- **Planned visual content / experiment:** Repeated no-cache layers versus append-only cached decode and per-layer layout; generate N tokens both ways.
- **Visual grammar:** Timeline and physical byte accounting. Owners: profiler clock / retained state.
- **Required counterexample / boundary:** wrong boundary or double-counted memory.
- **Test anchor:** Temperature-zero logits/token sequence agree, cache positions reset correctly, multiple sequence isolation; report prefill plus N-token decode and bytes.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-KVCACHE-GROWTH-001.
- **Next scene IDs:** FIG-CH21-MECHANISM-001 and FIG-CH21-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 22 — KV Cache Memory Mathematics

- **Reader question:** Predict cache capacity and traffic from model geometry, dtype, sequence length, and concurrency.
- **Planned visual content / experiment:** Geometry-to-bytes expansion and capacity budget; sweep context/concurrency/dtype and validate against instrumentation.
- **Visual grammar:** Timeline and physical byte accounting. Owners: profiler clock / retained state.
- **Required counterexample / boundary:** wrong boundary or double-counted memory.
- **Test anchor:** Overflow/unit tests, MHA/GQA cases, allocator overhead disclosure; no tok/s claim—capacity and physical bytes only.
- **Industrial refinement:** Count valid, allocated and reserved bytes separately. Add dense/GQA/sliding-window applicability limits and a quantized-KV metadata budget; defer MLA to Chapter 80.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-INDUSTRIAL-MEMORY-001.
- **Next scene IDs:** FIG-CH22-MECHANISM-001 and FIG-CH22-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 23 — One User Is Easy

- **Reader question:** Expose assumptions hidden by a synchronous single-request loop before concurrency invalidates them.
- **Planned visual content / experiment:** Single request owner/lifetime and synchronous call timeline; disconnect a client at three phases.
- **Visual grammar:** Swimlane and sequence lifecycle. Owners: scheduler / sequence / stream.
- **Required counterexample / boundary:** starvation, cancellation or queue pressure.
- **Test anchor:** Exactly one terminal result, bounded output, resource release on disconnect/error; baseline latency only.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH23-MECHANISM-001 and FIG-CH23-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 24 — The Inference Request State Machine

- **Reader question:** Define legal request transitions and who owns every resource in each state.
- **Planned visual content / experiment:** Full state graph with failure/cancel edges and resource table by state; inject error at every transition.
- **Visual grammar:** Swimlane and sequence lifecycle. Owners: scheduler / sequence / stream.
- **Required counterexample / boundary:** starvation, cancellation or queue pressure.
- **Test anchor:** Property tests for legal transitions, exactly-once terminal event, no resource leak/double release; measure queue versus execution time separately.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-INDUSTRIAL-OWNERSHIP-001.
- **Next scene IDs:** FIG-CH24-MECHANISM-001 and FIG-CH24-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 25 — Serving Multiple Users

- **Reader question:** Show why independent serial loops underutilize or oversubscribe shared model/hardware resources.
- **Planned visual content / experiment:** Multiple requests sharing weights but not state; sweep concurrency and observe saturation/queueing.
- **Visual grammar:** Swimlane and sequence lifecycle. Owners: scheduler / sequence / stream.
- **Required counterexample / boundary:** starvation, cancellation or queue pressure.
- **Test anchor:** Cross-request token/KV isolation, ordering-independent outputs, failure containment; report per-request tails and aggregate throughput.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH25-MECHANISM-001 and FIG-CH25-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 26 — Continuous Batching

- **Reader question:** Rebuild the physical work each iteration so active sequences share execution without waiting for request-batch completion.
- **Planned visual content / experiment:** One mixed prefill/decode iteration and sequence-slot timeline; replay staggered arrivals against serial/pool/batched baselines.
- **Visual grammar:** Swimlane and sequence lifecycle. Owners: scheduler / sequence / stream.
- **Required counterexample / boundary:** starvation, cancellation or queue pressure.
- **Test anchor:** Batched versus isolated greedy differential, positions, stop/cancel, failed-batch behavior; throughput, TTFT, ITL, tail latency, fairness with full manifest.
- **Industrial refinement:** Make chunked prefill a concrete scheduling policy with per-iteration token budget and decode latency tradeoff. Trace one long prompt beside short decodes.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-BATCH-TIMELINE-001.
- **Next scene IDs:** FIG-CH26-MECHANISM-001 and FIG-CH26-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 27 — Fairness, Backpressure, Cancellation, and Streaming

- **Reader question:** Keep a busy server bounded and responsive when clients, prompts, outputs, and failures differ.
- **Planned visual content / experiment:** Backpressure propagation and cancellation across API/runtime/provider; slow consumer, long prompt, and mixed-priority replay.
- **Visual grammar:** Swimlane and sequence lifecycle. Owners: scheduler / sequence / stream.
- **Required counterexample / boundary:** starvation, cancellation or queue pressure.
- **Test anchor:** No starvation under defined policy, memory bound, valid byte stream, exactly-once release; measure tails/fairness at saturation.
- **Industrial refinement:** Test fairness, bounded queues, cancellation and admission under a synthetic arrival trace; do not replace tail-latency analysis with average throughput.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH27-MECHANISM-001 and FIG-CH27-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 28 — Why Flat and Slot-Bound KV Caches Fail

- **Reader question:** Make fragmentation, fixed-slot capacity, duplicated prefixes, and affinity limits measurable.
- **Planned visual content / experiment:** Slot layouts under variable requests and duplicated system prompt; replay length distribution and calculate waste.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Allocation accounting and isolation remain correct; report utilization/copies rather than conflating with throughput.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH28-MECHANISM-001 and FIG-CH28-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 29 — Paging Comes to AI

- **Reader question:** Separate logical sequence order from physical KV placement while preserving attention semantics.
- **Planned visual content / experiment:** Logical sequence mapped to scattered physical blocks and OS analogy/breakpoints; vary block size on synthetic lengths.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Translation boundaries `B-1/B/B+1`, order preservation, invalid IDs; measure fragmentation versus metadata/lookups.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH29-MECHANISM-001 and FIG-CH29-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 30 — Building a Block Pool

- **Reader question:** Define allocation, reference, mutation, and reuse rules for physical KV storage.
- **Planned visual content / experiment:** Pool metadata/data split and reference state machine; concurrent allocate/release stress.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Zero/over-allocation, double-decref defense, refcount reuse iff zero, layer/offset bounds, sanitizer; benchmark allocator ops separately from attention.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH30-MECHANISM-001 and FIG-CH30-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 31 — Prefix Indexing and Radix Trees

- **Reader question:** Find the longest reusable token prefix and make cache ownership explicit.
- **Planned visual content / experiment:** Compressed radix branches with shared blocks and ownership graph; insert overlapping prompts and observe reference counts.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Exact/partial/no match, replacement, eviction decref, concurrent readers, model/config separation; report hit rate and tokens saved with workload.
- **Industrial refinement:** State every cache-key assumption: model revision, tokenization/template, adapter and relevant position/execution context. Prefix equality alone is not universal cache validity.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH31-MECHANISM-001 and FIG-CH31-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 32 — Shared Prefixes and Copy-on-Write

- **Reader question:** Explain why a shared partial tail is safe to read but unsafe to extend in place.
- **Planned visual content / experiment:** Two continuations before/after COW and mutation routing surface; disable COW to produce deterministic cross-request corruption.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Aligned and partial prefixes, source unchanged, destination valid region equal, backend copy differential; measure copy overhead by tail length.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH32-MECHANISM-001 and FIG-CH32-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 33 — Eviction, Pressure, and Admission

- **Reader question:** Decide which cached state to retain and whether a new request can safely enter under finite capacity.
- **Planned visual content / experiment:** Ownership/pin/eviction state machine and admission under pressure; replay skewed/uniform prefixes and forced exhaustion.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Never evict live/pinned references, atomic retry, no leaks/starvation, deterministic policy where promised; hit rate, evictions, wait, and physical bytes.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH33-MECHANISM-001 and FIG-CH33-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 34 — Paged Attention

- **Reader question:** Compute causal attention directly through a block table without reconstructing dense KV.
- **Planned visual content / experiment:** Query scan across scattered blocks and dense/paged semantic equivalence; randomize physical placement and block size.
- **Visual grammar:** Logical-to-physical memory map. Owners: sequence handles / block manager.
- **Required counterexample / boundary:** aliasing, stale handle or exhausted pool.
- **Test anchor:** Dense differential across B-1/B/B+1, MHA/GQA, causal offsets, scrambled tables, shared prefixes; measure overhead/memory without claiming production speed.
- **Industrial refinement:** Explicitly distinguish paged KV addressing, prefix reuse and tiled attention. Add quantized-KV read/dequantization to the layout contract, without claiming every backend supports it.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH34-MECHANISM-001 and FIG-CH34-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 35 — Where the Native Boundary Belongs

- **Reader question:** Decide which mechanism benefits from a native boundary without moving scheduling and ownership policy into opaque code.
- **Planned visual content / experiment:** Host/runtime versus native mechanism map and too-fine/too-broad boundary examples; measure empty-call and realistic bulk-call overhead.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Same oracle before/after boundary, panic/error translation, no hidden thread ownership; report crossover rather than “FFI is fast.”
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH35-MECHANISM-001 and FIG-CH35-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 36 — Designing a Stable Kernel ABI

- **Reader question:** Make shapes, buffers, ownership, versioning, errors, and workspace explicit across compiled components.
- **Planned visual content / experiment:** Rust-to-C ownership/lifetime diagram and version-size negotiation; compile mismatched client fixture and invalid shapes.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Layout/size assertions, null/overflow/short-buffer rejection, error detail lifetime, fuzz boundary; measure bulk-call overhead with workspace declared.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH36-MECHANISM-001 and FIG-CH36-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 37 — Arena Allocation and Hot-Path Memory Discipline

- **Reader question:** Eliminate unpredictable general allocation from kernel iterations while retaining auditable lifetime and alignment.
- **Planned visual content / experiment:** Arena regions/lifetimes and allocation timeline; stress alignment, exhaustion, reset, and repeated iterations.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Overflow, zero size, destruction order, sanitizer, no hot-loop allocator calls; report allocation latency/high-water under controlled workload.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH37-MECHANISM-001 and FIG-CH37-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 38 — Lock-Free Block Allocation and Refcounts

- **Reader question:** Move frequent block lifecycle operations native and concurrent without losing the refcount invariant.
- **Planned visual content / experiment:** Allocation state transition with linearization and contended cache lines; randomized multi-thread stress and forced exhaustion.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** No duplicate live ID, no lost block, reuse iff zero, ABA mitigation, thread-count differential, sanitizers; throughput versus safe locked oracle by contention.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH38-MECHANISM-001 and FIG-CH38-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 39 — Bulk KV Writes

- **Reader question:** Replace per-element host/native crossings with one validated write over token/head ranges.
- **Planned visual content / experiment:** Q/K/V range scattered across block tails and backend mutation surface; write across B-1/B/B+1 and convert dtypes.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Whole-range and scrambled tables, no partial mutation on invalid input, source alias policy, dense readback differential; bytes/s and call-count reduction.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH39-MECHANISM-001 and FIG-CH39-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 40 — Online Softmax

- **Reader question:** Produce exact softmax-weighted values without allocating a score vector for the full history.
- **Planned visual content / experiment:** Streaming score/value update and two-chunk merge; adversarial score ranges and chunk orders.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Dense softmax differential, overflow/underflow, empty/one position, GQA, tolerance by dtype; scratch bytes and runtime across sequence lengths.
- **Industrial refinement:** Connect the stable online softmax recurrence to tiled attention with an exact small oracle. Read the full FlashAttention algorithm before asserting kernel-level equivalence.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH40-MECHANISM-001 and FIG-CH40-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 41 — Split-K and Deterministic Attention Planning

- **Reader question:** Parallelize long-history attention while making task completion order irrelevant to numerical reduction order.
- **Planned visual content / experiment:** Deterministic task grid and out-of-order completion into indexed slots; randomize task order/thread count around thresholds.
- **Visual grammar:** Kernel dataflow and lifetime trace. Owners: caller / native allocation / kernel.
- **Required counterexample / boundary:** ABI mismatch, refcount or reduction failure.
- **Test anchor:** Serial/parallel equality or stated tolerance, split T-1/T/T+1, workspace bounds, task uniqueness, failure fallback; speedup/tail by sequence and threads.
- **Industrial refinement:** Make partition merge order, partial normalization state and determinism visible. Explain when scheduling or reduction changes floating-point results.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH41-MECHANISM-001 and FIG-CH41-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 42 — SIMD From First Principles

- **Reader question:** Map scalar loops to vector lanes while preserving tails, layout, and reduction semantics.
- **Planned visual content / experiment:** Scalar iterations packed into lanes and tail handling; inspect compiler output and sweep aligned/odd dimensions.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** All tails, alignment, NaNs/infinities policy, scalar differential, sanitizer; cycles/element and bandwidth with compiler/ISA recorded.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH42-MECHANISM-001 and FIG-CH42-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 43 — ARM NEON

- **Reader question:** Implement and validate the vector contract on ARM's baseline 128-bit SIMD.
- **Planned visual content / experiment:** NEON tile/register map; sweep head dimensions, contexts, and compiler flags on named hardware.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** Scalar differential, non-multiple tails, exact dispatch, cross-compile CI plus real-hardware record; report crossover and confidence.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH43-MECHANISM-001 and FIG-CH43-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 44 — x86 AVX2 and ISA Dispatch

- **Reader question:** Add a wider x86 path and select it safely at runtime without executing unsupported instructions.
- **Planned visual content / experiment:** Dispatch tree and AVX2 tile; run scalar-forced versus auto versus AVX2 on supported hardware.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** Unsupported-host safety, feature spoof/forced fallback, tails, scalar differential, thread counts; cycles/element plus frequency effects.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH44-MECHANISM-001 and FIG-CH44-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 45 — Thinking Like a GPU

- **Reader question:** Reframe a kernel as many work items with explicit memory spaces, synchronization, launch, and transfer costs.
- **Planned visual content / experiment:** CPU tasks versus GPU grid and memory hierarchy; paper-plan a shape and estimate launch/work threshold.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** Provider conformance requirements and asynchronous lifetime tests; no speed claim without implementation.
- **Industrial refinement:** Separate model graph, compiler IR, fused kernel and captured launch graph. Show a dependency before a performance optimization.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH45-MECHANISM-001 and FIG-CH45-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 46 — Metal and Unified Memory

- **Reader question:** Execute planned attention on Apple GPU while reasoning precisely about shared physical memory and synchronization.
- **Planned visual content / experiment:** Command lifecycle and shared-memory residency; Lab 67 Metal/CPU crossover sweep.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** Scalar differential by shape/dtype, unsupported fallback, command failure, buffer lifetime; warm/cold pipeline and synchronization included.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH46-MECHANISM-001 and FIG-CH46-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 47 — CUDA and Device Mirrors

- **Reader question:** Maintain coherent device-resident/mirrored state and run the same plan on a discrete NVIDIA GPU.
- **Planned visual content / experiment:** Host home/device mirror coherence and stream timeline; sweep resident versus transferred inputs where hardware exists.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** CPU differential, stale mirror injection, unsupported/error fallback overwrites workspace, multi-stream lifetime; include driver/device/clock/build.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH47-MECHANISM-001 and FIG-CH47-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 48 — Why a GPU Can Be Slower Than a CPU

- **Reader question:** Build an evidence-based shape gate from fixed overhead, work, transfer, and contention rather than brand assumptions.
- **Planned visual content / experiment:** Cost curves/crossover table and deterministic fallback tree; matched CPU/GPU sweep across context/query shapes.
- **Visual grammar:** Hardware execution and transfer diagram. Owners: host / device / completion event.
- **Required counterexample / boundary:** dispatch mismatch or uncharged transfer.
- **Test anchor:** Same outputs on both branches, T-1/T/T+1, forced modes, failure fallback; report losing cases and variance.
- **Industrial refinement:** Own the first bounded graph lowering/capture case study: dynamic shapes, specialization, buffer addresses, warmup and replay invalidation. Measure transfer/launch overhead before explaining a CPU/GPU result.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH48-MECHANISM-001 and FIG-CH48-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 49 — Prefix Caching

- **Reader question:** Reuse computed KV for repeated token prefixes across requests without violating model/config/lifetime semantics.
- **Planned visual content / experiment:** Cross-request prefix ownership and cold/warm timelines; shared-system-prompt workload with cache on/off.
- **Visual grammar:** Proposal, verification and rollback trace. Owners: sequence / draft / target cache.
- **Required counterexample / boundary:** rejection, invalid reuse or negative speedup.
- **Test anchor:** Cached/uncached logits, config/model/tokenizer keying, partial COW, eviction, tenant boundaries; TTFT/tokens saved/hit rate with cold control.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH49-MECHANISM-001 and FIG-CH49-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 50 — Sticky Slots as an Intermediate Design

- **Reader question:** Understand when context-bound prefix reuse offers high value with lower complexity and where it stops scaling.
- **Planned visual content / experiment:** Warm-slot routing and contrast with page-sharing radix; alternating conversations beyond slot capacity.
- **Visual grammar:** Proposal, verification and rollback trace. Owners: sequence / draft / target cache.
- **Required counterexample / boundary:** rejection, invalid reuse or negative speedup.
- **Test anchor:** Slot metadata never outlives KV, overlap leaves a fresh logit position, cross-conversation isolation; shared-prefix TTFT and churn workload.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH50-MECHANISM-001 and FIG-CH50-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 51 — Speculative Decoding

- **Reader question:** Reduce target-model decode iterations while preserving the target distribution exactly under the chosen algorithm.
- **Planned visual content / experiment:** Draft/verify/accept/rollback timeline; deterministic toy distributions with full accept/partial/reject.
- **Visual grammar:** Proposal, verification and rollback trace. Owners: sequence / draft / target cache.
- **Required counterexample / boundary:** rejection, invalid reuse or negative speedup.
- **Test anchor:** Distributional test or exact greedy special case, rollback positions/KV, RNG consumption, EOG/stop; tokens per target pass plus end-to-end latency.
- **Industrial refinement:** Read the full exact speculative-sampling paper; derive acceptance/correction and show first rejection plus KV rollback. Distinguish greedy verification from stochastic distribution preservation.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH51-MECHANISM-001 and FIG-CH51-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 52 — Prompt-Lookup Decoding

- **Reader question:** Propose tokens from repeated prompt n-grams without running a second model.
- **Planned visual content / experiment:** N-gram match to proposed run and target-logit predecessor mapping; RAG quote, code, and creative prompts.
- **Visual grammar:** Proposal, verification and rollback trace. Owners: sequence / draft / target cache.
- **Required counterexample / boundary:** rejection, invalid reuse or negative speedup.
- **Test anchor:** Greedy equivalence, mismatch token handling, rollback suffix, duplicate matches/tie policy, mixed sequences; acceptance, target calls, batch size, end-to-end latency.
- **Industrial refinement:** Contrast prompt lookup/ngram proposals with draft models; add a source-verified comparison sidebar for EAGLE/MTP with their model/training assumptions. These remain optional proposal mechanisms.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH52-MECHANISM-001 and FIG-CH52-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 53 — When Speculation Loses

- **Reader question:** Determine when proposal, verification batch inflation, rollback, and contention cost more than saved target steps.
- **Planned visual content / experiment:** Break-even surface and adaptive state timeline; sweep acceptance, lookahead, concurrency, model/provider.
- **Visual grammar:** Proposal, verification and rollback trace. Owners: sequence / draft / target cache.
- **Required counterexample / boundary:** rejection, invalid reuse or negative speedup.
- **Test anchor:** Gate changes performance only, never output; threshold/window boundaries and re-enable; end-to-end matched static-off/static-on/adaptive.
- **Industrial refinement:** Charge draft, verification, rollback and memory costs to useful accepted tokens; include a counterexample where speculation loses.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH53-MECHANISM-001 and FIG-CH53-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 54 — Why MoE Changes the Inference Engine

- **Reader question:** Explain how conditional expert activation changes weight demand, scheduling, batching, and correctness.
- **Planned visual content / experiment:** Token-router-expert-combine flow and dense/MoE byte demand; route synthetic tokens under uniform/skewed gates.
- **Visual grammar:** Routing and memory-tier swimlane. Owners: router / expert pool / transfer queue.
- **Required counterexample / boundary:** miss, eviction race or capacity pressure.
- **Test anchor:** Top-k/ties, weight ordering, combine weights, capacity/unsupported semantics; measure routing and expert compute separately.
- **Industrial refinement:** Trace top-k expert routing, dispatch, expert compute and combine; separate activation communication from expert-weight residency.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH54-MECHANISM-001 and FIG-CH54-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 55 — Models Larger Than Available VRAM

- **Reader question:** Quantify when weights, KV, scratch, and other processes cannot simultaneously reside in fast memory.
- **Planned visual content / experiment:** Full inference memory budget and tier hierarchy; vary model/cache/context on hypothetical and measured machines.
- **Visual grammar:** Routing and memory-tier swimlane. Owners: router / expert pool / transfer queue.
- **Required counterexample / boundary:** miss, eviction race or capacity pressure.
- **Test anchor:** Units/overflow, requested versus effective placement, OOM/fallback behavior; capacity/transfer model labeled estimate until measured.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH55-MECHANISM-001 and FIG-CH55-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 56 — Expert Storage and Paging

- **Reader question:** Store experts so their offsets are verifiable and fetch only routed weights without full expansion.
- **Planned visual content / experiment:** Container layout and miss-to-cache-fill path; pack mixed-size layer records and replay routes.
- **Visual grammar:** Routing and memory-tier swimlane. Owners: router / expert pool / transfer queue.
- **Required counterexample / boundary:** miss, eviction race or capacity pressure.
- **Test anchor:** Checksums/ranges, short reads, layer stride, duplicate IDs, quant block boundaries, oracle matvec; storage bytes/s and ceiling explicitly not inference.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH56-MECHANISM-001 and FIG-CH56-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 57 — Residency, Pinning, Eviction, and Queue Depth

- **Reader question:** Keep expert bytes valid while used, evictable when safe, and honestly report effective I/O concurrency.
- **Planned visual content / experiment:** Three-state residency machine and synchronous/batched I/O timelines; uniform/skewed routes, QD sweep, forced read error.
- **Visual grammar:** Routing and memory-tier swimlane. Owners: router / expert pool / transfer queue.
- **Required counterexample / boundary:** miss, eviction race or capacity pressure.
- **Test anchor:** Never evict pinned, unwind all-or-nothing, duplicate expert handling, requested/effective metrics, cache clear; hit/read bytes/bandwidth and no tok/s without model compute.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH57-MECHANISM-001 and FIG-CH57-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 58 — Toward Unified Inference Memory

- **Reader question:** Decide whether KV, recurrent state, expert weights, and scratch can share one placement/eviction abstraction without hiding their semantic differences.
- **Planned visual content / experiment:** Unified logical page kinds over tiers and mixed-budget eviction; compare static KV/expert partitions with dynamic policy in simulation.
- **Visual grammar:** Routing and memory-tier swimlane. Owners: router / expert pool / transfer queue.
- **Required counterexample / boundary:** miss, eviction race or capacity pressure.
- **Test anchor:** Dirty mutable state never discarded, immutable experts need no writeback, pin/failure/cancel rules; simulated hit/transfer/wait labeled model, not measured speed.
- **Industrial refinement:** Use an allocation ownership ledger across weight/KV/expert pools. Do not equate a unified policy with a physically unified address space.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH58-MECHANISM-001 and FIG-CH58-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 59 — Fast Wrong Answers Are Still Wrong

- **Reader question:** Explain why fluent output, determinism, and unit tests can coexist with serious inference defects.
- **Planned visual content / experiment:** Proof ladder and defect propagation chain; inject mask, position, stale-KV, and tensor-layout faults.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Define what each subsequent level proves; no performance result survives failed equivalence.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH59-MECHANISM-001 and FIG-CH59-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 60 — Scalar Oracles

- **Reader question:** Build references simple and independent enough to expose optimized-path bugs.
- **Planned visual content / experiment:** Oracle-to-implementations comparison; mutate each optimized path and verify oracle catches it.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Boundary shapes, extreme values, expected failures, oracle cross-check with hand math; oracles are not performance baselines.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH60-MECHANISM-001 and FIG-CH60-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 61 — Differential Testing

- **Reader question:** Systematically compare implementations across shapes, dtypes, layouts, orders, and providers.
- **Planned visual content / experiment:** Test matrix and failure-shrinking flow; randomized boundary-focused cases.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Coverage includes B/T thresholds, MHA/GQA, short/long, aligned/partial prefix, task orders; record failures, do not time as benchmark.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH61-MECHANISM-001 and FIG-CH61-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 62 — Numerical Determinism

- **Reader question:** Decide which outputs must be bitwise stable, tolerance-stable, or distributionally equivalent across order and hardware.
- **Planned visual content / experiment:** Completion-order versus combine-order; vary thread/task/provider order repeatedly.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Thread-count/order reproducibility, documented cross-provider tolerances, RNG sequence, gate thresholds; record determinism settings in benchmarks.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH62-MECHANISM-001 and FIG-CH62-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 63 — Concurrency Bugs That Still Produce Plausible Text

- **Reader question:** Find state races, cross-request contamination, stale metadata, and ordering errors that avoid crashes.
- **Planned visual content / experiment:** Two-request interleaving and stale-slot timeline; inject yields, reorder completions, cancel during shared batch.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Isolated-output differential under concurrency, no cross-request state, worker survives failure, terminal exactly once; performance secondary.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH63-MECHANISM-001 and FIG-CH63-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 64 — Ownership and Lifetime Failures

- **Reader question:** Prove every weight, context, page, mirror, expert lease, stream, and sampler has one coherent lifetime.
- **Planned visual content / experiment:** Ownership graphs for KV/expert/device stream and success/error/cancel unwind; inject every acquisition failure.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Zero leaked blocks/pins/tasks, no reuse while referenced, duplicate acquisition, idempotent cleanup, sanitizer; benchmark only after guard overhead understood.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH64-MECHANISM-001 and FIG-CH64-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 65 — Sanitizers, Fuzzing, and Boundary Testing

- **Reader question:** Exercise memory, parser, ABI, allocator, and numerical boundaries beyond hand-selected cases.
- **Planned visual content / experiment:** Input-to-parser/kernel containment and corpus lifecycle; seed with valid minimal cases then mutate.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Crashes, leaks, undefined behavior, hangs, invariant errors; fuzz executions are coverage evidence, not performance results.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH65-MECHANISM-001 and FIG-CH65-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 66 — Real-Model Equivalence

- **Reader question:** Demonstrate actual supported model semantics across prompt positions and decode, not merely component arithmetic.
- **Planned visual content / experiment:** Component-to-model proof ladder and corpus matrix; prompts crossing blocks/context shapes and repeated prefixes.
- **Visual grammar:** Counterexample and proof-boundary diagram. Owners: oracle / optimized implementation / harness.
- **Required counterexample / boundary:** plausible output that violates an invariant.
- **Test anchor:** Tokenizer/template, logits, greedy outputs, cached/uncached, batched/isolated, providers, repeated runs; only equivalent configurations advance performance gates.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH66-MECHANISM-001 and FIG-CH66-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 67 — Protocols and the AI Gateway

- **Reader question:** Normalize multiple wire protocols into one truthful engine request without leaking protocol quirks into execution policy.
- **Planned visual content / experiment:** Many protocols to one request object and response/error mapping; replay equivalent requests across adapters.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Schema validation, stop/sampling mapping, stream/sync semantic equivalence, error codes, usage; protocol overhead measured separately from model.
- **Industrial refinement:** Own grammar/JSON-schema constrained decoding: compiler/automaton state, allowed-token mask, unsatisfiable continuation, stop behavior and cancellation. Syntactic validity is not semantic truth.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH67-MECHANISM-001 and FIG-CH67-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 68 — Model Resolution and Routing

- **Reader question:** Resolve a user model identifier to an exact local artifact or provider with explicit policy and provenance.
- **Planned visual content / experiment:** Model-name decision tree and artifact identity flow; ambiguous aliases, missing artifact, offline provider.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** No silent provider/model substitution, canonical cache identity, capability errors, secret boundary; measure resolution separately, no inference claim.
- **Industrial refinement:** Own adapter identity, loading, residency and routing; introduce multimodal encoder/projector inputs and per-request state. Include reuse isolation across adapters and model revisions.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH68-MECHANISM-001 and FIG-CH68-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 69 — Streaming as a Systems Contract

- **Reader question:** Define ordering, UTF-8, backpressure, cancellation, terminal events, and usage for a reliable token stream.
- **Planned visual content / experiment:** Runtime-channel-wire pipeline and slow-client backpressure; disconnect at prefill/decode/final event.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Valid framing/UTF-8, ordered pieces, exactly one terminal outcome, no Done after error, cleanup; time first byte/token and blocked-producer behavior.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH69-MECHANISM-001 and FIG-CH69-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 70 — Metrics and Observability

- **Reader question:** Expose enough state to explain latency, cache behavior, saturation, failures, and provider selection without perturbing hot paths.
- **Planned visual content / experiment:** Causal observability map from symptom to subsystem; generate slow client, cache churn, provider fallback, and verify signals.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Monotonic counters, label bounds, terminal accounting, snapshot consistency, low overhead; benchmark instrumentation on/off.
- **Industrial refinement:** Name timestamps and units for TTFT, ITL, TPOT, E2E, queue delay and goodput. Use trace IDs and censoring/failure policy; never infer token timing solely from transport chunks.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH70-MECHANISM-001 and FIG-CH70-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 71 — Failure Containment

- **Reader question:** Prevent malformed input, one request, one model, one shared batch, or one provider failure from corrupting unrelated work.
- **Planned visual content / experiment:** Failure domains and recovery transitions; inject parser, model-load, decode, device, stream, and disk errors.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Unaffected requests remain isolated where contract permits, failed cache metadata invalidated, capacity recovered, errors observable; measure recovery and blast radius.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH71-MECHANISM-001 and FIG-CH71-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 72 — Security and Untrusted Models

- **Reader question:** Treat model files, templates, prompts, API inputs, local paths, native code, and secrets as distinct trust boundaries.
- **Planned visual content / experiment:** Trust-boundary/data-flow diagram and attack-to-control table; fuzz huge metadata, malicious paths, prompt/cache tenant collisions.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Reject over-budget/invalid artifacts safely, secret redaction, tenant cache partitioning, dependency audit; record security-control overhead where relevant.
- **Industrial refinement:** Threat-model untrusted model files, templates, adapters, tenant cache reuse, prompt logging and resource exhaustion. Pin a parser boundary and test rejection.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH72-MECHANISM-001 and FIG-CH72-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 73 — Benchmarking Without Lying to Yourself

- **Reader question:** Produce reproducible performance evidence that separates correctness, workload, state, hardware, and control.
- **Planned visual content / experiment:** Benchmark boundary/state-reset flow and metric decision table; deliberately contaminate prefix/model/OS caches and detect it.
- **Visual grammar:** Protocol and operational lifecycle. Owners: tenant / gateway / runtime / observer.
- **Required counterexample / boundary:** disconnect, overload, malformed artifact or SLO miss.
- **Test anchor:** Equivalence gate first, full manifest/raw output, matched stop/model/quantization, repeated order-randomized controls; this chapter's deliverable is the benchmark suite.
- **Industrial refinement:** Report SLO-qualified goodput, tail distributions, warm/cold phases, arrival process, concurrency, model, precision and hardware. No synthetic timing fixture becomes a speed claim.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH73-MECHANISM-001 and FIG-CH73-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 74 — Hermon's System Architecture

- **Reader question:** Establish the current system boundary and exact default/compatibility/preview/library/target topology at a recorded commit.
- **Planned visual content / experiment:** Source-verified system topology and three runtime modes; build/test normal and preview feature configurations where available.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Check doc claims against actual dispatch gates/call graph; no reused benchmark number without reproducer/metadata.
- **Industrial refinement:** Update the case-study ledger to the inspected Hermon pin without rewriting historical measurements. Keep CURRENT, PREVIEW, LIBRARY and TARGET distinct.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-ENGINE-ARCHITECTURE-001.
- **Next scene IDs:** FIG-CH74-MECHANISM-001 and FIG-CH74-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 75 — Why Hermon Did Not Rewrite Everything

- **Reader question:** Analyze how wrapping a proven engine enables product/runtime progress while native components mature behind gates.
- **Planned visual content / experiment:** Proven mechanism inside owned policy shell and rewrite/substitution risk map; inspect dependency/build modes.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Wrapper equivalence/real-model tests and explicit stub behavior; do not claim native superiority from architecture.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH75-MECHANISM-001 and FIG-CH75-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 76 — The Substitution Ladder

- **Reader question:** Show how reference structures become a default native path through evidence rather than a flag flip.
- **Planned visual content / experiment:** Seven-rung ladder with evidence gates and rollback; place current paged/native/MoE components on it.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Each rung names tests and measurement required; highlight model fixture/1,000-prompt/open integration gates.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH76-MECHANISM-001 and FIG-CH76-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 77 — Anatomy of the Hermon Source Tree

- **Reader question:** Teach a contributor where contracts live and which changes are local versus cross-layer.
- **Planned visual content / experiment:** Crate dependency/ownership graph and “change X, inspect Y” map; follow a protocol field, GGUF key, radix change, and ISA kernel.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Confirm unsafe lint boundaries, feature compilation, test fixture gates; no performance claim.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH77-MECHANISM-001 and FIG-CH77-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 78 — Follow One Request Through Hermon

- **Reader question:** Trace validation, model resolution, dispatch, admission, batched execution, stream backpressure, and terminal accounting on the default path.
- **Planned visual content / experiment:** Source-linked sequence diagram and resource ownership by phase; run a local fixture if available with logs/metrics.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Piece/Done/error contract, bounded channels, batch failure invalidation, worker recovery; timings only if full setup recorded.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** FIG-RUNTIME-SEQUENCE-001.
- **Next scene IDs:** FIG-CH78-MECHANISM-001 and FIG-CH78-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 79 — Follow One Token Through Hermon

- **Reader question:** Trace tokenization, chat template, prefill/decode batching, logits, sampling, UTF-8 buffering, KV mutation, and stream output.
- **Planned visual content / experiment:** Token lifecycle with positions and PLD verification branch; fixed greedy prompt trace if model fixture exists.
- **Visual grammar:** Pinned source-to-runtime architecture. Owners: API / dispatcher / worker / bridge.
- **Required counterexample / boundary:** mistaking available library code for the default path.
- **Test anchor:** Token bytes/UTF-8, logit-position selection, accepted/rejected draft KV, stop/EOG, isolated baseline; no performance beyond reproducible run.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH79-MECHANISM-001 and FIG-CH79-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 80 — Hybrid Transformer Architectures

- **Reader question:** Adapt the inference mental model when layers mix full attention, sliding/local attention, linear/recurrent mechanisms, and MoE.
- **Planned visual content / experiment:** Mixed layer stack and state-kind table; simulate memory/work under layer mixtures.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Per-layer semantics, position/mask, state reset, unsupported hard error; estimates labeled, real measurements only on implemented model.
- **Industrial refinement:** Separate externally deployed architectural variants from research hypotheses. Add typed multimodal input and MLA/sliding-window/hybrid state-accounting exceptions, sourced per actual model.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH80-MECHANISM-001 and FIG-CH80-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 81 — Recurrent State and STATE Pages

- **Reader question:** Manage mutable fixed-size recurrent/linear-attention state alongside append-oriented KV.
- **Planned visual content / experiment:** KV append versus STATE mutation and branch/COW; fork two continuations and reproduce alias corruption without COW.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Sequential oracle, reset/fork/rollback, cancel/error, provider mirror coherence; state bytes/update latency separately.
- **Industrial refinement:** A recurrent STATE page stores a bounded recurrence state, not a dense KV history. Define rollback/checkpoint behavior before using speculation.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH81-MECHANISM-001 and FIG-CH81-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 82 — Unified Memory Economics

- **Reader question:** Derive placement decisions across weights, KV, STATE, experts, scratch, and mirrors from value/cost rather than one cache policy.
- **Planned visual content / experiment:** Typed page catalog and eviction decision matrix; mixed chat/MoE/recurrent workloads under static/dynamic budgets.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Never lose dirty state, reserve forward progress, reproducible simulation inputs; results are simulation until implemented/measured.
- **Industrial refinement:** Compare physical tiers with explicit transfer latency/bandwidth and ownership. A cost model is an analytical prediction, not a measured migration policy.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH82-MECHANISM-001 and FIG-CH82-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 83 — Prefill/Decode Disaggregation

- **Reader question:** Decide when distinct workers/devices for compute-heavy prefill and memory-heavy decode outweigh KV transfer and coordination.
- **Planned visual content / experiment:** Prefill worker -> KV transfer -> decode worker with failure edges; sweep prompt/context/network bandwidth.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Exact model/revision/positions/layout, complete/atomic handoff, retry/recompute semantics; simulation estimates versus real two-node measurements clearly separated.
- **Industrial refinement:** Read after Chapter 84 in the proposed production order. Build a two-process KV-transfer reference with layout/version handshake, ownership acknowledgement, duplicate delivery and failure injection before any performance claim.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH83-MECHANISM-001 and FIG-CH83-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 84 — Multi-GPU Execution

- **Reader question:** Partition weights, operators, or experts across devices while accounting for collectives, memory, and failure.
- **Planned visual content / experiment:** TP all-reduce, pipeline stages, expert routing across GPUs; simulate latency by topology/batch.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Sharded versus single-device differential, collective order, partial failure, model/config identity; real measurements require exact topology.
- **Industrial refinement:** Read before Chapter 83 in the proposed production order. Separate TP/PP/DP/context/expert parallelism; add real two-worker correctness execution where hardware permits, with a CPU reference path and no GPU-scaling claim from simulation.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH84-MECHANISM-001 and FIG-CH84-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 85 — Multi-Node Inference

- **Reader question:** Extend execution and serving across machines with explicit partitioning, replication, state transfer, and failure domains.
- **Planned visual content / experiment:** Multi-node ownership/failure map and state transfer; inject loss, delay, node failure, duplicate terminal response.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Shard identity, exactly-once-visible stream semantics, lease expiry, recovery/recompute; simulation versus cluster results explicit.
- **Industrial refinement:** Extend the multi-worker contract to multi-node placement, collective failure, timeout and recovery. Require an actual distributed integration record; a simulator alone cannot establish production readiness.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH85-MECHANISM-001 and FIG-CH85-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 86 — The Inference Engine as a Database

- **Reader question:** Use database concepts to clarify planning, buffer management, indexes, operators, and distributed execution—and identify where the analogy fails.
- **Planned visual content / experiment:** Side-by-side pipelines and buffer/page mapping; apply a buffer-pool policy then identify failure on mutable KV/state.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Analogies checked against counterexamples; simulation inputs/results labeled.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH86-MECHANISM-001 and FIG-CH86-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 87 — Toward a Universal Inference Execution Protocol

- **Reader question:** Derive what a portable plan/provider/state interface would need before proposing any syntax or standard.
- **Planned visual content / experiment:** Model semantics -> planner -> provider graph and state-handle lifecycle; lower one attention step to two providers.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** Conformance oracle, unsupported capability, version mismatch, cancellation/failure, deterministic plan metadata; no performance claim from interface alone.
- **Industrial refinement:** Revisit IR and executable-plan contracts introduced in Chapters 45/48. Compare a minimal interchange boundary against concrete consumers; do not present a proposed protocol as an adopted standard.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH87-MECHANISM-001 and FIG-CH87-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 88 — What Comes After Today's Transformer Runtime?

- **Reader question:** Synthesize durable engine invariants for new architectures without predicting specific winners.
- **Planned visual content / experiment:** Inference-as-OS stack and research dependency tree; test one policy hypothesis in simulation rather than claim architecture.
- **Visual grammar:** Typed distributed state and communication map. Owners: worker / state shard / transport.
- **Required counterexample / boundary:** layout mismatch, lost owner or failed collective.
- **Test anchor:** State what evidence would falsify each proposal and which semantics must remain invariant; no speculative benchmark headlines.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH88-MECHANISM-001 and FIG-CH88-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 89 — Designing the Final Mini Engine

- **Reader question:** Freeze a coherent scope, supported model contract, architecture, ownership rules, and acceptance gates for the graduation runtime.
- **Planned visual content / experiment:** End-to-end architecture and owner/lifetime matrix; design review with failure walkthroughs.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** Acceptance matrix includes model equivalence, concurrency, security, performance, and production behavior; no code-complete claim yet.
- **Industrial refinement:** Select supported feature combinations and explicitly exclude unimplemented ones. Graduation scope is an evidence-backed engine, not a checklist of every surveyed vendor feature.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH89-MECHANISM-001 and FIG-CH89-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 90 — End-to-End Implementation

- **Reader question:** Assemble tokenizer, model loading, packed math, KV paging, batching, providers, sampling, and streaming without bypassing ownership surfaces.
- **Planned visual content / experiment:** Concrete module/call graph and one request/token/byte/owner trace; smoke workloads across supported modes.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** Build/format/unit/integration, explicit unsupported errors, leak/cancel/failure tests; collect baseline only after smoke equivalence.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH90-MECHANISM-001 and FIG-CH90-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 91 — Correctness Gate

- **Reader question:** Decide whether the integrated engine is semantically trustworthy enough to measure and expose.
- **Planned visual content / experiment:** Gate funnel and failure triage path; run boundary prompts/configs and injected errors.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** This entire chapter is the correctness record; any blocker prevents Chapter 92 performance conclusions.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH91-MECHANISM-001 and FIG-CH91-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 92 — Performance Gate

- **Reader question:** Determine where the correct engine meets, exceeds, or misses explicit workload/hardware targets.
- **Planned visual content / experiment:** Performance envelope and bottleneck timeline; required warm/cold, serial/batched, dense/paged, provider sweeps.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** Recheck outputs during benchmarks, publish manifests/raw results, estimates separated, unfavorable results retained.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH92-MECHANISM-001 and FIG-CH92-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 93 — Production Gate

- **Reader question:** Prove the engine behaves safely under load, cancellation, failures, hostile inputs, observability, and deployment lifecycle.
- **Planned visual content / experiment:** Deploy/readiness/drain lifecycle and fault-injection matrix; slow clients, OOM pressure, corrupt models, provider failure, restart/cache state.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** No leaks/deadlocks/starvation, bounded memory/queues, truthful metrics, graceful terminal behavior, recoverable rollback; publish operational evidence.
- **Industrial refinement:** Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH93-MECHANISM-001 and FIG-CH93-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).

## Chapter 94 — Replace One Hermon Component and Prove It

- **Reader question:** Demonstrate contribution-level mastery by substituting one bounded Hermon component through its real contracts and proving correctness/performance/rollback.
- **Planned visual content / experiment:** Before/after call and ownership path plus release/rollback ladder; component unit, runtime integration, real-model, concurrency, and matched performance A/B.
- **Visual grammar:** Acceptance-gate and integration diagram. Owners: component / test harness / release decision.
- **Required counterexample / boundary:** a failed gate incorrectly reported as completion.
- **Test anchor:** All Hermon gates relevant to the component, model fixture, failure/rollback, full benchmark metadata; do not claim merge/default unless repository state confirms it.
- **Industrial refinement:** Replace one narrowly bounded component only after semantic, ownership, ABI, failure and performance gates. Require a reversible integration and an honest status label.
- **Existing manifest assets (mixed status, not proof of chapter completion):** none yet.
- **Next scene IDs:** FIG-CH94-MECHANISM-001 and FIG-CH94-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).
