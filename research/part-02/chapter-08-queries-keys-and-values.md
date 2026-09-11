# Chapter 8 Research — Queries, Keys, and Values

**Status:** COMPLETE (verification closed 2026-09-12)

**Inspection date:** 2026-09-11

**Book baseline:** `3febdd4f0a5e75a8c53e83f706beb7e339403da8`

**Hermon baseline:** `2a3fd5214e17ca7283656108847d02dcd6cdf7c5`

**Pinned llama.cpp/GGML:** `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`

## Research question

Given one normalized residual activation, what exact learned transformations,
shape contracts, layouts, ownership rules, and verification evidence produce
raw query, key, and value heads without crossing into position or attention?

Chapter 8 must make every projected element traceable from an input component
and a weight row through Chapter 6 GEMV into a metadata-only head reshape. It
must stop before RoPE, query/key comparison, score scaling, masking, softmax,
value aggregation, or KV caching.

## Repository baseline verified

- Tensor Substrate v1 owns canonical row-major `f32` payloads and permits
  validated immutable strided and zero-stride views.
- `TensorView::reshape_view` is metadata-only, requires exact canonical
  row-major input, preserves the element count, and never copies payload.
- `gemv_reference` implements `[M,K] x [K] -> [M]`, accepts valid strided
  immutable views, accumulates in increasing-index `f32`, and returns a fresh
  canonical owner.
- Chapter 7 lookup returns an owned residual activation; `rms_norm_reference`
  returns a fresh normalized `[D]` owner.
- The historical ENGINE-1 fixture still maps `like` to hidden
  `[1.0,-0.5,2.0]`, logits approximately `[-0.7,0.1,0.4,2.2]`, then emits
  `Rust` followed by EOS.
- The incoming baseline has 178 Rust unit/integration tests and two compile-fail
  doctests, three chapter visual parity gates and 45 vector plates. The original
  September 4 research plan predated the Chapter 5–7 visual regeneration.

## Primary sources

### Transformer projections and multi-head geometry

- Vaswani et al., [Attention Is All You
  Need](https://arxiv.org/abs/1706.03762), especially section 3.2.2. The paper
  defines independently learned query, key, and value projections per head,
  concatenates head outputs, and uses model width 512 with eight 64-wide heads
  in its base configuration. Chapter 8 uses only the projection and head-shape
  premises, not the attention computation that follows.

### Multi-query attention

- Shazeer, [Fast Transformer Decoding: One Write-Head is All You
  Need](https://arxiv.org/abs/1911.02150). Multi-query attention keeps multiple
  query heads while sharing one key head and one value head. The paper's
  performance claims belong to its reported workload; this chapter uses MQA
  only as a shape and parameter-count architecture.

### Grouped-query attention

- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). GQA partitions
  query heads into groups, with one key and value head per group. One KV group
  gives MQA; as many KV groups as query heads gives MHA. The book's checked
  configuration therefore requires query-head divisibility by KV-head count.

### Official Llama implementation

- Meta's [Llama 3 model
  definition](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py)
  constructs bias-free Q output width `n_heads * head_dim` and K/V output width
  `n_kv_heads * head_dim`, then views results as token, head, and head-dimension
  axes. Its configuration defaults `n_kv_heads` to `n_heads`, assumes
  divisible grouping in integer divisions, and derives `head_dim = dim / n_heads`.
  Re-inspected the official raw model source on 2026-09-11; the educational
  API explicitly rejects non-divisible geometry instead of relying on it.
- This source supports the educational bias-free choice and generalized head
  geometry. It does not establish that every Transformer architecture is
  bias-free or that all model families derive head width identically.

## Mathematical specification

Let:

- `D` be model/residual width;
- `H_q` be query-head count;
- `H_kv` be shared key/value-head count;
- `D_h` be one head's width;
- `P_Q = H_q D_h` be flat query width;
- `P_KV = H_kv D_h` be flat key and value width.

The selected educational geometry requires:

```text
D > 0
H_q > 0
H_kv > 0
D_h > 0
D = H_q * D_h
H_q % H_kv = 0
```

Every product is checked before it becomes a shape or allocation request.
Given normalized residual `x_hat:[D]`, Chapter 6 orientation determines:

```text
W_Q:[P_Q,D]   q_flat = W_Q x_hat   [P_Q]
W_K:[P_KV,D]  k_flat = W_K x_hat   [P_KV]
W_V:[P_KV,D]  v_flat = W_V x_hat   [P_KV]
```

For projection kind `A` with output width `P`:

```text
a_i = sum over j=0..D of A[i,j] * x_hat[j],  0 <= i < P
```

This is semantic composition of `gemv_reference`, not a fourth numerical
kernel. The flat result is regrouped without moving payload:

```text
q_flat:[H_q*D_h]  -> Q:[H_q,D_h]
k_flat:[H_kv*D_h] -> K:[H_kv,D_h]
v_flat:[H_kv*D_h] -> V:[H_kv,D_h]

Q[h,j] aliases q_flat[h*D_h+j]
```

## Selected educational API

- Add a semantic `qkv` module above `linear`.
- `QkvConfig::try_new(D,H_q,H_kv,D_h)` validates positive dimensions,
  checked widths, `D=H_q*D_h`, and divisible grouping.
- `project_qkv_reference` accepts immutable input and weight views, validates
  their semantic ranks and exact shapes, invokes three existing GEMVs, and
  returns a named `QkvProjection`.
- `QkvProjection` owns three canonical tensors with shapes `[H_q,D_h]`,
  `[H_kv,D_h]`, and `[H_kv,D_h]`.
- Converting GEMV's owned flat `[P]` result to `[H,D_h]` transfers the same
  `Vec<f32>` allocation into new canonical metadata. There is no payload copy.
- Checked head access returns a rank-1 borrowed view of one activation row.
  The view aliases Q/K/V activation storage, never model parameter storage.
- No projection bias is accepted. This is a declared Llama-3-aligned model
  choice, not a statement about generic linear layers.
- Sequence Q/K/V shapes are derived and taught, but no sequence operator,
  prefill runtime, position operator, or attention computation is added.
- No packed candidate or timing benchmark is justified. The chapter records
  analytical FLOPs, parameter counts, and compulsory payload terms instead.

## Geometry classes

| Geometry | Condition | Query shape | Key/value shape |
| --- | --- | --- | --- |
| MHA | `H_q = H_kv` | `[H_q,D_h]` | `[H_q,D_h]` |
| MQA | `H_q > 1`, `H_kv = 1` | `[H_q,D_h]` | `[1,D_h]` |
| GQA | `1 < H_kv < H_q`, divisible | `[H_q,D_h]` | `[H_kv,D_h]` |

The group size `G = H_q/H_kv` describes future query-to-KV mapping. Chapter 8
validates and reports it but performs no matching or sharing operation.

## Analytical cost model

Under the conventional two-FLOP multiply/add model, one `[P,D] x [D]`
projection costs approximately `2PD` FLOPs. Three projections therefore cost:

```text
F_QKV approximately 2D(P_Q + 2P_KV) FLOPs
```

For MHA, `P_Q=P_KV=D`, giving approximately `6D^2` FLOPs and `3D^2`
bias-free parameters. For GQA the parameter count is
`D(P_Q+2P_KV)`. These are algorithmic counts, not retired instructions or
timings.

The transparent one-token path reads the shared input for each GEMV, reads all
three matrices, and writes three outputs. A packed graph might reuse dispatch
or input preparation, but no universal speed claim follows without a measured
candidate and complete benchmark record.

## Hermon findings

All findings below are pinned to Hermon
`2a3fd5214e17ca7283656108847d02dcd6cdf7c5`.

### CURRENT default path

- `crates/hermon-runtime/src/dispatch.rs` maps an unset
  `HERMON_RUNTIME_MODE` to `RuntimeMode::Batched`.
- `runtime_for` constructs `BatchedRuntime` around the llama.cpp model/context.
  Q/K/V execution on the default path is therefore delegated to llama.cpp;
  Hermon's Rust paged component code is not the default operator path.

### PREVIEW paged model path

- `crates/hermon-runtime/src/paged.rs::GgufLlamaForward::new` reads
  `embedding_length`, `head_count`, `head_count_kv`, and derived `head_dim`
  through `hermon-gguf`; it validates `H_q*D_h=D` and checked `H_kv*D_h`.
- It validates separate `blk.N.attn_q.weight`, `attn_k.weight`, and
  `attn_v.weight` tensor geometry. At the GGML boundary dimensions are recorded
  `[input_width,output_width]`, the reverse visual order from this book's
  semantic row-major `[output,input]` matrices.
- `forward_layer` normalizes token rows, submits Q/K/V names together through
  `project_bundle`, then separately applies position and continues into
  attention. The projection bundle is the only part relevant to Chapter 8.
- This path requires explicit paged runtime and GGUF feature gates, so it is
  PREVIEW rather than CURRENT.

### LIBRARY bridge

- `hermon-llamacpp::LlamaModel::matmul_bundle_with_session` accepts several
  named packed GGML weights with one shared F32 row batch. It validates common
  input width and individual output lengths, returns separate owned F32
  vectors, and reuses a `TensorSession`.
- `csrc/tensor_bridge.cpp::matmul_bundle_impl` copies the shared F32 input once,
  creates one `ggml_mul_mat` node per named weight, places them in one graph,
  computes it, and copies each output to caller-owned Rust vectors.
- The bridge is an integrated PREVIEW dependency of `GgufLlamaForward` and a
  reusable LIBRARY surface in its own right. It is not the default runtime.

## Pinned llama.cpp/GGML findings

The Hermon submodule resolves to
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

- `src/models/llama.cpp` normalizes a layer input, calls
  `llm_graph_context::build_qkv`, then applies RoPE to Q/K and passes Q/K/V to
  later attention construction. This ordering confirms the clean Chapter 8 to
  Chapter 9 boundary.
- `src/llama-graph.cpp::build_qkv` computes flat widths from head counts. It
  supports a fused `layer.wqkv` tensor or separate `wq`, `wk`, and `wv`
  tensors; optional architecture-specific biases and clamps can follow.
- Separate weights lower through `build_lora_mm`, whose base operation is
  `ggml_mul_mat`. Results are reshaped to GGML logical dimensions
  `[D_h,H_q,T]`, `[D_h,H_kv,T]`, and `[D_h,H_kv,T]`.
- A fused result is partitioned with offset `ggml_view_3d` nodes. Thus head
  formation can be metadata over one packed result, while the separate path
  uses three projections and three reshape nodes.
- `src/llama-model.cpp::create_tensor_qkv` accepts fused or separate tensor
  names and optional biases. `src/llama-model.cpp` reads KV-head metadata with
  query-head count as the default. `src/llama-hparams.cpp` derives K/V GQA
  widths as per-head width times KV-head count.
- `src/llama-arch.cpp` classifies attention Q/K/V weights as repeating-layer
  tensors associated with generic `GGML_OP_MUL_MAT` execution.

The model graph knows the semantic names Q, K, and V and their head geometry.
Once nodes reach backend scheduling, multiplication is a generic tensor
operation selected by dtype, layout, shape, and backend. This is the same
layering the educational API makes explicit at smaller scale.

## Real-model storage contrast

- The relevant Llama GGUF path normally names separate tensors
  `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, and `blk.N.attn_v.weight`.
- Pinned llama.cpp also supports architecture-specific combined
  `blk.N.attn_qkv.weight` storage.
- GGML tensors may remain quantized and backend-resident; their first physical
  dimension is the input width expected by `ggml_mul_mat`.
- The book uses three unpacked canonical F32 matrices in semantic `[P,D]`
  orientation. It teaches operator meaning, not GGUF byte layout or a promise
  of one universal production representation.

## Independent oracle plan

`code/reference/python/chapter08_qkv_oracle.py` expresses projection with
row dot products and reshape through nested Python lists. It will verify:

- a non-identity hand-computable MHA example;
- MHA, MQA, and GQA output shapes;
- flat index `h*D_h+j` to logical head coordinate;
- an embedding -> RMSNorm -> Q/K/V component fixture;
- identical raw projections for identical vectors labeled with two positions;
- representative invalid geometry and shape conditions.

The primary oracle uses Python's `math.fsum`; Rust comparisons use an explicit
absolute/relative tolerance because the engine inherits ordered F32 GEMV.

## Test plan

- Valid geometry: one/multiple heads, `D_h=1`, non-power-of-two `D`, MHA, MQA,
  GQA, mixed signs, negative and zero weights.
- Geometry failures: every zero dimension, checked query/KV width overflow,
  model-width disagreement, and non-divisible grouping.
- Tensor failures: wrong input/weight rank and each independent input/output
  width mismatch for Q, K, and V.
- Layout: strided input, independently strided Q/K/V weights, and a meaningful
  immutable zero-stride case.
- Shape/view: every flat-to-head coordinate, first/middle/last heads, `D_h=1`,
  out-of-range head, and aliasing of head views with activation storage.
- Ownership: output mutation cannot affect weights or normalized input.
- Composition: checked embedding -> RMSNorm -> Q/K/V -> head view agrees with
  the independent oracle.
- Limitation: identical normalized values at two conceptual positions produce
  identical raw Q/K/V.

## Planned canonical diagrams

The original eighteen topic prompts below were consolidated into ten canonical
plates rather than eighteen repetitive charts. `figures/chapter08.py` and
`figures/src/ch08-*.json` define them; SVG is the offline web surface and native
TikZ is the print surface. Generated text files are accessible descriptions,
not published character graphs. No animation is needed for this bounded
projection chapter; every mechanism is readable in static print.

1. normalized residual to three projections;
2. semantic operator to three GEMVs;
3. parameter versus activation ownership;
4. standard MHA geometry;
5. flat projection to head reshape;
6. physical head storage;
7. model width versus head width;
8. MHA versus MQA versus GQA;
9. GQA projection dimensions;
10. one-token component pipeline;
11. sequence projection shapes;
12. decode GEMV versus prefill GEMM;
13. shared input reuse;
14. separate versus packed QKV;
15. equation/shape/GEMV/head-view lowering;
16. Chapters 5-8 component composition;
17. position-blind Chapter 8 boundary;
18. source-classified Hermon/llama.cpp architecture map.

## Planned labs

Labs 39-48 cover one projection, all three projections, head reshape,
physical offsets, failure contracts, MHA/MQA/GQA geometry, component
composition, oracle comparison, position blindness, and analytical cost. The
existing future lab placeholders move forward without renumbering Labs 1-38.

## Open questions deliberately deferred

- Position indices, frequency schedules, coordinate pairs, and RoPE.
- Query/key compatibility, score scaling, masks, attention softmax, and value
  aggregation.
- Mapping query heads to KV heads during attention and all KV-cache storage.
- Sequence/prefill operator implementation and runtime scheduling.
- Fused/packed QKV candidates, quantized weights, mixed precision, SIMD, BLAS,
  accelerators, and backend dispatch.
- Architecture-specific projection biases, clamps, Q/K normalization, unequal
  key/value head widths, tensor parallelism, and low-rank adapters.

## Technical review — 2026-09-12

This is a recorded author review, not an independent external peer review.

- **Software:** private configuration fields protect validated geometry. All
  semantic shapes are checked before GEMV. Both dimension products and head
  bounds are checked before dependent offset arithmetic. Owned output transfer
  and borrowed cell addresses are tested; strided/broadcast operands remain valid.
- **ML:** raw QKV has no position or attention semantics. MHA/GQA/MQA preserves
  query count and changes K/V projection widths; same token ID is distinguished
  from same later-layer activation. Norm gain and epsilon precede the composed
  fixture; raw hand values are not substituted for normalized results.
- **Systems:** source-level operand accesses, unique payload, physical traffic,
  graph bundling, concatenation and fusion are explicitly different. Future
  dense KV counts use F16, while the actual teaching operator uses F32. No
  throughput, concurrent-user or quality claim is inferred from byte counts.
- **Graduate/research:** exact counting convention, rounding tolerance and
  analytical assumptions are declared. The initial oracle error bound is
  fixture-specific; no arbitrary-width stability proof is claimed. Projection
  propagates IEEE non-finite values and allocation failures retain Vec semantics.
- Re-inspected `dispatch.rs`, `paged.rs`, `linked.rs`, `tensor_bridge.cpp`,
  the native bridge equivalence test, `docs/CORE_ENGINE_ARCHITECTURE.md`, and
  pinned llama.cpp `build_qkv`. The real-model bridge test was inspected, not
  executed; no external model fixture or new performance candidate was used.

## Editorial and rendered review — 2026-09-12

- **Beginner:** the narrative introduces vector roles before equations, then
  traces a nonsymmetric hand example before ownership and production graphs.
  Ten illustrations answer named questions; the cost plate is an analytical
  table, not a speed chart. The Chapter 9 boundary is explicit.
- Checked 5,215 words, ten manuscript embeddings, 22 new display equations,
  cross-links, glossary and stable math IDs. Labs 39–48 share a workbench with
  separately numbered expected artifacts and deliberate breaks. Future lab
  placeholders moved to 49–61 without renumbering Labs 1–38.
- Reviewed all 19 standalone chapter pages and all ten plates in color and
  grayscale; native edition rendered across all 169 pages. A note touching
  the journey panel was moved below it, and the browser now rejects text
  crossing a Chapter 8 panel border. Equations and inline code stay on-page.
- Fixed a publication-link parser defect exposed by half-open intervals:
  an opening bracket in math could previously consume a later figure link.
  The parser now excludes nested opening brackets; the chapter build exercises
  the failing case. Native TikZ generation remains fail-closed and deterministic.

## Reproduction and outcome

- `cargo fmt --all -- --check`, `cargo check --workspace --all-targets`,
  `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`:
  193 unit/integration tests plus two compile-fail doctests; 15 new QKV tests.
- `scripts/check-qkv-visual-parity.py`: exact raw outputs; maximum composed
  absolute errors Q=3.33e-7, K=1.59e-7, V=1.18e-7; full vectors, head view,
  offsets, cost counts and ten scene links checked. Changed/truncated vectors
  are rejected. Existing three visual parity gates remain passing.
- `figures/build.py --check`: 55 scenes / 117 deterministic artifacts;
  `scripts/check-native-figures.py`: 55 native TikZ counterparts.
- Native edition 169 pages; conventional vector edition 164; standalone
  Chapter 8 19; atlas 55. Vector/glyph, page-bounds, offline SVG/MathML,
  1024/768/390px, keyboard/reduced-motion and panel-boundary gates pass.

Remaining limitation: source inspection and toy numerical parity do not prove
real-model compatibility or industrial serving readiness. The next executable
chapter is scalar RoPE; attention and caching remain unimplemented here.
