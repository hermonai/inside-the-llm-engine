# Chapter 7 Research — Embeddings and Normalization

Research date: 2026-09-03.

Starting book commit: `2220008b8f44c83b1dbdcc8ad9cc2bec7d6fde86`.

Hermon commit: `472a44cdb511b2dae6c9569e59543db8f8350b25`.

Pinned llama.cpp/GGML commit:
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

## Research question

How should the teaching engine turn a checked `TokenId` into an independently
owned model-width activation, normalize that activation with a transparent
RMSNorm definition, expose layout and finite-range limits, and preserve a clean
boundary before Q/K/V projections?

## Scope and exclusions

Chapter 7 adds two reference operators over Tensor Substrate v1:

- embedding lookup from a logical `[V,D]` table to an owned `[D]` activation,
  plus the checked sequence generalization `[T] -> [T,D]`;
- RMSNorm from input and learned-scale views `[D]` to a canonical owned `[D]`
  activation.

The implementation remains dependency-free and entirely safe Rust. It does not
add an operator graph, a tensor framework, mixed precision, a stabilized
sum-of-squares candidate, fusion, SIMD, threads, accelerators, Q/K/V, attention,
RoPE, KV state, model loading, or quantization. ENGINE-1's historical fixture
keeps its old numerical graph; only its manual embedding loop is routed through
the checked embedding operator.

## Primary sources

| Source | Direct evidence used |
| --- | --- |
| Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) and the [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf) | RMSNorm removes LayerNorm's recentering step and uses RMS-based rescaling; the paper motivates rescaling invariance and compares the operation with LayerNorm |
| Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450) | LayerNorm computes normalization statistics within one training case and applies learned gain and bias; this is the primary basis for the bounded LayerNorm contrast |
| PyTorch, [`RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | Official operator equation places epsilon inside the square root and applies a per-element learned scale; normalization is over the declared trailing dimensions |
| PyTorch, [`Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | Official lookup-table shape is `(num_embeddings, embedding_dim)`; integer indices select dense output vectors and preserve the input index shape with an added embedding dimension |
| Meta, [Llama 4 reference `model.py`](https://github.com/meta-llama/llama-models/blob/main/models/llama4/model.py) | A current model-family reference casts activations to float for mean-square and reciprocal-square-root work, casts back, then applies learned weights; its default epsilon is a model choice, not a universal constant |
| Netlib LAPACK, [`SLASSQ`](https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq_ga0596b4bfa745d0d1c5817d4790921cda.html) | Scaled sum-of-squares representations can avoid intermediate overflow and underflow; this provides the stable-algorithm contrast without importing it into the Chapter 7 kernel |
| Rust standard library, [`f32`](https://doc.rust-lang.org/std/primitive.f32.html) | `f32` has finite range, infinities, NaN, subnormals, and constants that make the teaching kernel's storage and arithmetic policy explicit |

The external sources specify mathematics and public APIs. Claims about Hermon
and llama.cpp below come from the pinned local source, not from changing online
documentation.

## Existing book and engine boundary

Chapter 2 owns text-to-token conversion. A `TokenId` is an integer symbol under
the tokenizer/model contract; its numeric magnitude is not a semantic feature.
Chapter 3's `TinyLanguageModel` stores an embedding table and manually copies
the selected row. Chapter 5 migrated those parameters into canonical
`OwnedTensor` storage with immutable `TensorView` access. Chapter 6 added
reference strided kernels and a canonical-only blocked candidate; ENGINE-1's
output projection now calls `gemv_reference`.

The Chapter 7 implementation should therefore not invent storage, token, or
linear-algebra abstractions. It should extract the row-selection behavior into
a checked operator and add normalization beside, not inside, the tensor
substrate. The tensor types continue to describe data; operator modules define
computation.

## Embedding semantics and memory

Let `V` be vocabulary size and `D` the model or embedding dimension. The
learned table is

$$
\mathbf{E}\in\mathbb{R}^{V\times D}.
$$

For token ID $t$ with $0\le t<V$, lookup returns

$$
\mathbf{x}=\mathbf{E}_{t,:}\in\mathbb{R}^{D},
\qquad x_j=E_{t,j},\quad 0\le j<D.
$$

Canonical row-major metadata is shape `[V,D]`, element strides `[D,1]`, and
base offset zero. The physical element offset is $tD+j$. The reference operator
must nevertheless use `TensorView::get2`; for a general validated view the
actual offset is `base + t*stride[0] + j*stride[1]`. This preserves Chapter 5's
logical-index contract and supports non-unit and zero strides.

Lookup is indexing plus `D` element reads and, under the selected ownership
policy, `D` output writes. It is not a dense matrix multiplication. A one-hot
matrix product is a mathematical equivalence, not the execution algorithm
taught or implemented here.

For token sequence

$$
\mathbf{t}=[t_0,\ldots,t_{T-1}],
$$

the output is

$$
\mathbf{X}\in\mathbb{R}^{T\times D},
\qquad X_{ij}=E_{t_i,j}.
$$

Order and repeated IDs are preserved. `T=0` is a valid empty request to the
sequence operator and yields canonical shape `[0,D]`; `V=0` and `D=0` remain
invalid table contracts.

## Ownership decision

A returned row view would avoid the `O(D)` copy, but its lifetime would be tied
to model parameters and it would alias immutable weight storage. Later residual
operations need request-local mutable state. Making arbitrary mutable parameter
aliases available would be a correctness defect.

The chosen API reads a borrowed immutable parameter view and returns a fresh
canonical `OwnedTensor`. The copy is visible in both code and documentation.
Downstream mutation of the activation cannot modify the embedding table, and
the activation can outlive the lookup borrow. Sequence lookup similarly owns
one canonical `[T,D]` allocation.

## Residual-stream dimension

The first embedding row establishes a model-space vector of width `D`. Chapter
7 uses *residual stream* for the persistent model-width activation carried
through a Transformer stack. It does not imply that every internal tensor has
width `D`; it says that operations returning to this stream must satisfy the
model-width boundary. This chapter only creates and normalizes the vector.

## RMS and RMSNorm definition

For input and learned scale

$$
\mathbf{x},\mathbf{w}\in\mathbb{R}^{D},\qquad D>0,
$$

define mean square, epsilon-stabilized inverse RMS, and output as

$$
\begin{aligned}
m_2 &= \frac{1}{D}\sum_{j=0}^{D-1}x_j^2,\\
r &= \frac{1}{\sqrt{m_2+\epsilon}},\\
y_i &= x_i r w_i,\qquad 0\le i<D.
\end{aligned}
$$

Here $epsilon$ is a positive finite scalar, $r$ is one scalar shared across
the vector, and $mathbf{w}$ is learned element-wise parameter data. Epsilon is
inside the square root. The different convention
$1/(\sqrt{m_2}+\epsilon)$ is not substituted.

The zero vector is valid: `r = 1/sqrt(epsilon)` is finite and every output is
still exactly zero. `D=0` is invalid because the mean square is undefined.
The teaching API rejects epsilon zero, negative, NaN, and either infinity.
Production runtimes may validate epsilon while loading model metadata rather
than on every operator call; that changes validation placement, not semantics.

## LayerNorm distinction

LayerNorm first computes a mean and centers the activation before a
variance-like scale normalization. RMSNorm does not subtract the mean; it
normalizes by root mean square and applies a learned scale. This bounded
contrast is enough to prevent the terms from being treated as aliases. Chapter
7 does not survey normalization families or training behavior.

## Precision and numerical policy

The teaching engine has one dtype. Inputs and learned scale are stored as
`f32`; multiplication, sum-of-squares accumulation, division, square root,
reciprocal, and output are `f32`. Reduction order is increasing logical index.
There is no implicit widening or fused operation contract.

This simple policy exposes two finite-range limits:

- sufficiently small nonzero `x*x` can underflow to zero, after which epsilon
  dominates the denominator;
- a finite `x` near `1e20` can produce an infinite square in `f32`, and several
  individually finite squares can overflow their accumulated sum.

The checked reference rejects non-finite input or weight values. It also
returns typed errors for non-finite squares, reduction state, inverse RMS, or
output. It does not silently return a plausible but false normalized vector.
This is still the naïve sum-of-squares algorithm: detection is not prevention.

`SLASSQ` demonstrates a scaled representation that avoids many intermediate
overflow and underflow cases. That algorithm is deliberately deferred. Adding
it now would obscure the direct equation-to-loop lowering and create a second
candidate without present optimization pressure. A future candidate must cross
the independent-oracle and tolerance gate.

Floating-point addition is non-associative. Scalar, SIMD, threaded, and GPU
reductions may combine squares in different orders, so future equivalence uses
absolute-plus-relative tolerance and rejects non-finite disagreement rather
than requiring universal bit identity.

## Reference operator contracts

The implementation uses focused `embedding` and `normalization` modules:

```text
embedding_lookup_reference(&TensorView, TokenId)
    -> Result<OwnedTensor [D], EmbeddingError>

embedding_sequence_reference(&TensorView, &[TokenId])
    -> Result<OwnedTensor [T,D], EmbeddingError>

rms_norm_reference(&TensorView, &TensorView, f32)
    -> Result<OwnedTensor [D], NormalizationError>
```

All accept validated immutable views. Embedding requires a rank-2 positive
`[V,D]` table and checked IDs. RMSNorm requires two rank-1 equal-length views,
positive `D`, and positive finite epsilon. Both support valid nonnegative
strides, including zero-stride broadcast views. Outputs are new canonical
owners. Errors are structured enums and wrap `TensorError` for substrate
failures; ordinary invalid calls do not panic.

ENGINE-1 maps the embedding operator's out-of-range error back to its existing
`ModelError::TokenOutOfRange`, preserving the public historical contract. It
does not insert RMSNorm into the Chapter 3 fixture, because doing so would
silently change the model and logits.

## Hand fixture and experiment results

The hand fixture is

$$
\mathbf{x}=[1,-2,3,-4],\qquad
\mathbf{w}=[1,0.5,2,-1],\qquad
\epsilon=10^{-5}.
$$

Its sum of squares is `30`, mean square is `7.5`, and the oracle computes
`r = 1/sqrt(7.50001)`. Both Python and Rust will publish the full output to a
stated tolerance.

The bounded scale experiment compares positive multipliers `1e-8`, `0.1`, `1`,
`10`, and `100`. Against the `alpha=1` Python result, maximum absolute deltas
were `2.1908698`, `0.000144584`, `0`, `0.000001446`, and `0.000001460`.
Approximate invariance holds only where mean square dominates epsilon.

The magnitude sweep uses `1e-20`, `1e-10`, `1`, `1e10`, and `1e20`. The first
two demonstrate epsilon-dominated output, `1` and `1e10` normalize near unit
magnitude, and finite `1e20` produces a typed F32 square-overflow error. Four
`1e19` values separately have finite squares but overflow the running F32 sum.
These findings are preserved rather than filtered.

No timing benchmark is planned. For one `[D]` RMSNorm call, the transparent
two-pass structure reads input for reduction, reads input and weight for output,
and writes output. Under an idealized cold-payload `f32` accounting that is
approximately `16D` bytes, excluding metadata, allocation, cache effects, and
write allocation. The exact scalar operation count depends on conventions for
square root and reciprocal and is less informative than the low-reuse dataflow.
A fabricated nanosecond comparison would not improve the chapter.

## Hermon source findings

All classifications were reverified at Hermon
`472a44cdb511b2dae6c9569e59543db8f8350b25` on 2026-09-03.

- **CURRENT:** `crates/hermon-runtime/src/dispatch.rs` maps an unset
  `HERMON_RUNTIME_MODE` to `Batched`. That default constructs the llama.cpp
  backed batched runtime, so the current production path delegates embedding,
  normalization, and the rest of the model graph to the pinned native engine.
- **PREVIEW:** `HERMON_RUNTIME_MODE=paged` is explicitly gated and described by
  its error text as preview. `crates/hermon-runtime/src/paged.rs` contains
  `GgufLlamaForward`, which validates model epsilon metadata, reads embedding
  rows through `model.tensor_row_f32`, and uses a visible Rust `rms_norm` loop
  for Hermon-owned forward work.
- **LIBRARY:** `crates/hermon-gguf/src/lib.rs::model_shape` verifies that
  `token_embd.weight` begins with GGML dimension `embedding_length` and derives
  vocabulary size from the next dimension. This is model-format metadata and
  does not mean the crate executes the default graph.
- **LIBRARY/PREVIEW bridge:** `hermon-llamacpp/src/linked.rs::tensor_row_f32`
  validates row bounds and materializes one CPU-resident 1-D/2-D model-tensor
  row as `Vec<f32>`. Its C++ bridge lives in `csrc/tensor_bridge.cpp` and
  supports conversion from packed types. The paged preview consumes it; the
  default graph need not round-trip embeddings through this Rust API.

The preview Rust RMSNorm sums `x*x` into `f32`, computes
`1/sqrt(sum/n + eps)`, then applies the learned gain. It uses debug assertions
for slice agreement because surrounding model construction establishes shapes.
It is relevant evidence for architecture contrast, not an API copied into the
checked educational engine.

## Pinned llama.cpp/GGML findings

The Hermon submodule pins
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

- `src/llama-graph.cpp` builds token embeddings with `ggml_get_rows(tok_embd,
  inp_tokens)` and routes RMS normalization through `build_norm`, passing
  `hparams.f_norm_rms_eps` to `ggml_rms_norm` before learned weight/bias graph
  operations.
- `ggml/src/ggml.c::ggml_get_rows` constructs a graph node whose first physical
  extent is the selected row width and whose remaining extents follow the index
  tensor. The result is F32 except for I32 input. This is graph semantics, not
  yet CPU execution.
- `ggml/src/ggml-cpu/ops.cpp` dispatches `GET_ROWS` by storage type. F32 rows are
  copied; F16/BF16 rows are converted; supported quantized rows are
  dequantized. Row indices are asserted in range within the kernel, after
  graph/model validation.
- The same file's F32 RMSNorm kernel requires the first tensor dimension to be
  contiguous, partitions outer rows across threads, and accumulates in
  `ggml_float`, which `ggml-cpu/vec.h` defines as `double` at this revision.
  The expression forms `x[i]*x[i]` from F32 operands before casting the product
  to that wider accumulator, then divides by row width and computes
  `1/sqrtf(mean + eps)`. The inspected code, not a blanket cross-backend claim,
  supports this CPU-path statement.
- GGML stores epsilon in operator parameters and asserts it is nonnegative.
The model graph later combines normalization with learned weights; a fused
RMSNorm-plus-multiply CPU path also exists. Backend-specific execution can
differ, so Chapter 7 describes only the pinned CPU implementation in detail.

The architecture is: llama graph builder -> GGML operator/tensor metadata ->
backend dispatch -> typed CPU row or RMSNorm kernel. This is more layered than
the teaching function but computes the same declared convention for the
supported F32 case.

## Correctness coverage

Thirty new Rust tests cover the Chapter 7 contracts. Embedding tests cover
first/middle/last IDs, `D=1`, `D>1`, wrong rank, empty
vocabulary, empty model dimension, out-of-range IDs, strided and zero-stride
tables, output ownership, sequence order/repetition, empty sequences, and an
invalid sequence member.

RMSNorm tests cover `D=1`, a hand vector, mixed signs, zero and uniform vectors,
non-unit weights, strided and zero-stride inputs/weights, wrong ranks, length
mismatch, `D=0`, every invalid epsilon class, non-finite values, finite large
values, square and reduction overflow, tiny-square underflow, output overflow,
and deterministic approximate scale invariance. All previous Rust and Python
regressions remain gates.

## Canonical diagrams

Fifteen Transformer diagrams answer the required questions: token to
model space, logical and physical table layout, parameters versus activations,
view versus copy, residual width, RMS derivation, two-pass lowering,
equation/shape/memory/loop mapping, zero-vector epsilon behavior, LayerNorm
contrast, precision flow, embedding versus output projection, the Chapter 7
engine boundary, and source-classified Hermon/llama.cpp execution mapping.

## Terminology

- **embedding table:** learned parameter tensor mapping vocabulary IDs to
  model-width rows;
- **embedding lookup:** checked row selection, not dense multiplication;
- **model dimension (`D`):** width of embeddings and the residual stream;
- **residual stream:** persistent model-width activation carried between
  Transformer sublayers;
- **root mean square (RMS):** square root of the arithmetic mean of squares;
- **RMSNorm:** RMS rescaling followed by learned element-wise scale;
- **epsilon:** positive scalar inside the RMSNorm square root in this contract;
- **learned scale:** immutable model parameter vector applied element-wise;
- **reduction precision:** dtype used to accumulate the sum of squares.

## Open questions deliberately deferred

- Whether a stabilized scaled-sum RMSNorm candidate is worthwhile under real
  model activation ranges.
- Mixed storage and accumulation dtypes, vector reduction trees, fusion, and
  backend dispatch.
- Whether a production engine should expose borrowed embedding rows for a
  strictly immutable consumer path.
- Weight tying between embeddings and output projection.
- Batch/sequence scheduling, device placement, and distributed embeddings.
- The projections that produce Q, K, and V, their orientation, and their
  head-shaped outputs. These are exactly the Chapter 8 boundary.
