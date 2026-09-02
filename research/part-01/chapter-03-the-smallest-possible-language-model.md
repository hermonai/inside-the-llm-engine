# Chapter 3 Research — The Smallest Possible Language Model

Inspection date: 2026-09-02.

## Question

What is the smallest genuine numerical model that accepts a model-specific
token ID and produces one score for every possible next token, while leaving
sampling and the autoregressive loop for Chapter 4?

The Chapter 3 boundary is:

```text
text -> tokenizer -> token IDs -> embedding row -> hidden vector
     -> output projection plus bias -> vocabulary logits
```

ENGINE-1 must replace ENGINE-0's hand-authored candidate score table. Its
logits must arise from stored `f32` parameters and visible scalar arithmetic.
It must not preselect an answer and manufacture scores around it, delegate to
the former fake model, implement training, or begin Chapter 4's probability
and sampling treatment.

## Scope and truth categories

- **ENGINE-1 / Chapter 3** is the dependency-free Rust teaching model designed
  below.
- **CURRENT** describes Hermon's default reachable path at commit `472a44c`.
- **PREVIEW** describes Hermon's explicitly gated Hermon-owned paged GGUF
  forward path.
- **LIBRARY** describes present tensor/kernel components that do not establish
  default end-to-end execution by their existence alone.
- **EXTERNAL** describes primary literature, official framework contracts, and
  pinned llama.cpp source.
- **INFERENCE** marks architectural conclusions drawn by this book.

The chapter does not claim that ENGINE-1 has Hermon's model architecture. The
analogy is only the stable semantic boundary: token IDs enter numerical model
execution and vocabulary logits leave it.

## Primary sources and recorded versions

| Subject | Recorded version | Primary evidence |
| --- | --- | --- |
| Neural language models and distributed word representations | Bengio, Ducharme, Vincent, and Jauvin, JMLR 3 (2003), pp. 1137–1155 | [A Neural Probabilistic Language Model](https://jmlr.org/papers/v3/bengio03a.html) and its [paper PDF](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) |
| Embedding lookup terminology and shape | PyTorch documentation inspected 2026-09-02 | [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding) |
| Affine output projection terminology and shape | PyTorch documentation inspected 2026-09-02 | [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) |
| llama.cpp token/decode/logit contract | Hermon pin `389ff61d77b5c71cec0cf92fe4e5d01ace80b797` | `vendor/llama.cpp/include/llama.h`, `src/llama-context.cpp`, and `src/llama-sampler.cpp` |
| Hermon request-to-logits boundary | Hermon `472a44cdb511b2dae6c9569e59543db8f8350b25` | paths recorded in “Inside Hermon” below |

Bengio et al. model multiple previous words through learned distributed
representations and a neural probability function. ENGINE-1 is deliberately
smaller: it conditions only on the final token and stops at logits. The paper
supports the broader historical point that neural language models and learned
word representations predate Transformers; it is not a specification for the
teaching fixture.

The framework documentation is corroborating terminology, not a dependency.
The Rust implementation uses no PyTorch, tensor framework, or matrix library.

## What a token ID contains

A `TokenId` is a categorical identity inside one vocabulary. Its integer value
does not give a distance, direction, likelihood, or semantic magnitude. Token
7 is not intrinsically more of anything than token 6. Arithmetic on the raw ID
would impose an ordering the vocabulary does not define.

An embedding table supplies the first numerical representation. With
vocabulary size `V` and hidden dimension `D`, let:

```text
E: [V, D] f32, contiguous row-major model parameters
x: scalar TokenId, request/step state
h: [D] f32, forward activation
```

Embedding lookup is row selection:

```text
h = E[x]
```

It reads `D` contiguous values beginning at `x * D`. It is not executed as a
dense multiplication in ENGINE-1. A one-hot vector times `E` is mathematically
equivalent, but materializing `V` mostly-zero values would obscure the actual
access pattern.

## Chosen model semantics

ENGINE-1 is a first-order neural next-token model:

```text
h = E[x]
z = W h + b
```

where:

| Name | Symbol | Shape | Dtype | Owner/lifetime | Inference-mutable? |
| --- | --- | --- | --- | --- | --- |
| Embedding | `E` | `[V,D]` | `f32` | model | no |
| Output projection | `W` | `[V,D]` | `f32` | model | no |
| Output bias | `b` | `[V]` | `f32` | model | no |
| Input token | `x` | scalar | `TokenId` | request/step | new per step |
| Hidden activation | `h` | `[D]` | `f32` | forward call | new per call |
| Logits | `z` | `[V]` | `f32` | forward call/result | new per call |

For vocabulary row `i`:

```text
z_i = b_i + sum from j=0 to D-1 of W[i,j] * h_j
```

The physical row-major offset is:

```text
offset(i, j) = i * D + j
```

The scalar implementation must therefore initialize an accumulator from
`b[i]`, walk one contiguous row of `W`, multiply each value by `h[j]`, and
write `z[i]`. Model semantics stay fixed if a later chapter replaces these
loops with SIMD or accelerator kernels.

This remains a language model because its task is next-token prediction and
its output contains one score for every vocabulary token. It is a first-order
model because only the final token affects its result:

```text
P(next token | complete history) is approximated by P(next token | final token)
```

That limitation is intentional and experimentally testable.

## Full sequence versus last-token API decision

ENGINE-1 accepts the full token sequence:

```rust
fn forward(&self, input: &[TokenId]) -> Result<ForwardPass, ModelError>
```

It rejects an empty sequence, selects `input.last()`, and documents that all
earlier positions are ignored.

Reasons:

- A sequence is the semantic input to next-token prediction and already exists
  in ENGINE-0 request state.
- Passing the full history makes the model's failure visible: two histories
  with the same last token return identical logits.
- Chapter 4 can append selected tokens without replacing the model boundary.
- The borrowed slice introduces no input allocation.

A `forward_last(TokenId)` API would describe today's arithmetic more narrowly,
but it would hide the context limitation at the caller and force the runtime to
own a model-specific “which position matters?” decision. This tradeoff is
documented rather than advertised as universal future-proofing.

## Typed output and numerical policy

`ForwardPass` owns:

- the input token used;
- the copied hidden activation;
- a typed `Logits` vector.

`Logits` exposes a borrowed slice and length. It is not an anonymous
`Vec<f32>` at subsystem boundaries. Construction is model-owned and rejects a
non-finite result (`NaN`, `+Inf`, or `-Inf`) as `NonFiniteLogit`. Real engines
may deliberately permit infinities for masking or processors, but ENGINE-1
has no such operation; a non-finite affine result is therefore an error.

Tests compare complete vectors with an absolute-plus-relative condition:

```text
|actual - expected| <= absolute_epsilon
                       + relative_epsilon * |expected|
```

The hand fixture happens to use exactly representable binary values in most
terms, but the tolerance establishes the correct contract before larger
reductions and different execution orders appear.

## Parameter construction and shape validation

`TinyLanguageModel::try_new` receives explicit `V`, `D`, `E`, `W`, and `b`.
Construction rejects:

- `V == 0`;
- `D == 0`;
- overflow while computing `V * D`;
- `len(E) != V * D`;
- `len(W) != V * D`;
- `len(b) != V`;
- non-finite parameter values.

Forward rejects an empty sequence and any final `TokenId >= V`. The model is
immutable during inference: `forward` takes `&self`; metrics belong outside
the model. No global mutable parameter state exists.

The tokenizer contract gains one canonical `vocabulary_size`. Runtime
construction validates:

```text
model vocabulary size == tokenizer vocabulary size == ModelContract size
```

The small ENGINE-1 vocabulary is a separate model-specific tokenizer rather
than pretending Chapter 2's byte-BPE IDs fit a four-row matrix. The byte oracle
and BPE remain available for Chapters 1–2 labs. Vocabulary coupling is explicit
instead of silently remapping IDs.

## Numerical fixture and independent oracle

The teaching vocabulary is:

```text
0 <eos>
1 I
2 like
3 Rust
```

`V = 4`, `D = 3`. Parameters are manually provided to isolate inference; a
real model normally obtains parameter values through training.

```text
E = [
  [ 0.0,  0.0,  0.0],   # <eos>
  [ 0.0,  1.0,  0.0],   # I
  [ 1.0, -0.5,  2.0],   # like
  [-1.0,  0.0,  0.0],   # Rust
]

W = [
  [-0.5,  0.4,  0.1],   # candidate <eos>
  [ 0.2,  0.2,  0.0],   # candidate I
  [ 0.3,  0.2,  0.1],   # candidate like
  [ 1.0, -0.4,  0.25],  # candidate Rust
]

b = [-0.2, 0.0, 0.0, 0.5]
```

For input `like = TokenId(2)`:

```text
h = E[2] = [1.0, -0.5, 2.0]

z_0 = -0.2 + (-0.5*1.0) + ( 0.4*-0.5) + (0.1*2.0)  = -0.7
z_1 =  0.0 + ( 0.2*1.0) + ( 0.2*-0.5) + (0.0*2.0)  =  0.1
z_2 =  0.0 + ( 0.3*1.0) + ( 0.2*-0.5) + (0.1*2.0)  =  0.4
z_3 =  0.5 + ( 1.0*1.0) + (-0.4*-0.5) + (0.25*2.0) =  2.2
```

Expected full vector:

```text
[-0.7, 0.1, 0.4, 2.2]
```

The independent plain-Python oracle will define these parameters and equations
separately, without importing or invoking ENGINE-1. Rust tests assert the full
vector, not merely its argmax. The fixture's `Rust` embedding also makes EOS
the unique argmax after `Rust`, allowing the inherited lifecycle to demonstrate
one text token followed by terminal EOS without hard-coding a step table.

## Selection boundary

The model returns logits. A separate deterministic selector scans them and
returns the index of the largest finite value, with first-index tie breaking.
The runtime converts the selected ID to an EOS or ordinary-text `Token` using
the model/tokenizer contract.

Argmax is temporary end-to-end infrastructure, not the Chapter 3 subject. It
does not convert scores into probabilities, and its presence does not make
sampling part of `Model::forward`. Chapter 4 will replace this minimal policy
with a rigorous logits-processor/sampler treatment.

## Parameter count, bytes, and access cost

With untied embedding and projection matrices:

```text
parameters = V*D + V*D + V = 2VD + V
f32 bytes  = (2VD + V) * 4
```

For `V=4`, `D=3`:

```text
parameters = 2*4*3 + 4 = 28
bytes      = 28 * 4 = 112 bytes
```

The embedding lookup reads one row: `D` values or 12 bytes in the fixture.
The output projection reads all `V*D` weights plus `V` biases, performs `V*D`
multiplications and approximately `V*D` additions, and writes `V` logits.
Its asymptotic work is `O(V*D)`.

For a hypothetical `V=50,000`, `D=4,096`:

```text
one matrix       = 204,800,000 parameters
untied total     = 409,650,000 parameters
f32 parameter    = 1,638,600,000 bytes
                  = about 1.526 GiB
```

This is arithmetic under the simplified two-matrix-plus-bias architecture, not
a measurement or a full-model estimate. Weight tying can reuse the embedding
for the output projection and reduce parameters, but ENGINE-1 keeps matrices
separate so input lookup and output scoring remain visible. Lower precision,
quantization, and optimized matrix multiplication are deferred.

## Parameters, activations, bytes, and owners

```text
MODEL LIFETIME, shareable and inference-immutable
  E [V,D] -> W [V,D] -> b [V]

FORWARD LIFETIME, request-local
  input TokenId -> h [D] -> z [V]

STREAM LIFETIME
  selected TokenId -> decoded bytes -> UTF-8 buffer -> text event
```

One model may later serve many requests because the weight vectors are read
through `&self`. Each forward pass owns fresh hidden/logit activation vectors;
accidental cross-request mutable activation sharing is absent.

Following one projection parameter:

```text
Rust fixture literal -> ModelWeights Vec<f32> -> contiguous row-major element
                     -> CPU load -> multiplication -> f32 accumulator
```

Every tensor introduced here must have a shape, dtype, layout, owner,
lifetime, and access pattern. Chapter 5 generalizes tensors; ENGINE-1 does not
build dynamic ranks, strides, broadcasting, devices, or autograd.

## Inside Hermon

Hermon commit: `472a44cdb511b2dae6c9569e59543db8f8350b25`.
Pinned llama.cpp commit: `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

### CURRENT default path

- `crates/hermon-runtime/src/dispatch.rs:52-99` makes `RuntimeMode::Batched`
  the default when `HERMON_RUNTIME_MODE` is absent.
- `crates/hermon-runtime/src/batched.rs` tokenizes requests, builds
  `hermon_llamacpp::Batch` entries, requests logits for selected batch rows,
  calls one shared context decode, and then samples per sequence.
- `crates/hermon-engine/src/engine.rs:229-281` shows the simpler blocking path:
  tokenize, construct a sampler, call context generation, decode token pieces,
  and stream text.
- `crates/hermon-llamacpp/src/linked.rs:612-628` passes token IDs through the
  C shim to `llama_decode`; `linked.rs:758-768` does the same for batches.
- `crates/hermon-llamacpp/csrc/shim.c:333-350` constructs a llama batch, calls
  `llama_decode`, and exposes `llama_get_logits_ith`.
- `linked.rs:863-867` keeps model execution and selection conceptually
  separate at the wrapper boundary even though the sampler consumes context
  logits through llama.cpp.

Hermon currently delegates the default real-model numerical graph and logits
storage to pinned llama.cpp. Hermon owns routing, request/runtime scheduling,
batch assembly, streaming, and lifecycle around that call. This is the
industrial version of the Chapter 3 contract, not of ENGINE-1's equations.

### PREVIEW and LIBRARY

- `crates/hermon-runtime/src/dispatch.rs:67-95` and `:499-556` classify the
  Hermon-owned paged runtime and real GGUF forward as explicitly gated by
  `HERMON_RUNTIME_MODE=paged` plus `HERMON_PAGED_GGUF=1`; greedy CPU inference
  is the stated current preview limit.
- `crates/hermon-runtime/src/paged.rs` owns a model-shaped forward path whose
  differential test is
  `crates/hermon-runtime/tests/gguf_paged_differential.rs`.
- `crates/hermon-engine/src/tensor.rs` and backend/kernel crates expose useful
  components, but their presence is **LIBRARY** evidence unless a reachable
  request path selects them.

### Pinned llama.cpp contract

`vendor/llama.cpp/include/llama.h:977-989` states that logits come from the
last `llama_decode` call, are stored as rows by requested batch output, and
have `n_vocab` columns. `llama_get_logits_ith` retrieves one row. In
`src/llama-context.cpp:3384-3400`, retrieval synchronizes computation before
returning the result pointer. This grounds the input/output contract without
claiming that llama.cpp implements ENGINE-1's one-embedding/one-projection
architecture.

## Planned implementation and verification

1. Add a small `model` module with typed errors, `Logits`, `ForwardPass`,
   `TinyLanguageModel`, scalar row lookup, and scalar projection.
2. Add a four-token model-bound tokenizer and expose vocabulary size from the
   tokenizer trait.
3. Evolve `Model::candidates` into `Model::forward`; evolve the selector to
   consume `Logits` and keep EOS classification outside model arithmetic.
4. Preserve request validation, cancellation, byte decoding, strict UTF-8
   framing, exactly-one terminal outcome, and no-emission-after-terminal.
5. Add the independent Python oracle and Rust differential test.
6. Add tests for every required dimension, parameter, ID, numeric, vocabulary,
   bias, repeatability, context-limitation, EOS, and lifecycle invariant.
7. Add a modest synthetic projection-scaling experiment that reports parameter
   bytes and timings as pedagogical local observations, with no extrapolation.

## Planned diagrams

- token ID to logits;
- embedding row lookup;
- one-logit dot product;
- tiny-model tensor shapes;
- parameters versus activations and memory;
- model semantics versus scalar execution;
- same-last-token context limitation.

All diagrams will remain plain text and within the repository width gate.

## Deferred claims and Chapter 4 handoff

Chapter 3 deliberately does not implement or derive:

- softmax or probabilities;
- temperature;
- random categorical sampling or seeds;
- top-k, top-p, min-p, or logits processors;
- a complete multi-step autoregressive explanation;
- Transformer context, attention, training, backpropagation, quantization,
  general tensors, BLAS, SIMD, Metal, CUDA, or GGUF.

Chapter 4 receives a typed finite logit vector, a separate temporary argmax
selector, token history, EOS identity, maximum-token limits, and the terminal
lifecycle. It must explain how logits become a numerically stable distribution,
how a policy selects a token, how the token is appended and fed back, and how
determinism, seeds, stop conditions, and terminal ownership interact.

## Open questions

- The repository-wide license remains undecided; no external parameter artifact
  or copied implementation is introduced.
- Later chapters must decide how much activation scratch is caller-provided
  versus model/context-owned. Fresh vectors are clearest at ENGINE-1 scale.
- Chapter 4 may rename the `engine0` crate when the curriculum benefits from a
  stable package name. Chapter 3 keeps the package path to avoid a mechanical
  migration unrelated to numerical semantics; milestone identity is stated in
  documentation and Git history.
