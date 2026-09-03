# Glossary

This is the initial reader-facing glossary. Chapter numbers indicate the first
planned introduction; definitions will gain cross-links as manuscript files
land.

### ABI

**Short:** An application binary interface between compiled components.
**Precise:** The calling convention, type layout, ownership, versioning, and
error contract that allows separately compiled host and native kernel code to
interoperate. First introduced: Chapter 36. Related: kernel, provider, arena.
Common confusion: an ABI is not merely a list of function names.

### Arena

**Short:** A larger allocation from which smaller hot-path regions are carved.
**Precise:** An allocator with an explicit lifetime and alignment policy that
amortizes general allocation and can make ownership auditable. First introduced:
Chapter 37. Related: workspace, block pool. Common confusion: arena allocation
does not make memory bounds or lifetime errors impossible.

### Artifact / model artifact

**Short:** Persistent serialized data from which a model can be loaded.
**Precise:** Configuration, tokenizer data, tensor metadata, and weight bytes
whose format and revision must be interpreted and validated before execution.
First introduced: Chapter 1. Related: running model, GGUF, weights. Common
confusion: an artifact is inert representation, not a process that can answer a
request.

### Arithmetic intensity

**Short:** Floating-point work divided by bytes moved across a named memory
boundary. **Precise:** Chapter 6 uses ideal compulsory-payload FLOP/byte models
to reason about potential reuse; actual cache or DRAM traffic requires separate
measurement. First introduced: Chapter 6. Related: FLOP, working set, Roofline
model. Common confusion: ideal logical bytes are not measured hardware traffic.

### Batch / continuous batching

**Short:** A batch is work executed together; continuous batching rebuilds that
work at iteration boundaries as requests arrive and finish.
**Precise:** A physical token batch can contain prompt or decode positions from
multiple logical sequences. First introduced: Chapters 25–26. Related: sequence,
prefill, decode, scheduler. Common confusion: continuous batching is not the
same as waiting to fill a static request batch.

### Backend

**Short:** An implementation of model operations for a hardware/runtime
substrate.
**Precise:** A backend supplies supported operations, shapes, dtypes, placement,
workspace, execution, synchronization, errors, and fallback behavior for CPU,
Metal, CUDA, or another substrate. First introduced: Chapter 1; formalized
Chapter 45. Related: provider, kernel. Common confusion: compiling a backend
does not prove that a request selected it or that it is faster for a shape.

### Block table

**Short:** A mapping from a sequence's logical token blocks to physical storage
blocks.
**Precise:** Ordered metadata containing physical block identifiers and logical
length, interpreted by paged attention and lifetime management. First
introduced: Chapter 29. Related: logical block, physical block, KV block.
Common confusion: it does not contain the KV vectors themselves.

### Blocking / tiling

**Short:** Partitioning an operation into bounded subproblems intended to
improve data reuse. **Precise:** ENGINE-2 tiles the M, K, and N axes of canonical
row-major GEMM and clamps every tail while accumulating into a fresh output.
First introduced: Chapter 6. Related: working set, cache locality, matrix
multiplication. Common confusion: a tile size is workload- and machine-specific,
not a universally faster constant.

### Context

**Short:** The model/runtime state needed to advance one or more sequences.
**Precise:** Depending on an engine, a context may own execution scratch, KV
state, sequence identifiers, device resources, and sampler-related state. First
introduced: Chapter 20. Related: sequence, KV cache. Common confusion: context
window (a token limit) and execution context (a state owner) are different.

### Concurrency

**Short:** The number of requests simultaneously admitted or active.
**Precise:** A workload property that affects queueing, memory, scheduling, and
throughput but need not equal one physical execution batch's size. First
introduced: Chapter 1. Related: request, batch, throughput. Common confusion:
concurrency does not prove parallel hardware execution at every instant.

### Control plane / data plane

**Short:** The control plane decides and coordinates; the data plane moves and
transforms inference data.
**Precise:** Validation, routing, admission, scheduling, stopping, and accounting
are control work, while loading weights, executing model operations, moving
state, and emitting bytes are data work. First introduced: Chapter 1. Related:
inference engine, provider, backend. Common confusion: these roles need not be
separate processes or machines.

### Copy-on-write (COW)

**Short:** Share storage for reads, but copy it before a writer mutates shared
state.
**Precise:** A partial shared KV tail must become privately owned before a
continuation appends into unused positions. First introduced: Chapter 32.
Related: prefix cache, refcount. Common confusion: complete immutable blocks do
not need copying merely because they are shared.

### Decode

**Short:** The workload phase that advances active sequences with newly
generated token positions.
**Precise:** Decode usually presents few query tokens per sequence while reading
an increasing history of KV state; its arithmetic intensity and scheduling
shape differ from prefill. First introduced: Chapter 1 as a preview; formalized
Chapter 20. Common confusion: token decoding into text bytes is a separate
tokenizer task.

### Differential test

**Short:** A controlled comparison of two independent implementations.
**Precise:** It supplies identical inputs, defines tolerances/ordering, and
compares an optimized, concurrent, or real-model path with an oracle. First
introduced: Chapter 61. Related: oracle, equivalence. Common confusion: comparing
a function with itself under another flag may not be independent.

### Embedding

**Short:** A vector representation selected for a token identifier.
**Precise:** A checked token ID selects one row from learned parameter table
`[vocab_size, model_dimension]`; Chapter 7 materializes that row as an owned
model-width activation. First introduced: Chapter 3; formalized Chapter 7.
Related: token, hidden state, residual stream. Common confusion: an embedding
is neither a token's human-language definition nor a dense matrix product.

### Epsilon

**Short:** A configured scalar that defines a numerical operator near a
singularity. **Precise:** Chapter 7 RMSNorm requires finite `epsilon > 0` and
adds it to mean square inside the square root, making zero-vector normalization
defined. First introduced: Chapter 7. Related: RMSNorm, model metadata. Common
confusion: epsilon's value and placement are part of model semantics, not
interchangeable implementation decoration.

### Eviction / pin / residency

**Short:** Residency says where bytes live; a pin prevents temporary eviction;
eviction removes a resident copy under policy.
**Precise:** These are placement states distinct from ownership and reference
lifetime. First introduced: Chapters 33 and 57. Common confusion: an unpinned
page is eligible for eviction, not necessarily unowned or immediately free.

### Expert / mixture of experts (MoE)

**Short:** An MoE layer routes each token to a subset of alternative parameter
blocks called experts.
**Precise:** Gating selects top-k expert feed-forward transformations, changing
weight residency, I/O, batching, and failure economics. First introduced:
Chapter 54. Common confusion: sparse expert activation does not imply the whole
model or its storage is sparse.

### FLOP / FLOP/s / GFLOP/s

**Short:** FLOP counts floating-point work; FLOP/s and GFLOP/s are rates.
**Precise:** Chapter 6 models GEMM as approximately $2MKN$ FLOPs and reports an
effective rate from elapsed time; one GFLOP/s is $10^9$ FLOP/s. First
introduced: Chapter 6. Related: arithmetic intensity, benchmark. Common
confusion: operation estimates, retired instructions, and elapsed time are not
interchangeable.

### GEMV / GEMM / matrix multiplication

**Short:** GEMV multiplies a matrix by a vector; GEMM multiplies two matrices.
**Precise:** ENGINE-2 defines GEMV as `[M,K] × [K] -> [M]` and GEMM as `[M,K] ×
[K,N] -> [M,N]`, with contracts narrower than full BLAS. First introduced:
Chapter 6. Related: dot product, inner dimension, kernel. Common confusion:
single-vector and multi-column workloads offer different weight reuse.

### GGML / GGUF

**Short:** GGUF is a model container format associated with the GGML ecosystem.
**Precise:** A GGUF file contains typed metadata, tensor descriptors, alignment,
and packed tensor bytes; GGML also names tensor/runtime conventions and kernels.
First introduced: Chapters 14–15. Common confusion: parsing GGUF does not prove
correct execution of every encoded architecture or quantization.

### GQA / MHA / MQA

**Short:** Attention head geometries with, respectively, grouped, one-to-one,
or globally shared KV heads relative to query heads.
**Precise:** The query-head-to-KV-head mapping changes KV shape and bandwidth
without eliminating distinct query heads. First introduced: Chapter 8. Common
confusion: fewer KV heads do not mean fewer query outputs.

### Hidden state

**Short:** The per-token vector transformed through model layers.
**Precise:** An activation with shape commonly `[tokens, hidden_dim]` whose
storage, dtype, and batching layout vary by execution plan. First introduced:
Chapter 3. Related: embedding, logits. Common confusion: it is not the same as
KV state or recurrent state.

### Inference engine

**Short:** The system that advances generation requests to observable terminal
outcomes using a running model.
**Precise:** It owns or coordinates model resolution, admission, request state,
execution, selection, stopping, streaming, failure, accounting, and resource
release. First introduced: Chapter 1. Related: inference runtime, server,
running model. Common confusion: the engine is not identical to model weights,
a forward function, a hardware backend, or an HTTP server.

### Inference runtime

**Short:** Stateful machinery that advances one or more inference requests.
**Precise:** A runtime owns active request progress and execution resources;
this book uses *engine* for the wider request-to-outcome responsibility, while
external projects may use the terms differently. First introduced: Chapter 1.
Related: inference engine, request, state. Common confusion: a language runtime
and an inference runtime are different uses of “runtime.”

### Inter-token latency (ITL)

**Short:** Elapsed time between consecutive observable output events or tokens.
**Precise:** A distribution over named emission boundaries whose token/piece
unit, aggregation, concurrency, and workload must be disclosed. First
introduced: Chapter 1. Related: latency, TTFT, throughput. Common confusion: ITL
does not include the initial wait for the first token.

### Kernel

**Short:** A bounded numerical or data-movement operation.
**Precise:** A kernel has explicit input/output layouts, dtype, workspace,
provider constraints, error behavior, and correctness contract. First
introduced: Chapter 6; native boundary formalized Chapter 35. Related: provider,
ABI. Common confusion: kernels can run on scalar CPU, SIMD CPU, or accelerators.

### Latency

**Short:** Elapsed time between two named lifecycle events.
**Precise:** A measurement such as arrival-to-admission, queue delay, TTFT, ITL,
or request completion, reported with endpoints, workload, concurrency, and
distribution. First introduced: Chapter 1. Related: TTFT, ITL, throughput.
Common confusion: an unlabeled “latency” number is not self-defining.

### KV cache / KV block

**Short:** Per-layer key and value vectors retained for earlier positions; a KV
block is a fixed-capacity physical unit of that storage.
**Precise:** For each layer and visible prior token, the cache preserves K/V
projections so decode need not recompute them. First introduced: Chapter 21;
blocked layout Chapter 29. Common confusion: KV caching does not cache logits or
avoid attention reads over history.

### Logit

**Short:** An unnormalized score for a vocabulary item.
**Precise:** The model output vector before probability normalization and
sampling, commonly shape `[vocab_size]` per selected position. First introduced:
Chapter 3. Related: sampler. Common confusion: logits are not probabilities.

### Logical block / physical block

**Short:** A logical block is a sequence-relative token range; a physical block
is allocator-owned storage.
**Precise:** A block table maps logical order to reusable physical identifiers,
allowing non-contiguous storage and sharing. First introduced: Chapter 29.
Common confusion: equal logical indices in two sequences need not map to the
same physical block.

### Online softmax

**Short:** A numerically stable softmax accumulated without storing every
score.
**Precise:** Streaming updates maintain a running maximum, normalization sum,
and rescaled value accumulator. First introduced: Chapter 40. Common confusion:
it changes storage and reduction organization, not attention semantics.

### Oracle

**Short:** An independent, clarity-first correctness reference.
**Precise:** A scalar or trusted implementation designed to reveal defects in a
more optimized path and paired with explicit tolerances. First introduced:
Chapter 3; formalized Chapter 60. Related: differential test. Common confusion: the production path
cannot validate itself merely by being deterministic.

### Paged attention / paged KV

**Short:** Attention over KV addressed through block tables rather than one
contiguous sequence allocation.
**Precise:** Logical positions are translated to fixed-capacity physical blocks,
enabling allocation flexibility and prefix sharing while preserving causal
attention semantics. First introduced: Chapters 29 and 34. Common confusion:
“paged” does not imply OS virtual-memory faults or NVMe storage.

### Prefill

**Short:** The workload phase that processes prompt positions and creates
reusable state.
**Precise:** Prefill often has many query tokens and matrix-matrix-friendly
work, unlike one-step decode. First introduced: Chapter 1 as a preview;
formalized Chapter 20. Common confusion: prefill is not only tokenization or
model loading.

### Prefix cache / radix tree

**Short:** A prefix cache reuses computed state for matching token prefixes; a
radix tree is a compressed prefix index.
**Precise:** Cache entries have independent ownership, eviction, and partial
tail rules. First introduced: Chapter 31. Common confusion: text equality is
insufficient if tokenization/model/configuration differs.

### Provider

**Short:** A selectable inference capability or destination.
**Precise:** A provider exposes routing/placement identity, model availability,
credentials or locality where relevant, and a capability that may be realized
by one or more hardware backends. First introduced: Chapter 1; hardware-provider
contracts formalized Chapter 45. Common confusion: a provider route and the
backend selected beneath it are not the same decision.

### Quantization

**Short:** A representation that encodes weights or state with fewer/lower-
precision values plus reconstruction metadata.
**Precise:** Block layouts, scales, zero points, grouping, accumulation dtype,
and compatible kernels jointly define semantics. First introduced: Chapter 16.
Common confusion: “4-bit” does not fully identify a format or quality level.

### RoPE

**Short:** Rotary positional embedding applied to paired query/key dimensions.
**Precise:** Position-dependent rotations encode relative positional structure
without adding a learned vector to the hidden state. First introduced: Chapter
9. Common confusion: implementation details such as dimension ordering and
scaling are part of model support.

### Request / inference request

**Short:** One bounded generation intent and its mutable lifecycle state.
**Precise:** It joins input/configuration with admission, generation history,
selection and stopping state, cancellation, timings, output ordering, and a
terminal outcome. First introduced: Chapter 1; state machine formalized Chapter
24. Related: sequence, stream, terminal. Common confusion: a request is not the
same as a conversation, sequence slot, or physical batch.

### Running model

**Short:** A loaded model ready to perform model work.
**Precise:** Validated model semantics plus resident or addressable weights and
execution resources, normally shared read-only across requests. First
introduced: Chapter 1. Related: artifact, inference engine, backend. Common
confusion: a running model still does not own the full request lifecycle.

### Residual stream

**Short:** The model-width activation carried between Transformer sublayers.
**Precise:** For each token position it has width `D`; sublayers may create
other internal shapes but values returning to the residual stream must satisfy
that width and request-local ownership contract. First introduced: Chapter 7.
Related: embedding, hidden state, RMSNorm. Common confusion: persistent through
layers does not mean model-lifetime or durable storage.

### RMS / RMSNorm

**Short:** RMS is root mean square; RMSNorm rescales an activation by reciprocal
epsilon-stabilized RMS and applies learned element-wise scale. **Precise:** For
`x,w:[D]`, Chapter 7 defines `y_i = x_i*w_i / sqrt(mean(x^2)+epsilon)` with
positive `D`, finite positive epsilon, explicit reduction precision, and no
mean centering. First introduced: Chapter 7. Related: residual stream, epsilon,
LayerNorm. Common confusion: RMSNorm is not LayerNorm and does not place epsilon
outside the square root in this contract.

### Sampler

**Short:** The policy that selects the next token from logits.
**Precise:** It may transform logits using temperature, penalties, top-k/top-p,
constraints, and RNG state before selection. First introduced: Chapter 4.
Common confusion: greedy argmax and stochastic sampling have different state
and reproducibility contracts.

### Autoregressive generation / token feedback

**Short:** Repeatedly choose a next token, append it, and use it in the next
model evaluation. **Precise:** At step `t`, a causal model scores token
`x_(t+1)` from visible history; the runtime selects and commits one token so the
next step conditions on the enlarged sequence. First introduced: Chapter 4.
Related: forward pass, sampler, sequence. Common confusion: autoregressive
feedback does not prove that a particular model representation uses all prior
tokens.

### Categorical distribution / categorical sampling

**Short:** A probability distribution over a finite set and a draw selecting
one member. **Precise:** ENGINE-1 maps a uniform draw in `[0,1)` to the first
cumulative probability boundary strictly greater than it. First introduced:
Chapter 4. Related: probability distribution, sampler, PRNG. Common confusion:
the RNG creates a draw; model-derived probabilities determine interval sizes.

### Greedy decoding

**Short:** Select the token with the largest current logit. **Precise:** A
deterministic `O(V)` argmax policy with an explicit tie rule; ENGINE-1 chooses
the lowest token ID among equal maxima and does not compute softmax. First
introduced: Chapter 4. Related: argmax, sampler. Common confusion: a locally
maximal next token does not prove a globally maximal complete sequence.

### Logit processing

**Short:** Ordered policy transformations between raw logits and selection.
**Precise:** Temperature, masks, top-k/top-p, penalties, and constraints can
change the candidate distribution while raw model logits remain separate
evidence. First introduced: Chapter 4. Related: sampler, logits. Common
confusion: processor order is part of inference behavior, not interchangeable
configuration decoration.

### max_new_tokens

**Short:** The maximum number of new output tokens a request may commit.
**Precise:** It counts generated non-EOS tokens separately from prompt length;
reaching it is a successful stop reason, not a model error. First introduced:
Chapter 4. Related: request, stop reason. Common confusion: it is not total
sequence length.

### Probability distribution / renormalization

**Short:** Non-negative mass summing to one; renormalization restores unit mass
after filtering. **Precise:** If retained mass is `s>0`, each retained
probability becomes `p_i/s` and removed candidates remain zero. First
introduced: Chapter 4. Related: softmax, categorical sampling. Common confusion:
sampling from a truncated distribution still requires defined normalization.

### PRNG / seed

**Short:** A pseudorandom number generator deterministically advances state;
a seed initializes it. **Precise:** Reproduction also depends on algorithm,
consumption order, engine/model/tokenizer versions, numeric path, and policy.
ENGINE-1 owns one non-cryptographic SplitMix64 state per request. First
introduced: Chapter 4. Related: sampler state. Common confusion: one seed does
not guarantee identical output across systems.

### Sampler configuration / sampler state

**Short:** Configuration is immutable policy; state is mutable progress.
**Precise:** ENGINE-1 configuration names greedy or stochastic parameters,
while request-local state owns its PRNG and successful-sample count. First
introduced: Chapter 4. Related: sampler, request, seed. Common confusion:
shareable configuration does not make mutable RNG state shareable.

### Softmax / stable softmax

**Short:** Softmax normalizes scores into probabilities; stable softmax first
subtracts the largest score. **Precise:** `p_i=exp(z_i-m)/sum_j exp(z_j-m)`
with `m=max(z)` preserves exact softmax while keeping exponential arguments
non-positive. First introduced: Chapter 4. Related: logits, probability
distribution. Common confusion: greedy argmax does not require softmax.

### Temperature

**Short:** A positive sampling parameter that scales logit gaps before
normalization. **Precise:** Stochastic ENGINE-1 computes `z'_i=z_i/T` for
finite `T>0`; lower values sharpen and higher values flatten without changing
ordering. First introduced: Chapter 4. Related: softmax, sampler. Common
confusion: temperature changes distribution shape; the RNG supplies randomness.

### Top-k / top-p / nucleus sampling

**Short:** Top-k retains a fixed candidate count; top-p retains the smallest
high-probability prefix reaching a mass threshold. **Precise:** ENGINE-1 applies
top-k in logit space, stable softmax, then top-p in probability space with the
crossing candidate included, followed by renormalization. First introduced:
Chapter 4. Related: logit processing, sampler. Common confusion: top-p does not
keep `p` percent of vocabulary entries.

### Sequence

**Short:** One logical ordered token history advanced by the runtime.
**Precise:** It owns or references request state, token positions, KV mappings,
sampler state, stopping state, and output lifecycle. First introduced: Chapter
23. Common confusion: sequence, request, conversation, and physical batch are
related but not identical.

### SIMD

**Short:** Single instruction, multiple data execution within a CPU core.
**Precise:** ISA-specific vector lanes, alignment, tails, dispatch, reduction
order, and memory access define a SIMD kernel. First introduced: Chapter 42.
Common confusion: compiler auto-vectorization is one route, not the definition.

### Speculative decoding

**Short:** Propose multiple tokens cheaply, then verify them with the target
model.
**Precise:** Correct verification accepts a matching prefix and samples the
first mismatch under the target distribution while rolling back rejected state.
First introduced: Chapter 51. Common confusion: it reduces target steps only
when acceptance compensates for verification overhead.

### State

**Short:** Information whose current value affects future execution.
**Precise:** In inference this includes immutable model configuration and
weights plus mutable request history, positions, cached intermediates, sampler,
stopping, timing, and resource ownership; qualify which state and owner are
meant. First introduced: Chapter 1. Related: request, context, ownership. Common
confusion: “state” is not one undifferentiated buffer.

### Stream

**Short:** An ordered sequence of progress events ending in a defined terminal
outcome.
**Precise:** The engine-to-consumer contract for token or text-piece events,
ordering, backpressure/error behavior, and completion/cancellation/failure.
First introduced: Chapter 1; production semantics formalized Chapter 69.
Related: token, terminal, request. Common confusion: channel close alone does
not identify why generation ended.

### Terminal outcome

**Short:** The single final result of a request: completed, cancelled, or
failed.
**Precise:** An exactly-once lifecycle transition after which no progress event
may be emitted and owned request resources become releasable; successful
completion also records a stop reason. First introduced: Chapter 1. Related:
stream, stop reason, cancellation. Common confusion: stopping because of a
token limit is completion, while error and cancellation are distinct.

### Throughput

**Short:** Completed work per unit time.
**Precise:** Requests or tokens divided by elapsed time for a defined workload,
population, concurrency, and accounting boundary. First introduced: Chapter 1.
Related: latency, concurrency, fairness. Common confusion: aggregate throughput
is not the inverse of one request's latency.

### Time to first token (TTFT)

**Short:** Time from a disclosed request boundary to the first observable
output token or piece.
**Precise:** Commonly admission/arrival to first emission, including the named
queue, preparation, model, selection, and emission spans chosen by the
measurement. First introduced: Chapter 1. Related: latency, ITL. Common
confusion: TTFT endpoints differ across tools and must be stated.

### Token / token ID

**Short:** A token is one identity in a particular model vocabulary; a token ID
is its numeric representation.
**Precise:** Its ordinary piece may correspond to a word, subword, punctuation,
whitespace, one or more bytes, or an incomplete UTF-8 fragment; control tokens
may have no ordinary text bytes. First introduced: Chapter 2. Related:
vocabulary, tokenizer, special token. Common confusion: a token is not
necessarily a word, character, Unicode scalar value, byte, or streamed piece.

### BOS / EOS / PAD / UNK

**Short:** Common special-token roles for sequence beginning, sequence ending,
padding, and unknown input.
**Precise:** Their IDs, insertion/removal rules, and relationship to stopping
are model-specific; UNK represents information the ordinary encoder could not
preserve. First introduced: Chapter 2. Related: special token, byte fallback.
Common confusion: PAD is not automatically EOS, and byte fallback does not make
the configured UNK identity disappear.

### BPE / merge rule

**Short:** Byte-pair encoding segments input by applying a learned, ranked set
of adjacent-symbol merges.
**Precise:** Inference encoding begins from configured base symbols and
deterministically applies fixed pair/rank rules until none remains; training,
pre-tokenization, byte mapping, and normalization are separate parts of the
artifact. First introduced: Chapter 2. Related: tokenizer, vocabulary. Common
confusion: BPE encoding does not learn new merges from a user's prompt.

### Byte / byte fallback

**Short:** A byte is an eight-bit storage unit; byte fallback represents input
outside ordinary pieces with byte-token identities.
**Precise:** A complete 256-byte fallback alphabet can cover arbitrary bytes at
the vocabulary stage, but earlier normalization may already have changed the
input and decoded bytes may still be malformed UTF-8. First introduced: Chapter
2. Related: UTF-8, UNK. Common confusion: byte fallback does not imply one token
per byte after merges or unconditional surface-text round trip.

### Chat template

**Short:** The model-specific serialization from structured messages to model
input.
**Precise:** It orders role/content data and inserts separators, turn endings,
and optional generation prefixes under the tokenizer's special-token contract.
First introduced: Chapter 2. Related: special token, model contract. Common
confusion: a plausible `role: content` flattening is not interchangeable with
the template used during training.

### Decode buffer

**Short:** Per-request bytes retained until complete valid output text exists.
**Precise:** Token pieces append to a UTF-8 framing buffer; complete valid
prefixes may emit while a valid incomplete suffix remains bounded to at most
three bytes. First introduced: Chapter 2. Related: decoding, UTF-8, stream.
Common confusion: one generated token need not create one text event.

### Encoding / decoding

**Short:** Encoding maps input bytes/text to token IDs; decoding maps IDs to
configured byte pieces and reconstructed output.
**Precise:** The transforms include the tokenizer's normalization,
pre-tokenization, vocabulary model, post-processing, and decoder rules; exact
round trip is conditional on those semantics. First introduced: Chapter 2.
Related: tokenizer, normalization. Common confusion: model workload phase
*decode* and tokenizer ID-to-byte decoding are different uses of the word.

### Normalization

**Short:** A configured transform applied before token segmentation.
**Precise:** It may apply a Unicode normalization form, case or accent changes,
whitespace rules, dummy prefixes, or other mappings, some of which are
irreversible at the original-byte surface. First introduced: Chapter 2.
Related: Unicode scalar value, tokenizer. Common confusion: Unicode does not
require every tokenizer to normalize input automatically.

### SentencePiece / Unigram tokenizer

**Short:** SentencePiece is a tokenizer toolkit/model format; Unigram is one
segmentation model it can store.
**Precise:** SentencePiece supports BPE, Unigram, word, and character model
types plus embedded normalization and special-symbol configuration. A Unigram
model scores candidate pieces and searches segmentations rather than applying a
ranked BPE merge list. First introduced: Chapter 2. Related: BPE, vocabulary.
Common confusion: *SentencePiece* and *Unigram* are not synonyms.

### Special token

**Short:** A vocabulary identity with model/control semantics rather than
ordinary user-text semantics.
**Precise:** BOS/EOS/PAD/UNK, role, separator, and end-of-turn identities must
be inserted or parsed only through an authorized model-specific surface. First
introduced: Chapter 2. Related: chat template, token ID. Common confusion: a
diagnostic marker spelling in ordinary user text is not itself authority to
insert the control identity.

### Tokenizer

**Short:** The configured mapping between input text/bytes and model vocabulary
identifiers.
**Precise:** Tokenizer identity includes normalization, pre-tokenization,
vocabulary/model rules, byte/unknown behavior, special-token semantics,
post-processing, and decode rules bound to a model revision. First introduced:
Chapter 2. Related: vocabulary, model artifact, chat template. Common confusion:
the algorithm name alone does not identify the tokenizer.

### Unicode scalar value / UTF-8

**Short:** A Unicode scalar value is a Unicode code point excluding surrogates;
UTF-8 encodes each scalar as one to four bytes.
**Precise:** Grapheme, scalar, byte, and token boundaries are independent;
ill-formed byte sequences cannot be interpreted as Unicode characters. First
introduced: Chapter 2. Related: byte, decode buffer. Common confusion: a
user-perceived character may contain several scalar values, and one token piece
may contain only part of one scalar's UTF-8 bytes.

### Vocabulary

**Short:** The finite model-specific catalog that gives token IDs meaning.
**Precise:** It contains ordinary pieces and possibly control, unknown, byte,
unused, or other entries, together with configuration used to select IDs. First
introduced: Chapter 2. Related: token ID, tokenizer, embedding. Common
confusion: equal numeric IDs or visible pieces in two vocabularies are not
interchangeable.

### Activation

**Short:** Numerical data produced while executing a model for a particular
input. **Precise:** A typed, shaped intermediate or output such as ENGINE-1's
hidden vector or logits, with a lifetime distinct from immutable parameters.
First introduced: Chapter 3. Related: parameter, hidden state, logits. Common
confusion: activations are not model weights merely because both use floating-
point storage.

### Argmax

**Short:** The index of the largest value in a collection. **Precise:** A
deterministic selection rule with an explicit tie policy; in ENGINE-1 it
consumes logits outside `Model::forward`. First introduced: Chapter 3. Related:
logit, sampler. Common confusion: argmax does not normalize scores or assign
probability one to its result.

### Bias

**Short:** An additive learned parameter in an affine operation. **Precise:**
ENGINE-1's `b:[V]` initializes one accumulator per output token in
`z = W h + b`. First introduced: Chapter 3. Related: output projection,
parameter. Common confusion: an operation with nonzero bias is affine even
though ML libraries often call it a linear layer.

### Dot product

**Short:** A sum of pairwise products between equal-length vectors. **Precise:**
ENGINE-2 reduces rank-1 `f32` views in ascending logical index order into an
`f32` accumulator; the empty dot product is zero. ENGINE-1 uses the operation
inside each projection row. First introduced: Chapter 3; formalized Chapter 6.
Related: GEMV, GEMM, projection. Common confusion: different lengths are an
error, not an invitation to truncate or broadcast.

### Dtype

**Short:** The element type used to store or compute numerical values.
**Precise:** A dtype determines representation and precision; storage and
accumulation dtypes may differ. Tensor Substrate v1 implements `f32` while
keeping dtype explicit. First introduced: Chapter 3; formalized Chapter 5.
Related: tensor, quantization. Common confusion: shape does not imply dtype,
and equal byte widths do not imply equal interpretations.

### Element count / storage extent

**Short:** Element count is the product of shape dimensions; storage extent is
the physical span reachable through a view's base offset and strides.
**Precise:** Both use checked `usize` arithmetic; a strided view may require a
larger backing slice than its logical element count, while any zero dimension
makes the logical count zero. First introduced: Chapter 5. Related: shape,
stride, bounds. Common confusion: logical count, required storage length, and
byte count are three different quantities.

### Forward pass

**Short:** One execution of model semantics on supplied input. **Precise:** In
ENGINE-1, a forward pass selects `E[x]` and computes `W h + b`, producing
request-local hidden and logit activations without mutating parameters. First
introduced: Chapter 3. Related: model, activation. Common confusion: a forward
pass is not by itself token sampling or a complete generation loop.

### Matrix / vector

**Short:** A matrix is a two-dimensional numerical array; a vector is a
one-dimensional numerical array. **Precise:** Their logical shapes define valid
index relationships, while dtype, layout, and owner define physical execution.
First introduced: Chapter 3. Related: tensor, dot product. Common confusion:
matrix notation alone does not specify row-major bytes.

### Ownership / aliasing

**Short:** Ownership controls a storage lifetime; aliasing means two logical
objects can reach overlapping storage.
**Precise:** An `OwnedTensor` owns its `Vec<f32>`, immutable views may overlap
through shared borrows, and mutable access in Tensor Substrate v1 is restricted
to one exclusive canonical owner view. First introduced: Chapter 5. Related:
view, copy, lifetime. Common confusion: sharing storage does not transfer its
ownership, and physical overlap is safe for reads but dangerous for
uncontrolled mutation.

### Output projection

**Short:** The model operation that produces one score per vocabulary token.
**Precise:** ENGINE-1 uses `W:[V,D]`, `h:[D]`, and `b:[V]` to produce logits
`z:[V]`. First introduced: Chapter 3. Related: logits, bias. Common confusion:
the projection scores candidates; it does not select one.

### Parameter / weight

**Short:** Persistent numerical data that defines model behavior. **Precise:**
ENGINE-1's embedding, projection weights, and bias are validated at model
construction and read immutably during inference. First introduced: Chapter 3.
Related: activation, model artifact. Common confusion: parameters do not
normally change during an inference forward pass.

### Rank / axis / dimension

**Short:** Tensor rank counts axes; a dimension is an axis's length.
**Precise:** Shape `[2,3,4]` has tensor rank 3 and dimension lengths 2, 3, and
4. First introduced: Chapter 5. Related: shape, tensor. Common confusion:
tensor rank is unrelated to the linear-algebra rank of a matrix.

### Row-major layout

**Short:** Matrix rows occupy contiguous physical storage. **Precise:** For a
`[V,D]` ENGINE-1 matrix, logical element `(i,j)` has flat offset `i*D+j`.
First introduced: Chapter 3. Related: layout, matrix, tensor. Common confusion:
logical shape does not force row-major layout.

### Scalar

**Short:** One numerical value, represented as a rank-0 tensor with shape `[]`.
**Precise:** The empty product gives a scalar tensor one logical element and
canonical empty strides. First introduced: Chapter 5. Related: vector, matrix,
tensor, rank. Common confusion: shape `[]` is a scalar, while shape `[1]` is a
one-element vector.

### Slice

**Short:** A bounded view over part of a tensor along an axis.
**Precise:** Tensor Substrate v1 slicing changes a view's base offset and one
dimension while retaining its storage and strides. First introduced: Chapter
5. Related: view, stride, base offset. Common confusion: a slice need not copy
elements or become contiguous.

### Shape

**Short:** The ordered sizes of a tensor's logical dimensions. **Precise:** A
shape constrains indexing and compatible operations; ENGINE-1 rejects parameter
counts that do not match `[V,D]` and `[V]`. First introduced: Chapter 3.
Related: tensor, dimension. Common confusion: shape is an executable contract,
not merely documentation.

### Stride / base offset

**Short:** A stride is the storage step for incrementing one axis; the base
offset is where a view's logical origin begins.
**Precise:** Tensor Substrate v1 uses element strides and computes physical
offset as the checked base plus the dot product of logical indices and strides.
First introduced: Chapter 5. Related: shape, layout, view. Common confusion:
frameworks such as NumPy and GGML expose byte strides, so units must never be
assumed.

### Tensor

**Short:** Numerical data interpreted with shape, dtype, layout, and ownership.
**Precise:** A tensor contract also names location, lifetime, and valid access;
Chapter 3 uses only contiguous row-major CPU `f32` arrays. First introduced:
Chapter 3; formalized Chapter 5. Related: shape, dtype, layout. Common
confusion: a raw `Vec<f32>` does not by itself say which tensor it represents.

### Tensor layout / contiguity

**Short:** Layout maps logical indices to physical storage; contiguity is one
specific layout property.
**Precise:** Tensor Substrate v1 calls a view contiguous only when its strides
exactly equal canonical row-major strides for its shape. First introduced:
Chapter 5. Related: stride, row-major layout, view. Common confusion: a valid
strided view is not necessarily contiguous, and contiguous is not a universal
performance claim.

### Transpose / reshape

**Short:** Transpose permutes axes; reshape changes dimensions while preserving
logical element order.
**Precise:** A rank-2 transpose in Tensor Substrate v1 swaps shape and stride
metadata without moving values. `reshape_view` is allocation-free and accepts
only a canonical contiguous source with the same checked element count. First
introduced: Chapter 5. Related: view, copy, contiguity. Common confusion: a
transpose is not a reshape, and a reshape operation must not conceal a copy.

### View / copy

**Short:** A view borrows existing element storage; a copy owns newly
materialized elements.
**Precise:** A `TensorView` owns shape, stride, and base-offset metadata but
borrows its payload. `to_contiguous` traverses logical order into a new
canonical `OwnedTensor`. First introduced: Chapter 5. Related: ownership,
aliasing, transpose. Common confusion: metadata allocation does not mean the
element payload was copied.

### Data reuse / spatial locality / temporal locality

**Short:** Reuse performs more useful work with data already nearby; spatial
locality visits nearby addresses, while temporal locality revisits data soon.
**Precise:** Chapter 6's `ikj` walk streams row-major B/C rows and reuses one A
scalar; blocking bounds A/B/C regions to encourage reuse in a memory hierarchy.
First introduced: Chapter 6. Related: working set, arithmetic intensity, loop
order. Common confusion: favorable source access does not prove a particular
cache-hit count.

### Equivalence gate / performance gate

**Short:** Equivalence proves acceptable results; a performance gate tests a
correct candidate under a declared workload. **Precise:** ENGINE-2 compares
blocked GEMM with hand fixtures, an independent oracle, typed edge cases, and a
documented tolerance before release timing. First introduced: Chapter 6.
Related: oracle, differential test, benchmark. Common confusion: faster output
is irrelevant when semantic equivalence has not passed.

### Inner dimension

**Short:** The shared dimension contracted by a linear algebra product.
**Precise:** For `[M,K] × [K,N]`, both K dimensions must agree and each output
cell reduces K pairwise products. First introduced: Chapter 6. Related: GEMM,
shape. Common confusion: equal total element counts do not establish compatible
inner dimensions.

### Loop order

**Short:** The nesting order in which a multidimensional computation visits
indices. **Precise:** `ijk` and `ikj` implement the same GEMM equation but expose
different row-major operand and output access patterns. First introduced:
Chapter 6. Related: spatial locality, blocking. Common confusion: equal FLOP
counts do not imply equal byte movement or elapsed time.

### Memory bandwidth

**Short:** The rate at which bytes cross a named memory boundary. **Precise:**
Bandwidth must identify the level, direction, workload, and measurement; the
Chapter 6 compulsory-byte model is not a bandwidth measurement. First
introduced: Chapter 6. Related: arithmetic intensity, Roofline model. Common
confusion: storage size and transfer rate are different quantities.

### Microkernel / packing

**Short:** Packing rearranges data for execution; a microkernel computes a
small register-oriented core tile. **Precise:** They are production GEMM
optimization rungs previewed but not implemented by scalar ENGINE-2. First
introduced: Chapter 6. Related: blocking, SIMD, kernel. Common confusion:
packing has allocation/movement cost and is not hidden free speed.

### Multiply-accumulate (MAC)

**Short:** Multiply two operands and add the product into an accumulator.
**Precise:** Dot, GEMV, and GEMM are reductions of MAC contributions; storage,
product, and accumulator dtype plus ordering define numerical behavior. First
introduced: Chapter 6. Related: dot product, FMA, reduction. Common confusion:
a source MAC expression does not guarantee one fused hardware instruction.

### Reference kernel / optimized kernel

**Short:** A reference kernel prioritizes transparent semantics; an optimized
kernel changes execution under a proved contract. **Precise:** ENGINE-2's
reference GEMM accepts valid strided views, while blocked GEMM accepts only
canonical row-major views and must pass equivalence before measurement. First
introduced: Chapter 6. Related: equivalence gate, kernel, oracle. Common
confusion: reference does not mean disposable, and optimized does not mean
universally faster.

### Roofline model

**Short:** A model bounding performance by compute peak and bandwidth times
arithmetic intensity. **Precise:** $P\le\min(P_{peak},BI)$ previews whether a
workload may be compute- or movement-constrained at a named boundary. First
introduced: Chapter 6. Related: arithmetic intensity, memory bandwidth. Common
confusion: an uncalibrated diagram is not a measurement of the machine.

### Tile

**Short:** One bounded subrange used by a blocked computation. **Precise:** An
ENGINE-2 GEMM tile has positive M, K, and N extents; final tail tiles are
clamped when dimensions are not divisible. First introduced: Chapter 6.
Related: blocking, working set. Common confusion: a tile is not necessarily a
physical storage allocation.

### Working set

**Short:** Data actively needed during a region of execution. **Precise:**
Blocking attempts to keep useful A, B, and accumulating C regions reusable in
faster memory, without promising residency in a specific cache. First
introduced: Chapter 6. Related: tile, locality, cache. Common confusion: a
formula for tile payload does not prove the hardware retains it.

### Workspace

**Short:** Temporary storage required by a planned computation.
**Precise:** Workspace size, alignment, indexing, lifetime, and reduction order
are part of a kernel contract and may be host or device resident. First
introduced: Chapter 36. Common confusion: workspace is not persistent model or
KV state.
