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

### Batch / continuous batching

**Short:** A batch is work executed together; continuous batching rebuilds that
work at iteration boundaries as requests arrive and finish.
**Precise:** A physical token batch can contain prompt or decode positions from
multiple logical sequences. First introduced: Chapters 25–26. Related: sequence,
prefill, decode, scheduler. Common confusion: continuous batching is not the
same as waiting to fill a static request batch.

### Block table

**Short:** A mapping from a sequence's logical token blocks to physical storage
blocks.
**Precise:** Ordered metadata containing physical block identifiers and logical
length, interpreted by paged attention and lifetime management. First
introduced: Chapter 29. Related: logical block, physical block, KV block.
Common confusion: it does not contain the KV vectors themselves.

### Context

**Short:** The model/runtime state needed to advance one or more sequences.
**Precise:** Depending on an engine, a context may own execution scratch, KV
state, sequence identifiers, device resources, and sampler-related state. First
introduced: Chapter 20. Related: sequence, KV cache. Common confusion: context
window (a token limit) and execution context (a state owner) are different.

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
shape differ from prefill. First introduced: Chapter 4, formalized Chapter 20.
Common confusion: token decoding into text bytes is a separate tokenizer task.

### Differential test

**Short:** A controlled comparison of two independent implementations.
**Precise:** It supplies identical inputs, defines tolerances/ordering, and
compares an optimized, concurrent, or real-model path with an oracle. First
introduced: Chapter 61. Related: oracle, equivalence. Common confusion: comparing
a function with itself under another flag may not be independent.

### Embedding

**Short:** A vector representation selected for a token identifier.
**Precise:** A row of learned model weights with shape `[hidden_dim]`, forming
the initial hidden state for a token position. First introduced: Chapter 7.
Related: token, hidden state. Common confusion: an embedding is not a token's
human-language definition.

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

### Kernel

**Short:** A bounded numerical or data-movement operation.
**Precise:** A kernel has explicit input/output layouts, dtype, workspace,
provider constraints, error behavior, and correctness contract. First
introduced: Chapter 35. Related: provider, ABI. Common confusion: kernels can
run on scalar CPU, SIMD CPU, or accelerators.

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
Chapter 60. Related: differential test. Common confusion: the production path
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
work, unlike one-step decode. First introduced: Chapter 20. Common confusion:
prefill is not only tokenization or model loading.

### Prefix cache / radix tree

**Short:** A prefix cache reuses computed state for matching token prefixes; a
radix tree is a compressed prefix index.
**Precise:** Cache entries have independent ownership, eviction, and partial
tail rules. First introduced: Chapter 31. Common confusion: text equality is
insufficient if tokenization/model/configuration differs.

### Provider

**Short:** A hardware/runtime implementation of an execution capability.
**Precise:** A provider advertises supported shapes/dtypes, placement and
workspace rules, fallback behavior, and conformance tests. First introduced:
Chapter 45. Common confusion: a compiled GPU backend need not be selected or
faster for a particular shape.

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

### Sampler

**Short:** The policy that selects the next token from logits.
**Precise:** It may transform logits using temperature, penalties, top-k/top-p,
constraints, and RNG state before selection. First introduced: Chapter 4.
Common confusion: greedy argmax and stochastic sampling have different state
and reproducibility contracts.

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

### Token / tokenizer

**Short:** A token is a vocabulary identifier; a tokenizer maps text/bytes to
identifiers and back.
**Precise:** Tokenization includes normalization, pre-tokenization, model rules,
special tokens, and byte decoding whose exact behavior is part of model
semantics. First introduced: Chapter 2. Common confusion: tokens are not words.

### Workspace

**Short:** Temporary storage required by a planned computation.
**Precise:** Workspace size, alignment, indexing, lifetime, and reduction order
are part of a kernel contract and may be host or device resident. First
introduced: Chapter 36. Common confusion: workspace is not persistent model or
KV state.
