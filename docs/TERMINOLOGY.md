# Terminology System

`GLOSSARY.md` is the reader-facing definition catalog. This file governs how
authors introduce and maintain terms.

## Entry contract

Every important term records: term, short definition, precise definition,
first-introduced chapter, related terms, and common confusion. Add the glossary
entry before a chapter reaches TECH-REVIEW.

## Canonical distinctions

- A **token** is a model vocabulary identifier; it is not necessarily a word,
  character, or byte.
- A **Unicode scalar value**, a UTF-8 **byte**, a vocabulary **token ID**, and a
  streamed text **piece** are distinct units. Name the one being counted.
- **Encoding/decoding** can mean tokenizer IDs-to-bytes work; **decode** also
  names the later model workload phase. Qualify the tokenizer operation when
  both meanings appear nearby.
- **SentencePiece** names a toolkit/model format that can contain BPE or
  Unigram semantics; do not use it as a synonym for either algorithm.
- **Special-token insertion** is an explicit trusted operation. Marker-looking
  ordinary text does not become a control identity unless a model-specific
  parsing surface deliberately authorizes that behavior.
- A **chat template** is model-input serialization bound to a tokenizer/model
  revision, not merely presentation formatting.
- **Prefill** evaluates prompt positions and creates reusable per-position
  state; **decode** advances active sequences with newly generated positions.
- A **logit** is an unnormalized model score; a **probability** is a normalized
  non-negative mass. Logit processing, probability filtering, and token
  selection are distinct stages.
- **Greedy decoding** is deterministic argmax selection and does not require
  softmax. **Categorical sampling** draws from a probability distribution and
  therefore owns mutable pseudorandom state.
- **Temperature** rescales logits before softmax; **top-k** retains a fixed
  number of candidates, while **top-p** retains the smallest probability-ranked
  prefix whose cumulative mass reaches the threshold.
- A **seed** initializes a particular PRNG contract. It does not promise
  identical output across engine versions, numerical backends, or models.
- **Tensor rank** is the number of logical axes. It is not the linear-algebra
  rank of a matrix. An **axis** names a logical direction; a **dimension** is
  that axis's length.
- **Shape** is the ordered list of logical dimension lengths. **Layout** maps
  legal logical indices to physical storage; equal shapes do not imply equal
  layouts.
- A **stride** is the storage step for incrementing one axis. Always name its
  unit: Tensor Substrate v1 uses element strides, while NumPy and GGML expose
  byte strides.
- **Contiguous** means canonical row-major strides in Tensor Substrate v1. It
  does not mean merely addressable, dense-looking, or fast on every traversal.
- A **view** borrows existing element storage with its own shape/layout
  metadata. A **copy** owns newly materialized elements. Never describe an
  operation as a view if it may silently allocate element storage.
- **Aliasing** means two logical objects can reach overlapping storage.
  Immutable aliasing is allowed; mutable access requires exclusive ownership
  in the Chapter 5 API.
- **Transpose** permutes axes and may be metadata-only. **Reshape** changes the
  logical dimensions without changing element order and is a no-copy view only
  when its layout contract permits. **Slicing** narrows an axis and may change
  the base offset.
- A **dot product** reduces equal-length vectors. **GEMV** contracts `[M,K] ×
  [K]` into `[M]`; **GEMM** contracts `[M,K] × [K,N]` into `[M,N]`. The shared
  **inner dimension** must agree; equal element counts are insufficient.
- **FLOP** is an amount of floating-point work; **FLOP/s** and **GFLOP/s** are
  rates. Do not equate the conventional `2MKN` GEMM work model with retired
  instructions.
- **Arithmetic intensity** is FLOPs per byte at a named boundary. Mark
  compulsory-payload calculations as ideal models, not measured traffic or
  memory bandwidth.
- **Spatial locality** concerns nearby addresses; **temporal locality** concerns
  reuse over time. **Loop order** and **blocking/tiling** can expose locality,
  but neither term promises a cache hit or speedup for every workload.
- A **reference kernel** prioritizes transparent semantics. An **optimized
  kernel** may narrow layout or shape support only through an explicit contract
  and must pass an **equivalence gate** before a performance gate.
- **Packing** changes layout and costs movement/storage. A **microkernel** is a
  small register-oriented compute core. Chapter 6 previews both terms but
  ENGINE-2 implements neither.
- **EOS** is a sampled control token that ends generation; `max_new_tokens`
  counts committed generated tokens. Keep both distinct from prompt length and
  future text stop-string handling.
- A **sequence** is one logical token history. A **physical token batch** is the
  work assembled for one forward execution and may mix phases/sequences.
- The **KV cache** stores per-layer key/value vectors for prior positions. It
  does not cache logits or eliminate reading visible history during attention.
- A **logical block** is a sequence-relative range; a **physical block** is an
  allocator-owned storage object; a **block table** maps between them.
- A **prefix cache** indexes reusable computed state. A **radix tree** is one
  possible index; **sticky slots** reuse state bound to a context and are not a
  general page-sharing prefix cache.
- **MHA**, **GQA**, and **MQA** describe query-to-KV-head geometry. Do not use
  them interchangeably.
- A **provider** exposes a selectable inference capability or placement, while
  a **backend** implements model operations on a hardware/runtime substrate and
  a **kernel** is one bounded computation. Projects may use these names
  differently; qualify external usage. None of the terms implies GPU execution.
- An **oracle** is an independent correctness reference; a **differential
  test** compares implementations under controlled inputs and tolerances.
- **Residency** says where bytes currently live; **ownership** says who controls
  lifetime/mutation; **pinning** temporarily prevents eviction.

## Naming rules

Use `ENGINE-N` for curriculum milestones, `part-NN` and
`chapter-NN-topic.md` for manuscript paths, and uppercase project status labels.
Use “Hermon-owned paged engine” only for the specific gated path verified in
Hermon; do not shorten a PREVIEW into an apparently CURRENT “Hermon engine.”

## Review

Terminology review checks first introduction, pluralization, hyphenation,
acronym expansion, diagram labels, code identifiers, and glossary cross-links.
When an external project uses a conflicting term, retain its name while stating
the book's canonical equivalent.
