# Canonical Diagram Inventory

Status `canonical` means technically and visually reviewed. Source is the
repository-native Unicode text artifact; chapter embeds may be abbreviated but
must preserve its semantics.

| ID | Diagram | Ch. | Type | Question answered | Source | Status |
| --- | --- | ---: | --- | --- | --- | --- |
| D001 | Request to token | 1 | control flow | How can a request end? | [`runtime/request-to-token.txt`](runtime/request-to-token.txt) | canonical |
| D002 | Model versus engine | 1 | architecture | What is artifact data versus running state? | [`runtime/model-vs-engine.txt`](runtime/model-vs-engine.txt) | canonical |
| D003 | Inference stack | 1 | architecture | Where are the serving boundaries? | [`runtime/inference-stack.txt`](runtime/inference-stack.txt) | canonical |
| D004 | Token, byte, owner | 1 | data flow | Which three journeys organize the book? | [`runtime/token-byte-owner.txt`](runtime/token-byte-owner.txt) | canonical |
| D005 | Latency decomposition | 1 | performance | Which intervals make up request latency? | [`runtime/latency-decomposition.txt`](runtime/latency-decomposition.txt) | canonical |
| D006 | Control/data planes | 1 | architecture | Who decides and what bytes move? | [`runtime/control-plane-data-plane.txt`](runtime/control-plane-data-plane.txt) | canonical |
| D007 | Hermon request path | 1 | control flow | What path and status are source-verified? | [`runtime/hermon-current-request-path.txt`](runtime/hermon-current-request-path.txt) | canonical |
| D008 | Text to token IDs | 2 | data flow | How do text units differ? | [`tokenizer/text-unicode-bytes-tokens.txt`](tokenizer/text-unicode-bytes-tokens.txt) | canonical |
| D009 | BPE merge | 2 | control flow | How do ranked merges produce pieces? | [`tokenizer/bpe-merge-process.txt`](tokenizer/bpe-merge-process.txt) | canonical |
| D010 | Chat template | 2 | data flow | How does chat become tokenizable input? | [`tokenizer/chat-template-pipeline.txt`](tokenizer/chat-template-pipeline.txt) | canonical |
| D011 | Token ownership | 2 | ownership | Who owns token and byte buffers? | [`tokenizer/engine0-token-ownership.txt`](tokenizer/engine0-token-ownership.txt) | canonical |
| D012 | Binding contract | 2 | architecture | Which artifacts must agree? | [`tokenizer/model-tokenizer-template-contract.txt`](tokenizer/model-tokenizer-template-contract.txt) | canonical |
| D013 | Token to byte stream | 2 | data flow | How does a generated ID become text? | [`tokenizer/token-to-byte-stream.txt`](tokenizer/token-to-byte-stream.txt) | canonical |
| D014 | Partial UTF-8 boundary | 2 | state machine | When may buffered bytes be emitted? | [`tokenizer/utf8-partial-token-boundary.txt`](tokenizer/utf8-partial-token-boundary.txt) | canonical |
| D015 | Special-token trust | 2 | trust boundary | Who may create control identities? | [`tokenizer/special-token-trust-boundary.txt`](tokenizer/special-token-trust-boundary.txt) | canonical |
| D016 | Token-count impact | 2 | systems flow | What changes with sequence length? | [`tokenizer/token-count-system-impact.txt`](tokenizer/token-count-system-impact.txt) | canonical |
| D017 | ID to logits | 3 | data flow | How does one ID produce vocabulary scores? | [`model/token-id-to-logits.txt`](model/token-id-to-logits.txt) | canonical |
| D018 | Embedding lookup | 3 | tensor shape | Which row becomes the hidden vector? | [`model/embedding-row-lookup.txt`](model/embedding-row-lookup.txt) | canonical |
| D019 | One logit | 3 | data flow | Which terms produce one score? | [`model/one-logit-dot-product.txt`](model/one-logit-dot-product.txt) | canonical |
| D020 | Parameters/activations | 3 | ownership | Which values outlive a forward pass? | [`model/parameters-vs-activations.txt`](model/parameters-vs-activations.txt) | canonical |
| D021 | Semantics/execution | 3 | architecture | What must stay invariant across kernels? | [`model/semantics-vs-execution.txt`](model/semantics-vs-execution.txt) | canonical |
| D022 | Tiny-model shapes | 3 | tensor shape | What are all ENGINE-1 shapes? | [`model/tiny-model-tensor-shapes.txt`](model/tiny-model-tensor-shapes.txt) | canonical |
| D023 | Context limitation | 3 | data flow | Why do histories with one last ID collide? | [`model/context-limitation.txt`](model/context-limitation.txt) | canonical |
| D024 | ENGINE-1 overview | 3 | architecture | Where do weights and request state meet? | [`model/engine-1-overview.txt`](model/engine-1-overview.txt) | canonical |
| D025 | Sampling pipeline | 4 | control flow | In what order are logits processed? | [`sampling/sampling-pipeline.txt`](sampling/sampling-pipeline.txt) | canonical |
| D026 | Categorical intervals | 4 | data flow | How does one draw select one ID? | [`sampling/categorical-intervals.txt`](sampling/categorical-intervals.txt) | canonical |
| D027 | Autoregressive state | 4 | state machine | What changes after each sampled token? | [`sampling/autoregressive-state.txt`](sampling/autoregressive-state.txt) | canonical |
| D028 | Two requests | 4 | ownership | What is shared and request-local? | [`sampling/model-two-requests.txt`](sampling/model-two-requests.txt) | canonical |
| D029 | Part I token | 4 | data flow | What is one token's end-to-end path? | [`sampling/part1-follow-token.txt`](sampling/part1-follow-token.txt) | canonical |
| D030 | Part I byte | 4 | data flow | Where do bytes enter and leave? | [`sampling/part1-follow-byte.txt`](sampling/part1-follow-byte.txt) | canonical |
| D031 | Part I owner | 4 | ownership | Who owns each mutable resource? | [`sampling/part1-follow-owner.txt`](sampling/part1-follow-owner.txt) | canonical |
| D032 | Stable softmax | 4 | numerical flow | How are finite probabilities computed? | [`sampling/stable-softmax.txt`](sampling/stable-softmax.txt) | canonical |
| D033 | Terminal state machine | 4 | state machine | How is exactly one outcome enforced? | [`sampling/generation-terminal-state-machine.txt`](sampling/generation-terminal-state-machine.txt) | canonical |
| D034 | Shape and strides | 5 | memory layout | Which metadata defines a tensor view? | [`tensor/shape-and-strides.txt`](tensor/shape-and-strides.txt) | canonical |
| D035 | Logical/physical | 5 | memory layout | How do coordinates differ from storage? | [`tensor/logical-vs-physical.txt`](tensor/logical-vs-physical.txt) | canonical |
| D036 | Row-major offsets | 5 | memory layout | How are row-major offsets derived? | [`tensor/row-major-offsets.txt`](tensor/row-major-offsets.txt) | canonical |
| D037 | Contiguous/strided | 5 | memory layout | How can equal shapes traverse differently? | [`tensor/contiguous-vs-strided.txt`](tensor/contiguous-vs-strided.txt) | canonical |
| D038 | View versus copy | 5 | ownership | Which operation allocates elements? | [`tensor/view-vs-copy.txt`](tensor/view-vs-copy.txt) | canonical |
| D039 | Transpose view | 5 | tensor shape | How can transpose avoid movement? | [`tensor/transpose-view.txt`](tensor/transpose-view.txt) | canonical |
| D040 | Tensor ownership | 5 | ownership | Who owns storage and who borrows it? | [`tensor/tensor-ownership.txt`](tensor/tensor-ownership.txt) | canonical |
| D041 | Tensor lifetime | 5 | ownership | When may owners and views be released? | [`tensor/tensor-memory-lifetime.txt`](tensor/tensor-memory-lifetime.txt) | canonical |
| D042 | Follow element | 5 | data flow | How does an index reach one element? | [`tensor/follow-the-element.txt`](tensor/follow-the-element.txt) | canonical |
| D043 | Follow byte | 5 | data flow | How does an element offset become an address? | [`tensor/follow-the-byte.txt`](tensor/follow-the-byte.txt) | canonical |
| D044 | Follow owner | 5 | ownership | How does storage ownership change? | [`tensor/follow-the-owner.txt`](tensor/follow-the-owner.txt) | canonical |
| D045 | Reshape view | 5 | tensor shape | When can grouping change without copy? | [`tensor/reshape-view.txt`](tensor/reshape-view.txt) | canonical |
| D046 | Storage extent | 5 | memory layout | Why can element count understate storage? | [`tensor/storage-extent.txt`](tensor/storage-extent.txt) | canonical |
| D047 | Dot product | 6 | numerical flow | How does one accumulator form? | [`linear/dot-product-multiply-accumulate.txt`](linear/dot-product-multiply-accumulate.txt) | canonical |
| D048 | GEMV shapes | 6 | tensor shape | Which GEMV dimension contracts? | [`linear/gemv-shape-contract.txt`](linear/gemv-shape-contract.txt) | canonical |
| D049 | GEMM shapes | 6 | tensor shape | Which GEMM dimension contracts? | [`linear/gemm-shape-contract.txt`](linear/gemm-shape-contract.txt) | canonical |
| D050 | One GEMM cell | 6 | numerical flow | Which operands produce one cell? | [`linear/gemm-one-output-cell.txt`](linear/gemm-one-output-cell.txt) | canonical |
| D051 | Weight orientation | 6 | tensor shape | Which weight axis names outputs? | [`linear/weight-orientation.txt`](linear/weight-orientation.txt) | canonical |
| D052 | Row-major access | 6 | memory layout | Which accesses are adjacent? | [`linear/row-major-access.txt`](linear/row-major-access.txt) | canonical |
| D053 | Loop order | 6 | performance | How does loop order alter locality? | [`linear/loop-order-ijk-vs-ikj.txt`](linear/loop-order-ijk-vs-ikj.txt) | canonical |
| D054 | Cache tiling | 6 | performance | How does blocking create reuse? | [`linear/cache-reuse-and-tiling.txt`](linear/cache-reuse-and-tiling.txt) | canonical |
| D055 | Reference/blocked | 6 | architecture | How do kernel contracts differ? | [`linear/reference-vs-blocked-kernel.txt`](linear/reference-vs-blocked-kernel.txt) | canonical |
| D056 | Optimization ladder | 6 | control flow | In what order should optimization proceed? | [`linear/optimization-ladder.txt`](linear/optimization-ladder.txt) | canonical |
| D057 | GEMV/GEMM reuse | 6 | performance | Why do their reuse regimes differ? | [`linear/gemv-vs-gemm-reuse.txt`](linear/gemv-vs-gemm-reuse.txt) | canonical |
| D058 | Roofline | 6 | performance | What bounds attainable throughput? | [`linear/roofline-concept.txt`](linear/roofline-concept.txt) | canonical |
| D059 | ENGINE-2 stack | 6 | architecture | Where do kernels sit in the engine? | [`linear/engine-2-kernel-stack.txt`](linear/engine-2-kernel-stack.txt) | canonical |
| D060 | Follow FLOP | 6 | numerical flow | Where does one multiply-add contribute? | [`linear/follow-the-flop.txt`](linear/follow-the-flop.txt) | canonical |
| D061 | Follow byte | 6 | data flow | How do operand bytes reach arithmetic? | [`linear/follow-the-byte.txt`](linear/follow-the-byte.txt) | canonical |
| D062 | Follow reuse | 6 | performance | Where are reuse opportunities? | [`linear/follow-the-reuse.txt`](linear/follow-the-reuse.txt) | canonical |
| D063 | Candidate gates | 6 | control flow | When may a candidate replace reference? | [`linear/reference-candidate-gates.txt`](linear/reference-candidate-gates.txt) | canonical |
| D064 | Token to model space | 7 | data flow | Where does text become numerical activation? | [`transformer/token-to-model-space.txt`](transformer/token-to-model-space.txt) | canonical |
| D065 | Embedding logical layout | 7 | tensor shape | Which logical table row does one ID select? | [`transformer/embedding-logical-layout.txt`](transformer/embedding-logical-layout.txt) | canonical |
| D066 | Embedding physical layout | 7 | memory layout | How does `E[t,j]` reach a physical byte? | [`transformer/embedding-physical-layout.txt`](transformer/embedding-physical-layout.txt) | canonical |
| D067 | Parameters and activations | 7 | ownership | Which Chapter 7 values have model or request lifetime? | [`transformer/parameters-vs-activations.txt`](transformer/parameters-vs-activations.txt) | canonical |
| D068 | Embedding view or copy | 7 | ownership | Why does lookup return an owned activation? | [`transformer/embedding-view-vs-copy.txt`](transformer/embedding-view-vs-copy.txt) | canonical |
| D069 | Residual-stream width | 7 | tensor shape | What invariant does model dimension `D` establish? | [`transformer/residual-stream-width.txt`](transformer/residual-stream-width.txt) | canonical |
| D070 | RMS calculation | 7 | numerical flow | How do squares become one reciprocal RMS factor? | [`transformer/rms-calculation-pipeline.txt`](transformer/rms-calculation-pipeline.txt) | canonical |
| D071 | RMSNorm two passes | 7 | data flow | Why does the reference read input twice? | [`transformer/rmsnorm-two-pass.txt`](transformer/rmsnorm-two-pass.txt) | canonical |
| D072 | Equation to loop | 7 | numerical flow | How do equation, shapes, offsets, and loops align? | [`transformer/equation-to-loop.txt`](transformer/equation-to-loop.txt) | canonical |
| D073 | Epsilon and zero vector | 7 | control flow | How does positive epsilon define zero-vector behavior? | [`transformer/epsilon-zero-vector.txt`](transformer/epsilon-zero-vector.txt) | canonical |
| D074 | LayerNorm and RMSNorm | 7 | numerical flow | Which statistic and centering does each operator use? | [`transformer/layernorm-vs-rmsnorm.txt`](transformer/layernorm-vs-rmsnorm.txt) | canonical |
| D075 | Normalization precision | 7 | numerical flow | Which Chapter 7 stages use `f32` and can fail? | [`transformer/normalization-precision-flow.txt`](transformer/normalization-precision-flow.txt) | canonical |
| D076 | Embedding and projection | 7 | data flow | Why is row lookup different from output GEMV? | [`transformer/embedding-vs-output-projection.txt`](transformer/embedding-vs-output-projection.txt) | canonical |
| D077 | Transformer Primitives v1 | 7 | architecture | Which tested primitives now connect IDs to normalized vectors? | [`transformer/chapter07-engine-architecture.txt`](transformer/chapter07-engine-architecture.txt) | canonical |
| D078 | Hermon and llama.cpp path | 7 | architecture | Which relevant paths are CURRENT, PREVIEW, or LIBRARY? | [`transformer/hermon-llamacpp-normalization-path.txt`](transformer/hermon-llamacpp-normalization-path.txt) | canonical |
