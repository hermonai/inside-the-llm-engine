# Unicode Text Diagram Library

Canonical diagrams are polished Unicode text artifacts that remain readable in
source, terminals, AI context, and grayscale print. Follow
[`docs/DIAGRAM_STYLE.md`](../docs/DIAGRAM_STYLE.md); validate with
`scripts/check-diagram-style.py` and `scripts/check-diagram-width.py`. The full
metadata inventory is [`INDEX.md`](INDEX.md).

## Runtime — Chapter 1

- [`request-to-token.txt`](runtime/request-to-token.txt) — traces a request through success, cancellation, and failure.
- [`model-vs-engine.txt`](runtime/model-vs-engine.txt) — separates an immutable model artifact from runtime-owned execution state.
- [`inference-stack.txt`](runtime/inference-stack.txt) — locates library, engine, server, service, provider, and backend boundaries.
- [`token-byte-owner.txt`](runtime/token-byte-owner.txt) — previews the book's token, byte, and owner journeys.
- [`latency-decomposition.txt`](runtime/latency-decomposition.txt) — aligns total latency with named, measurable request intervals.
- [`control-plane-data-plane.txt`](runtime/control-plane-data-plane.txt) — distinguishes lifecycle decisions from tensor and byte movement.
- [`hermon-current-request-path.txt`](runtime/hermon-current-request-path.txt) — maps Hermon's verified path with explicit truth-status labels.

## Tokenization — Chapter 2

- [`text-unicode-bytes-tokens.txt`](tokenizer/text-unicode-bytes-tokens.txt) — distinguishes visible text, Unicode scalars, bytes, and token IDs.
- [`bpe-merge-process.txt`](tokenizer/bpe-merge-process.txt) — shows deterministic ranked pair merging.
- [`chat-template-pipeline.txt`](tokenizer/chat-template-pipeline.txt) — shows how structured chat becomes model input.
- [`engine0-token-ownership.txt`](tokenizer/engine0-token-ownership.txt) — identifies owners of input IDs and decoded byte state.
- [`model-tokenizer-template-contract.txt`](tokenizer/model-tokenizer-template-contract.txt) — binds model, tokenizer, template, and special IDs.
- [`token-to-byte-stream.txt`](tokenizer/token-to-byte-stream.txt) — follows generated IDs through bytes and UTF-8 framing.
- [`utf8-partial-token-boundary.txt`](tokenizer/utf8-partial-token-boundary.txt) — shows why token pieces may need buffering before text emission.
- [`special-token-trust-boundary.txt`](tokenizer/special-token-trust-boundary.txt) — prevents untrusted surface text from forging control identities.
- [`token-count-system-impact.txt`](tokenizer/token-count-system-impact.txt) — connects token count to work, memory, context, and scheduling.

## Tiny model — Chapter 3

- [`token-id-to-logits.txt`](model/token-id-to-logits.txt) — follows one ID through gather and projection to logits.
- [`embedding-row-lookup.txt`](model/embedding-row-lookup.txt) — identifies the selected embedding row and resulting vector.
- [`one-logit-dot-product.txt`](model/one-logit-dot-product.txt) — expands one projection row into multiply-accumulate terms.
- [`parameters-vs-activations.txt`](model/parameters-vs-activations.txt) — separates model-lifetime weights from forward-lifetime values.
- [`semantics-vs-execution.txt`](model/semantics-vs-execution.txt) — separates numerical meaning from implementation strategy.
- [`tiny-model-tensor-shapes.txt`](model/tiny-model-tensor-shapes.txt) — records every ENGINE-1 tensor shape.
- [`context-limitation.txt`](model/context-limitation.txt) — exposes the model's last-token-only defect.
- [`engine-1-overview.txt`](model/engine-1-overview.txt) — joins model weights, request history, forward computation, and sampler state.

## Sampling — Chapter 4

- [`sampling-pipeline.txt`](sampling/sampling-pipeline.txt) — orders logit processing, normalization, RNG draw, and selection.
- [`categorical-intervals.txt`](sampling/categorical-intervals.txt) — maps a uniform random draw onto cumulative probability intervals.
- [`autoregressive-state.txt`](sampling/autoregressive-state.txt) — shows feedback and the state mutated by each generation step.
- [`model-two-requests.txt`](sampling/model-two-requests.txt) — separates shared model parameters from per-request sampler state.
- [`part1-follow-token.txt`](sampling/part1-follow-token.txt) — summarizes one token's complete Part I path.
- [`part1-follow-byte.txt`](sampling/part1-follow-byte.txt) — summarizes bytes from prompt input to streamed text.
- [`part1-follow-owner.txt`](sampling/part1-follow-owner.txt) — summarizes ownership across model, request, sampler, and stream.
- [`stable-softmax.txt`](sampling/stable-softmax.txt) — derives finite probabilities through max shifting.
- [`generation-terminal-state-machine.txt`](sampling/generation-terminal-state-machine.txt) — enforces exactly one terminal request outcome.

## Tensor substrate — Chapter 5

- [`shape-and-strides.txt`](tensor/shape-and-strides.txt) — pairs logical dimensions with physical stride metadata.
- [`logical-vs-physical.txt`](tensor/logical-vs-physical.txt) — distinguishes logical coordinates from storage positions.
- [`row-major-offsets.txt`](tensor/row-major-offsets.txt) — derives offsets for a canonical row-major matrix.
- [`contiguous-vs-strided.txt`](tensor/contiguous-vs-strided.txt) — contrasts dense traversal with a strided logical view.
- [`view-vs-copy.txt`](tensor/view-vs-copy.txt) — distinguishes borrowed metadata from newly owned elements.
- [`transpose-view.txt`](tensor/transpose-view.txt) — shows transpose as shape/stride exchange without movement.
- [`tensor-ownership.txt`](tensor/tensor-ownership.txt) — names owner and immutable/mutable borrowers.
- [`tensor-memory-lifetime.txt`](tensor/tensor-memory-lifetime.txt) — places owners and views on a lifetime timeline.
- [`follow-the-element.txt`](tensor/follow-the-element.txt) — traces a logical element to one checked physical offset.
- [`follow-the-byte.txt`](tensor/follow-the-byte.txt) — connects dtype width and element offset to byte address.
- [`follow-the-owner.txt`](tensor/follow-the-owner.txt) — traces allocation, borrowing, materialization, and release.
- [`reshape-view.txt`](tensor/reshape-view.txt) — shows legal contiguous regrouping without a copy.
- [`storage-extent.txt`](tensor/storage-extent.txt) — proves why reachable extent can exceed logical element count.

## Linear algebra — Chapter 6

- [`dot-product-multiply-accumulate.txt`](linear/dot-product-multiply-accumulate.txt) — expands a dot product into an ordered accumulator trace.
- [`gemv-shape-contract.txt`](linear/gemv-shape-contract.txt) — states matrix-vector shapes and the contracted dimension.
- [`gemm-shape-contract.txt`](linear/gemm-shape-contract.txt) — states matrix-matrix shapes and the contracted dimension.
- [`gemm-one-output-cell.txt`](linear/gemm-one-output-cell.txt) — connects one output cell to one row-column dot product.
- [`weight-orientation.txt`](linear/weight-orientation.txt) — fixes the output-row weight convention used by ENGINE-2.
- [`row-major-access.txt`](linear/row-major-access.txt) — maps row-major coordinates to contiguous physical access.
- [`loop-order-ijk-vs-ikj.txt`](linear/loop-order-ijk-vs-ikj.txt) — contrasts access and reuse under two equivalent loop nests.
- [`cache-reuse-and-tiling.txt`](linear/cache-reuse-and-tiling.txt) — shows how blocking keeps a working tile reusable.
- [`reference-vs-blocked-kernel.txt`](linear/reference-vs-blocked-kernel.txt) — contrasts broad reference and narrow blocked contracts.
- [`optimization-ladder.txt`](linear/optimization-ladder.txt) — orders correctness-preserving optimization stages.
- [`gemv-vs-gemm-reuse.txt`](linear/gemv-vs-gemm-reuse.txt) — explains why GEMV and GEMM expose different reuse.
- [`roofline-concept.txt`](linear/roofline-concept.txt) — relates arithmetic intensity to analytical throughput ceilings.
- [`engine-2-kernel-stack.txt`](linear/engine-2-kernel-stack.txt) — locates public kernels, implementations, tensors, and model use.
- [`follow-the-flop.txt`](linear/follow-the-flop.txt) — traces one multiply and accumulation into an output cell.
- [`follow-the-byte.txt`](linear/follow-the-byte.txt) — traces operand bytes from storage through reuse.
- [`follow-the-reuse.txt`](linear/follow-the-reuse.txt) — distinguishes spatial and temporal reuse opportunities.
- [`reference-candidate-gates.txt`](linear/reference-candidate-gates.txt) — gates candidate selection on contract, equivalence, and evidence.

## Transformer primitives — Chapter 7

- [`token-to-model-space.txt`](transformer/token-to-model-space.txt) — locates the discrete-to-numerical boundary.
- [`embedding-logical-layout.txt`](transformer/embedding-logical-layout.txt) — shows which `[V,D]` row one ID selects.
- [`embedding-physical-layout.txt`](transformer/embedding-physical-layout.txt) — maps logical embedding coordinates to storage.
- [`parameters-vs-activations.txt`](transformer/parameters-vs-activations.txt) — separates model and request ownership.
- [`embedding-view-vs-copy.txt`](transformer/embedding-view-vs-copy.txt) — explains the owned lookup output.
- [`residual-stream-width.txt`](transformer/residual-stream-width.txt) — establishes the model-width boundary.
- [`rms-calculation-pipeline.txt`](transformer/rms-calculation-pipeline.txt) — derives reciprocal RMS.
- [`rmsnorm-two-pass.txt`](transformer/rmsnorm-two-pass.txt) — traces reduction then scaling.
- [`equation-to-loop.txt`](transformer/equation-to-loop.txt) — connects notation, shapes, indexing, and loops.
- [`epsilon-zero-vector.txt`](transformer/epsilon-zero-vector.txt) — defines zero-vector and epsilon behavior.
- [`layernorm-vs-rmsnorm.txt`](transformer/layernorm-vs-rmsnorm.txt) — distinguishes the two operators.
- [`normalization-precision-flow.txt`](transformer/normalization-precision-flow.txt) — identifies `f32` stages and failures.
- [`embedding-vs-output-projection.txt`](transformer/embedding-vs-output-projection.txt) — separates lookup from GEMV.
- [`chapter07-engine-architecture.txt`](transformer/chapter07-engine-architecture.txt) — summarizes the Chapter 7 milestone.
- [`hermon-llamacpp-normalization-path.txt`](transformer/hermon-llamacpp-normalization-path.txt) — classifies relevant industrial paths.

Future topic directories reserve canonical locations for accelerators,
distributed execution, kernels, memory, MoE, and scheduling.
