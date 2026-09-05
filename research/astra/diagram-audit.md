# Per-diagram disposition

All 78 historical artifacts are preserved. These decisions govern their
next visual regeneration; KEEP does not assert that a new vector exists.

| ID | Source | Disposition | Reason / next check |
| --- | --- | --- | --- |
| D001 | [runtime/request-to-token.txt](../../diagrams/runtime/request-to-token.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D002 | [runtime/model-vs-engine.txt](../../diagrams/runtime/model-vs-engine.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D003 | [runtime/inference-stack.txt](../../diagrams/runtime/inference-stack.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D004 | [runtime/token-byte-owner.txt](../../diagrams/runtime/token-byte-owner.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D005 | [runtime/latency-decomposition.txt](../../diagrams/runtime/latency-decomposition.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D006 | [runtime/control-plane-data-plane.txt](../../diagrams/runtime/control-plane-data-plane.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D007 | [runtime/hermon-current-request-path.txt](../../diagrams/runtime/hermon-current-request-path.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D008 | [tokenizer/text-unicode-bytes-tokens.txt](../../diagrams/tokenizer/text-unicode-bytes-tokens.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D009 | [tokenizer/bpe-merge-process.txt](../../diagrams/tokenizer/bpe-merge-process.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D010 | [tokenizer/chat-template-pipeline.txt](../../diagrams/tokenizer/chat-template-pipeline.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D011 | [tokenizer/engine0-token-ownership.txt](../../diagrams/tokenizer/engine0-token-ownership.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D012 | [tokenizer/model-tokenizer-template-contract.txt](../../diagrams/tokenizer/model-tokenizer-template-contract.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D013 | [tokenizer/token-to-byte-stream.txt](../../diagrams/tokenizer/token-to-byte-stream.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D014 | [tokenizer/utf8-partial-token-boundary.txt](../../diagrams/tokenizer/utf8-partial-token-boundary.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D015 | [tokenizer/special-token-trust-boundary.txt](../../diagrams/tokenizer/special-token-trust-boundary.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D016 | [tokenizer/token-count-system-impact.txt](../../diagrams/tokenizer/token-count-system-impact.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D017 | [model/token-id-to-logits.txt](../../diagrams/model/token-id-to-logits.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D018 | [model/embedding-row-lookup.txt](../../diagrams/model/embedding-row-lookup.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D019 | [model/one-logit-dot-product.txt](../../diagrams/model/one-logit-dot-product.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D020 | [model/parameters-vs-activations.txt](../../diagrams/model/parameters-vs-activations.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D021 | [model/semantics-vs-execution.txt](../../diagrams/model/semantics-vs-execution.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D022 | [model/tiny-model-tensor-shapes.txt](../../diagrams/model/tiny-model-tensor-shapes.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D023 | [model/context-limitation.txt](../../diagrams/model/context-limitation.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D024 | [model/engine-1-overview.txt](../../diagrams/model/engine-1-overview.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D025 | [sampling/sampling-pipeline.txt](../../diagrams/sampling/sampling-pipeline.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D026 | [sampling/categorical-intervals.txt](../../diagrams/sampling/categorical-intervals.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D027 | [sampling/autoregressive-state.txt](../../diagrams/sampling/autoregressive-state.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D028 | [sampling/model-two-requests.txt](../../diagrams/sampling/model-two-requests.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D029 | [sampling/part1-follow-token.txt](../../diagrams/sampling/part1-follow-token.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D030 | [sampling/part1-follow-byte.txt](../../diagrams/sampling/part1-follow-byte.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D031 | [sampling/part1-follow-owner.txt](../../diagrams/sampling/part1-follow-owner.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D032 | [sampling/stable-softmax.txt](../../diagrams/sampling/stable-softmax.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D033 | [sampling/generation-terminal-state-machine.txt](../../diagrams/sampling/generation-terminal-state-machine.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D034 | [tensor/shape-and-strides.txt](../../diagrams/tensor/shape-and-strides.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D035 | [tensor/logical-vs-physical.txt](../../diagrams/tensor/logical-vs-physical.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D036 | [tensor/row-major-offsets.txt](../../diagrams/tensor/row-major-offsets.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D037 | [tensor/contiguous-vs-strided.txt](../../diagrams/tensor/contiguous-vs-strided.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D038 | [tensor/view-vs-copy.txt](../../diagrams/tensor/view-vs-copy.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D039 | [tensor/transpose-view.txt](../../diagrams/tensor/transpose-view.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D040 | [tensor/tensor-ownership.txt](../../diagrams/tensor/tensor-ownership.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D041 | [tensor/tensor-memory-lifetime.txt](../../diagrams/tensor/tensor-memory-lifetime.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D042 | [tensor/follow-the-element.txt](../../diagrams/tensor/follow-the-element.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D043 | [tensor/follow-the-byte.txt](../../diagrams/tensor/follow-the-byte.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D044 | [tensor/follow-the-owner.txt](../../diagrams/tensor/follow-the-owner.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D045 | [tensor/reshape-view.txt](../../diagrams/tensor/reshape-view.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D046 | [tensor/storage-extent.txt](../../diagrams/tensor/storage-extent.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D047 | [linear/dot-product-multiply-accumulate.txt](../../diagrams/linear/dot-product-multiply-accumulate.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D048 | [linear/gemv-shape-contract.txt](../../diagrams/linear/gemv-shape-contract.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D049 | [linear/gemm-shape-contract.txt](../../diagrams/linear/gemm-shape-contract.txt) | REDRAW | Incorrect operand-edge implication or mismatched reduction connectors; see visual audit. |
| D050 | [linear/gemm-one-output-cell.txt](../../diagrams/linear/gemm-one-output-cell.txt) | REDRAW | Incorrect operand-edge implication or mismatched reduction connectors; see visual audit. |
| D051 | [linear/weight-orientation.txt](../../diagrams/linear/weight-orientation.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D052 | [linear/row-major-access.txt](../../diagrams/linear/row-major-access.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D053 | [linear/loop-order-ijk-vs-ikj.txt](../../diagrams/linear/loop-order-ijk-vs-ikj.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D054 | [linear/cache-reuse-and-tiling.txt](../../diagrams/linear/cache-reuse-and-tiling.txt) | REDRAW | Incorrect operand-edge implication or mismatched reduction connectors; see visual audit. |
| D055 | [linear/reference-vs-blocked-kernel.txt](../../diagrams/linear/reference-vs-blocked-kernel.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D056 | [linear/optimization-ladder.txt](../../diagrams/linear/optimization-ladder.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D057 | [linear/gemv-vs-gemm-reuse.txt](../../diagrams/linear/gemv-vs-gemm-reuse.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D058 | [linear/roofline-concept.txt](../../diagrams/linear/roofline-concept.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D059 | [linear/engine-2-kernel-stack.txt](../../diagrams/linear/engine-2-kernel-stack.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D060 | [linear/follow-the-flop.txt](../../diagrams/linear/follow-the-flop.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D061 | [linear/follow-the-byte.txt](../../diagrams/linear/follow-the-byte.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D062 | [linear/follow-the-reuse.txt](../../diagrams/linear/follow-the-reuse.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D063 | [linear/reference-candidate-gates.txt](../../diagrams/linear/reference-candidate-gates.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D064 | [transformer/token-to-model-space.txt](../../diagrams/transformer/token-to-model-space.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D065 | [transformer/embedding-logical-layout.txt](../../diagrams/transformer/embedding-logical-layout.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D066 | [transformer/embedding-physical-layout.txt](../../diagrams/transformer/embedding-physical-layout.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D067 | [transformer/parameters-vs-activations.txt](../../diagrams/transformer/parameters-vs-activations.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D068 | [transformer/embedding-view-vs-copy.txt](../../diagrams/transformer/embedding-view-vs-copy.txt) | CONVERT TO UML | Actual owner composition and borrow dependencies; retain text contract. |
| D069 | [transformer/residual-stream-width.txt](../../diagrams/transformer/residual-stream-width.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D070 | [transformer/rms-calculation-pipeline.txt](../../diagrams/transformer/rms-calculation-pipeline.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D071 | [transformer/rmsnorm-two-pass.txt](../../diagrams/transformer/rmsnorm-two-pass.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D072 | [transformer/equation-to-loop.txt](../../diagrams/transformer/equation-to-loop.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D073 | [transformer/epsilon-zero-vector.txt](../../diagrams/transformer/epsilon-zero-vector.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D074 | [transformer/layernorm-vs-rmsnorm.txt](../../diagrams/transformer/layernorm-vs-rmsnorm.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D075 | [transformer/normalization-precision-flow.txt](../../diagrams/transformer/normalization-precision-flow.txt) | EXPAND INTO SEQUENCE | Separate before/transition/after; preserve terminal or numerical boundary. |
| D076 | [transformer/embedding-vs-output-projection.txt](../../diagrams/transformer/embedding-vs-output-projection.txt) | KEEP | Focused question and useful terminal form; add vector companion after semantic review. |
| D077 | [transformer/chapter07-engine-architecture.txt](../../diagrams/transformer/chapter07-engine-architecture.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
| D078 | [transformer/hermon-llamacpp-normalization-path.txt](../../diagrams/transformer/hermon-llamacpp-normalization-path.txt) | CONVERT TO DATAFLOW | Separate data, control, state and status; use UML when software structure is primary. |
