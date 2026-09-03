# Transformer Diagrams

Canonical tensor-shape and operator flows. Chapter 7 establishes the first
Transformer primitives without beginning Q/K/V or attention.

- [`token-to-model-space.txt`](token-to-model-space.txt) — locates the discrete-to-numerical boundary.
- [`embedding-logical-layout.txt`](embedding-logical-layout.txt) — identifies the selected logical `[V,D]` row.
- [`embedding-physical-layout.txt`](embedding-physical-layout.txt) — maps an embedding coordinate to element and byte offsets.
- [`parameters-vs-activations.txt`](parameters-vs-activations.txt) — separates model-lifetime weights from request-lifetime activations.
- [`embedding-view-vs-copy.txt`](embedding-view-vs-copy.txt) — records the owned-output policy and its cost.
- [`residual-stream-width.txt`](residual-stream-width.txt) — states the model-width boundary established by `D`.
- [`rms-calculation-pipeline.txt`](rms-calculation-pipeline.txt) — derives reciprocal RMS from one vector.
- [`rmsnorm-two-pass.txt`](rmsnorm-two-pass.txt) — shows the reduction and scaling passes.
- [`equation-to-loop.txt`](equation-to-loop.txt) — lowers symbols through shapes and indexing into loops.
- [`epsilon-zero-vector.txt`](epsilon-zero-vector.txt) — contrasts rejected zero epsilon with a defined zero-vector result.
- [`layernorm-vs-rmsnorm.txt`](layernorm-vs-rmsnorm.txt) — distinguishes centering from RMS-only scaling.
- [`normalization-precision-flow.txt`](normalization-precision-flow.txt) — exposes the `f32` path and finite-range gates.
- [`embedding-vs-output-projection.txt`](embedding-vs-output-projection.txt) — contrasts row selection with vocabulary GEMV.
- [`chapter07-engine-architecture.txt`](chapter07-engine-architecture.txt) — summarizes Transformer Primitives v1.
- [`hermon-llamacpp-normalization-path.txt`](hermon-llamacpp-normalization-path.txt) — maps verified CURRENT, PREVIEW, and LIBRARY paths.
