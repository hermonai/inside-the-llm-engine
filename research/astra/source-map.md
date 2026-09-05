# Production source refresh

Inspected 2026-09-05 after fetching both origins. Book starts at `8588ff8`.
Hermon HEAD and origin/main both resolve to
`472a44cdb511b2dae6c9569e59543db8f8350b25`; pinned llama.cpp/GGML is
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`. No upstream source was modified.
This is source reconnaissance, not a new runtime benchmark or real-model test.

| Concept | Exact source at inspected revision | Classification / verified boundary |
| --- | --- | --- |
| Runtime selection | [dispatch.rs](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/dispatch.rs) `RuntimeMode::from_env`, runtime construction | CURRENT: unset/unknown selects Batched. Paged remains gated PREVIEW. |
| Continuous batching | [batched.rs](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/batched.rs) `step`, `drain`/admission, worker loop | CURRENT: one worker owns context and batch; prompt chunks and one-token decode contributions can coexist. Capacity can defer work; columns in prototype are conceptual iterations. |
| Model loading/GGUF | [llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/src/llama-model-loader.cpp) `gguf_init_from_file` | CURRENT upstream path reads model metadata/tensors; file existence does not prove all model-family semantics. |
| Tensor representation | [ggml.h](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/include/ggml.h) `struct ggml_tensor` | CURRENT upstream: `ne` elements, `nb` byte strides, type/block-aware layout. Book TensorView strides count elements. |
| QKV/RoPE/attention | [models/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/src/models/llama.cpp) `build_qkv`, `ggml_rope_ext`, `build_attn` | CURRENT for this architecture: norm then QKV, rotate Q and K, then attention. Family-specific modes/scaling remain outside simple rotation prototype. |
| Projection graph | [llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/src/llama-graph.cpp) `build_qkv` | CURRENT upstream supports separate/fused weights, optional bias and views. Book planned bias-free row-major contract is an explicit simplification. |
| Context/KV | [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/src/llama-kv-cache.cpp) | CURRENT upstream cache implementation; dense growth prototype counts valid payload, not allocator overhead or actual physical GGML cache layout. |
| Graph execution | [llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/src/llama-context.cpp) `ggml_backend_sched_alloc_graph`, `ggml_backend_sched_graph_compute_async` | CURRENT upstream graph reaches backend scheduler. |
| Backend scheduling | [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/src/ggml-backend.cpp) `ggml_backend_sched_split_graph` | CURRENT library mechanism; device placement depends on configured providers and supported operations. |
| CPU matrix kernel | [ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/src/ggml-cpu/ggml-cpu.c) `ggml_compute_forward_mul_mat` | CURRENT CPU implementation selected by backend; quantized vector-dot traits differ from educational F32 GEMV. |
| Packed data | [ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/src/ggml-quants.c) | LIBRARY implementation of quantization formats, not proof that all formats reach every provider. |

The source-code zoom is model semantics → Hermon runtime/bridge → llama.cpp
graph → GGML operation → backend scheduler → configured kernel/provider.
Memory ownership crosses this sequence; it is not a second data-processing
operator. Do not connect a CPU kernel directly to a GPU and imply migration.

Open verification: actual model/provider matrix, paged preview equivalence,
memory allocator overhead, scheduling fairness under load, backend fallback
measurements. These need dedicated chapter experiments. The first milestone
makes none of those untested claims. See the preserved
[initial inventory](../hermon/README.md) for the broader historical context.
