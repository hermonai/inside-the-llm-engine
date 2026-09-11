# Industrial source map

Audit date: 2026-09-10. This ledger supplements, and does not replace, the
historical [Astra source map](astra/source-map.md). No external engine was
built or benchmarked in this audit. DOCUMENTED means a primary source describes
a capability; it does not prove deployment quality or compatibility with every
model, device, precision or feature combination. A repository HEAD is not a
release recommendation. Moving documentation is dated separately from code pins.

## Inspected revisions

| System | Revision observed | Inspection and boundary |
| --- | --- | --- |
| Hermon | `2a3fd5214e17ca7283656108847d02dcd6cdf7c5` | Local clean checkout; architecture document, dispatcher selection, batched state fields and API delegation inspected. Not fetched or benchmarked. |
| Hermon's vendored llama.cpp | `389ff61d77b5c71cec0cf92fe4e5d01ace80b797` | Vendor pin identified; do not substitute latest upstream behavior for this pin. |
| llama.cpp / GGML | `434ddbbc0e30522e897670681e503b797c12b7c1` | Upstream HEAD API snapshot; pinned server README inspected. Kernel implementation not audited. |
| vLLM | `2a02f6efe319c885e3ccbcecde402e0028f9ec1e` | HEAD API snapshot only; feature claims below come from separately dated official docs. |
| SGLang | `2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed` | HEAD API snapshot only; feature claims below come from separately dated official docs. |
| TensorRT-LLM | `430f24fdb4e01241ddc1219af27d70a31eb04f32` | HEAD API snapshot and pinned README feature/configuration sections inspected; disaggregation details from moving docs. |

## Claim-level evidence

| ID | Primary source | What was actually checked | Reuse boundary |
| --- | --- | --- | --- |
| H1 | [Hermon dispatcher](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/crates/hermon-runtime/src/dispatch.rs) | `RuntimeMode::from_env`: default/unknown selects Batched; pool compatibility; paged preview gate. | CURRENT route selection, not all provider claims. |
| H2 | [Hermon architecture](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/docs/CORE_ENGINE_ARCHITECTURE.md) | Batched worker owns model context; paged path distinguished from default. | Document statements retain CURRENT / PREVIEW / LIBRARY distinctions. |
| H3 | [Batched state](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/crates/hermon-runtime/src/batched.rs) | Per-sequence progress, sticky slot prefix fields, prompt-lookup configuration. | Field inspection, not a complete scheduling proof. |
| H4 | [API delegation](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/crates/hermon-api/src/engine_route.rs) | Delegation into Dispatcher streaming. | API call boundary only. |
| L1 | [Pinned llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/434ddbbc0e30522e897670681e503b797c12b7c1/tools/server/README.md) | Feature list and continuous-batching options. | DOCUMENTED: CPU/GPU quantized inference, multi-user batching, multimodal input, schema-constrained JSON, speculation, monitoring. No feature-combination test. |
| V1 | [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) | Body: choice, regex, JSON schema, grammar and backend selection. | DOCUMENTED on audit date; supported grammar syntax depends on backend. |
| V2 | [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | Body explicitly marks experimental; separate instances permit distinct prefill/decode tuning. | EXPERIMENTAL documented feature, not universal production readiness. |
| V3 | [vLLM documentation index](https://docs.vllm.ai/en/latest/) | Feature taxonomy and links inspected. | DISCOVERY ONLY for LoRA, quantization, parallelism, caching; chapter authors must inspect linked implementation/configuration before asserting support. |
| S1 | [SGLang disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) | Rationale, prefill/decode roles, router and Mooncake/NIXL transfer options. | DOCUMENTED; no transport performance validated. |
| S2 | [SGLang LoRA serving](https://docs.sglang.io/docs/advanced_features/lora) | Adapter selection via compatible and native APIs; backend choice. | DOCUMENTED; no quoted vendor speedup is adopted. |
| T1 | [Pinned TensorRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM/blob/430f24fdb4e01241ddc1219af27d70a31eb04f32/README.md) | Kernel/serving overview and configuration categories. | DOCUMENTED: parallelism, LoRA, speculation, prefix caching, CUDA graphs, chunked context. Not a backend support matrix. |
| T2 | [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html) | Separate pools, transfer abstraction and layout-conversion caveats. | DOCUMENTED; model/transport/layout restrictions must be rechecked before implementation. |
| P1 | [FlashAttention](https://arxiv.org/abs/2205.14135v2) | Primary abstract: exact attention with tiling to reduce HBM/SRAM traffic. | Abstract-level principle; no benchmark transfer or complete kernel proof. |
| P2 | [PagedAttention](https://arxiv.org/abs/2309.06180v1) | Primary abstract: KV memory management inspired by virtual memory. | Abstract-level principle; not a claim that paging replaces attention arithmetic. |
| P3 | [Speculative decoding](https://arxiv.org/abs/2211.17192v2) | Primary abstract: draft computation plus target verification can preserve target distribution. | Abstract-level claim; acceptance/rejection derivation still requires full-paper review in Chapter 51. |

## Evidence discipline for the new artwork

The four industrial plates are reference designs and analytical fixtures, not
diagrams reverse-engineered from any single vendor. Timing values and byte counts
are synthetic and checked by `figures/industrial.py`. Their ownership boundaries
are educational design contracts. No arrow implies Hermon has implemented a new
scheduler, distributed runtime, graph compiler or cache manager.

Before a vendor-specific chapter is marked TECH-REVIEW, add an inspected file or
configuration at a pin, test the actual feature combination, and record model,
hardware, precision, workload, command and result. A documentation-navigation
entry alone is insufficient evidence.
