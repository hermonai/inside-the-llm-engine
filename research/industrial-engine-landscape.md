# Industrial engine landscape

2026-09-10. This is a curriculum evidence matrix, not a ranking or benchmark.
Source IDs resolve in the [source map](industrial-source-map.md). D = documented;
C = inspected current path; P = preview; N = not assessed in this audit.
N never means unsupported. Scope below deliberately distinguishes a vendor's
documented offering from the teaching engine's still-planned implementation.

| Capability / boundary | Hermon | llama.cpp / GGML | vLLM | SGLang | TensorRT-LLM | Durable principle / chapter home |
| --- | --- | --- | --- | --- | --- | --- |
| Default request path | C H1/H4 | D L1 | N | N | N | API is not a model operator; 1, 24, 74, 78 |
| CPU and accelerator execution | See H2, build-dependent | D L1 | N | N | D T1 | Provider selection needs proof of execution; 42–48 |
| Continuous batching | C fields H3; H2 design | D L1 | N | N | N | Re-form work at iteration boundaries; 25–27 |
| Slot-local prefix reuse | C fields H3 | N | N | N | N | Reuse policy is not physical paging; 49–50 |
| Paged runtime integration | P H1/H2 | N | N | N | N | Integration status separate from library availability; 28–34, 76 |
| Prefix caching | Not generalized from H3 | N | Discovery V3 | N | D T1 | Reuse identity includes model and relevant execution context; 31–33, 49 |
| Quantized inference | H2 route only | D L1 | Discovery V3 | N | D T1 configuration | Distinguish weight, activation and KV precision; 16–17, 22, 34 |
| Graph replay / compilation | N | N | N | N | D T1 configuration | IR, fusion and captured execution are different objects; 45, 48, 87 |
| Chunked prompt processing | H3 progress fields | N | N | N | D T1 | Bound prompt work per iteration; 20, 26–27 |
| Speculative generation | H3 opt-in prompt lookup | D L1 | Discovery V3 | N | D T1 | Optional verified branch with rollback, not mandatory stage; 51–53 |
| Structured output | N | D L1 JSON | D V1 | N | N | Constraint state belongs to the sequence/sampler contract; 4, 67 |
| Adapters | N | N | Discovery V3 | D S2 | D T1 | Adapter identity affects routing, memory and reuse; 68 |
| Multimodal input | N | D L1 | Discovery V3 | N | N | Encoders/projectors add typed inputs; 68, 80 |
| MoE kernels / parallelism | H2 library distinction | N | Discovery V3 | N | D T1 | Expert routing moves activations, not merely file offsets; 54–58, 84 |
| Tensor/pipeline/context/expert parallelism | N | N | Discovery V3 | N | D T1 | Partition placement determines communication; 84–85 |
| Prefill/decode separation | TARGET, not default H1 | N | Experimental D V2 | D S1 | D T2 | Transfer cost and ownership are first-class; 83 |
| Monitoring | API contract only H4 | D L1 | Discovery V3 | N | N | Use a consistent metric boundary; 19–20, 69–73 |

The important comparison is architectural, not the number of checked boxes.
Hermon's inspected default route is a batched worker delegated through its
bridge, while its paged route is explicitly gated. llama.cpp's pinned server
describes a lightweight native serving surface. vLLM documents constrained
generation and labels its disaggregated prefill path experimental. SGLang's
documentation makes adapters and disaggregated roles explicit. TensorRT-LLM's
README exposes graph, parallelism and serving choices as configuration concerns.
These are distinct teaching entry points, not evidence that one project wins.

## Research still required at chapter-production time

Read pinned kernel/layout implementations for FlashAttention, quantized KV,
GGML graph allocation, vLLM block ownership, SGLang radix/HiCache and modern
speculative variants such as EAGLE and MTP. Inspect safetensors' actual format
specification before extending Chapter 14. Check feature intersections—for
example adapter plus prefix reuse plus speculation—rather than adding isolated
feature names. None of those deeper source audits is claimed complete here.
