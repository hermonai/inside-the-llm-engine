# Inside the LLM Engine

## The Systems Engineering of Large Language Model Inference

An open technical book about how large-language-model inference engines work —
and why every modern engine is built the way it is.

Start with a token, build a numerical model and assemble a decoder you can
inspect. Then follow its weights, intermediate values and cached state into
real hardware and a multi-user service. The book teaches the computation
before asking the reader to optimize it.

Later chapters explain batching, paged KV memory, prefix caching,
quantization, FlashAttention, speculative decoding and distributed serving
as ways of spending one resource to relieve another:

| Resource | Where it binds |
| --- | --- |
| Memory bandwidth — bytes moved per token | decode |
| Memory capacity — bytes resident | batch size, context length, model size |
| Compute — FLOPs | prefill, large batches |
| Latency — time to first token, time per output token | the user |

Every technique enters as a response to a measured constraint and is charged
for what it spends, with the two outcomes that matter: a correct answer, and
tokens per dollar at a latency target. When next year's method arrives, it
slots into the same ledger.

## What is in it

59 chapters in ten parts — see [BOOK.md](BOOK.md):

- **Build and understand a language model:** the former appendices now form
  Part I, with detailed token/tensor construction, causal attention, the FFN,
  a complete small decoder, checkpoint formats, numerics, hardware and
  measurement. Native LaTeX illustrations and worked answers throughout.
- **The physics of inference:** the roofline, prefill versus decode, the KV
  cache, and napkin math for latency, throughput and cost.
- **Making one request fast:** GEMM on real hardware, FlashAttention and decode
  attention, integer and FP8/MXFP4/NVFP4 quantization, KV quantization, and how
  kernels are written now.
- **Serving many requests:** the request lifecycle, continuous batching, paged
  KV, prefix caching, chunked prefill, scheduling under pressure, structured
  output and multi-LoRA serving.
- **Beyond one token per step:** speculative decoding, the EAGLE family and
  multi-token prediction, and when speculation loses.
- **Architectures that reshape the engine:** GQA and multi-head latent
  attention, mixture of experts, hybrid state-space models, long context,
  reasoning workloads and multimodal inference.
- **Scaling out:** tensor/pipeline/context parallelism, wide expert
  parallelism, disaggregated serving and the KV cache as a storage tier.
- **The hardware you have:** offloading, streaming experts from disk, unified
  memory.
- **Correctness and measurement**, **production**, real-engine anatomies and a
  capstone mini engine.

Part I works from small examples to executable checks. Systems chapters
introduce their question in plain language, then examine a real measurement, do the napkin math before the
code, derives the mechanism, charges it in a constraint ledger, runs a small
self-contained lab, tells what actually happened in a production engine —
failures included — and ends with a dated frontier watch and exercises.

The construction path uses the dependency-free Rust engine and independent
oracles, plus a standard-library Python decoder for the attention-to-generation
bridge. No model download or GPU is required for its small examples.

## Status

The second edition began on 2026-09-23. It restructures a first edition whose
nine written chapters were initially moved behind the systems material.
The author's 2026-09-28 revision brings that construction to the front.
The later systems chapters continue their depth pass;
[docs/STATUS.md](docs/STATUS.md) says exactly what
exists and what has been verified. Frontier claims are checked against primary
sources and logged with dates in [research/FRONTIER.md](research/FRONTIER.md).

## Build

```bash
make textbook-check     # structure and consistency checks
make textbook           # PDF: XeLaTeX, latexmk and biber (TeX Live 2025)
(cd code/mini-engine && cargo test --workspace)
python3 -m unittest discover -s code/foundations -p 'test_*.py'
```

## Repository

| Path | Contents |
| --- | --- |
| `tex/` | The book: foundations, systems chapters, TikZ figures, worked problems |
| `code/` | `mini-engine` (Rust), reference oracles (Python), labs |
| `research/` | Frontier ledger, measurement records, first-edition research notes |
| `docs/` | The chapter plan and status |
| `archive/` | Superseded first-edition governance and plans |

Authors and AI agents start with [AGENTS.md](AGENTS.md) and
[AUTHORING.md](AUTHORING.md).

Printed chapter numbers follow the reading order. Lab and source IDs remain
stable: the old `ch14-request-lifecycle` is now printed Chapter 31.
The complete mapping is in [docs/STRUCTURE.md](docs/STRUCTURE.md).

## Hermon

[Hermon](https://github.com/hermonai/hermon), a Rust inference engine with a C
kernel layer, supplies measured case studies — including optimizations that
failed and why. It is evidence, not the subject: the book compares it with
vLLM, SGLang, TensorRT-LLM and llama.cpp from their own primary sources.

## Companion

[Training Large Language Models From Scratch](https://github.com/mapleaiorg/llm-train-book)
covers the other side of the model-artifact boundary. This book does not teach
training.

## Contributing and licence

Corrections, reviews, measurements on hardware we do not have, and labs are
welcome; read [CONTRIBUTING.md](CONTRIBUTING.md). No licence has been selected
yet. Until the maintainers choose prose and code licences, normal copyright
applies; do not assume Hermon's Apache-2.0 licence transfers to this
repository.
