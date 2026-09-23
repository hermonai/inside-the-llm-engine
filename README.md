# Inside the LLM Engine

## The Systems Engineering of Large Language Model Inference

An open technical book about how large-language-model inference engines work —
and why every modern engine is built the way it is.

A 7-billion-parameter model on a consumer GPU generates a token by reading all
of its weights from memory, once, while the chip's arithmetic units sit mostly
idle. Nearly everything in a modern engine — batching, paged KV memory, prefix
caching, quantization, FlashAttention, speculative decoding, disaggregated
serving — is a way of spending one scarce resource to relieve another. This
book is organised around that accounting:

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

42 chapters in nine parts — see [BOOK.md](BOOK.md):

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

Each chapter opens with a real measurement, does the napkin math before the
code, derives the mechanism, charges it in a constraint ledger, runs a small
self-contained lab, tells what actually happened in a production engine —
failures included — and ends with a dated frontier watch and exercises.

Appendix A, *The Transformer From Scratch*, is the build-it-yourself path: a
dependency-free Rust engine with 220 tests and independent Python oracles.

## Status

The second edition began on 2026-09-23. It restructures a first edition whose
nine written chapters become Appendix A and Chapter 14. The chapters are
being drafted breadth-first; [docs/STATUS.md](docs/STATUS.md) says exactly what
exists and what has been verified. Frontier claims are checked against primary
sources and logged with dates in [research/FRONTIER.md](research/FRONTIER.md).

## Build

```bash
make textbook-check     # structure and consistency checks
make textbook           # PDF: XeLaTeX, latexmk and biber (TeX Live 2025)
(cd code/mini-engine && cargo test --workspace)
```

## Repository

| Path | Contents |
| --- | --- |
| `tex/` | The book: chapters, appendices, TikZ figures, worked problems |
| `code/` | `mini-engine` (Rust), reference oracles (Python), labs |
| `research/` | Frontier ledger, measurement records, first-edition research notes |
| `docs/` | The chapter plan and status |
| `archive/` | Superseded first-edition governance and plans |

Authors and AI agents start with [AGENTS.md](AGENTS.md) and
[AUTHORING.md](AUTHORING.md).

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
