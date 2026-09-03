# Inside the LLM Engine

## From First Token to Production-Grade Inference

Build and understand an industrial LLM inference engine with Rust, C, GGUF,
quantization, KV caching, paged attention, continuous batching, SIMD, GPU
kernels, speculative decoding, MoE paging, and distributed serving.

This open engineering book follows an LLM request from text to streamed tokens
and follows model and state bytes from a GGUF file through memory, kernels, and
hardware. The reader progressively builds `mini-engine`: first a token
generator, then a real model runner, and finally a production-shaped inference
runtime.

[Part I, Chapters 1–4](manuscript/part-01/README.md), and the
standard-library-only [ENGINE-1](code/mini-engine/README.md) are complete. They
establish the request-to-terminal lifecycle, tokenizer/chat contract,
byte-safe output, a real token-ID-to-logits model, request-owned sampling, and
the complete autoregressive feedback loop. Part II is now in progress:
[Chapter 5](manuscript/part-02/chapter-05-tensors-without-magic.md) adds the
checked Tensor Substrate v1. [Chapter 6](manuscript/part-02/chapter-06-matrix-multiplication-the-engine-room.md)
builds ENGINE-2's reference and blocked scalar linear-algebra kernels and
migrates the existing projection through GEMV. [Chapter 7](manuscript/part-02/chapter-07-embeddings-and-normalization.md)
adds checked single/sequence embedding and RMSNorm as Transformer Primitives
v1 while preserving the original tiny-model regression. Chapter 8 is next.

```text
                         INSIDE THE LLM ENGINE

  "Explain quantum computing."                         immutable model data
              │                                                ║
              ▼                                                ▼
       ┌─────────────┐      token IDs                  ┌────────────────┐
       │  TOKENIZER  │ ──────────────────────────────▶ │ GGUF / WEIGHTS │
       └─────────────┘                                 │ packed / quant │
              │                                        └────────┬───────┘
              ▼                                                 ║
       ┌─────────────┐       request state                      ║
       │ API / STREAM│ ──────────────────────────────────┐      ║
       └──────┬──────┘                                  │      ║
              │                                         ▼      ▼
              │                                  ┌──────────────────────┐
              └────────────────────────────────▶ │ REQUEST RUNTIME      │
                                                 │ admission / scheduler│
                                                 │ prefix / KV ownership│
                                                 └──────────┬───────────┘
                                                            │
                                               physical token batch
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ MODEL FORWARD PASS   │
                                                 │ tensors / attention  │
                                                 └──────────┬───────────┘
                                                            │
                                                 execution plan + state
                                                            ▼
                                     ┌──────────────────────┼────────────────┐
                                     ▼                      ▼                ▼
                                ┌──────────┐           ┌──────────┐     ┌──────────┐
                                │ CPU SIMD │           │  Metal   │     │CUDA/ROCm │
                                └─────┬────┘           └────┬─────┘     └────┬─────┘
                                      └──────────┬──────────┴────────────────┘
                                                 │
                                              logits
                                                 ▼
                                         sampler ──▶ token
                                                 │
                                  decode bytes ──▶ stream ──▶ repeat

  Legend:  ──▶ control flow    ══▶ bulk data    [state] mutable ownership
```

## Why this book exists

Transformer explanations often stop at equations. Serving documentation often
starts after the model has become a black box. Kernel guides, memory managers,
and production API manuals live in separate worlds. An inference engine has to
make all of them agree: tensor semantics, byte layouts, ownership, scheduling,
hardware execution, failure behavior, correctness, and measured performance.

The book teaches those connections from the inside out. Every major optimization
begins with the failure of a simpler design. Every performance claim needs a
reproducer. Every optimized path needs an oracle.

## What readers will build

The curriculum advances through eleven named milestones:

```text
ENGINE-0  token generator
    └──▶ ENGINE-1  tiny neural language model
          └──▶ ENGINE-2  linear algebra kernel layer
                └──▶ ENGINE-3  real GGUF model runner
                      └──▶ ENGINE-4  KV-cached decoder
                            └──▶ ENGINE-5  concurrent inference server
                                  └──▶ ENGINE-6  continuous-batched runtime
                                        └──▶ ENGINE-7  paged-KV runtime
                                              └──▶ ENGINE-8  native kernel runtime
                                                    └──▶ ENGINE-9  accelerated runtime
                                                          └──▶ ENGINE-10 production system
```

Labs move through four levels: **CHECK** a concept, **BUILD** it, **BREAK** it
deliberately, and **EXTEND** it with a measured improvement.

## Who this is for

- Programmers who have called an LLM API and want to know what happens beneath it.
- Systems developers moving into numerical and inference engineering.
- ML engineers moving below framework abstractions.
- Inference engineers studying quantization, cache design, scheduling, kernels,
  accelerators, MoE, or distributed execution.
- Infrastructure architects reasoning about latency, throughput, memory tiers,
  isolation, and serving economics.

The early parts establish tensor and model prerequisites. Later parts assume
comfort with Rust, C, systems programming, and measurement, with appendices
providing focused refreshers.

## The progression

The manuscript contains **15 parts and 94 chapters**, followed by 14 reference
appendices. It moves through conceptual inference, a Transformer from scratch,
GGUF and quantization, KV caching, serving and scheduling, paged memory, native
kernels, accelerators, modern decode optimization, MoE, correctness,
production, the Hermon case study, frontier architecture, and a graduation
implementation.

- [BOOK.md](BOOK.md) is the readable table of contents.
- [The detailed authoring outline](docs/OUTLINE.md) specifies every chapter.
- [The roadmap](docs/ROADMAP.md) explains the staged build.
- [The current status](docs/STATUS.md) distinguishes scaffolding from completed work.

## Hermon: production evidence, not the subject of the book

[Hermon](https://github.com/hermonai/hermon) is the primary production
reference architecture. It supplies real examples of request routing,
continuous batching, KV ownership, native kernel boundaries, accelerator
selection, negative performance results, and release gates. The teaching
engine begins smaller and does not copy Hermon.

Claims about Hermon are classified as current, preview, library-only, target,
historical, external, or inferred. The current inventory is recorded in
[research/hermon/README.md](research/hermon/README.md) against a specific commit.
Source code outranks planning documents for claims about what executes today.

## Major subjects

Tokens and sampling; tensor shapes; Transformer inference; GGUF; quantization;
packed matrix multiplication; profiling; prefill and decode; KV cache geometry;
continuous batching; backpressure and cancellation; paged KV; prefix radix
trees; copy-on-write; eviction; native C ABIs; arenas; online softmax; SIMD;
Metal, CUDA, and other providers; speculative decoding; MoE expert paging;
differential testing; observability; security; distributed inference; and
future inference-memory and execution protocols.

## Repository map

```text
manuscript/   chapter prose, organized by part
code/         reference examples, mini-engine, and experiments
diagrams/     reusable plain-text architecture artifacts
research/     evidence logs and source inventories
labs/         CHECK / BUILD / BREAK / EXTEND engineering exercises
docs/         constitution, outline, policies, workflow, roadmap, and status
scripts/      repository checks and reproducibility helpers
```

Start with [the book constitution](docs/BOOK_CONSTITUTION.md) before drafting.
Contributors and AI agents should also read [AGENTS.md](AGENTS.md),
[the source policy](docs/SOURCE_POLICY.md), and
[the chapter contract](docs/CHAPTER_CONTRACT.md). Mathematical and visual
artifacts follow [the math style](docs/MATH_STYLE.md) and
[the diagram style](docs/DIAGRAM_STYLE.md).

## Status

Phase 0 established the repository and editorial architecture. Phase 1 is
complete. Phase 2 is in progress: Chapters 5–7, Tensor Substrate v1, ENGINE-2
plus Transformer Primitives v1, Labs 16–38, and their source-verified research,
independent oracles, 78 Unicode diagrams, and four performance records pass
their gates. The Chapter 1–6 diagram/math retrofit is complete; Chapter 8 is
next. See
[docs/STATUS.md](docs/STATUS.md) for the authoritative ledger.

## Contributing and license

Corrections, technical review, diagrams, reproducible experiments, portability
work, exercises, and implementations are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a change.

No project license has been selected yet. Until maintainers explicitly choose
the prose and code licenses, normal copyright restrictions apply. The decision
is tracked in [docs/STATUS.md](docs/STATUS.md); do not assume Hermon's
Apache-2.0 license transfers to this separate repository.

This book is under active development. Its manuscript, APIs, code, diagrams,
and curriculum may evolve as implementations are tested and reviewed. Current
engine behavior and frontier proposals are kept visibly separate.
