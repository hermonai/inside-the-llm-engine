# Inside the LLM Engine

The active manuscript is now [authored LaTeX](tex/README.md), with editable
TikZ figures, numbered equations and worked solutions. Run `make textbook-check`
and `make textbook`. The nine-chapter textbook revision includes a rewritten
Chapter 1, substantive early-chapter corrections, 65 native mechanism figures
and 27 new worked synthesis problems. It is a partial editorial rewrite, not a
claim that the entire planned book or production engine is complete.

## From model weights and KV memory to industrial inference serving.

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
[Chapter 5](tex/chapters/ch05.tex) adds the
checked Tensor Substrate v1. [Chapter 6](tex/chapters/ch06.tex)
builds ENGINE-2's reference and blocked scalar linear-algebra kernels and
migrates the existing projection through GEMV. [Chapter 7](tex/chapters/ch07.tex)
adds checked single/sequence embedding and RMSNorm as Transformer Primitives
v1 while preserving the original tiny-model regression.
[Chapter 8](tex/chapters/ch08.tex) adds checked
raw Q/K/V projections, MHA/GQA/MQA geometry, borrowed head views and an
independent component oracle. [Chapter 9](tex/chapters/ch09.tex)
adds standard RoPE with explicit pairing, partial rotation, failure-safe
in-place mutation and an independent complex-number oracle. Chapter 10,
causal self-attention, is next.

The [Chapter 5 visual pilot](figures/chapter05-atlas.md) carries one matrix
through storage, transpose, reshape, slicing, copying and Rust ownership.
Eight embedded vector figures, formatted equations and an executable
Rust/Python/figure parity check connect the explanation to the implementation.
Use the [publication build](docs/FIGURE_BUILD.md) for the complete ebook and
standalone chapter PDFs. The [Chapter 6 visual edition](figures/chapter06-atlas.md)
now follows one executable dot/GEMV/GEMM fixture through addresses, loop order,
tiling, numerical equivalence and production source. Fourteen canonical plates
and two step sequences retain all three historical benchmark records, including
losses. The [Chapter 7 visual edition](figures/chapter07-atlas.md) adds ten
canonical plates and an executable two-pass RMSNorm trace. Chapter 8 adds ten
more native-vector plates connecting projection arithmetic, head layouts,
ownership, graph bundles and analytical serving-memory implications. The
historical native export contains nine chapters across 187 pages; the active
textbook edition is built independently from `tex/`. Chapter 9 adds
ten geometric TikZ plates covering rotations, frequency planes, relative
position, coordinate layout, numerical precision and production contracts.

Companion: [Training Large Language Models From Scratch](https://github.com/mapleaiorg/llm-train-book),
*From tokens and gradients to industrial distributed training.* The two books
meet at a versioned model-artifact contract; see [the boundary](docs/TRAINING_INFERENCE_BOUNDARY.md).

The [industrial architecture recalibration](docs/INDUSTRIAL_CURRICULUM_AUDIT.md)
preserves those milestones and proposes stronger compiler, adapter, structured
decoding and distributed-systems coverage. Read the four new
[reference architecture plates](figures/industrial-atlas.md) or the
[94-chapter illustration plan](figures/CHAPTER_ILLUSTRATION_PLAN.md).
These planning/reference artifacts do not mark future chapters complete.

Begin with one request and its terminal event in [Chapter 1](tex/chapters/ch01.tex).
The later [industrial architecture plates](figures/industrial-atlas.md) show the
planned system boundaries; they are reference designs, not current engine capabilities.

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

| Milestone | Reader-built capability |
| --- | --- |
| ENGINE-0 | Token generator |
| ENGINE-1 | Tiny numerical model and autoregressive loop |
| ENGINE-2 | Checked linear-algebra kernel layer |
| ENGINE-3 | Real GGUF model runner (planned) |
| ENGINE-4 | KV-cached decoder (planned) |
| ENGINE-5 | Concurrent inference server (planned) |
| ENGINE-6 | Continuous-batched runtime (planned) |
| ENGINE-7 | Paged-KV runtime (planned) |
| ENGINE-8 | Native kernel runtime (planned) |
| ENGINE-9 | Accelerated runtime (planned) |
| ENGINE-10 | Production system (planned) |

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

The planned curriculum contains **15 parts and 94 chapters**, followed by 14 reference
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

| Location | Purpose |
| --- | --- |
| `tex/` | Active authored chapters, native figures and worked problems |
| `code/` | Reference examples, mini-engine and experiments |
| `research/` | Evidence logs and source inventories |
| `labs/` | CHECK / BUILD / BREAK / EXTEND exercises |
| `docs/` | Contracts, outline, policies and status |
| `manuscript/`, `diagrams/` | Historical Markdown and text-diagram evidence |
| `scripts/` | Repository and textbook verification |

Start with [the book constitution](docs/BOOK_CONSTITUTION.md) before drafting.
Contributors and AI agents should also read [AGENTS.md](AGENTS.md),
[the source policy](docs/SOURCE_POLICY.md), and
[the chapter contract](docs/CHAPTER_CONTRACT.md). Mathematical and visual
artifacts follow [the math style](docs/MATH_STYLE.md) and
[the diagram style](docs/DIAGRAM_STYLE.md).

## Status

Phase 0 established the repository and editorial architecture. Phase 1 is
complete. Phase 2 is in progress: Chapters 5–9, Tensor Substrate v1, ENGINE-2,
Transformer Primitives v1, checked QKV and standard RoPE, Labs 16–58, and their source-verified
research pass their gates. There are 218 unit/integration tests plus two
compile-fail doctests, five chapter parity gates and 65 atlas plates. The
Chapter 1–6 diagram/math retrofit is complete; Chapter 10 is next. See
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

## Visual regeneration workstream

The first visual-regeneration milestone adds an [illustrated atlas](figures/ATLAS.md),
editable SVG sources, three step animations, and PDF/HTML build infrastructure.
See the [book audit](ASTRA_BOOK_AUDIT.md), [visual audit](ASTRA_VISUAL_AUDIT.md),
and [bounded regeneration plan](ASTRA_REGENERATION_PLAN.md). Nine chapters are
written; later prototype diagrams do not imply completed implementations.
The [shared textbook standard](TEXTBOOK_STANDARD.md) now governs new authoring.
The Markdown/HTML exporter is historical and does not regenerate the active LaTeX manuscript.
