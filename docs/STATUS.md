# Project Status

## Chapter 9 - verified RoPE milestone, 2026-09-12

COMPLETE: first-principles rotation and relative-position derivations, checked
adjacent/split-half scalar RoPE, partial prefixes, F64 phases with F32 storage,
failure-atomic in-place mutation, an independent complex-number oracle and
complete embedding/RMSNorm/QKV/position composition. Ten canonical geometric
TikZ plates and Labs 49-58 accompany the chapter.

All 218 unit/integration tests plus two compile-fail doctests pass, including
25 new RoPE tests. Rust format/check/Clippy, all five chapter parity gates,
65-scene generation, native TikZ parity, industrial-plan/mutation gates,
structure/links/math/legacy diagram checks and browser QA pass. Maximum oracle
discrepancy is below 1e-6 for the checked fixture outputs.

The native LaTeX ebook is 187 pages across nine written chapters; conventional
vector PDF is 182 pages, standalone Chapter 9 is 19 pages, and the atlas has
65 plates (52 canonical chapter figures, nine prototypes, four architecture
references). All new pages and plates were reviewed; color/grayscale, offline
MathML/SVG, 1024/768/390px and page-bound checks pass. A long inline vector was
moved to display math; the native log has no overfull boxes or missing glyphs.
See [the technical/editorial review](../research/part-02/chapter-09-position-rope.md).

Next: **Chapter 10 - Causal Self-Attention**. No attention, KV cache, model-specific
scaling, optimized RoPE candidate or new benchmark is claimed. The training
book remains unchanged. Dated entries below preserve milestone-time counts
and next-step recommendations; this entry and the current ledger supersede them.

## Chapter 8 — verified QKV milestone, 2026-09-12

COMPLETE: a 5,215-word projection chapter with ten canonical vector/TikZ
illustrations, checked one-token bias-free QKV and borrowed head views,
independent Python oracle, raw/composed numerical traces, and Labs 39–48.
The full suite passes 193 unit/integration tests plus two compile-fail doctests;
Rust format/check/Clippy and all four chapter parity gates pass. The source
review retains Hermon's CURRENT batched / PREVIEW paged distinction at
`2a3fd521`, with pinned llama.cpp `389ff61d` (inspected September 11).

The native `.tex` edition is now 169 pages / eight written chapters. The
standalone Chapter 8 is 19 pages; the atlas has 55 plates, including 42
canonical chapter figures, nine prototypes and four architecture references.
PDF vector/glyph/page-bound checks and offline 1024/768/390px browser checks
pass. The rendered review corrected a note/panel overlap and added a panel
boundary regression gate. See [the review record](../research/part-02/chapter-08-queries-keys-and-values.md).

This supersedes the historical next-task statements below. No attention,
position operator, KV cache, GGUF loader or new performance candidate was
implemented. The next new curriculum chapter is **Chapter 9 — Position: RoPE
From First Principles**. The paired training book is unchanged in this milestone.

## Chapter 7 visual regeneration — 2026-09-11

COMPLETE: connect the existing lookup and RMSNorm operators with ten canonical
vector plates, an input-only fixture, an independent oracle and a two-pass
step sequence. The 178 unit/integration tests and two compile-fail doctests pass.
Three Rust/Python/figure parity gates and browser/PDF checks pass. The atlas now
has 45 plates: 32 canonical, nine prototypes and four architecture references.
Industrial drafts are integrated; Chapter 8 research is preserved.
Chapter 8 remains the next new curriculum chapter; no attention work is added.

The paired training book owns an executable CPU core and local resume contract;
see [the boundary](TRAINING_INFERENCE_BOUNDARY.md). A portable
[LaTeX edition](../publication/latex/README.md) ships native TikZ plates and
geometric legacy diagrams, without publishing character-box graphs.

## Chapter 6 industrial regeneration — 2026-09-08

COMPLETE: the 9,984-word Chapter 6 now embeds fourteen canonical vector plates
and two accessible step sequences. One input-only fixture connects the real
Rust kernels, independent F32 oracle, addresses, loop contributions and tile
tails. D049/D050/D054 and two adjacent legacy diagrams have repaired operand
semantics. Labs 22–29 retain their numbering and add deterministic checkpoints.

All 173 unit/integration tests, two compile-fail doctests, six Python oracles,
six release examples, both chapter parity gates and repository/publication
checks pass. The current full ebook is 137 pages; Chapter 6 is 33 pages;
Chapter 5 remains 23 pages; the atlas is 31 vector plates (22 canonical,
nine prototypes). Offline HTML passes 1024/768/390px, keyboard and reduced-motion
checks. Historical benchmark records and numerical engine implementations are
unchanged. The PDF builder also repairs two older off-page inline source paths.

See the [research and review record](../research/astra/chapter06-regeneration.md),
[Chapter 6 atlas](../figures/chapter06-atlas.md), and
[publication instructions](FIGURE_BUILD.md).
Next bounded regeneration: **Chapter 7 — Embeddings and Normalization**.
Do not begin it in this milestone. Chapter 8 research/status work below is
preserved separately and is not complete.

## Chapter 5 visual regeneration — 2026-09-07

COMPLETE: the 7,468-word tensor pilot now embeds eight canonical vector figures,
formatted mathematics, actual Rust ownership UML and a five-state executable
Rust/Python/fixture contract. Labs 16-21 retain their original exercises and add
visual checkpoints. The suite at that milestone passed 167 unit/integration tests and two
compile-fail doctests; all five existing oracles and historical generation pass.
The ebook at that milestone was 123 pages, the standalone chapter 23, and the visual atlas
17. Offline HTML passes desktop/tablet/phone checks. See the
[review record](../research/astra/chapter05-regeneration.md) and
[chapter atlas](../figures/chapter05-atlas.md).

That milestone handed off to Chapter 6, now complete; Chapter 7 regeneration
comes next, before completing Chapter 8. The older curriculum ledger and in-progress Chapter 8
work below are preserved; this milestone does not complete any unwritten chapter.

## Visual regeneration milestone — 2026-09-06

The bounded first regeneration milestone is COMPLETE: audits of seven chapters,
78 legacy diagrams and 112 display equations; ten semantic visual prototypes;
three playable sequences; verified PDF/HTML builds; and expanded figure/math/link
checks. See [validation](../research/astra/validation.md) and the
[regeneration plan](../ASTRA_REGENERATION_PLAN.md). The next regeneration task
was the Chapter 5 tensor pilot, now complete; Chapter 6 has also passed and Chapter 7 follows.
The curriculum ledger below retains its existing chapter statuses; this visual
milestone does not mark any unwritten chapter complete.

Last updated: 2026-09-04.

## Phase 0 ledger

| Area | Status | Evidence / next gate |
| --- | --- | --- |
| Repository bootstrap | COMPLETE | Empty public repository cloned and structured |
| Book constitution and policies | COMPLETE | Core editorial/source/code/math/style/benchmark contracts created |
| Master outline | OUTLINED | 15 parts, 94 chapter authoring specifications; review again before each phase |
| Public README and BOOK | COMPLETE | Launch-facing overview and table of contents agree |
| Glossary and terminology | IN PROGRESS | Chapters 1–7 system, token, numerical model, sampling, tensor-memory, linear-algebra, embedding, normalization, and streaming terms added; expand with each chapter |
| Hermon reconnaissance | COMPLETE | Initial map plus Chapters 1–7 request/tokenizer/logit/sampling/tensor/kernel/embedding/normalization boundaries verified at `hermon` commit `472a44c` |
| Manuscript part indexes | COMPLETE | 15 part contracts plus appendices scaffolded |
| Diagram system | COMPLETE | Seventy-eight inventoried canonical Unicode diagrams; shared grammar plus automated style and display-width gates |
| Diagram/math retrofit | COMPLETE | Chapters 1–6 audited; 11 diagrams added, 2 redesigned, 47 equation blocks standardized, and 18 explicit shape declarations added |
| Research system | COMPLETE | Inventories and note templates established |
| Code project | COMPLETE | ENGINE-2 plus Transformer Primitives v1 provide dependency-free checked kernels, embedding lookup, and RMSNorm over Tensor Substrate v1; 173 unit/integration tests plus two doctests pass |
| Initial CI | COMPLETE | Structure, links, diagram style/width, math structure, Rust format/check/test/Clippy workflow added |
| License | PLANNED | Maintainers must choose prose and code licensing; no license inferred from Hermon |

Phase 0 remains complete as repository architecture. Phase 1 completion is
tracked separately below.

## Phase 1 ledger

| Scope | Status | Evidence / next gate |
| --- | --- | --- |
| Part I (Ch. 1–4) | COMPLETE | Four reviewed chapters, complete ENGINE-1 generation loop, Labs 1–15, and independent numerical oracles |
| Chapter 1 — The Missing Half of AI | COMPLETE | 6,913-word reviewed chapter, primary-source research, seven canonical diagrams |
| Chapter 2 — From Text to Tokens | COMPLETE | 7,373-word reviewed chapter, primary-source research, nine canonical diagrams, two-tokenizer comparison |
| Chapter 3 — The Smallest Possible Language Model | COMPLETE | 6,373-word reviewed chapter, primary-source research, eight canonical diagrams, full-vector Python oracle |
| Chapter 4 — Logits, Sampling, and the Autoregressive Loop | COMPLETE | 6,306-word reviewed chapter, primary-source research, nine canonical diagrams, fixed-draw Python oracle |
| ENGINE-0 | COMPLETE | Dependency-free tokenized request/runtime/stream lifecycle; byte oracle, BPE, chat/template contract, strict UTF-8 framing; 37 tests and full Rust gate pass |
| ENGINE-1 | COMPLETE | Immutable model logits; separate greedy and stochastic selection; stable softmax, temperature, top-k/top-p, categorical sampling, request-owned seeded RNG, feedback, and single terminal owner; 83 tests at the Phase 1 boundary, 173 unit/integration tests plus two doctests in the current full suite |
| Lab 1 — Generate One Token Manually | COMPLETE | Independent candidate oracle plus CHECK/BUILD/BREAK/EXTEND exercise |
| Labs 2–4 — Tokenization / UTF-8 / chat template | COMPLETE | Hand BPE, split-byte streaming, malformed terminal policy, and wrong-template experiments |
| Labs 5–8 — Numerical forward / causality / context / shape | COMPLETE | Full hand logits, one-weight intervention, same-last-token proof, and typed malformed-shape failures |
| Labs 9–15 — Sampling / feedback / failure | COMPLETE | Stable softmax, temperature, fixed-draw categorical selection, top-k/top-p, full-loop tracing, seeded reproduction, and typed sampler failures |

## Phase 2 ledger

| Scope | Status | Evidence / next gate |
| --- | --- | --- |
| Part II (Ch. 5–13) | IN PROGRESS | Chapters 5–9 complete; causal attention is next |
| Chapter 5 — Tensors Without Magic | COMPLETE | 7,468-word regenerated chapter, eight embedded vector plates, thirteen retained legacy diagrams, fresh production source review, traversal record and independent visual/offset oracles |
| Tensor Substrate v1 | COMPLETE | Owned canonical `f32` tensors, immutable strided views, exclusive canonical mutation, checked indexing/extent arithmetic, explicit materialization, and ENGINE-1 parameter migration |
| Labs 16–21 — Tensor memory | COMPLETE | Hand offsets, metadata transpose, reshape gate, non-contiguous copy, overflow failures, and aliasing/mutation exercises |
| Chapter 6 — Matrix Multiplication: The Engine Room | COMPLETE | 9,984-word regenerated chapter, fourteen embedded vector plates, two step sequences, seventeen retained Unicode diagrams, three unchanged historical records and independent Rust/Python/visual parity |
| ENGINE-2 / Linear Algebra Kernel Layer v1 | COMPLETE | Strided dot/GEMV/GEMM reference kernels, canonical-only blocked scalar GEMM, explicit layout/ownership/error contracts, and ENGINE-1 projection migration |
| Labs 22–29 — Linear algebra kernels | COMPLETE | Hand dot/GEMV/GEMM, loop-order offsets, tile tails, typed failures, deterministic equivalence, rounding checks, visual checkpoints and GEMV/GEMM measurement |
| Chapter 7 — Embeddings and Normalization | COMPLETE | 6,034-word reviewed chapter, primary-source and Hermon/llama.cpp trace, fifteen canonical diagrams, independent oracle, and scale/magnitude experiments |
| Transformer Primitives v1 | COMPLETE | Checked single and sequence embedding lookup plus explicit two-pass F32 RMSNorm with immutable strided inputs, canonical owned outputs, typed numerical failures, and unchanged historical model behavior |
| Labs 30–38 — Embeddings and normalization | COMPLETE | Table layout, checked lookup, view/copy ownership, hand RMS, implementation, epsilon, scale, magnitude, and Rust/Python equivalence exercises |
| Chapter 8 — Queries, Keys, and Values | COMPLETE | Checked QKV/head API, 15 new Rust tests, independent oracle and full component trace, ten canonical native-vector plates, Labs 39–48 and source-classified production review |

| Chapter 9 - Position: RoPE From First Principles | COMPLETE | Checked two-layout/partial-prefix RoPE, 25 new tests, independent complex oracle, full positioned Q/K/V trace, ten TikZ plates and Labs 49-58 |

## Curriculum status

| Scope | Status | Milestone |
| --- | --- | --- |
| Part I (Ch. 1–4) | COMPLETE | ENGINE-1 is the smallest complete autoregressive inference engine |
| Part II (Ch. 5–13) | IN PROGRESS | Chapters 5–9, Tensor Substrate v1, ENGINE-2, Transformer Primitives v1, checked QKV and standard RoPE complete; Chapter 10 is next |
| Part III (Ch. 14–18) | PLANNED | ENGINE-3 |
| Part IV (Ch. 19–22) | PLANNED | ENGINE-4 |
| Part V (Ch. 23–27) | PLANNED | ENGINE-5 / ENGINE-6 |
| Part VI (Ch. 28–34) | PLANNED | ENGINE-7 |
| Part VII (Ch. 35–41) | PLANNED | ENGINE-8 |
| Part VIII (Ch. 42–48) | PLANNED | ENGINE-9 |
| Part IX (Ch. 49–53) | PLANNED | Decode optimization |
| Part X (Ch. 54–58) | PLANNED | MoE / inference memory |
| Part XI (Ch. 59–66) | PLANNED | Correctness regime |
| Part XII (Ch. 67–73) | PLANNED | ENGINE-10 |
| Part XIII (Ch. 74–79) | PLANNED | Hermon case study |
| Part XIV (Ch. 80–88) | PLANNED | Frontier architecture |
| Part XV (Ch. 89–94) | PLANNED | Graduation project |
| Appendices A–N | PLANNED | Reference material |

## Open decisions

1. Select licenses for prose, diagrams, and code; decide whether one or
   separate licenses are appropriate.
2. Choose the small, redistributable model fixtures for later equivalence labs.
3. Keep Markdown/offline HTML and native LaTeX/TikZ editions synchronized;
   finish the deeper editorial regeneration of legacy Chapters 1–4.

## Next recommended task

Complete only Chapter 10 - Causal Self-Attention. Start with positioned Q/K
and unchanged V. Derive scaled scores, causal visibility, stable softmax and
value mixing, including explicit MHA/GQA/MQA geometry. Implement a checked
dense scalar reference, independent oracle, tests, labs and native TikZ plates.
Inspect current production source without claiming a paged backend is the
default. Do not begin persistent KV caching, FFN, a full decoder, GGUF,
quantization, SIMD, GPU execution or autograd.
