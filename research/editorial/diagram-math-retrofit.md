# Diagram and Math Retrofit — Chapters 1–6

Date: 2026-09-03

Scope: editorial infrastructure and completed Chapters 1–6 only

Starting book commit: `f14b61001144bffb848a0ab88943d262c8f1c9c5`

## Question

Can the completed foundation chapters use one publication-ready visual and
mathematical language before Part II introduces denser tensor equations and
architecture diagrams?

## Sources reviewed

- Repository governance: `AGENTS.md`, `docs/BOOK_CONSTITUTION.md`,
  `docs/CHAPTER_CONTRACT.md`, `docs/CODE_POLICY.md`, `docs/MATH_STYLE.md`,
  `docs/STYLE_GUIDE.md`, `docs/AUTHORING_WORKFLOW.md`, and
  `docs/AI_AUTHORING.md`.
- Curriculum/status: `README.md`, `BOOK.md`, `docs/OUTLINE.md`,
  `docs/STATUS.md`, `research/README.md`, and Chapters 1–6.
- All 52 canonical diagram files present at the starting commit and their topic
  indexes.
- Four independent Python numerical oracles for Chapters 3–6.
- Hermon local source at `472a44cdb511b2dae6c9569e59543db8f8350b25`,
  equal to local `origin/main` on 2026-09-03. Rechecked
  `crates/hermon-api/src/lib.rs`, `crates/hermon-runtime/src/{lib,dispatch,batched}.rs`,
  and `crates/hermon-core/src/provider.rs` for the request-path diagram.

## Audit result

Before the retrofit there were 52 canonical diagrams and 46 display-math
blocks. Chapters 1–4 contained no display math even though their central
latency, token-domain, projection, probability, and sampling relations were
written as monospaced text. Canonical diagrams were generally stronger than
their older chapter embeds; remaining defects included legacy connectors, a
wrong `│` in one arithmetic expansion, vocabulary-symbol collision risk, and
missing inventories/purpose statements.

## Changes

- Added `docs/DIAGRAM_STYLE.md` with one visual grammar, eight diagram classes,
  truth-status rules, width/accessibility requirements, and review workflow.
- Expanded `docs/MATH_STYLE.md`; added `docs/MATH_INDEX.md`.
- Added `diagrams/INDEX.md` and one-line purposes for every canonical artifact
  in `diagrams/README.md`.
- Added 11 canonical diagrams: three runtime, two tokenizer, one model, two
  sampling, two tensor, and one linear-kernel gate.
- Substantially redesigned two existing canonical diagrams:
  `model/token-id-to-logits.txt` and `sampling/autoregressive-state.txt`.
- Normalized nine additional existing canonical diagrams for connectors,
  arithmetic marks, or `V_vocab` shape labels.
- Replaced the most important legacy manuscript embeds in Chapters 1–4 with
  Unicode box drawing and consistent arrows.
- Standardized 47 added or materially changed display-equation blocks; all 79
  display blocks in Chapters 1–6 were reviewed. Added 18 explicit
  `\mathbb{R}` shape declarations (2 existed before; 20 now remain).
- Added automated diagram inventory/style and math-structure checks, and made
  the width check report its maximum observed line.

## Notation decisions

| Concern | Decision |
| --- | --- |
| Vocabulary size | $V_{\mathrm{vocab}}$; reserve bold $\mathbf{V}$ for later attention values |
| Temperature | $\tau$; top-p threshold is $\tau_p$ |
| Scalars/vectors/matrices | italic lowercase, bold lowercase, bold uppercase |
| Dimensions | uppercase $B,T,M,N,K,D,H_q,H_{kv},D_h$ |
| Indices | $i,j,k$ for coordinates/reductions; $a$ for a generic tensor axis |
| Tensor storage | shape/stride/base metadata is distinct from its owner |
| Precision | storage dtype and accumulation dtype are separate contracts |
| Estimates | `\approx` for conventional FLOP models; equality only for definitions/exact counts |
| Units | seconds, bytes, requests/s, tokens/s, FLOP/s, and FLOP/byte appear in formulas |

## Verification and exceptions

The four Python oracles contain 26 explicit assertion sites and are rerun as
part of this retrofit. Rust format/check/test/Clippy, structure, links, diagram
style/width, credential, large-file, and research/status checks are also
required before completion.

Deliberate exceptions:

- Programming-language signatures keep source-native `->` syntax; the Unicode
  arrow rule applies to diagrams, not compilable code.
- Small array literals stay in code/text form when they are data rather than a
  derivation.
- Existing bracket shape notation stays in code contracts and diagrams; formal
  operator contracts additionally use membership notation.
- Not every equation is placed in `MATH_INDEX.md`; only cross-chapter reusable
  contracts receive stable semantic IDs.

No Chapter 7 content, RMSNorm, Q/K/V, attention, RoPE, KV caching, or later
engine milestone was introduced.
