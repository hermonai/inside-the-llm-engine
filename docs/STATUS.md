# Project Status

Last updated: 2026-09-02.

## Phase 0 ledger

| Area | Status | Evidence / next gate |
| --- | --- | --- |
| Repository bootstrap | COMPLETE | Empty public repository cloned and structured |
| Book constitution and policies | COMPLETE | Core editorial/source/code/math/style/benchmark contracts created |
| Master outline | OUTLINED | 15 parts, 94 chapter authoring specifications; review again before each phase |
| Public README and BOOK | COMPLETE | Launch-facing overview and table of contents agree |
| Glossary and terminology | IN PROGRESS | Chapters 1–2 system, token, Unicode, template, and streaming terms added; expand with each chapter |
| Hermon reconnaissance | COMPLETE | Initial map plus Chapters 1–2 request/tokenizer paths verified at `hermon` commit `472a44c` |
| Manuscript part indexes | COMPLETE | 15 part contracts plus appendices scaffolded |
| Diagram system | COMPLETE | Policy and area indexes created; canonical diagrams begin with chapters |
| Research system | COMPLETE | Inventories and note templates established |
| Code project | COMPLETE | Tokenized ENGINE-0 is dependency-free Rust and passes 37 tests; numerical ENGINE-1 remains planned |
| Initial CI | COMPLETE | Structure, links, diagrams, Rust format/check/test/Clippy workflow added |
| License | PLANNED | Maintainers must choose prose and code licensing; no license inferred from Hermon |

Phase 0 remains complete as repository architecture. Phase 1 completion is
tracked separately below.

## Phase 1 ledger

| Scope | Status | Evidence / next gate |
| --- | --- | --- |
| Part I (Ch. 1–4) | IN PROGRESS | Chapters 1–2 complete; Chapters 3–4 remain |
| Chapter 1 — The Missing Half of AI | COMPLETE | 6,874-word reviewed chapter, primary-source research, four canonical diagrams |
| Chapter 2 — From Text to Tokens | COMPLETE | 7,306-word reviewed chapter, primary-source research, seven canonical diagrams, two-tokenizer comparison |
| ENGINE-0 | COMPLETE | Dependency-free tokenized request/runtime/stream lifecycle; byte oracle, BPE, chat/template contract, strict UTF-8 framing; 37 tests and full Rust gate pass |
| Lab 1 — Generate One Token Manually | COMPLETE | Independent candidate oracle plus CHECK/BUILD/BREAK/EXTEND exercise |
| Labs 2–4 — Tokenization / UTF-8 / chat template | COMPLETE | Hand BPE, split-byte streaming, malformed terminal policy, and wrong-template experiments |

## Curriculum status

| Scope | Status | Milestone |
| --- | --- | --- |
| Part I (Ch. 1–4) | IN PROGRESS | Tokenized ENGINE-0 complete; ENGINE-1 begins in Chapter 3 |
| Part II (Ch. 5–13) | PLANNED | ENGINE-2 |
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
3. Decide the publication toolchain only after Markdown-first manuscript needs
   are demonstrated.

## Next recommended task

Execute only Chapter 3 — The Smallest Possible Language Model. Replace the fake
ENGINE-0 candidate table itself with genuine numerical ENGINE-1 inference:
token ID -> embedding lookup -> hidden vector -> output projection -> one logit
per vocabulary item. Separate immutable learned parameters from runtime
activations, make every shape explicit, reject invalid IDs/shapes, and verify
the result with an independent hand-computable oracle. Do not preserve the fake
integer score table behind a wrapper, and do not begin Chapter 4 sampling or the
autoregressive loop.
