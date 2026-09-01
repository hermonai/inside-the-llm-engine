# Project Status

Last updated: 2026-09-02.

## Phase 0 ledger

| Area | Status | Evidence / next gate |
| --- | --- | --- |
| Repository bootstrap | COMPLETE | Empty public repository cloned and structured |
| Book constitution and policies | COMPLETE | Core editorial/source/code/math/style/benchmark contracts created |
| Master outline | OUTLINED | 15 parts, 94 chapter authoring specifications; review again before each phase |
| Public README and BOOK | COMPLETE | Launch-facing overview and table of contents agree |
| Glossary and terminology | OUTLINED | Initial core terms exist; expand with each chapter |
| Hermon reconnaissance | COMPLETE | Initial map verified at `hermon` commit `472a44c`; must be refreshed before manuscript use |
| Manuscript part indexes | COMPLETE | 15 part contracts plus appendices scaffolded |
| Diagram system | COMPLETE | Policy and area indexes created; canonical diagrams begin with chapters |
| Research system | COMPLETE | Inventories and note templates established |
| Code project | PLANNED | Layout selected; no teaching engine implementation yet |
| Initial CI | PLANNED | Deferred until the first executable language/tooling lands; Phase 0 uses repository scripts |
| License | PLANNED | Maintainers must choose prose and code licensing; no license inferred from Hermon |

Phase 0 is complete as repository architecture. It does not imply any manuscript
chapter or engine milestone is complete.

## Curriculum status

| Scope | Status | Milestone |
| --- | --- | --- |
| Part I (Ch. 1–4) | PLANNED | ENGINE-0 / ENGINE-1 |
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
2. Select the first Rust workspace dependency policy when ENGINE-0 begins.
3. Choose the small, redistributable model fixtures for later equivalence labs.
4. Decide the publication toolchain only after Markdown-first manuscript needs
   are demonstrated.

## Next recommended task

Begin Phase 1 with a bounded Chapter 1 research note. Verify the contemporary
inference-stack framing using primary sources, refine the Chapter 1 outline only
where evidence requires it, design the ENGINE-0 interface and Lab 1 oracle, and
leave Chapter 2 assumptions explicit. Do not mass-generate manuscript prose.
