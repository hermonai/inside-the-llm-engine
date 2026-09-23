# Status

Second edition, begun 2026-09-23 on branch `second-edition`. The first
edition's status ledger, with its dated milestone entries, is
`archive/docs/STATUS.md`.

## Where the book is

**Pass 1 (skeleton) not yet started.** The direction is encoded: the plan is
[`STRUCTURE.md`](STRUCTURE.md), the policy is [`../AUTHORING.md`](../AUTHORING.md),
and superseded governance is in `archive/`. The manuscript is still the first
edition's nine chapters at their old paths.

## Verified baseline (2026-09-23, before restructuring)

| Check | Result |
| --- | --- |
| `make textbook-check` | pass: 9 chapters, 65 native figures, 27 worked problems |
| `make textbook` | pass: 202-page PDF, XeLaTeX (not LuaLaTeX) |
| `cargo test --workspace` in `code/mini-engine` | 220 pass: 218 unit/integration + 2 compile-fail doctests |
| All 17 CI steps run locally | pass |
| Manuscript size | 56,737 words in `tex/chapters/ch01.tex`–`ch09.tex` |

## Chapter status

Statuses: PLANNED, SKELETON, ZERO, FULL, VERIFIED (`AUTHORING.md` §11).

| Part | Chapters | Status |
| --- | --- | --- |
| I The Physics of Inference | 1–6 | PLANNED |
| II Making One Request Fast | 7–13 | PLANNED |
| III Serving Many Requests | 14–21 | PLANNED (14 inherits first-edition Chapter 1) |
| IV Beyond One Token per Step | 22–24 | PLANNED |
| V Architectures That Reshape the Engine | 25–30 | PLANNED |
| VI Scaling Out | 31–34 | PLANNED |
| VII Inference on the Hardware You Have | 35–37 | PLANNED |
| VIII Correctness and Measurement | 38–39 | PLANNED |
| IX Production and Capstone | 40–42 | PLANNED |
| Appendix A | A.1–A.8 written (first edition), A.9–A.11 planned | first-edition text at old paths |
| Appendices B–G | — | PLANNED |

## Next

1. Start `research/FRONTIER.md`; verify the topics Parts I and II depend on.
2. Move the first edition's chapters with `git mv`: Chapters 2–9 to Appendix A,
   Chapter 1 to Chapter 14. Keep all 220 tests passing.
3. Skeleton every chapter and appendix; restructure
   `tex/inside-the-llm-engine.tex` and `scripts/check-textbook.py`.
4. Zero-draft Part I, beginning with Chapter 3.

## Decisions for the author

- The first edition's nine chapters become Appendix A (and Chapter 14). This is
  the largest editorial call in the plan; see `STRUCTURE.md`.
- Work happens on branch `second-edition`, created from `astra-visual-rewrite`
  at `c8b8158`; neither branch has been pushed or merged.
- The project still has no licence (unchanged from the first edition).
