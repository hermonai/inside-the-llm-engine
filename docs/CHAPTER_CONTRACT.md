# Chapter Contract

Every completed chapter must deliver a coherent learning unit and a verified
step in the engine curriculum. Editorial judgment may combine headings, but it
may not omit the underlying obligations without recording why.

## Required content

1. An opening problem and why it matters to an inference engine.
2. A concrete mental model before advanced formalism.
3. A first-principles derivation and consistent core terminology.
4. At least one meaningful ASCII system/data/ownership diagram.
5. Mathematical treatment where relevant, including dimensions and cost.
6. A reference implementation or explicitly labeled pseudocode.
7. A walkthrough that follows data, token, byte, or owner.
8. The failure mode or limit of the naive design.
9. The production design and its tradeoffs.
10. A source-verified Hermon case study when the topic maps to Hermon.
11. Correctness invariants and tests.
12. Performance analysis or a reproducible benchmark when meaningful.
13. Common mistakes and representative engineering failures.
14. CHECK, BUILD, BREAK, and EXTEND exercises where appropriate.
15. Further exploration, summary, next build step, and primary references.

## Chapter metadata

Before drafting, the outline and research note must agree on purpose,
prerequisites, key question, concepts, mathematics, systems and hardware
concepts, implementation work, Hermon and external connections, diagrams,
experiments, correctness, benchmark, misconceptions, failure cases,
deliverable, and what the next chapter assumes.

## Callout conventions

Use blockquotes with a bold label. The first line states the claim or task; the
body supplies evidence or action.

> **FIRST PRINCIPLE**
> A durable invariant from which the design follows.

> **BUILD IT**
> A reader implementation task with a testable result.

> **INSIDE HERMON — CURRENT/PREVIEW/LIBRARY/HISTORICAL**
> A commit-verified production case study whose status is explicit.

> **PROVE IT**
> A correctness property, oracle, or boundary test.

> **ENGINEERING FAILURE**
> A real or representative failure, diagnosis, and lesson.

> **PERFORMANCE LAB**
> A reproducible measurement following `BENCHMARK_POLICY.md`.

> **FRONTIER**
> A future design or open question, never presented as shipped behavior.

## Definition of done

- Research note cites evidence and closes or names material open questions.
- Code builds, formats, and tests at the chapter's milestone.
- Optimized numerical output is checked against an independent oracle.
- Benchmark metadata and raw results are durable and reproducible.
- Diagrams render in a monospaced view without color or external images.
- Terms match `TERMINOLOGY.md` and new terms update `GLOSSARY.md`.
- Technical review and editorial review are separate recorded passes.
- Links and chapter transitions are checked.
- `STATUS.md` is updated only after the gates actually pass.
