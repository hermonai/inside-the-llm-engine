# Source-first textbook revision: author review

Review date: 2026-09-13. Scope: Inside the LLM Engine, DNA Computing and
Evolutor. The local training book is not changed. This record reports
author-agent checks, not an independent specialist review or reader study.

## Teaching decision

The inspected DNA and Evolutor accepted LaTeX editions use a concrete
question, a small running example, defined mathematical objects, mechanism
figures, code and worked answers. The new inference edition adopts these
general qualities with original material. Sebastian Raschka's
[official companion repository](https://github.com/rasbt/LLMs-from-scratch)
and book listing informed the requested cumulative build-from-scratch
approach; no prose, figures or signature expression are reproduced.

An engineer should be able to predict a result, derive it, run a check, break
an assumption, and explain the resulting failure before meeting the next
abstraction. Attractive layout is necessary but does not establish correctness.

## Inference changes and checks

- Authoring: nine editable LaTeX chapters, numbered equations and figures,
  Pagella/Heros typography, a working glossary and selective index.
- Prose: Chapter 1 is rewritten around a real executable request. Chapters
  2–9 are imported reviewed material with corrections and new worked
  synthesis, not eight fully rewritten chapters.
- Interfaces: distinguish TinyBpeTokenizer probes from TinyLmTokenizer;
  update Request sampling state, ForwardPass and OwnedTensor descriptions.
  Remove obsolete fake-model claims from the active early narrative.
- Illustrations: 52 reviewed semantic plates retained as cropped editable
  TikZ, plus 13 original figures. Remove 28 mechanically rendered legacy
  diagrams from the active edition and replace early mechanisms with native
  geometry, equations or precise prose.
- Exercises: 27 new problems with numerical answers, derivations,
  counterexamples or explicit open-task rubrics.
- Actual CLI: default EOS, budget limit, cancellation, injected failure and
  immediate EOS; exactly one terminal owner in each admitted request.
  The emitted leading space in ` Rust` is part of the vocabulary contract.
  Standalone `like` is rejected, and the checker verifies this too.
- Independent arithmetic: projection intervention; UTF-8 completion;
  probability renormalization; strided offsets and extents; GEMM tails and
  FLOPs; RMSNorm; GQA head ownership; analytical KV bytes; relative RoPE.
- Regressions: 218 Rust unit/integration tests plus two compile-fail
  doctests; format and Clippy; all five chapter visual/numerical parity gates;
  repository structure and links.

No engine algorithm or historical benchmark is changed in this revision.
The retained production/source tours are dated snapshots at their recorded
commits, not fresh assertions about today's external default behavior.

## DNA and Evolutor

The new combined LaTeX candidates directly include accepted Chapters 1–2,
and import the reviewed Chapter 3 once into independently editable LaTeX.
Each adds a worked bridge and one original native TikZ figure while retaining
twelve reviewed Chapter 3 vector plates and twelve worked exercises.

DNA derives twelve tests versus three dependent stages, and states capacity,
early-stop and physical-measurement assumptions. Evolutor derives the scalar
forward pass, input/weight sensitivities, a one-parameter update and its finite
difference. Existing numerical suites and new preservation checks pass.
The checker verifies 184 prior source hashes for DNA and 180 for Evolutor.
Old review records and old PDFs are not rewritten.

A pre-existing roadmap freshness failure was isolated: the frozen accepted
planning generator still emitted Chapter 3 as planned, while the repository
had a standalone review candidate. A new planning view changes only the
current roadmap, retains the historical validation logic, and has tests
preventing Chapter 3 from being falsely promoted to acceptance.

## Visual and build review

All three candidates build from LaTeX without a Markdown/Pandoc stage.
Every page is rendered with Poppler and checked for off-page word bounds.
Color contact sheets cover all pages; grayscale samples cover mechanism,
matrix and chart pages. New figures receive full-page detail inspection.
The review corrected a math-package ordering issue, crowded request-state
labels, projection edge routing, an intersecting adjoint label, duplicated
list markers and isolated worked-solution spill pages.

Build/check/render records are produced under `build/textbook/`. They are
diagnostics, not acceptance certificates. The build fails on overfull boxes,
missing characters and unresolved references. Fonts and vector geometry
remain native in the PDF; no generated raster typography is used.

## What remains

This is a first series-style revision, not proof of industry-leading quality
or completion of any whole book. Inference needs Chapter 10 onward and deeper
editorial review of retained material. DNA and Evolutor need Chapter 3
specialist review and cumulative acceptance before Chapter 4. Real readers,
independent technical reviewers, licensing decisions and release validation
remain necessary. New authoring follows `TEXTBOOK_STANDARD.md`.
