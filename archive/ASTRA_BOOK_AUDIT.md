# Book audit: visual regeneration baseline

Inspected 2026-09-05. Starting book commit: `8588ff839174c31754385403765c6eb2a127ab43`.
Working branch: `astra-visual-rewrite`. Seven completed chapters, 94 planned
chapter specifications, 38 labs, four runnable Rust examples, five Python
numerical oracles, 163 Rust tests, and 78 registered text diagrams exist.
Chapter 8 is RESEARCHING; QKV is not implemented. Later prototype plates are
EDUCATIONAL MODEL specifications, not claims that those operators have shipped.

## Preservation record

The starting working tree includes edits to `docs/STATUS.md`,
`diagrams/runtime/model-vs-engine.txt`, and Chapter 1, plus an untracked
Chapter 8 research note. All four pre-existing changes are retained without
staging; only the new regeneration section of the status file is committed. Existing
experiments, failed performance hypotheses, historical numerical fixtures,
and Git history remain intact. The new milestone does not complete Prompt 9.

## Chapter decisions

| Chapter | Classification | Retained strength | Revision and acceptance evidence |
| --- | --- | --- | --- |
| 1: Missing Half | KEEP + VISUAL REBUILD | Request lifecycle, terminal ownership, three journeys | Add progressive engine zoom and an actual runtime sequence; distinguish historical ENGINE-0 from today's numerical path. Preserve local edits. |
| 2: Text to Tokens | KEEP + VISUAL REBUILD | Bytes versus Unicode, special-token trust, strict streaming | Sequence one split UTF-8 scalar across two emissions; show special identity versus literal spelling. Keep BPE and malformed-byte tests. |
| 3: Smallest Model | KEEP + VISUAL REBUILD | Full hand logits and causal weight intervention | One continuous ID/row/dot/logit plate. Preserve D=3 historical fixture; do not silently replace it with the proposed D=4 visual fixture. |
| 4: Sampling | KEEP + VISUAL REBUILD | Fixed draws, logit/probability separation, commit boundary | Align CDF intervals with exact endpoint policy; show cancellation before and after commit. Retain request-owned RNG tests. |
| 5: Tensors | KEEP + VISUAL REBUILD | Checked strides, extent, views, copies, ownership | Reuse one matrix across logical/physical/transpose/copy panels. Add UML composition and borrow dependencies; do not invent inheritance. |
| 6: Matrix Multiplication | TECHNICAL REVISION | Reference and blocked equivalence, honest measurements | Correct D049/D050/D054 edge semantics before graphical migration. Add one-row GEMV sequence and carry reduction order into locality plates. |
| 7: Embedding/Normalization | KEEP + VISUAL REBUILD | Explicit F32 failures, epsilon, two-pass implementation | Show a shared numerical vector through reduction and scaling; distinguish real-valued formula from F32 intermediates and typed errors. |

No whole chapter currently warrants MAJOR REWRITE, MERGE, SPLIT, MOVE, or
HISTORICAL ONLY. Chapter 6 needs a bounded visual technical revision rather
than replacement of its tested arithmetic. The historical model remains a
deliberate teaching limitation. Chapters 8–94 are plans, not manuscripts to audit.

## Mathematics

The [equation ledger](research/astra/equation-audit.md) enumerates every display
block with its nearest section, source line, code/oracle mapping, and visual
review obligation. Scalar prose/inline equations are reviewed by the semantic
families in `docs/MATH_INDEX.md`; a syntax scan is not a proof of every claim.
Central shape, offset, sampling, GEMM, and RMSNorm equations agree with the
existing numerical tests. The real-number associativity statement in Chapter 6
is correctly qualified immediately afterward; do not flag it as an F32 identity.

Main weaknesses are navigation and visual binding: equations have chapter-level
links to diagrams and tests, but lack stable per-equation figure references.
The first PDF build found one form-feed character replacing the start of
`\frac` and six missing backslashes before `\mathbf` in Chapter 7. These seven
presentation defects are corrected in this milestone. The prior math checker
only covered Chapters 1–6; it now discovers every written chapter and rejects
control characters and these malformed inline commands. The complete inventory
contains 112 display blocks and 29 explicit real-valued shape declarations.
Some shape declarations use plain capitals while policy prefers bold matrices.
Byte/FLOP models correctly require assumptions; preserve their analytical label.
The next chapter pass must explicitly inspect local symbol definitions and
attach individual numerical examples, rather than claiming this inventory alone
provides full mathematical recertification.

## Code

The Rust workspace is dependency-free and forbids unsafe code. `tensor.rs`
owns storage metadata; `linear.rs` separates broad strided reference kernels
from canonical-only blocked execution; embedding returns an owned activation;
normalization exposes F32 range failures. No competing tensor layer is needed.
The [example ledger](research/astra/code-audit.md) includes every source example
and oracle. Existing tests exercise invalid shapes, overflow, view aliasing,
independent ownership, zero dimensions, tiled tails, sampling and terminal state.
Four examples are experiments, not new production implementations. Similar
Rust/Python expressions are intentional independent oracles, not duplication
to remove. Manuscript code fences mix excerpts, commands, and pseudocode;
they must not all be advertised as independently compilable programs.

## Publication and evidence

No PDF/HTML build existed at the starting commit. Markdown, Unicode diagrams,
and display mathematics are the authoritative current edition. The first new
publication build renders all seven chapters plus a separate prototype atlas.
It does not claim to publish the unwritten 87 chapters. License selection is
still an existing maintainer decision; no license is invented for this pass.

Fresh production inspection is in the [source map](research/astra/source-map.md).
Source availability establishes a path, not real-model equivalence or measured
performance. No model download or accelerator benchmark is needed for this
bounded editorial/infrastructure milestone.

## Reader review

Beginner: isolate objects before showing equations. Software engineer: expose
ownership and error branches. ML engineer: state weight orientation and GQA
geometry. Systems programmer: trace strides and bytes. Graduate student:
distinguish algebra from rounding. Researcher: retain source commits and open
equivalence gates. The visual storyboard must answer all six perspectives
without putting every detail on the opening plate.
