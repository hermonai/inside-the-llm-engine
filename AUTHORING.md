# Authoring *Inside the LLM Engine*

This is the book's only policy document. It replaces about thirty earlier
policy, contract, style and planning files, which are preserved in `archive/`.
Keep it under 3,000 words: when a rule is added, cut or merge another. The
chapter plan is [`docs/STRUCTURE.md`](docs/STRUCTURE.md); progress is
[`docs/STATUS.md`](docs/STATUS.md).

## 1. What the book is

*Inside the LLM Engine: The Systems Engineering of Large Language Model
Inference* is the book an engineer reads to understand why every modern
inference engine is built the way it is, and that stays useful as techniques
change. The model is *Designing Data-Intensive Applications*: durable
principles, real systems as evidence, honest trade-offs. The book does not
teach training, and it does not re-derive the Transformer in the main text —
Appendix A does that for readers who want to build one.

## 2. The spine: the constraint ledger

Every engine negotiates four scarce resources — **memory bandwidth** (bytes
moved per token), **memory capacity** (bytes resident), **compute** (FLOPs) and
**latency** (TTFT, time per output token) — for two outcomes: **correctness**
and **cost** (goodput: tokens per dollar at a latency target).

Every technique enters as a response to a *measured* constraint and is charged
for what it spends. Each chapter therefore ends its mechanism with a
**constraint ledger**: a small table, typeset with the `ConstraintLedger`
environment, with one row per resource the technique touches:

```latex
\begin{ConstraintLedger}
Memory bandwidth & buys & 4-bit weights read 0.56 bytes per parameter instead of 2 \\
Correctness      & spends & perplexity rises; measure it per model \\
\end{ConstraintLedger}
```

A row says *buys*, *spends*, *moves* (shifts the bottleneck elsewhere) or
*risks* (for correctness), and the note says by how much when that is known.
A chapter whose technique appears to spend nothing has not found its cost yet.

## 3. Chapter anatomy

Every chapter has the same eight parts, in this order. Headings may be
specific ("Eighty-five tokens per second"), but the parts must be recognisable.

1. **Opening measurement** — a real, surprising, reproducible number, with its
   setup in a measurement record. Lead with the phenomenon.
2. **Napkin math** — the back-of-envelope cost model before any code, using
   real model and hardware numbers. This is the most transferable skill in the
   field; practise it every chapter.
3. **The mechanism** — derived, drawn and explained from first principles.
4. **The constraint ledger** — §2.
5. **Build it small** — one focused, runnable, self-contained lab (§10).
6. **What actually happened** — a real case from a production engine's
   history, failures included (§7).
7. **Frontier watch** — a dated, sourced note on where the technique is
   heading, backed by `research/FRONTIER.md` entries (§6).
8. **Exercises** — prediction, derivation, implementation and falsification
   problems, with worked answers for the closed ones in `tex/worked/`.

Target 5,000–8,000 words. A chapter that needs more is two chapters; a chapter
that needs much less may be a section of its neighbour.

## 4. Voice

- Lead with the phenomenon, not the caveat. Put a qualification exactly where
  it changes what the reader should believe — once — not in every paragraph.
- Concrete before abstract; numbers before notation; mechanism before
  optimization. Name real models, real hardware and real engines.
- Short declarative sentences for claims; derivations may take their time.
- No "simply", "obviously", "just", "magic". Define a term before relying on it.
- Analogies (virtual memory, databases, CPUs) state where they break.
- Write original prose. Learn from Kleppmann's structure and Raschka's
  build-to-understand approach; imitate nobody's wording.

## 5. Numbers and evidence

Every number in the book is exactly one of:

- **measured** — produced by a recorded run. The text names the hardware,
  model and quantization; `research/measurements/` holds the record: date,
  commit or binary version, hardware, OS/driver, model file and quantization,
  command, workload (prompt and output lengths, batch/concurrency),
  repetitions and statistic, raw output. Re-measure before printing a number
  someone else measured.
- **derived** — computed from stated inputs; show the arithmetic or cite the
  equation, and keep the inputs sourced.
- **illustrative** — chosen to teach; say so.

A zero draft may carry a number it has not yet measured only inside
`\todo{measure: …}`. Vendor peak numbers are *specifications*, not
measurements: cite the datasheet and state the precision and sparsity the peak
assumes.

Correctness precedes speed. A benchmark of an engine that produced a wrong
answer is void. Compare like with like: model, quantization, stopping rule,
workload and concurrency. Never multiply independent speedups and call the
product measured. Preserve negative results with their hypothesis and setup;
a failed optimization is evidence.

## 6. Staying at the frontier — mandatory

Training data has a cutoff and inference moves month to month. **Never write a
frontier claim from memory.** Before drafting or deepening any chapter in
Parts II, IV, V or VI — and before any frontier-watch section anywhere:

1. Search for the current state of the chapter's techniques, then read primary
   sources: papers, official documentation, source code at a pinned commit.
2. Record each finding in `research/FRONTIER.md`: technique, one-line
   description, primary source with link, **date checked**, maturity
   (*research*, *shipped in* a named engine or product, or *industry
   standard*), and target chapter.
3. Cite primary sources in the chapter. Blog posts and vendor benchmark claims
   are leads, not evidence.

Hardware specifications and model configurations are frontier claims too:
verify them against the datasheet or the model's published config and log
them. A frontier-watch section states the date it was checked; refreshing
those sections is how the book stays current.

## 7. Real engines, and Hermon

Code outranks documents. Before stating what an engine does, read its source
at a recorded commit and classify the claim:

| Label | Meaning |
| --- | --- |
| DEFAULT | runs on ordinary traffic with default configuration |
| PREVIEW | implemented, behind a flag or environment variable |
| LIBRARY | callable code that no request path reaches |
| DESIGN | designed or documented, not built |

"The crate exists" does not mean a request reaches it. "A test passes" does not
mean a model family is supported. A roadmap does not outrank a gate in source.

[Hermon](https://github.com/hermonai/hermon) is the book's evidence base, not
its subject. Its value is rare: a real engine's measured results, including
the failures, recorded with numbers in its design documents. Use them in
*What actually happened* sections, re-checked against source at a pinned
commit. Never imply Hermon is the reference implementation of a technique it
only prototypes; compare it with vLLM, SGLang, TensorRT-LLM and llama.cpp from
their own primary sources.

## 8. Mathematics

- Every equation names its symbols, shapes and units at first use. Declare
  shapes by membership: $\mathbf{W}\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$.
- Scalars italic ($x$), vectors bold lowercase ($\mathbf{x}$), matrices and
  tensors bold uppercase ($\mathbf{W}$).
- Canonical symbols, used the same way in every chapter (Appendix F keeps the
  full table): $L$ layers; $d$ model width; $d_{\mathrm{ff}}$ FFN width;
  $H_q$, $H_{kv}$ query and key/value heads; $d_h$ head width; $V$ vocabulary
  size; $B$ concurrent sequences; $S$ context length in tokens; $P$
  parameters and $P_{\mathrm{act}}$ active parameters; $b$ bytes per element
  (with a subscript: $b_w$ weights, $b_{kv}$ cache); $\beta$ memory bandwidth
  (bytes/s); $\pi$ peak compute (FLOP/s); $I$ arithmetic intensity
  (FLOP/byte). Appendix A predates this table and writes $V_{\mathrm{vocab}}$.
- Units on every byte, bandwidth, time and throughput quantity. GB is
  $10^9$ bytes and GiB is $2^{30}$; tools disagree, so say which.
- Use $\approx$ for estimates and $\le$ for bounds; equality only for exact
  counting conventions, which are stated ("a multiply–add is two FLOPs").
- A hand calculation that serves as evidence has an executable mirror: a
  test, an oracle in `code/reference/`, or a check in `scripts/check-textbook.py`.

## 9. Figures

- Native TikZ (or pgfplots for data), in `tex/figures/<name>.tex`, placed with
  `\EngineFigure{name}{caption}{label}`.
- Draw the thing that changes: tensor cells, memory addresses, timelines, state
  transitions, a roofline with measured points. No decorative boxes; every
  arrow has a meaning the caption or a legend states.
- Captions state what the reader should infer and where the figure stops
  being true. Data plots label each point *measured* or *derived*.
- Colour is reinforced by labels, patterns or line styles; check grayscale.
- A zero draft states a figure it has not drawn with `\FigureIntent{what it
  shows and why}`.

## 10. Labs

Each chapter has one lab in `code/labs/chNN-<slug>/`: a README (predict, run,
explain, break) and the smallest code that shows the mechanism.

- **Self-contained.** It may depend on `code/mini-engine` as a library, but it
  must not require having built any previous chapter's lab.
- **Runs on a laptop CPU** in about a minute by default. GPU variants are
  optional extensions, clearly marked.
- **Has an oracle.** Every optimized path is checked against an independent,
  obviously-correct computation before anything is timed.
- Rust by default; Python for clarity-first oracles and plots; C where a
  native boundary is the lesson.

A zero draft states the lab with `\LabIntent{…}` instead of building it.

## 11. Workflow: breadth first, then depth

The first edition finished each chapter to final polish before starting the
next and stalled. This edition works in passes over the whole book:

| Pass | A chapter is… | Status |
| --- | --- | --- |
| 1 | a file with its section headings and a one-paragraph intent | SKELETON |
| 2 | 1,500–3,000 words: core argument, key numbers (`\todo{measure}` where unmeasured), figure and lab intents | ZERO |
| 3 | full length with figures, lab, exercises, verified frontier watch | FULL |
| 4 | reproduced, technically reviewed and edited | VERIFIED |

Work part by part, Part I first. Deepen the frontier-heavy chapters first
because they date fastest. Keep moving between passes; do not gold-plate one
chapter. VERIFIED requires a review by someone other than the drafting agent;
one agent's repeated passes are not independent review.

Per chapter, the only required artifacts are the chapter, its figures, its lab,
its worked problems and its `FRONTIER.md` entries. No per-chapter audits,
storyboards, atlases or parity reports.

## 12. Build and checks

```bash
make textbook-check     # structure, anatomy, figures, worked problems, CLI trace
make textbook           # full PDF with XeLaTeX + latexmk + biber (TeX Live)
(cd code/mini-engine && cargo test --workspace)
```

`scripts/check-textbook.py` reads the chapter list from the main file, checks
that every chapter exists and has the eight anatomy parts once it leaves
SKELETON, that every referenced figure exists as TikZ, and it reports the
`\todo` count. `make textbook` fails on overfull boxes, missing glyphs and
unresolved references. CI (`.github/workflows/ci.yml`) also runs the
mini-engine tests, its Rust/Python parity checks, and the frozen-history
checks. Keep LaTeX; do not migrate formats again.

References go in `tex/references.bib` and are cited with `\cite{key}`; use
primary sources.

## 13. Repository map

| Path | Role |
| --- | --- |
| `tex/` | The book: `chapters/`, `appendices/`, `figures/`, `worked/`, `references.bib` |
| `code/mini-engine/` | Dependency-free Rust teaching engine (Appendix A, lab substrate) |
| `code/reference/` | Independent Python oracles |
| `code/labs/` | One self-contained lab per chapter |
| `research/FRONTIER.md` | Dated, sourced frontier ledger |
| `research/measurements/` | Records behind measured numbers |
| `research/` (other) | First-edition research notes and benchmark records |
| `labs/` | First-edition labs 1–58 (Appendix A) |
| `manuscript/`, `diagrams/`, `figures/`, `publication/` | Frozen first-edition formats; checked by CI, not edited |
| `archive/` | Superseded governance and plans, mirroring original paths |

## 14. Rules that do not bend

- **Git:** atomic conventional commits; `git diff --check` before each. Never
  rewrite history, force-push or run destructive cleanup. Never push unless
  the author asks in that session.
- **Secrets:** the repository has a public remote. Never commit hostnames, IP
  addresses, credential paths, tokens or passwords. Describe measurement
  hardware by model and configuration, never by host name.
- **Honesty:** a working component is not an integrated system; a target is
  not a shipped feature; an illustrative number is not a measurement.
- **Preserve:** never delete a chapter, test, failed measurement or historical
  record. Superseded material moves to `archive/` with a note.
