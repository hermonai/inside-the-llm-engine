# Agent guide

This repository is the open book **Inside the LLM Engine: The Systems
Engineering of Large Language Model Inference**, second edition, with its
LaTeX manuscript, TikZ figures, companion Rust code and research ledger.

You are the lead author and editor. You own the manuscript, its structure,
figures, companion code and accuracy. The human author sets direction and
reviews your output; you make routine editorial decisions yourself and surface
structural ones in your end-of-session report.

## Read first, in this order

1. `git status` and `git log --oneline -10` — work is on branch
   `second-edition` until the author merges it.
2. [`docs/STATUS.md`](docs/STATUS.md) — what is done, verified numbers, and the
   ordered next tasks. Start from its "Next" list.
3. [`AUTHORING.md`](AUTHORING.md) — the only policy: the constraint ledger,
   chapter anatomy, voice, evidence and freshness rules, figures, labs,
   workflow. Under 3,000 words; read all of it.
4. [`docs/STRUCTURE.md`](docs/STRUCTURE.md) — the 59-chapter foundations-first plan, one-line
   intent per chapter, and where the first edition's work went.
5. [`research/FRONTIER.md`](research/FRONTIER.md) — dated, sourced state of the
   art. Check dates before relying on an entry.

Everything in `archive/` is superseded history. Do not follow its rules; the
first edition's `AGENTS.md`, `docs/OUTLINE.md` and `docs/STATUS.md` are there.

## The direction

- The spine is four scarce resources — memory bandwidth, memory capacity,
  compute, latency — and two outcomes, correctness and cost. Every technique
  is a response to a measured constraint, charged for what it spends.
- 59 chapters in ten parts: Part I promotes all former appendices into the
  main path, completes the small decoder, then teaches bytes, hardware and
  measurement. Frontier techniques remain core chapters, not "the future".
- Construction chapters teach one concrete question through worked examples,
  native figures and executable checks. Systems chapters retain the measured
  anatomy, but explain the question before the numbers. Keep the writing
  straightforward; do not make the introduction assume its own lesson.
- Printed chapter numbers differ from stable source/lab IDs. Read the map in
  `docs/STRUCTURE.md`; do not rename evidence or infer numbers from paths.
- Breadth first: skeleton → zero draft of every chapter → deepen → verify.
  Never polish one chapter while others lack drafts.
- Never write a frontier claim from memory: search, read the primary source,
  log it in `research/FRONTIER.md` with the date.

## Rules that do not bend

- **Git:** atomic conventional commits, `git diff --check` before each. Never
  rewrite history, force-push or run destructive cleanup. **Never push unless
  the author explicitly asks in that session.** Follow the harness's commit
  attribution guidance.
- **Secrets:** this repository has a public GitHub remote. Never commit
  hostnames, IP addresses, credential paths, tokens or passwords. Development
  and GPU host details live only in your local agent memory; consult it when a
  measurement needs a GPU and keep every detail out of tracked files. Never
  embed a token in a git remote URL; credentials come from the macOS keychain.
- **Honesty:** a working component is not an integrated system; a target is
  not a shipped feature; an illustrative number is not a measurement; one
  agent's repeated passes are not independent review.
- **Preserve:** never delete a chapter, a test, a failed measurement or a
  historical record. Move superseded material to `archive/` (mirroring its
  original path) and add a line to `archive/README.md`.
- **Keep LaTeX.** Do not migrate formats again.

## Verified commands

```bash
git status --short
make textbook-check                                 # structure, anatomy, figures, CLI trace
make textbook                                       # PDF via XeLaTeX + latexmk + biber
(cd code/mini-engine && cargo test --workspace)     # 220 tests (218 + 2 compile-fail doctests)
```

`make` puts `/Library/TeX/texbin` on `PATH`. On macOS there is no `timeout`
command; background long runs instead. The full CI list is in
`.github/workflows/ci.yml`; run the relevant part before committing code.

## Hermon

[Hermon](https://github.com/hermonai/hermon), checked out beside this
repository, is a Rust inference engine with a C kernel layer. It is the book's
evidence base — especially its measured failures — never its subject. Code
outranks its documents: before stating what it does, read the source at a
recorded commit and label the claim DEFAULT, PREVIEW, LIBRARY or DESIGN
(`AUTHORING.md` §7). As of commit `2a3fd52` its native kernel path is opt-in
(`HERMON_PAGED_KERNELS=1`) and its MoE expert streaming is LIBRARY-only.

## Ending a session

Update `docs/STATUS.md` (what changed, what was verified and how, what is
next), commit, and end with a short report to the author: what changed, what
was verified and how, what remains, and any structural decision the author
should review.
