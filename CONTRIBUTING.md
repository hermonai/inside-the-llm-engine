# Contributing

Inside the LLM Engine welcomes corrections, technical reviews, plain-text
diagrams, working implementations, portability improvements, reproducible
benchmarks, exercises, paper summaries, hardware experiments, and—later—
translations.

## Before starting

Read `AGENTS.md`, `docs/STATUS.md`, `docs/ROADMAP.md`, and the relevant entry in
`docs/OUTLINE.md`. Search existing issues and inspect the current Git state.
For substantial work, choose one bounded chapter, experiment, or infrastructure
task and record its status so two contributors do not rewrite the same area.

## Evidence requirements

- Architecture claims about a current system require source or canonical-doc
  citations. Hermon claims must follow `docs/SOURCE_POLICY.md` and record the
  inspected commit.
- Performance claims must follow `docs/BENCHMARK_POLICY.md`. Include the raw
  result or a durable pointer to it; identify the control and do not compound
  independent ratios as though they were measured end to end.
- Code changes require tests appropriate to their risk. Optimized numerical
  paths require comparison with an independent oracle before benchmarking.
- Pseudocode must be labeled. Compilable examples must not omit the hard
  correctness path without explaining the simplification.

## Chapter changes

Use the lifecycle in `docs/AUTHORING_WORKFLOW.md` and the completion criteria in
`docs/CHAPTER_CONTRACT.md`. A chapter PR should normally include its research
note, diagrams, code/tests, references, and `docs/STATUS.md` update. Exercises
should include CHECK, BUILD, BREAK, and EXTEND levels when the subject supports
them.

## Style and structure

Follow `docs/STYLE_GUIDE.md`, `docs/MATH_STYLE.md`, and
`docs/TERMINOLOGY.md`. Prefer ASCII diagrams for essential architecture and
keep reusable sources in `diagrams/`. Do not conflate current, preview, and
future behavior.

## Pull requests

Keep changes coherent and avoid unrelated rewrites. Use conventional commit
subjects. In the PR description, state the problem, evidence, correctness
checks, measurements if any, limitations, and affected curriculum milestone.
Run `git diff --check`, repository checks, and relevant language tests.

## License note

The project has not selected prose or code licenses. Contributions cannot be
accepted under an assumed license until maintainers resolve that decision. A
future contribution agreement or explicit license choice may be required; see
`docs/STATUS.md`.
