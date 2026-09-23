# Contributing

*Inside the LLM Engine* welcomes corrections, technical reviews, measurements
on hardware the authors do not have, labs, exercises and fixes to the
companion code.

## Before starting

Read [AGENTS.md](AGENTS.md), [AUTHORING.md](AUTHORING.md) and
[docs/STATUS.md](docs/STATUS.md). The chapter plan is
[docs/STRUCTURE.md](docs/STRUCTURE.md). For substantial work, claim one chapter
or task in an issue so two people do not rewrite the same area.

## Evidence

- Every number is *measured*, *derived* or *illustrative* (`AUTHORING.md` §5).
  A measured number comes with a record in `research/measurements/`: hardware,
  software versions, model and quantization, command, workload, repetitions and
  raw output.
- Frontier claims — anything about current techniques, engines, hardware or
  models — cite a primary source and have a dated entry in
  `research/FRONTIER.md` (`AUTHORING.md` §6).
- Claims about what an engine does cite its source at a pinned commit and say
  whether the behaviour is DEFAULT, PREVIEW, LIBRARY or DESIGN (`AUTHORING.md` §7).
- Optimized numerical code is checked against an independent oracle before it
  is benchmarked. A benchmark of a wrong answer is void.

## Pull requests

Keep changes coherent, use conventional commit subjects, and run
`git diff --check`, `make textbook-check` and the relevant tests. State the
problem, the evidence, the checks you ran and any limitations. Never include
host names, IP addresses, credentials or machine-specific paths.

## Licence

The project has not selected prose or code licences. Contributions cannot be
accepted under an assumed licence until the maintainers resolve that decision.
