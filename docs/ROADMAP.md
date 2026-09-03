# Roadmap

The phases describe curriculum and software maturity, not release dates.

| Phase | Scope | Exit condition |
| --- | --- | --- |
| 0 | Repository and editorial architecture | Constitution, policies, 94-chapter outline, part scaffolds, Hermon inventory, README, status, and consistency gate complete |
| 1 | Part I — conceptual inference foundation | ENGINE-0/1, reference token/logit/sampling labs, reviewed prose |
| 2 | Part II — Transformer from scratch | ENGINE-2 produces verified next-token logits for a tiny model |
| 3 | Part III — real model / GGUF | ENGINE-3 parses, loads, and runs a supported real GGUF artifact with equivalence tests |
| 4 | Parts IV–V — KV and serving | ENGINE-4/5/6 demonstrate cached decode, request lifecycle, streaming, and continuous batching |
| 5 | Part VI — paged inference | ENGINE-7 owns mapped KV blocks, prefixes, COW, eviction, and paged attention |
| 6 | Parts VII–VIII — native kernels and hardware | ENGINE-8/9 add stable C ABI, oracle-backed SIMD, and gated providers |
| 7 | Parts IX–X — modern decode and MoE | Prefix/speculation and expert paging are implemented, broken deliberately, and measured |
| 8 | Parts XI–XII — correctness and production | ENGINE-10 passes correctness, protocol, observability, security, and benchmark gates |
| 9 | Part XIII — Hermon case study | Current source-verified request/token walkthrough at a recorded commit |
| 10 | Part XIV — frontier architecture | Today/near-term/frontier/research boundaries reviewed; no proposal presented as current |
| 11 | Graduation implementation | Final mini-engine integrated end to end; one Hermon component replacement experiment scoped |
| 12 | Full technical review | Independent review closes factual, mathematical, code, and evidence defects |
| 13 | Editorial review | Progression, terminology, diagrams, exercises, and accessibility pass |
| 14 | Release candidate | Links, builds, tests, benchmark reproductions, licensing, and publication artifacts pass |

## Phase 0 quality gate

A fresh agent must be able to locate the mission, audience, build progression,
chapter sequence and contract, current Hermon boundary, source and benchmark
rules, Unicode text-diagram policy, code layout, current status, and next task without chat
history. The planning documents must not contradict one another.

## Dependency chain

Part I establishes the autoregressive loop; Part II establishes Transformer
semantics; Part III binds semantics to bytes; Parts IV–VI bind state to
workloads and memory ownership; Parts VII–X bind execution to native/hardware
and modern model behavior; Parts XI–XII prove and operate the system; Part XIII
audits a production architecture; Part XIV derives future systems from learned
invariants; Part XV integrates the result.
