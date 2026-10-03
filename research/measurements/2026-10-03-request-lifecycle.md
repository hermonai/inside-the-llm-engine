# Request ownership and scaled cancellation replay — 2026-10-03

Used by printed Chapter 31, stable source/lab ID `ch14-request-lifecycle`.
The historical real-model measurement remains in
[the 2026-09-24 record](2026-09-24-m1-part3.md). No GPU or model loaded here.

## Setup and source identity

- Apple M1, ordinary desktop session, macOS 26.6.2 (25G83).
- Rust 1.92.0 (`ded5c06cf`), Cargo 1.92.0.
- Baseline repository commit `7f7b9bc` plus the lab changes identified below.
- The exact timed/tested source is preserved at commit `d48a8c9`. A later
  comment-only clarification describes tick overruns more accurately;
  it does not change the scheduler or any measured runtime operation.
- `src/main.rs` SHA-256:
  `9fb7d40548458de657776dfceab19951cfba8601fc390b208b100a78568b2d27`.
- `src/ownership.rs` SHA-256:
  `40d5f88dfce3c0a935a1b602f702d7a4befb3cef73dbce4f2c24553b6c635b53`.
- Dependency-free CPU-thread replay. A 4 ms tick stands for one step;
  100 ms polling slices, client close after 350 ms, budget 1,500,
  streamed neighbor budget 500, monitor period 2 ms for 2,000 ms.
- Two default invocations and one 3 ms slice invocation; individual rows,
  not percentile estimates. Release timing is monotonic. Delay columns
  are rounded milliseconds, not precise zero or hard latency bounds.
- Other local checks/build work occurred during this desktop session.
  Sleep overshoot and scheduling are uncontrolled. This is a mechanism
  experiment, not a capacity benchmark. The observer's count snapshot
  precedes its close timestamp; there is no atomic cross-thread cut.

Commands, from `code/labs`:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release -p ch14-request-lifecycle
cargo run --release -p ch14-request-lifecycle
cargo run --release -p ch14-request-lifecycle -- --slice-ms 3
cargo run --release -p ch14-request-lifecycle -- --contracts-only --scenarios 100000 --seed 1
cargo run --release -p ch14-request-lifecycle -- --contracts-only --scenarios 100000 --seed 7
```

## Raw timing rows

Rows below transcribe the program's reported count/delay columns without
normalization. Values are abandoned tokens / close-to-release milliseconds.

| Handler / other traffic | Default run 1 | Default run 2 | 3 ms slice run |
| --- | ---: | ---: | ---: |
| Poll / idle | 16 / 68 | 14 / 58 | 0 / 2 |
| Poll / 500-token stream | 437 / 1749 | 437 / 1756 | 0 / 0 |
| Poll / monitor | 438 / 1754 | 438 / 1754 | 0 / 4 |
| Event / idle | 0 / 1 | 0 / 2 | 0 / 3 |
| Event / 500-token stream | 0 / 3 | 0 / 2 | 0 / 1 |
| Event / monitor | 0 / 3 | 0 / 2 | 0 / 0 |
| Stream client leaves after 20 | 1 / 7 | 1 / 8 | 1 / 8 |

Common deterministic protocol output:

```text
Protocol model (illustrative integer ms): idle expiry 2100, fixed deadline 100
cancel selected: reusable=false, delivery=Some(Terminal(Cancelled))
work retired: reusable=true, late delivery=None
```

Each timing invocation also produced this seeded contract summary:

```text
Part 2: the terminal contract on 10000 random scenarios (seed 1)
45109 requests: cancelled 5494, end of sequence 9856, failed 2665, token budget 27094
every request ended exactly once, nothing followed its end, streamed tokens
arrived in order and complete, uncancelled requests ended as the oracle
predicted, and no request made a token after its cancellation
```

Contracts-only output, both with the common protocol prelude:

```text
seed 1: checked 100000 scenarios, 450990 requests
seed 7: checked 100000 scenarios, 450155 requests
```

Two earlier development invocations before the final unit-example test was
added retained these rows. Runtime mechanisms were unchanged; final runs
above correspond to the full source hashes.

| Handler / other traffic | Development 1 | Development 2 |
| --- | ---: | ---: |
| Poll / idle | 15 / 62 | 15 / 64 |
| Poll / stream | 437 / 1751 | 437 / 1749 |
| Poll / monitor | 433 / 1755 | 438 / 1754 |
| Event / idle | 0 / 2 | 0 / 2 |
| Event / stream | 0 / 1 | 0 / 1 |
| Event / monitor | 0 / 1 | 0 / 3 |
| Streaming leaver | 1 / 8 | 1 / 8 |

The earlier short-slice development invocation reported poll rows 1/6,
0/0, 0/1; event rows 0/0, 0/1, 0/1; streaming leaver 1/7.
These are retained observations, not extra independent samples.

## Correctness and falsification

All seventeen lifecycle tests pass after restoring mutations; formatting
and clippy pass. Full labs-workspace tests and clippy also pass.

- Closed-form natural-end oracle: 5 budgets × 7 EOS boundaries ×
  7 failure boundaries × 2 streaming modes = 490 specifications.
- Single-owner protocol: 5 commands to depth 7 = 78,125 sequences.
  Completed responsive traces drain one terminal; incomplete traces are
  not asserted to have terminated.
- Idle timeout: all 256 subsets of eight notification times checked against
  an independently enumerated quiet-interval oracle. Chosen tie rule:
  timeout wins a notification exactly at expiry.
- Boundary cases: full deque cancellation, natural prefix drain, late
  retirement, stale generation/serial, duplicate reply, duplicate active
  admission, mixed-ID/status/late-output traces, cancellation precedence.
- Unit examples: wire traffic, fixed-cache byte-seconds, mean occupancy,
  queue fill time and payload bytes recomputed in a test.

Two intentional mutations were executed and restored:

```text
Disable the observed-cancellation guard in check:
checker_rejects_natural_end_after_observed_cancellation ... FAILED
assertion failed: check(&run(s, None), &s, Some(1)).is_err()
exit code 101

Remove the pending-ticket condition from Owner::reusable:
cancel_can_publish_terminal_but_not_reuse_pending_work ... FAILED
assertion failed: !o.reusable()
exit code 101
```

The first development quiet-interval oracle failed at an exact-deadline
notification (`left 3, right 6`). Its predicate had counted the notification
at expiry even though the implementation's chosen tie rule expires first.
The oracle now enumerates wakes strictly before expiry; all 256 schedules
pass. This is a specification correction, not an OS timing guarantee.

## Limits

No socket protocol, real inference, GPU interruption/fencing, remote admission,
distributed exactly-once delivery or crash recovery is demonstrated.
Exhaustive short sequences and seeded scenario testing are complementary
checks, not independent technical review or proof of production concurrency.
ASGI and pinned engine sources were re-read; upstream issue 57061 was not
reproduced locally. The chapter distinguishes the reported failure from
the inspected source boundary and the book's own protocol model.

## Book validation and layout

The final 712-page XeLaTeX/biber build has no overfull boxes, missing glyphs
or unresolved references. Its SHA-256 is
`77efe40836c17e383b63f5bb7b042c84473f4bd4fcdc95da193a4c9c4002bb4a`.
The source fingerprints are in `build/textbook/build-record.json`; all
29 distinct files rendered by `\CodeLines` are included, not only the
foundation listings. Read-only checks exercise duplicate listing paths,
missing sources and outside-repository rejection.

All 712 pages were rendered and passed the visible word-bounds audit in
`build/lifecycle-release-review`. Chapter 31 occupies PDF pages 475--494
(printed 469--488). All twenty pages were inspected in colour and grayscale.
The new wait diagram's crowded label, ledger heading placement and split
worked answers were corrected. Every problem/solution pair is now kept
together. The relevant contents and bibliography pages were inspected too.
This is author layout QA, not independent technical review or a new manual
inspection of every unchanged page.

The textbook gate reports 31 FULL and 28 ZERO chapters, 182 native figures,
122 worked problems and five remaining TODOs. Structure/preservation/secrets,
relative links, the full foundation and Rust workspaces, formatting/clippy,
canonical diagram/style, deterministic figures, native parity, industrial
visual mutation guards and all five Rust/visual parity scripts pass locally.
The earlier independent ggml and native C sanitizer observations remain
historical; no new GPU, model or remote CI execution is claimed.
