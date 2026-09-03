# Chapter 5 Exploratory Traversal-Order Record

## Question and validity gate

Does iterating one canonical row-major `[N,N]` `f32` allocation in physical
order differ measurably from visiting the same elements column by column on
this machine? The tensor oracle and 108-test Rust suite passed before timing.
Both traversal functions must produce the exact same `f64` checksum or the
measurement is invalid.

This is an access-order probe, not matrix multiplication and not a universal
cache-speed claim.

## Reproducer

- Book/code commit: `756b1b48c0d3f3687d52cd6d571045a281297854`
- Date: 2026-09-03
- Harness: `code/mini-engine/crates/engine0/examples/chapter05_traversal.rs`
- Command:

```sh
cargo run --manifest-path code/mini-engine/Cargo.toml --release \
  -p engine0 --example chapter05_traversal -- 2048 7
```

## Environment

- Machine/CPU: Apple M1, 8 physical / 8 logical cores
- RAM: 16 GiB
- GPU: not used
- Storage: not material after process initialization
- OS: macOS 26.6.2, build 25G83
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, LLVM 21.1.3
- Host target: `aarch64-apple-darwin`
- Build: Cargo `release`; repository defaults, no additional `RUSTFLAGS`
- Tensor: canonical row-major `[2048,2048]`, 4,194,304 `f32` values, 16 MiB
- Values: `storage[index] = (index mod 251) as f32`
- Work: read every value once per traversal; accumulate as `f64`
- Warmup: one full row-major and one full column-wise traversal
- Repetitions: 7 per order, execution order alternated each repetition
- Statistic: median elapsed nanoseconds
- Concurrency: one process, one thread, no model/runtime requests
- Cache state: warm process; hardware cache contents uncontrolled between runs
- Control: exact checksum equality, `524280621.0`

## Raw result

```text
shape=[2048,2048] repetitions=7
checksum=524280621.0
row_major_median_ns=4163875
column_wise_median_ns=13692583
```

## Interpretation and limits

On this single run, physical-order traversal had a lower median. Both loops
performed the same additions over the same values; only visit order changed.
The result is consistent with spatial locality, but does not isolate cache
levels, prefetching, compiler decisions, frequency, or competing load. It must
not be generalized into a fixed ratio, a different processor, a kernel result,
or end-to-end inference performance.
