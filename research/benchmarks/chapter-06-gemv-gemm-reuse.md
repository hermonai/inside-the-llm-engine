# Chapter 6 Exploratory GEMV/GEMM Reuse Record

## Question and validity gate

With a fixed row-major `512×512` weight matrix, what changes as one input
column grows to 8 and 64 columns? The `N=1` case uses a scalar GEMV loop; larger
`N` uses ENGINE-2's scalar blocked GEMM. Correctness checks, the independent
oracle, 133 Rust tests, `cargo check`, and warning-denying Clippy passed before
timing.

This experiment compares workload regimes and reuse opportunities. It does
not claim that different kernels execute identical instructions, and the
ideal intensity column is an analytic compulsory-traffic lower-bound model,
not measured memory traffic.

## Reproducer and environment

- Book/code commit: `03e08a877be445d70a211996a8eb735a982e5c0f`
- Date: 2026-09-03
- Harness and command: `code/mini-engine/crates/engine0/examples/chapter06_bench.rs`; `cargo run --manifest-path code/mini-engine/Cargo.toml --release -p engine0 --example chapter06_bench`
- Machine/CPU: Apple M1, arm64; RAM: 16 GiB; GPU: not used
- OS: macOS 26.6.2, build 25G83
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, LLVM 21.1.3
- Build: Cargo `release`, defaults, no extra `RUSTFLAGS`
- Inputs: deterministic `f32`; `M=K=512`; `N` shown below
- Execution: one process, one thread, warm process; output allocation included;
  cache contents, frequency, and competing host load uncontrolled
- Statistic: median wall-clock nanoseconds; repetitions shown below

## Raw result

```text
n,repetitions,kernel,median_ns,gflops,ideal_flop_per_byte
1,15,gemv,191416,2.739,0.498 # checksum=-22.585403
8,9,blocked_gemm,3294459,1.273,3.879 # checksum=-21.333565
64,5,blocked_gemm,6217167,5.397,25.600 # checksum=-1.752068
```

The model is

$$
I_{ideal}=\frac{2MKN}{4(MK+KN+MN)}.
$$

It counts one compulsory read of each input and one output write. Actual scalar
code can transfer more data through the cache hierarchy.

## Interpretation and limits

The ideal reuse opportunity rises sharply with `N`, but measured scalar
throughput did not rise monotonically: the narrow `N=8` blocked case achieved
less GFLOP/s than GEMV, while `N=64` exceeded both. Loop overhead, tile shape,
and short contiguous runs can overwhelm the analytic reuse advantage. The
record therefore supports two bounded claims: batching creates *potential*
weight reuse, and shape-specific measurement is necessary. It does not turn
ideal arithmetic intensity into a bandwidth measurement or a promise about a
production inference runtime.
