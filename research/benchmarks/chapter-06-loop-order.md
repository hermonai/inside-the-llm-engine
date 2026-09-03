# Chapter 6 Exploratory Loop-Order Record

## Question and validity gate

For the same scalar row-major square matrix product, how does the
equation-shaped `i,j,k` traversal compare with `i,k,j` on this machine? Before
timing, the harness checks every result element with
`|actual-expected| <= 1e-4 + 1e-5 |expected|`. The independent Python oracle,
133-test Rust suite, `cargo check`, and warning-denying Clippy run passed.

This is a single-machine locality experiment, not a BLAS comparison or an
end-to-end inference result.

## Reproducer and environment

- Book/code commit: `03e08a877be445d70a211996a8eb735a982e5c0f`
- Date: 2026-09-03
- Harness: `code/mini-engine/crates/engine0/examples/chapter06_bench.rs`
- Command: `cargo run --manifest-path code/mini-engine/Cargo.toml --release -p engine0 --example chapter06_bench`
- Machine/CPU: Apple M1, arm64; RAM: 16 GiB; GPU: not used
- OS: macOS 26.6.2, build 25G83
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, LLVM 21.1.3
- Build: Cargo `release`, repository defaults, no additional `RUSTFLAGS`
- Inputs: deterministic canonical row-major `f32`; one process and one thread
- Work: allocation and zero-initialization included; two floating operations
  counted per inner iteration; warm process; cache contents uncontrolled
- Statistic: median wall-clock nanoseconds; repetitions shown below

## Raw result

```text
size,repetitions,ijk_ns,ikj_ns,ijk_gflops,ikj_gflops,speedup
64,15,185375,54791,2.828,9.569,3.38x # checksum=15.758824
128,9,1645000,286375,2.550,14.646,5.74x # checksum=-19.102465
256,5,15638708,1709000,2.146,19.634,9.15x # checksum=-8.965134
```

## Interpretation and limits

`i,k,j` was faster in all three observed cases, and its advantage grew over
this size range. That is consistent with walking rows of the right operand and
output contiguously instead of using a long-stride right-operand access in the
innermost loop. The result does not isolate cache capacity, prefetching,
compiler vectorization, frequency, allocator effects, or system load. The
ratios are observations on this exact build and must not be presented as
portable constants.
