# Chapter 6 Exploratory Blocking and Crossover Record

## Question and validity gate

How does the scalar ENGINE-2 blocked kernel respond to tile size, and at what
small square size does its bookkeeping stop costing more than the direct
`i,j,k` loop on this machine? Every candidate is checked against that reference
within `1e-4 + 1e-5 |expected|` per element before measurement. The independent
Python oracle, 133-test Rust suite, `cargo check`, and warning-denying Clippy
run passed.

## Reproducer and environment

- Book/code commit: `03e08a877be445d70a211996a8eb735a982e5c0f`
- Date: 2026-09-03
- Harness and command: `code/mini-engine/crates/engine0/examples/chapter06_bench.rs`; `cargo run --manifest-path code/mini-engine/Cargo.toml --release -p engine0 --example chapter06_bench`
- Machine/CPU: Apple M1, arm64; RAM: 16 GiB; GPU: not used
- OS: macOS 26.6.2, build 25G83
- Rust: `rustc 1.92.0 (ded5c06cf 2025-12-08)`, LLVM 21.1.3
- Build: Cargo `release`, defaults, no extra `RUSTFLAGS`
- Inputs: deterministic canonical row-major `f32`; one process, one thread
- Work: output allocation/zeroing and API metadata included; warm process;
  hardware cache contents and competing host load uncontrolled
- Statistic: median wall-clock nanoseconds

## Raw result: tile sweep

Shape is `M=K=N=192`; seven repetitions per candidate.

```text
tile,median_ns,gflops,relative_to_ijk
8,10900458,1.299,0.56x # checksum=48.355335
16,4394583,3.221,1.40x # checksum=48.355335
24,2482750,5.702,2.47x # checksum=48.355335
32,2809958,5.038,2.18x # checksum=48.355335
48,1976375,7.162,3.10x # checksum=48.355335
64,1428667,9.908,4.29x # checksum=48.355335
```

## Raw result: crossover probe

The blocked candidate uses `BlockSize { m: 32, k: 32, n: 32 }`.

```text
size,repetitions,ijk_ns,blocked_ns,speedup
8,31,333,1000,0.33x # checksum=1.473601
16,31,2459,2834,0.87x # checksum=1.986403
32,21,19375,12500,1.55x # checksum=-0.397974
64,15,178333,93125,1.91x # checksum=0.320800
128,9,1668500,755625,2.21x # checksum=14.711495
```

## Interpretation and limits

The best tested tile at `192³` was 64, not the API default of 32; this is a
useful negative result against treating one teaching default as universally
optimal. For the fixed 32 tile, blocked execution lost at sizes 8 and 16 and
won from 32 upward in this probe. That bracket is not an immutable crossover:
shape, compiler, cache state, CPU, tile anisotropy, and measurement noise all
move it. The kernel is scalar and does not claim production-grade packing,
SIMD, threading, or cache-model tuning.
