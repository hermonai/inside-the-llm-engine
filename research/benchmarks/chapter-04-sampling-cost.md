# Chapter 4 Sampling-Cost Probe

Status: exploratory teaching measurement, not a production LLM benchmark.

## Question

How does ENGINE-1's straightforward scalar sampling work scale with vocabulary
size when measured separately from model forward execution?

## Reproducer

- Book commit: `5e38b8b612394e56df6aa240a6fe660e0331b51e`
- Date: 2026-09-03
- Command: `cargo run --release -q -p engine0 --example chapter04_sampling_cost`
- Harness:
  `code/mini-engine/crates/engine0/examples/chapter04_sampling_cost.rs`
- Compiler: `rustc 1.92.0 (ded5c06cf 2025-12-08)`
- Build: Cargo `--release`, default target features
- Machine: MacBook Pro (`MacBookPro17,1`), Apple M1, 8 CPU cores
  (4 performance + 4 efficiency), 16 GiB RAM
- OS: macOS 26.6.2 (25G83)
- GPU: not used
- Model/quantization: none; deterministic finite synthetic `f32` logits
- Workload: one sampling call over `V` logits; no tokenizer, model forward,
  stream, concurrency, or cache state
- Configuration: temperature 1.0; top-k 40; top-p 0.9; seed `0x5eed`
- Warmup: 32 calls per repetition
- Repetitions: 7 after one discarded warmup repetition
- Statistic: median integer nanoseconds per call
- Iterations: 50,000 (`V=16`), 10,000 (`V=256`), 500 (`V=4,096`)

## Raw result

```text
vocab,iterations,greedy_ns,softmax_categorical_ns,top_k_40_ns,top_p_0_9_ns
16,50000,26,341,472,597
256,10000,284,2832,5295,6817
4096,500,5137,41467,107582,127452
```

## Interpretation

Greedy performs one maximum scan. The stochastic baseline adds temperature
scaling, stable softmax, probability validation, RNG, and a cumulative scan.
The teaching top-k path sorts candidate indexes; top-p sorts probability mass
and renormalizes. Their increasing cost is consistent with those operations.

This probe does **not** establish end-to-end token latency and must not be
extrapolated to production LLMs. A real model forward, vocabulary size,
provider implementation, memory placement, vectorization, and batching can all
change the proportion. Its pedagogical result is narrower: sampling work is
measurable, operation order matters, and the clarity-first sorting paths have a
visible cost that later implementations may optimize only after equivalence.
