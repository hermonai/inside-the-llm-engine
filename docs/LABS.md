# Engineering Labs

Labs are executable curriculum gates. Each lab records prerequisites, chapter,
expected artifact, oracle, failure injection, measurement (if any), and cleanup.

| Lab | Primary chapter | Build | Break / prove |
| --- | ---: | --- | --- |
| 1. Generate one token manually | 4 | Compute logits and sample one token | Match a hand-computable oracle; alter seed/logit order |
| 2. Implement naive attention | 10 | Dense causal scalar attention | Expose mask and stability failures |
| 3. Decode with and without KV | 21 | Add per-layer KV reuse | Greedy logits agree across both paths |
| 4. Parse a GGUF tensor directory | 15 | Bounds-checked metadata/tensor index | Reject truncation, overflow, and invalid alignment |
| 5. Measure quantized matvec bandwidth | 17 | Packed matvec benchmark | Separate bytes, dequantization, compute, and control |
| 6. Build a continuous batching simulator | 26 | Iteration scheduler | Inject skew, cancellation, and head-of-line blocking |
| 7. Implement a block allocator | 30 | Allocate/incref/decref/free | Boundary, exhaustion, and concurrent lifetime tests |
| 8. Demonstrate prefix COW corruption | 32 | Share aligned prefix pages | Disable COW on partial tail and reproduce corruption |
| 9. Compare scalar and SIMD attention | 42–44 | ISA-specialized microkernel | Differential shapes, dispatch fallback, sanitizer |
| 10. Measure CPU/GPU attention crossover | 48 | Shape-gated provider A/B | Find the launch/transfer break-even, including losing cases |
| 11. Build an expert pager | 56–57 | Pack, acquire, pin, prefetch, evict | Inject short reads, queue pressure, and pin leaks |
| 12. Cancel during decode | 27/69/71 | End-to-end cancellation propagation | Prove leases, slots, and streams release exactly once |
| 13. Run model equivalence tests | 66 | Compare real-model token logits | Vary context, GQA, quantization, and reduction tolerance |
| 14. Benchmark without cache contamination | 73 | Reproducible warm/cold harness | Detect accidental prefix/model/OS-cache leakage |

Every lab evolves through CHECK, BUILD, BREAK, and EXTEND prompts. Performance
labs follow `BENCHMARK_POLICY.md`; numerical labs use independent oracles.
