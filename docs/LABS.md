# Engineering Labs

Labs are executable curriculum gates. Each lab records prerequisites, chapter,
expected artifact, oracle, failure injection, measurement (if any), and cleanup.

| Lab | Primary chapter | Build | Break / prove |
| --- | ---: | --- | --- |
| [1. Generate one token manually](../labs/lab-01-generate-one-token-manually.md) | 1 / 4 | Predict ENGINE-0's candidate selection; revisit with logits | Match a hand-computable oracle; inject cancellation/failure and alter score order |
| [2. Tokenize by hand](../labs/lab-02-tokenize-by-hand.md) | 2 | Apply a fixed ranked BPE table | Change rank/prerequisite and catch the segmentation change |
| [3. Stream UTF-8 across token boundaries](../labs/lab-03-stream-utf8-across-tokens.md) | 2 | Buffer byte fragments until valid text exists | Reject malformed and incomplete terminal sequences without lossy replacement |
| [4. Use the wrong chat template](../labs/lab-04-use-the-wrong-chat-template.md) | 2 | Compare structured-template and naive IDs | Prove ordinary marker text cannot insert controls; reject a mismatched tokenizer |
| [5. Calculate a forward pass by hand](../labs/lab-05-forward-pass-by-hand.md) | 3 | Derive embedding and every logit | Compare the full vector with independent Python and Rust oracles |
| [6. Change one weight](../labs/lab-06-change-one-weight.md) | 3 | Predict one parameter's effect | Distinguish row-local projection changes from embedding fan-out |
| [7. Same last token, same output](../labs/lab-07-same-last-token-same-output.md) | 3 | Compare different histories | Prove ENGINE-1 uses only the final token |
| [8. Break the shape](../labs/lab-08-break-the-shape.md) | 3 | Construct malformed parameters | Require typed dimension, count, finite-value, ID, and vocabulary failures |
| [9. Stable softmax by hand](../labs/lab-09-stable-softmax-by-hand.md) | 4 | Normalize a fixed logit vector safely | Reproduce naive overflow and prove shift invariance |
| [10. Change temperature](../labs/lab-10-temperature.md) | 4 | Compare one vector at three temperatures | Reject zero, negative, infinite, and NaN stochastic temperatures |
| [11. Select with a fixed draw](../labs/lab-11-fixed-categorical-draw.md) | 4 | Map a draw through cumulative intervals | Exercise exact boundaries and malformed distributions |
| [12. Top-k versus top-p](../labs/lab-12-top-k-vs-top-p.md) | 4 | Derive fixed-count and adaptive-mass sets | Expose invalid edges and order dependence |
| [13. Trace the autoregressive loop](../labs/lab-13-build-the-autoregressive-loop.md) | 4 | Generate `Rust` then EOS from real logits | Inject budget, cancellation, model, and decode failures |
| [14. Change the seed](../labs/lab-14-change-the-seed.md) | 4 | Repeat a seeded stochastic trace | Prove request-local RNG ownership and bounded reproducibility |
| [15. Break the sampler](../labs/lab-15-break-the-sampler.md) | 4 | Exercise typed sampler failures | Reject silent fallback and prove exactly-once termination |
| [16. Calculate tensor offsets by hand](../labs/lab-16-offset-by-hand.md) | 5 | Derive canonical strides and offsets | Reject wrong-rank and out-of-bounds indices |
| [17. Transpose without copy](../labs/lab-17-transpose-without-copy.md) | 5 | Swap rank-2 shape/stride metadata | Prove storage aliases and logical order changes |
| [18. Reshape a contiguous view](../labs/lab-18-reshape-view.md) | 5 | Borrow one allocation through compatible shapes | Reject count mismatch and non-contiguous input |
| [19. Copy a non-contiguous view](../labs/lab-19-non-contiguous-copy.md) | 5 | Materialize logical order canonically | Prove the new owner is independent |
| [20. Break shape arithmetic](../labs/lab-20-break-shape-arithmetic.md) | 5 | Exercise checked size and extent paths | Reject overflow, short storage, and bad strides |
| [21. Mutation and aliasing](../labs/lab-21-mutation-and-aliasing.md) | 5 | Mutate through one exclusive canonical borrow | Demonstrate compiler rejection of conflicting borrows |
| 22. Implement naive attention | 10 | Dense causal scalar attention | Expose mask and stability failures |
| 23. Decode with and without KV | 21 | Add per-layer KV reuse | Greedy logits agree across both paths |
| 24. Parse a GGUF tensor directory | 15 | Bounds-checked metadata/tensor index | Reject truncation, overflow, and invalid alignment |
| 25. Measure quantized matvec bandwidth | 17 | Packed matvec benchmark | Separate bytes, dequantization, compute, and control |
| 26. Build a continuous batching simulator | 26 | Iteration scheduler | Inject skew, cancellation, and head-of-line blocking |
| 27. Implement a block allocator | 30 | Allocate/incref/decref/free | Boundary, exhaustion, and concurrent lifetime tests |
| 28. Demonstrate prefix COW corruption | 32 | Share aligned prefix pages | Disable COW on partial tail and reproduce corruption |
| 29. Compare scalar and SIMD attention | 42–44 | ISA-specialized microkernel | Differential shapes, dispatch fallback, sanitizer |
| 30. Measure CPU/GPU attention crossover | 48 | Shape-gated provider A/B | Find the launch/transfer break-even, including losing cases |
| 31. Build an expert pager | 56–57 | Pack, acquire, pin, prefetch, evict | Inject short reads, queue pressure, and pin leaks |
| 32. Cancel during decode | 27/69/71 | End-to-end cancellation propagation | Prove leases, slots, and streams release exactly once |
| 33. Run model equivalence tests | 66 | Compare real-model token logits | Vary context, GQA, quantization, and reduction tolerance |
| 34. Benchmark without cache contamination | 73 | Reproducible warm/cold harness | Detect accidental prefix/model/OS-cache leakage |

Every lab evolves through CHECK, BUILD, BREAK, and EXTEND prompts. Performance
labs follow `BENCHMARK_POLICY.md`; numerical labs use independent oracles.
