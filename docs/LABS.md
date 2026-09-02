# Engineering Labs

Labs are executable curriculum gates. Each lab records prerequisites, chapter,
expected artifact, oracle, failure injection, measurement (if any), and cleanup.

| Lab | Primary chapter | Build | Break / prove |
| --- | ---: | --- | --- |
| [1. Generate one token manually](../labs/lab-01-generate-one-token-manually.md) | 1 / 4 | Predict ENGINE-0's candidate selection; revisit with logits | Match a hand-computable oracle; inject cancellation/failure and alter score order |
| [2. Tokenize by hand](../labs/lab-02-tokenize-by-hand.md) | 2 | Apply a fixed ranked BPE table | Change rank/prerequisite and catch the segmentation change |
| [3. Stream UTF-8 across token boundaries](../labs/lab-03-stream-utf8-across-tokens.md) | 2 | Buffer byte fragments until valid text exists | Reject malformed and incomplete terminal sequences without lossy replacement |
| [4. Use the wrong chat template](../labs/lab-04-use-the-wrong-chat-template.md) | 2 | Compare structured-template and naive IDs | Prove ordinary marker text cannot insert controls; reject a mismatched tokenizer |
| 5. Implement naive attention | 10 | Dense causal scalar attention | Expose mask and stability failures |
| 6. Decode with and without KV | 21 | Add per-layer KV reuse | Greedy logits agree across both paths |
| 7. Parse a GGUF tensor directory | 15 | Bounds-checked metadata/tensor index | Reject truncation, overflow, and invalid alignment |
| 8. Measure quantized matvec bandwidth | 17 | Packed matvec benchmark | Separate bytes, dequantization, compute, and control |
| 9. Build a continuous batching simulator | 26 | Iteration scheduler | Inject skew, cancellation, and head-of-line blocking |
| 10. Implement a block allocator | 30 | Allocate/incref/decref/free | Boundary, exhaustion, and concurrent lifetime tests |
| 11. Demonstrate prefix COW corruption | 32 | Share aligned prefix pages | Disable COW on partial tail and reproduce corruption |
| 12. Compare scalar and SIMD attention | 42–44 | ISA-specialized microkernel | Differential shapes, dispatch fallback, sanitizer |
| 13. Measure CPU/GPU attention crossover | 48 | Shape-gated provider A/B | Find the launch/transfer break-even, including losing cases |
| 14. Build an expert pager | 56–57 | Pack, acquire, pin, prefetch, evict | Inject short reads, queue pressure, and pin leaks |
| 15. Cancel during decode | 27/69/71 | End-to-end cancellation propagation | Prove leases, slots, and streams release exactly once |
| 16. Run model equivalence tests | 66 | Compare real-model token logits | Vary context, GQA, quantization, and reduction tolerance |
| 17. Benchmark without cache contamination | 73 | Reproducible warm/cold harness | Detect accidental prefix/model/OS-cache leakage |

Every lab evolves through CHECK, BUILD, BREAK, and EXTEND prompts. Performance
labs follow `BENCHMARK_POLICY.md`; numerical labs use independent oracles.
