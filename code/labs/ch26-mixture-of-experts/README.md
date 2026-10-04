# Chapter 43 lab — Mixture of Experts

## Predict
For 128 experts and top-8 routing, estimate how many distinct experts a 32-token and
64-token batch activates. For the three-token fixture, predict which expert is hottest.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
`reference` evaluates assignments directly. `grouped` deliberately reorders them by
expert and uses the saved token ID as the inverse map during combine. Their equality
demonstrates that grouping is an execution-layout optimization, not a semantic change.

## Break
`broken_without_inverse_map` discards token identity after grouping and is caught by a
test. The gate-pairing test shows another subtle failure: sorting expert IDs separately
from their gates changes the model even if every intended expert still executes.

## Extend
Add a per-expert capacity. Implement (1) token dropping and (2) dropless overflow, then
construct two batches where adding a batch-mate changes the first token only in the
dropping implementation.

## Limits
This lab does not implement a learned router, real FFNs, quantization, grouped GPU GEMM,
or all-to-all communication. It isolates the route/dispatch/combine correctness contract
and the expert-union arithmetic.
