# Chapter 38 lab — Mixed LoRA and residency

## Predict
For a batch containing base-only rows plus rank-1 and rank-3 adapters, predict whether
row order should change numerical results. Predict what happens when a device slot is
reused while an old request still holds its previous generation.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
`dynamic` evaluates the low-rank path. `merged_oracle` independently materializes the
effective dense weight and is the numerical oracle. `Residency` demonstrates that a
physical slot is not an adapter identity: a generation and immutable adapter ID travel
with every lease.

## Break
1. Ignore `adapter_index` and use adapter 0 for every row: mixed-row parity fails.
2. Drop the LoRA scale: heterogeneous-rank parity fails.
3. Validate only the residency slot number: stale-handle test fails.
4. Allow eviction while pinned: pin/eviction test fails.
5. Silently substitute `None` on a missing adapter: add a request-level test proving
   missing identity must queue/error rather than execute the base model.

## Limits
The arithmetic is scalar f64 and the residency cache contains metadata only. It does
not benchmark GPU gather kernels, DMA, tensor parallelism, quantization or real model
quality.
