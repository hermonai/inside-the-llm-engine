# Chapter 35 lab — Chunked Prefill

## Predict
For a 10,240-token prompt with a 6,144-token prefix hit, predict the first chunk under
a 1,024-token budget with 64 pending decodes.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
The lab separates scheduled work from committed progress. Prefix-hit tokens count as
logical context but not new prefill compute. A minimum prefill share demonstrates one
simple anti-starvation policy.

## Break
1. Advance `committed` before checking `succeed`: rollback test must fail.
2. Charge `prompt` instead of `prompt-hit`: prefix accounting test must fail.
3. Remove `min_prefill_share`: the saturation/fairness test must fail.
4. Remove the budget cap: `budget_is_never_exceeded` must fail.

## Limits
This is a CPU scheduler/accounting model. It does not model tensor kernels, real GPU
timings, paged-KV allocation, multimodal atomicity or distributed P/D transfer.
