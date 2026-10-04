# Chapter 42 lab — Shrinking KV

## Predict
Compute the 28-layer/8-KV-head/128-wide BF16 payload before running. Then map eight
query heads onto two K/V heads.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
The byte tests reproduce the architecture formula. The numerical test evaluates the
MLA value path in two different operation orders: project-every-history-entry versus
reduce-first/project-once.

## Break
Change `kv_head` to `q_head % hkv`; the mapping test catches the plausible wrong grouping.
Make the projection matrix depend on token index: the absorption identity no longer
applies because there is no single fixed linear map to factor outside the sum.

## Limits
This is not a DeepSeek checkpoint, GPU kernel, RoPE implementation or quality benchmark.
It isolates head ownership, byte accounting and the fixed-linear-map absorption
invariant.
