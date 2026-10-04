# Chapter 34 lab — Prefix Caching and RadixAttention

## Predict
Before running, predict the reusable length for same-tenant `[1,2,3,4,9]` after
caching `[1,2,3,4,5,6]`, and for the identical request in another namespace.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
The weak hash is intentionally unsafe as an identity proof. The cache verifies exact
namespace and token ancestry. This makes collisions observable in a deterministic unit
test rather than relying on probability.

## Break
1. Remove the namespace comparison: the tenant-isolation test must fail.
2. Change `verified_exact_hit` to trust only `weak_hash`: the collision test must fail.
3. Change eviction to remove the most recently touched entry: the LRU test must fail.
4. Next exercise: integrate Chapter 33 generation handles and pins. Tests must reject
   stale generations, make pin acquisition atomic, and forbid eviction of pinned state.

## Limits
This lab models index correctness and eviction metadata, not real tensor KV payloads,
GPU kernels, cryptographic hashing, distributed routing, or hybrid-model checkpoints.
