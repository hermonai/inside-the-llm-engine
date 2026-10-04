# Chapter 40 lab — Draft-tree geometry

## Predict
For parent links `[-,0,0,1,2]`, write each root-to-node path before running.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
The ancestry mask is generated from parent links. `path_tokens` is an independent
semantic reconstruction. Their equality proves the packed tree represents the same
histories as separate causal paths.

## Break
Replace `ancestry_mask` with `triangular_mask`; the sibling-leak test fails.
Set logical depth equal to packed row + 1; the depth test fails. Change scores and
observe the deterministic budget selection.

## Limits
This lab does not reproduce EAGLE neural weights, target logits, GPU kernels, exact
EAGLE-2 expansion, or speedups. It isolates the tree representation contract those
systems require.
