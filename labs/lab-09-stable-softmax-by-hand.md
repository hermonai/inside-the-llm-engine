# Lab 9 — Stable Softmax by Hand

Chapter: 4. Prerequisite: Chapter 3 logits. Artifact: one checked probability
vector. Oracle: `code/reference/python/chapter04_sampling_oracle.py`.

## CHECK

For logits `[1,2,3]`, find `m=max(z)`, subtract it, compute each exponential,
sum the numerators, and normalize. Verify every probability is non-negative
and the sum is approximately one.

Expected result:
`[0.0900305732,0.2447284711,0.6652409558]`.

## BUILD

Run:

```sh
python3 code/reference/python/chapter04_sampling_oracle.py --logits 1 2 3
cargo test -p engine0 stable_softmax_matches_the_three_logit_oracle
```

Explain why subtracting one constant leaves the exact distribution unchanged.

## BREAK

Compute naive softmax for `[1000,999,998]`. Record the overflow, then use the
stable formula and verify finite probabilities.

## EXTEND

Add several constants to `[1,2,3]`. State a tolerance and test shift invariance.
Cleanup: revert experimental fixtures unless promoted with a test.

