# Lab 36 — RMSNorm Scale Experiment

**Chapter:** 7. **Level:** EXTEND.

## Objective

Measure where positive scale invariance is a useful approximation and where
epsilon makes it visibly false.

## Prerequisites

Labs 33–35.

## Setup

Use `x=[1,-2,3,-4]`, fixed learned scale, epsilon `1e-5`, and multipliers
`1e-8`, `0.1`, `1`, `10`, and `100`.

## Build

Run the Chapter 7 Rust example and Python oracle. For every multiplier, report
the maximum absolute output difference from the `alpha=1` result.

## Expected observation

Factors `10` and `100` remain close to baseline; `0.1` differs more; `1e-8`
differs dramatically because epsilon dominates mean square. Exact invariance is
not claimed.

## Explanation

Without epsilon, positive scaling cancels between numerator and RMS denominator
in real arithmetic. Adding a fixed epsilon introduces an absolute scale, which
matters when `alpha^2 * mean_square` is comparable to or below epsilon.

## Verification

```bash
cd code/mini-engine
cargo run --release --example chapter07_scale_and_stress
cd ../..
python3 code/reference/python/chapter07_embedding_rmsnorm_oracle.py
```

Both programs must show the same trend; low-order decimals may differ because
one follows `f32` and one uses Python's wider mathematical oracle.

## Break / prove

Write “RMSNorm is scale invariant” without qualification, then use the `1e-8`
row as a counterexample. Replace the statement with its explicit conditions.

## Extend

Sweep around `sqrt(epsilon / mean_square(x))` and locate the crossover.

## Cleanup

Commit only compact textual findings, not large generated sweeps.
