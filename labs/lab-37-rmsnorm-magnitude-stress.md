# Lab 37 — RMSNorm Magnitude Stress

**Chapter:** 7. **Level:** BREAK.

## Objective

Expose underflow, epsilon dominance, square overflow, and reduction overflow in
the transparent `f32` reference algorithm.

## Prerequisites

Lab 36 and the Chapter 6 non-associativity discussion.

## Setup

Use alternating-sign two-element vectors with magnitudes `1e-20`, `1e-10`,
`1`, `1e10`, and `1e20`. Add a four-element `1e19` fixture for accumulated
sum overflow.

## Build

Predict whether each `x*x` fits `f32`, whether the sum fits, and whether epsilon
dominates. Run the Rust example and focused tests.

## Expected observation

Small values remain finite but are scaled mainly by epsilon. `1e10` normalizes
successfully. `1e20` is finite but its square overflows and returns
`NonFiniteSquare`. Several finite `1e19` squares overflow their sum and return
`NonFiniteReduction`.

## Explanation

Detecting a failed naïve reduction is not the same as preventing it. Scaled
sum-of-squares algorithms can extend the finite range, but Chapter 7 keeps the
equation-shaped kernel and makes its boundary explicit.

## Verification

```bash
cd code/mini-engine
cargo test --test transformer_primitives rmsnorm_reports_f32
cargo run --release --example chapter07_scale_and_stress
```

## Break / prove

Remove the square-finiteness check and observe how an infinite denominator can
produce a misleading zero or NaN. Restore the typed failure.

## Extend

Study Netlib `SLASSQ` and sketch its `(scale,sumsq)` invariant. Do not silently
replace the reference operator.

## Cleanup

Retain the finite-range regression fixtures.
