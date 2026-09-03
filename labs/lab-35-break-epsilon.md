# Lab 35 — Break Epsilon

**Chapter:** 7. **Level:** BREAK.

## Objective

Prove that epsilon is semantic operator metadata with a validated domain, not
decorative syntax.

## Prerequisites

Labs 33–34.

## Setup

Use the zero vector, unit weights, and epsilon values `1e-5`, `0`, `-1`, NaN,
positive infinity, and negative infinity.

## Build

Predict denominator and result behavior before calling the Rust operator. Map
each invalid value to `NormalizationError::InvalidEpsilon`.

## Expected observation

Positive finite epsilon makes the zero-vector reciprocal finite and the output
zero. Every other listed epsilon is rejected before reduction.

## Explanation

With epsilon zero, the zero-vector reciprocal is undefined. Negative epsilon
can make the square-root argument negative. NaN and infinities violate the
finite numerical contract. A model-specific constant is a configuration
choice; positive-finite validation is the operator policy.

## Verification

```bash
cd code/mini-engine
cargo test --test transformer_primitives rmsnorm_rejects_every_nonpositive_or_nonfinite_epsilon
cargo test --test transformer_primitives rmsnorm_zero_vector_is_finite_and_zero
```

## Break / prove

Temporarily accept epsilon zero. Run the zero-vector case and observe the
non-finite intermediate or output. Restore validation.

## Extend

Compare `sqrt(mean_square + epsilon)` with `sqrt(mean_square) + epsilon` for a
very small vector. Name the convention before comparing numbers.

## Cleanup

Do not retain permissive epsilon validation.
