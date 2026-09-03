# Lab 33 — Compute RMS by Hand

**Chapter:** 7. **Level:** CHECK.

## Objective

Derive the scalar normalization factor from a four-element vector before using
the operator implementation.

## Prerequisites

Chapter 7's mean-square and RMS definitions.

## Setup

Use `x=[1,-2,3,-4]`, `D=4`, and `epsilon=1e-5`.

## Build

Calculate each square, their sum, mean square, stabilized denominator, and
reciprocal. Keep epsilon inside the square root. Then use learned scale
`w=[1,0.5,2,-1]` to calculate all four outputs.

## Expected observation

The squares sum to `30`, the mean square is `7.5`, and the reciprocal factor is
approximately `0.365148`. Sign comes from `x` and `w`; squaring only determines
the shared magnitude factor.

## Explanation

Squaring prevents signs from cancelling, the mean makes the statistic
dimension-normalized, square root restores the input's units, and reciprocal
turns the denominator into a reusable scale.

## Verification

```bash
python3 code/reference/python/chapter07_embedding_rmsnorm_oracle.py
```

Match the full vector within the oracle's stated tolerance.

## Break / prove

Move epsilon outside the square root. Calculate the new result and prove it is
a different operator, even when the numerical difference is small here.

## Extend

Repeat with the zero vector and explain why the output remains zero despite a
large finite reciprocal factor.

## Cleanup

Label approximate decimal values with `≈`; retain exact symbolic steps.
