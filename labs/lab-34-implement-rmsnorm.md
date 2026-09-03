# Lab 34 — Implement RMSNorm

**Chapter:** 7. **Level:** BUILD.

## Objective

Lower the RMSNorm equation into a checked two-pass scalar operator over logical
rank-one views.

## Prerequisites

Lab 33, Tensor Substrate v1 indexing, and floating-point tolerance.

## Setup

Open `engine0::normalization::rms_norm_reference`. Identify validation, pass
one reduction, scalar reciprocal RMS, pass two learned scaling, and output
allocation.

## Build

Annotate which data each pass reads and why output cannot be finalized until
the reduction is complete. Confirm input, weight, accumulator, square root, and
output all use `f32` in the teaching implementation.

## Expected observation

The reference accepts canonical, strided, and zero-stride logical views and
always returns a new canonical `[D]` owner. It performs no implicit layout copy
before the operator.

## Explanation

The first pass reduces `D` values to one scalar. The second pass broadcasts
that scalar while applying `D` learned weights. This dataflow differs from a
matrix product even though both touch floating-point tensors.

## Verification

```bash
cd code/mini-engine
cargo test --test transformer_primitives rmsnorm_matches_the_hand_calculated_mixed_sign_example
cargo test --test transformer_primitives rmsnorm_accepts_strided_input_and_weight
```

## Break / prove

Use the physical slice directly and show why the strided fixture detects that
hidden layout assumption.

## Extend

Write an operation/traffic ledger without timing: input read twice, weight read
once, output written once under the simple cold-payload model.

## Cleanup

Restore checked logical indexing.
