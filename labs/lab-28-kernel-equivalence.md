# Lab 28 — Kernel Equivalence

**Chapter:** 6. **Level:** CHECK.

## Prerequisites

Reference and blocked GEMM plus floating-point tolerance.

## Build

Generate deterministic matrices over `M,K,N = 0..6`. Run the reference kernel
and blocked kernels with `[1,1,1]`, `[2,3,4]`, and `[5,4,3]` tiles. Calculate
the maximum absolute and relative error for every case.

## Oracle

All shapes and values must agree under

$$
|a-r| \le 10^{-5} + 10^{-5}|r|.
$$

The committed Rust property grid is deterministic and requires no random-test
framework.

## Break / prove

Reverse one K-tile traversal in a disposable change. Find an input where low
bits differ, then explain why non-associativity makes tolerance necessary even
when both algorithms implement the same real-number equation.

## Extend

Report exact equality separately from tolerance equality. Do not loosen the
tolerance merely to silence an unexplained discrepancy.

## Visual checkpoint — Chapter 6 regeneration

Run the new F32 reassociation and explicit FMA tests. Explain zero versus one in the associativity example and zero versus -2^-26 for separate versus fused rounding. Do not change production accumulation or widen tolerances.

Use the [shared fixture](../code/reference/fixtures/chapter06-visual.json) and
run `python3 scripts/check-linear-visual-parity.py` from the repository root.
Report values, shapes, address units and the evidence boundary, not only PASS.

## Cleanup

Restore ascending K traversal and retain any small diagnostic fixture that
clarifies reduction order.
