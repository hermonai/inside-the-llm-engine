# Lab 38 — Compare Rust Against the Python Oracle

**Chapter:** 7. **Level:** CHECK.

## Objective

Use independently expressed mathematics to validate the Rust embedding and
RMSNorm contracts without treating either implementation as self-proving.

## Prerequisites

Labs 30–37 and the repository numerical-equivalence policy.

## Setup

Read the Rust test fixture and
`code/reference/python/chapter07_embedding_rmsnorm_oracle.py`. Identify their
shared inputs but different expression: checked tensor traversal versus Python
lists, `math.fsum`, and a separate binary32 simulator.

## Build

Run both suites. Compare selected rows, the hand-calculated vector, zero-vector
behavior, scale trends, and magnitude classifications. Record exact equality
only for integer-like lookup values and zeros; use the declared tolerance for
RMSNorm output.

## Expected observation

Both paths agree on semantics and trend. Decimal low bits can differ because
the Rust kernel performs every stage in `f32` while the primary Python oracle
uses wider floats and `math.fsum`.

## Explanation

An independent oracle reduces correlated implementation errors. It does not
prove all possible inputs, so typed boundary tests and deterministic property
fixtures remain necessary.

## Verification

```bash
python3 code/reference/python/chapter07_embedding_rmsnorm_oracle.py
cd code/mini-engine
cargo test --test transformer_primitives
```

The oracle must print `PASS`, and all 30 Rust Chapter 7 tests must pass.

## Break / prove

Transpose two learned weights in Rust. Confirm full-vector comparison fails
even though a scalar RMS-only check would pass.

## Extend

Add a small deterministic vector family and report maximum absolute and
relative error separately. Never loosen tolerance before diagnosing a failure.

## Cleanup

Remove deliberate mismatches and retain only bounded fixtures.
