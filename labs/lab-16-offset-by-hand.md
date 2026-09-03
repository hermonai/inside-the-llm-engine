# Lab 16 — Calculate Tensor Offsets by Hand

**Chapter:** 5. **Level:** CHECK.

## Prerequisites

Read Chapter 5 through canonical row-major strides.

## Build

For shape `[2,3,4]`, derive strides `[12,4,1]`. Calculate offsets for
`[0,0,0]`, `[0,2,3]`, and `[1,2,3]`, then verify them with
`checked_offset` in `engine0::tensor`.

## Oracle

Run `python3 code/reference/python/chapter05_tensor_oracle.py`. The final index
must map to element offset `23`; all results use exact integer equality.

## Break / prove

Pass a rank-2 index and an axis index equal to its dimension. Require typed
`RankMismatch` and `IndexOutOfBounds` errors rather than a panic.

## Extend

Choose a rank-4 shape, derive its strides, and prove the largest valid logical
index maps to `element_count - 1`.

## Cleanup

Revert any temporary test-only shape changes.
