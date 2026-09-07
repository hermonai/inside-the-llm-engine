# Lab 16 — Calculate Tensor Offsets by Hand

**Chapter:** 5. **Level:** CHECK.

## Visual contract checkpoint

Study the [canonical plate](../figures/generated/tensor.svg) and its
[text equivalent](../figures/generated/tensor.txt).
For the visual fixture A = [[1,2,3],[4,5,6]], derive element offset 5 and byte offset 20 for A[1,2]. Then return to the rank-3 exercise above.

From the repository root, run `python3 scripts/check-tensor-visual-parity.py`.
This supplements, rather than replaces, the original exercise and oracle.

## Reading prerequisite

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
