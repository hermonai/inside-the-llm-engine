# Lab 18 — Reshape a Contiguous View

**Chapter:** 5. **Level:** BUILD.

## Visual contract checkpoint

Study the [canonical plate](../figures/generated/tensor-reshape.svg) and its
[text equivalent](../figures/generated/tensor-reshape.txt).
Compare reshape [3,2] with transpose [3,2]: their values at [0,1] are 2 and 4. Explain why reshaping the transpose to [6] returns NonContiguous rather than silently copying.

From the repository root, run `python3 scripts/check-tensor-visual-parity.py`.
This supplements, rather than replaces, the original exercise and oracle.

## Reading prerequisite

Labs 16–17 and strict canonical contiguity.

## Build

Start with shape `[2,3]`. Create no-copy reshape views with shapes `[3,2]`,
`[6]`, and `[1,6]`; record each canonical stride vector.

## Oracle

All compatible reshapes visit physical offsets `[0,1,2,3,4,5]` in logical
order and share references with the owner.

## Break / prove

Require `[4,2]` to fail with `ReshapeElementMismatch`. Transpose the source and
require reshape to fail with `NonContiguous`; v1 never silently copies.

## Extend

Reshape `[0,4]` to another zero-element shape and explain why no index is valid.

## Cleanup

Keep only passing checked examples.
