# Lab 19 — Copy a Non-Contiguous View

**Chapter:** 5. **Level:** BUILD.

## Prerequisites

Lab 17 and the view/copy allocation table.

## Build

Transpose a `[2,3]` tensor, then call `to_contiguous`. Compare the strided
view's logical order with the new owner's physical storage.

## Oracle

Both must read `[A,D,B,E,C,F]`; the copy must report shape `[3,2]`, strides
`[2,1]`, and independent element references.

## Break / prove

Mutate the copy through its exclusive canonical view. Prove the source owner is
unchanged. This distinguishes value equality from storage aliasing.

## Extend

Materialize a bounded row slice and account for copied bytes exactly.

## Cleanup

Restore the deterministic fixture values.
