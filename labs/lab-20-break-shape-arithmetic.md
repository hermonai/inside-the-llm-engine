# Lab 20 — Break Shape Arithmetic

**Chapter:** 5. **Level:** BREAK.

## Prerequisites

Checked products, storage extent, and typed tensor errors.

## Build

Call size helpers with `[usize::MAX,2]`; do not allocate. Construct shape
`[2,3]` with five values. Construct a `[2,2]` view with strides `[100,1]` over
four values.

## Oracle

Require `ShapeOverflow`, `StorageLengthMismatch`, and `InvalidViewExtent`,
respectively. Run the 108-test-or-later workspace suite.

## Break / prove

Replace one checked operation locally with wrapping arithmetic and identify
which invariant disappears. Do not commit the unsafe variant.

## Extend

Exercise base-offset and extent addition near `usize::MAX` and require
`OffsetOverflow`.

## Cleanup

Revert the deliberately wrapping implementation.
