# Lab 17 — Transpose Without Copy

**Chapter:** 5. **Level:** BUILD.

## Prerequisites

Lab 16 and the distinction between shape and strides.

## Build

Construct a `[2,3]` owner over `[A,B,C,D,E,F]` and call `transpose`. Record
source metadata `[2,3] / [3,1]` and transposed metadata `[3,2] / [1,3]`.

## Oracle

Logical transpose order must be `[A,D,B,E,C,F]`, while corresponding source
and transpose elements have the same reference identity. No element allocation
occurs.

## Break / prove

Try transpose on rank 1 and require `ExpectedMatrix`. Pretend the transpose has
strides `[2,1]`; explain the finite but wrong logical result.

## Extend

Use `to_contiguous` and show that the new owner stores
`[A,D,B,E,C,F]` canonically.

## Cleanup

Remove any intentionally incorrect stride fixture.
