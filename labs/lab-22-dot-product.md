# Lab 22 — Dot Product

**Chapter:** 6. **Level:** CHECK.

## Prerequisites

TensorView rank, shape, logical indexing, and `f32` accumulation.

## Build

Calculate `[1, 2, 3] · [4, 5, 6]` by expanding all three products. Then encode
the vectors as rank-1 `OwnedTensor` values and call `dot_reference`.

## Oracle

The hand result and Rust result are both `32`. Run
`python3 code/reference/python/chapter06_matmul_oracle.py` and the `linear`
integration tests as independent controls.

## Break / prove

Pass lengths 2 and 3, then pass a rank-2 left operand. Identify
`LengthMismatch` and `RankMismatch`; neither case may truncate, broadcast, or
panic. Also prove the dot product of two length-zero vectors is `0`.

## Extend

Construct a stride-2 vector view and predict which physical elements its three
logical indices read before executing it.

## Cleanup

Keep useful tests; remove ad hoc print statements.
