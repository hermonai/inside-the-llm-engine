# Lab 26 — Blocked GEMM

**Chapter:** 6. **Level:** BUILD.

## Prerequisites

Lab 25 and the idea of a bounded working set.

## Build

Multiply `[5,7] × [7,3]` with block dimensions `[4,4,2]`. Mark the full and
tail ranges on M, K, and N, then inspect `matmul_blocked`'s clamped endpoints.

## Oracle

Compare every element with `matmul_reference` using
`1e-5 + 1e-5 |reference|`. Run the deterministic shape grid, which includes
zero, smaller-than-tile, exact-tile, and tail cases.

## Break / prove

Replace one clamped endpoint with an unconditional tile endpoint in a
disposable branch. Demonstrate why the `[5,7] × [7,3]` fixture catches the
overrun or missing-tail defect.

## Extend

Try the bounded tile sweep in the release harness. Explain the observed winner
as specific to this shape, compiler, and CPU.

## Cleanup

Restore checked endpoints and remove the deliberately broken branch.
