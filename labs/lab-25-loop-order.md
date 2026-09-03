# Lab 25 — Loop Order

**Chapter:** 6. **Level:** BUILD.

## Prerequisites

Canonical row-major offsets and Lab 24.

## Build

Implement or inspect scalar `i,j,k` and `i,k,j` products. For a tiny `[2,3] ×
[3,2]` case, record the flat offsets visited in `B` and `C` by each innermost
loop. Verify both outputs before discussing speed.

## Oracle

Both traversals must match `matmul_reference` within Chapter 6 tolerance. The
offset sequences must show stride `N` through a `B` column for `i,j,k`, versus
stride 1 across a `B` row and `C` row for `i,k,j`.

## Break / prove

Deliberately use `k*K+j` for the right offset. Choose `K != N` so the error is
observable as a wrong value or checked bound, not masked by a square matrix.

## Extend

Run the release loop-order benchmark. Report the machine, commit, repetitions,
median, and correctness gate; do not turn one host's ratio into a universal
claim.

## Cleanup

Remove temporary offset logging if it would contaminate timing.
