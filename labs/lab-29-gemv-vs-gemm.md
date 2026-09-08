# Lab 29 — GEMV Versus GEMM

**Chapter:** 6. **Level:** EXTEND.

## Prerequisites

Labs 25–28 and `docs/BENCHMARK_POLICY.md`.

## Build

Hold one `[512,512]` weight matrix fixed. Measure multiplication by one vector,
then right-hand matrices with 8 and 64 columns. Record elapsed time, effective
GFLOP/s, and the ideal compulsory-traffic arithmetic intensity model.

## Oracle

Run the release Chapter 6 harness and preserve its checksums, environment,
commit, repetitions, and medians in the benchmark record. Correctness gates
must pass before timing.

## Break / prove

Treat ideal bytes as measured traffic or compare only total latency across
different `N`; explain why each produces a misleading conclusion. Inspect the
recorded `N=8` negative result rather than hiding it.

## Extend

Vary a bounded set of rectangular shapes. Form a hypothesis about weight reuse
and kernel overhead, then distinguish the model from observations.

## Visual checkpoint — Chapter 6 regeneration

Reproduce chart values from the historical CSV: GFLOP/s 2.739,1.273,5.397 for N=1,8,64. Separate recorded time, derived throughput, modeled ideal intensity, and unmeasured memory traffic. Retain the N=8 loss.

Use the [shared fixture](../code/reference/fixtures/chapter06-visual.json) and
run `python3 scripts/check-linear-visual-parity.py` from the repository root.
Report values, shapes, address units and the evidence boundary, not only PASS.

## Cleanup

Do not commit machine-scale raw artifacts. Keep the small textual record and
reproducer in Git.
