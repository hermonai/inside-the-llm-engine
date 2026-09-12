# Labs 49-58 - RoPE position workbench

Prerequisites: Chapters 5-8, the scalar Rust workspace, and Python 3.10+.
All inputs are tiny synthetic fixtures; no model download or GPU is required.
Keep intentional bugs on a disposable local branch and restore only your own
edits after each break. Do not commit a failing candidate as the reference.

Run commands from the repository root unless the command changes directory:

```sh
cd code/mini-engine
cargo test --test rope
cargo run --quiet --example chapter09_rope_trace
```

From the repository root, run `python3 scripts/check-rope-visual-parity.py`.
The input-only fixture is `code/reference/fixtures/chapter09-rope.json`;
the independent oracle is `code/reference/python/chapter09_rope_oracle.py`.
Record your outputs and explanations in your own lab notebook, not in the
canonical fixture. Analytical counts below are not timing measurements.

## Lab 49 - Build the two-dimensional map

**CHECK.** Use [1,2], frequency 1 and positions 0/1. Derive the images of both
basis vectors and combine them. Expected position-one output is approximately
[-1.142640,1.922076]; squared norm remains 5 in real arithmetic.

**Artifact:** a typeset derivation, coordinate sketch, and full numeric pair.
**BREAK:** substitute degrees for radians and record the discrepancy.
**Cleanup:** restore radian evaluation and pass the hand-computed test.

## Lab 50 - Use two clocks, not one

**CHECK.** With R=4 and base 100, derive frequencies [1,0.1] and rotate
[1,2,3,4]. Compare every coordinate with the complex oracle.

**Artifact:** frequency/angle/output table at positions 0,1,7.
**BREAK:** use frequency 1 for both planes; show that norm tests still pass.
**Cleanup:** restore the plane-index exponent and pass coordinate parity.

## Lab 51 - Prove relative position and its sign

**BUILD.** Use query [1,2,3,4], key [2,-1,1,3], positions 2 and 5.
Calculate both rotated-dot and key-minus-query routes. Expect approximately
13.558047, with tolerance 1e-5 + 1e-5 times the oracle magnitude.

**Artifact:** a transpose-by-transpose derivation and both raw outputs.
**BREAK:** reverse the relative-position sign; explain the changed score.
**Cleanup:** restore the sign and run the relative-inner-product test.

## Lab 52 - Reorder the coordinate convention

**BUILD.** Map adjacent [a0,b0,a1,b1] to split-half [a0,a1,b0,b1]. Rotate both
representations and prove equivalence after the same output permutation.

**Artifact:** explicit input/output permutation and four checked coordinates.
**BREAK:** change pairing without permuting input. Demonstrate unchanged norms
but wrong values. **Cleanup:** restore the paired operator/data convention.

## Lab 53 - Preserve the partial tail

**CHECK.** Use head width 5, rotary width 4, and a negative-zero final value.
Prove that the tail keeps its F32 bits. Explain why frequencies divide by R.

**Artifact:** input/output bit patterns and named failures for R=0,3,6.
**BREAK:** silently truncate odd R or use head width in the exponent.
**Cleanup:** restore checked configuration and the documented prefix policy.

## Lab 54 - Make a late failure atomic

**BUILD.** Put two F32 maximum values in the final pair of the second head;
use base 100 and position 7. Expect NonFiniteOutput at head 1, pair 1.

**Artifact:** complete before/after bit snapshots, the named error, and the
allocation pointer. **BREAK:** remove arithmetic preflight so earlier pairs
are written before the late error. **Cleanup:** restore preflight and verify
all bits, not only the failed pair. Allocation aborts remain outside this test.

## Lab 55 - Separate logical coordinates from storage

**BUILD.** Create an offset view with strides [8,2] and a broadcast view with
strides [0,1]. Compare fresh canonical results with equal contiguous values.

**Artifact:** logical-to-physical offset table and all output coordinates.
**BREAK:** iterate raw storage as if every view were contiguous.
**Cleanup:** restore validated logical indexing; check that inputs remain unchanged.

## Lab 56 - Identify the precision boundary

**CHECK.** Convert 2^24 and 2^24+1 to F32 and F64. Explain which conversion
loses the distinction. Derive the chord-length phase sensitivity bound.

**Artifact:** actual conversion results and a distinction between conversion
accuracy, phase accuracy and model quality. **BREAK:** claim that finite output
at 2^53 proves long-context quality; name the missing evidence.
**Cleanup:** retain only the narrower tested numerical claims.

## Lab 57 - Follow the composed token

**BUILD.** Run the embedding/RMSNorm/QKV/RoPE example at position seven and
compare every Q/K/V coordinate with the independent oracle. V must match the
pre-RoPE projection; Q and K must change from the raw composed outputs.

**Artifact:** full trace with maximum error per owner and position metadata.
**BREAK:** rotate V, or rotate K twice. Record the first divergence.
**Cleanup:** restore exactly one Q/K rotation and pass the parity gate.

## Lab 58 - Design a measurable next candidate

**EXTEND.** Specify a sine/cosine table for T=32768, R=128, F32 coefficients.
The coefficient payload is 16 MiB. State its configuration key, precision,
sharing scope, table-build cost, warm lookup boundary, and reseeding policy
if proposing recurrence instead.

**Artifact:** an experiment protocol, not invented timing results. Include
hardware/build metadata, positions, head geometry, input layout, preflight and
allocation policies, warmup/repetitions, accuracy gates, and raw-result format.
**BREAK:** compare a warm table with a compute path that includes unrelated
model loading. **Cleanup:** align boundaries and label all costs analytical
until the candidate is implemented and measured.

## Completion gate

Run the focused Rust suite, full workspace tests, and the independent parity
script after removing all deliberate bugs. Keep the standard RoPE reference
unchanged when exploring scaling or optimization candidates. Chapter 10 starts
with these positioned Q/K owners and unchanged V; attention remains future work.
