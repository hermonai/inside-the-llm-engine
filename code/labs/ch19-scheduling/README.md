# Chapter 36 lab — Scheduling Under Pressure

## Predict
With six blocks and two three-block requests, predict whether both tenants can make
progress and whether physical allocation can exceed six.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
Admission, scheduling and preemption are separate methods. The invariant recomputes
physical ownership from the running set rather than trusting the `used` counter.

## Break
1. Allocate before the queue-bound check: early-rejection ownership test should fail.
2. Do not subtract victim blocks: capacity invariant fails.
3. Schedule only tenant 0: fairness test fails.
4. Increment service twice per tick: conservation test fails.

## Limits
This lab uses unique blocks and equal tenant weights. The chapter exercise adds shared
refcounts, deadlines, output uncertainty, swap/recompute cost and bounded cascades.
