# Chapter 41 lab — When speculation loses

## Predict
For each trace row, compute `(committed * baseline_ms) / (draft + verify + overhead)`
and decide whether speculation should remain optional under KV/SLO pressure.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
`oracle_speedup` is the independent arithmetic oracle. The controller score adds
resource/SLO policy. The separation prevents a policy implementation from becoming its
own correctness oracle.

## Break
Increase verification time while keeping committed tokens fixed: acceptance is unchanged
but speedup falls. Raise KV pressure above 0.90: optional speculation is disabled.
Reduce hysteresis margin to zero and feed alternating noisy arm scores to observe churn.

## Limits
This is a deterministic control/mechanism lab, not a GPU benchmark. It does not model
real kernel nonlinearities, arrivals, full paged-KV allocation, or a production
TurboSpec implementation.
