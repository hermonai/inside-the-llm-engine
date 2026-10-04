# Chapter 39 lab — Speculative decoding contract

## Predict
For `p=[.5,.3,.2]` and `q=[.6,.1,.3]`, calculate accept mass and residual mass
before running the code. Then predict the state after drafting five positions from
frontier 80 and rejecting the fourth draft.

## Run
```bash
cargo fmt --check
cargo test
cargo run
```

## Explain
`exact_output_mass` enumerates the mathematical probability law; it does not use
random sampling. `wrong_output_mass` is a deliberate negative control. `Frontiers`
separates committed, drafted, evaluated and published positions.

## Break
- Replace the residual with `p`: `wrong_residual_is_detected_without_monte_carlo`
  demonstrates the distribution error.
- Leave `evaluated` at the draft frontier after `reconcile`: rollback test fails.
- Allow `publish_to` beyond `committed`: publication safety test fails.
- Add grammar state, mutate it for every draft, and omit rollback: the Chapter 37
  grammar transaction test you add should fail.

## Limits
This is a probability/state oracle, not a neural model or performance benchmark.
It does not model GPU kernels, target logits, tree attention or asynchronous device
retirement.

After three accepted proposals from frontier 80 and a correction at 84,
committed history ends at 84 but retained target KV ends at 83. The correction
has not yet been fed into the model. Publication cannot move backward; exposing
a committed token is not evidence that its KV has been materialized.
