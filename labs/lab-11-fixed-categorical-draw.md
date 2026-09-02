# Lab 11 — Select With a Fixed Random Draw

Chapter: 4. Artifact: cumulative intervals and one selected token. Oracle:
`categorical_select` in independent Python and Rust implementations.

## CHECK

For probabilities `[0.20,0.30,0.50]`, write the half-open interval owned by
each token. Determine the result for `r=0.63` before running code.

## BUILD

Run:

```sh
python3 code/reference/python/chapter04_sampling_oracle.py \
  --logits 0.2 0.3 0.5 --draw 0.63
cargo test -p engine0 categorical_selection_uses_half_open_cumulative_intervals
```

The Python command softmaxes logits; for the exact probability exercise, call
its `categorical_select([.2,.3,.5],.63)` from a Python shell.

## BREAK

Try draws `-0.1`, `1.0`, and NaN, then probabilities that are negative or do
not sum to one. Every case must fail rather than inventing a token.

## EXTEND

Test exact boundaries `0.2` and `0.5` and the largest representable value below
one. Explain the final-positive-candidate rounding fallback.
