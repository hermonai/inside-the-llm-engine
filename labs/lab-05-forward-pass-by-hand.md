# Lab 5 — Calculate an ENGINE-1 Forward Pass by Hand

Primary chapter: 3. Level: CHECK / BUILD.

## Goal

Calculate every activation and logit for input token `like`, then compare the
complete vector with both independent implementations.

## Prerequisites and artifact

Read Chapter 3 through the hand calculation. Record one worksheet containing
the selected embedding row, four dot products, four bias additions, the final
logit vector, and the argmax only as a closing check.

## CHECK

Using `V=4`, `D=3`, and the parameters in
`code/reference/python/chapter03_oracle.py`, calculate:

```text
h = E[2]
z_i = b_i + sum_j W[i,j]h_j, for i in 0..4
```

Predict the full vector before running code. A selected token alone is not an
acceptable oracle.

## BUILD

Run:

```sh
python3 code/reference/python/chapter03_oracle.py
cd code/mini-engine
cargo test -p engine0 full_logits_match_independent_hand_oracle -- --exact
cargo run -p engine0 -- --trace 'I like'
```

The Python oracle and Rust test must report `[-0.7, 0.1, 0.4, 2.2]` within the
documented tolerance. The trace must show the same embedding and logits before
selection.

## BREAK

Swap two projection rows without changing the biases. Explain why the same
four numbers can now attach to different vocabulary identities. Confirm that a
correct-looking maximum is insufficient if row-to-token ownership is wrong.

## EXTEND

Choose another valid input token. Calculate its hidden vector and full logits,
add the expected vector to a new test, and state which parameter bytes the
forward pass reads.

## Cleanup

Revert experimental parameter changes. Keep only a new named test if it adds a
distinct invariant.
