# Lab 7 — Prove the Context Limitation

Primary chapter: 3. Level: CHECK / BUILD / BREAK / EXTEND.

## Goal

Demonstrate experimentally that ENGINE-1 ignores every history position except
the final token.

## CHECK

Compare these token sequences:

```text
A = [I, like]
B = [Rust, <eos>, like]
```

Predict the hidden vector and logits for each. State why different lengths and
prefix identities cannot affect the result.

## BUILD

Run:

```sh
cd code/mini-engine
cargo test -p engine0 same_last_token_produces_same_logits_despite_different_history
```

Inspect `Model::forward`: it accepts the full borrowed slice but selects
`input.last()` before embedding lookup.

## BREAK

Change only the final token in sequence B to `Rust`. Confirm the result changes.
This control prevents the weak conclusion that the model ignores all input.

## EXTEND

Write two natural-language histories that end in the same token but should
plausibly predict different continuations. Explain the kind of context-dependent
representation Part II must eventually provide. Do not implement attention.

## Cleanup

Remove temporary test cases unless they establish a new boundary condition.
