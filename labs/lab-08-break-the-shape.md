# Lab 8 — Break the Model Shape

Primary chapter: 3. Level: BREAK / PROVE.

## Goal

Treat dimensions and parameter counts as executable inference contracts.

## CHECK

For `V=4`, `D=3`, predict the required lengths of `E`, `W`, and `b`, the total
parameter count, and the `f32` byte count.

## BREAK

Construct one model for each malformed case:

- `V=0`;
- `D=0`;
- 11 embedding values instead of 12;
- 13 projection values instead of 12;
- three bias values instead of four;
- one `NaN` parameter;
- input `TokenId(4)`;
- a tokenizer reporting a different vocabulary size.

Run the relevant test group:

```sh
cd code/mini-engine
cargo test -p engine0 --test model
cargo test -p engine0 tokenizer_model_vocabulary_mismatch_is_rejected_before_execution
```

Every case must return a typed error. A panic, silent truncation, or ignored
extra parameter fails the lab.

## BUILD

Add one boundary case not listed above. Assert the exact error variant and the
shape values it reports.

## EXTEND

Explain which validation belongs at model load, runtime construction, and each
forward call. Identify checks that can be paid once rather than on every dot
product without weakening correctness.

## Cleanup

Keep generally useful boundary tests. Remove malformed fixture files; this lab
does not define a serialization format.
