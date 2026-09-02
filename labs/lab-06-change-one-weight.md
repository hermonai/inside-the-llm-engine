# Lab 6 — Change One Projection Weight

Primary chapter: 3. Level: CHECK / BUILD / BREAK / EXTEND.

## Goal

Connect one stored parameter to exactly one output logit's arithmetic.

## CHECK

For the `like` hidden vector `[1.0, -0.5, 2.0]`, change only `W[3,0]` from
`1.0` to `1.5`. Before running anything, predict:

- which logit changes;
- its direction and exact delta;
- which logits remain unchanged;
- whether the embedding changes.

## BUILD

Run the focused Rust test:

```sh
cd code/mini-engine
cargo test -p engine0 changing_one_projection_weight_changes_only_its_output_row
```

Inspect the fixture in `tests/model.rs`. The first three logits must remain
`[-0.7, 0.1, 0.4]`; the fourth must change from `2.2` to `2.7`.

## BREAK

Change `W[3,1]` instead. A negative hidden component reverses the intuitive
direction: increasing this weight lowers the fourth logit. Calculate the delta
before testing it.

## EXTEND

Change one embedding value rather than one projection weight. Predict which
logits can change and explain why this intervention fans out across output
rows, unlike changing one row-specific projection weight.

## Oracle and cleanup

The oracle is direct scalar arithmetic, not the selected token. Restore the
built-in fixture after the experiment.
