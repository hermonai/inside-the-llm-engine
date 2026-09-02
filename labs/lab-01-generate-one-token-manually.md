# Lab 1 — Generate One Token Manually

**Primary chapter:** Chapter 1, revisited with numerical logits in Chapters 3–4
**Prerequisite:** Rust toolchain and Chapter 1 through “Build ENGINE-0”  
**Artifact:** one predicted token, one captured trace, and one deliberate
failure explanation  
**Oracles:** historical [`engine-0-oracle.md`](../code/reference/engine-0-oracle.md)
and current [`chapter03_oracle.py`](../code/reference/python/chapter03_oracle.py)
**Measurement:** pedagogical lifecycle timestamps only; this is not a benchmark  
**Cleanup:** `cargo clean` is optional; do not commit `target/`

## CHECK — predict before executing

Do not run the code yet. Read the two candidate tables in the historical
oracle, then the Chapter 3 full-logit oracle. Git history preserves the exact
ENGINE-0 source; current `main` executes ENGINE-1.

1. Apply the greedy rule: choose the greatest score.
2. Write down the first token ID and text.
3. Decide whether `<eos>` should appear as a text token.
4. Write the expected trace order from admission through terminal.

The historical first token is derivable with integer comparison alone. The
current first token is derivable from four hand-computable dot products. If you
need Transformer mathematics, the exercise boundary has failed.

## BUILD — observe the lifecycle

From the repository root:

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace 'I like'
```

Compare the semantic events with your prediction. Microsecond values will
differ across runs. Identify these intervals:

```text
queue delay       = execution_started - admitted
time to first     = first_token_emitted - admitted
ready/emit gap    = first_token_emitted - first_token_ready
request latency   = terminal - submitted_to_runtime
```

ENGINE-0 is synchronous, so its queue interval is bookkeeping rather than a
production queue measurement. The point is to name endpoints before timing.

Run the correctness suite:

```sh
cargo test --workspace
```

## BREAK — force a non-success terminal

Try cancellation before the first model step:

```sh
cargo run -p engine0 -- --trace --cancel-at 0 'I like'
```

Then inject a model failure after the first emitted token:

```sh
cargo run -p engine0 -- --trace --fail-at 1 'I like'
```

For each run, answer:

- How many terminal events appear?
- Are any tokens emitted after terminal?
- Is failure distinguishable from successful completion?
- Which object owns the decision to stop?

The injected failure command exits nonzero. That is expected evidence, not a
lab failure.

## EXTEND — change policy without changing lifecycle

Change one output-projection weight in `TinyLanguageModel`'s Chapter 3 fixture.
Before running, update the complete expected logit vector. Do not change
`Runtime::generate`, `StreamEvent`, or the terminal state machine.

Then restore the source and add a test for tied highest logits. ENGINE-1's
temporary greedy selector keeps the first vocabulary position on a tie. Explain
why a deterministic tie rule belongs to selection policy rather than request
lifecycle.

## Completion gate

Lab 1 is complete when:

- your manual first token matches the executable;
- you can name every trace transition and its owner;
- completed, cancelled, and failed runs each have exactly one terminal;
- you can explain why timestamps are observations but not benchmark claims;
- `cargo test --workspace` passes after restoring or completing your changes.
