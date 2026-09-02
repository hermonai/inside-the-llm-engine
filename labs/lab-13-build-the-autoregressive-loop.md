# Lab 13 — Trace the Autoregressive Loop

Chapter: 4. Artifact: a complete token/history/logit/terminal trace.

## CHECK

Start with prompt `I like`, token IDs `[1,2]`. Chapter 3 proved that final token
`like` produces logits `[-0.7,0.1,0.4,2.2]`. Predict the greedy token and the
new history. Then compute the next step from final token `Rust`.

## BUILD

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace --max-tokens 8 'I like'
```

The real model must produce:

```text
[I, like] -> Rust -> [I, like, Rust] -> <eos> -> completed once
```

Only `Rust` is user-visible. EOS is neither decoded nor emitted as text.

## BREAK

Run with `--max-tokens 1`, `--cancel-at 1`, and `--fail-at 1`. Verify the final
event is exactly one budget completion, cancellation, or failure respectively,
with no later token.

## EXTEND

Draw the commit point. Inject a tokenizer/UTF-8 failure after token commit and
explain why the result contains the committed token plus one failed terminal.
