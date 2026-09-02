# mini-engine

This Rust workspace advances from ENGINE-0 through ENGINE-10. The current
milestone remains **ENGINE-0**, evolved through Chapter 2: request lifecycle and
text/token/byte boundaries are executable, while the candidate model is still
deliberately fake.

ENGINE-0 now establishes:

- `TokenId` and a byte-oriented, fallible `Tokenizer` contract;
- an independent one-byte/one-ID oracle;
- a deterministic teaching BPE with all-byte fallback and fixed ranked merges;
- ordinary encoding separated from explicit special-token insertion;
- typed chat messages, `TinyChatTemplate`, and a model/tokenizer/template
  identity contract;
- token IDs in `Request` and runtime-owned generated identity history;
- token events separated from valid UTF-8 text events;
- a strict per-request UTF-8 framer for partial, malformed, and incomplete
  token-byte pieces;
- the Chapter 1 completed/cancelled/failed terminal invariant and named trace
  points, now including encode/decode/buffer stages.

It intentionally has no embeddings, learned parameters, hidden vectors,
projection, logits, Transformer, GGUF reader, production server, accelerator,
or scheduler. `DemoModel` still returns a hand-computable candidate table whose
integer scores are not logits. Chapter 3 replaces that model rather than
wrapping it.

```sh
cd code/mini-engine
cargo run -p engine0 -- tokenize lower
cargo run -p engine0 -- decode 259
cargo run -p engine0 -- --trace 'What color is the sky?'
cargo test --workspace
```

The Chapter 2 fixtures live under [`fixtures/tokenizer`](fixtures/tokenizer/).
Independent oracles are
[`engine-0-oracle.md`](../reference/engine-0-oracle.md) and
[`chapter-02-tokenizer-oracles.md`](../reference/chapter-02-tokenizer-oracles.md).
Guided work is in [Labs 1–4](../../docs/LABS.md).

The workspace has no external Rust dependencies. The real-tokenizer comparison
was run in a temporary Python environment and is recorded as text under
[`research/part-01/tokenizer-comparison.md`](../../research/part-01/tokenizer-comparison.md).
