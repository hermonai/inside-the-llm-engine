# mini-engine

This Rust workspace advances from ENGINE-0 through ENGINE-10. The current
milestone is **ENGINE-0**: a deterministic request-to-token runtime with no
external dependencies.

ENGINE-0 establishes these substitution seams:

- `Request` and runtime-owned `GenerationState`;
- `Model` as a candidate source;
- `Selector` as token-selection policy;
- `TokenSink` and ordered `StreamEvent` values;
- completed, cancelled, and failed terminal outcomes;
- named lifecycle timing and trace points.

It intentionally has no tokenizer, neural network, GGUF reader, network server,
GPU path, or production scheduler. Chapters 2–4 replace the fake model-facing
pieces without rewriting the request lifecycle.

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace 'What color is the sky?'
cargo test --workspace
```

The hand-computable oracle is
[`code/reference/engine-0-oracle.md`](../reference/engine-0-oracle.md), and the
guided exercise is [Lab 1](../../labs/lab-01-generate-one-token-manually.md).
