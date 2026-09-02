# mini-engine

This dependency-free Rust workspace advances from ENGINE-0 through ENGINE-10.
The current milestone is the complete **ENGINE-1**: the Chapter 2
text/token/byte boundary feeds a genuine numerical model, and Chapter 4 turns
its logits into an autoregressive generation loop.

ENGINE-1 establishes:

- `TokenId` and byte-oriented, fallible `Tokenizer` contracts;
- Chapter 2's independent byte oracle, teaching BPE, explicit special tokens,
  typed chat template, model contract, and strict UTF-8 stream framing;
- a model-bound four-token vocabulary: `<eos>`, `I`, `like`, and `Rust`;
- immutable row-major `f32` embedding `[V,D]`, projection `[V,D]`, and bias
  `[V]` parameters;
- a typed `Model::forward(&[TokenId]) -> ForwardPass` boundary;
- visible scalar `h = E[x]` and `z = W h + b` execution;
- typed finite raw `Logits` kept separate from sampling workspaces;
- a separate greedy argmax mode with lowest-token-ID tie breaking;
- stable softmax in `f64`, temperature, top-k in logit space, top-p in
  probability space, renormalization, and fixed-draw categorical selection;
- request-owned SplitMix64 state with a narrow, versioned seed-reproduction
  contract and no cryptographic claim;
- construction-time shape, parameter-count, finite-value, and
  tokenizer/model vocabulary validation;
- request-local hidden/logit activations and model-shared immutable weights;
- real token feedback through `model.forward` until EOS, `max_new_tokens`,
  cancellation, or failure;
- the completed/cancelled/failed exactly-once terminal invariant and opt-in
  trace points for raw tensors, processed probabilities, RNG draws, selected
  IDs, commit, bytes, and text.

The Chapter 3 fixture uses `V=4`, `D=3`, 28 parameters, and 112 bytes of `f32`
payload. Input `like` produces the independently verified vector
`[-0.7, 0.1, 0.4, 2.2]`. The old fake candidate table is not part of the
current source path; Git history preserves the ENGINE-0 milestone.

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace 'I like'
cargo run -p engine0 -- --trace --sample --temperature 1 --top-k 3 \
  --top-p .9 --seed 42 'I like'
cargo run -p engine0 -- tokenize lower
cargo run -p engine0 -- decode 259
cargo test --workspace
```

The generic `tokenize` and `decode` inspection commands continue to use the
Chapter 2 teaching BPE. The default generation path uses the paired ENGINE-1
four-token tokenizer, so its accepted prompt domain is intentionally tiny.

Independent references:

- [`chapter03_oracle.py`](../reference/python/chapter03_oracle.py) implements
  the equations separately in plain Python;
- [`chapter04_sampling_oracle.py`](../reference/python/chapter04_sampling_oracle.py)
  independently proves the sampling stages with artificial draws;
- [`chapter-02-tokenizer-oracles.md`](../reference/chapter-02-tokenizer-oracles.md)
  retains the tokenizer cases;
- [`engine-0-oracle.md`](../reference/engine-0-oracle.md) records the historical
  fake candidate milestone.

Guided work is in [Labs 1–15](../../docs/LABS.md). Chapter 5 will begin a small
checked tensor/view substrate for ENGINE-2. ENGINE-1 intentionally adds no
training, Transformer, general tensor framework, BLAS, accelerator, or GGUF
dependency.
