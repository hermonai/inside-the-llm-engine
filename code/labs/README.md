# Chapter labs

One lab per chapter (AUTHORING.md §10): the smallest runnable program that
shows the chapter's mechanism, with an oracle that checks every optimized path
before anything is timed. Each runs on a laptop CPU in about a minute.

Rust labs are crates in this workspace and may use `code/mini-engine` as a
library; Python labs use only the standard library. From this directory:

```bash
cargo test --workspace                       # every lab's oracle checks, small sizes
cargo run --release -p ch01-fake-model       # a lab at full size
python3 -m unittest discover -s chNN-<slug>  # a Python lab's checks
```

Each lab's README follows the same four steps: **predict** (write the number
down before running), **run**, **explain** (with the chapter's napkin math),
and **break** (a change that makes the mechanism fail or disappear).

| Chapter | Lab | Language |
| --- | --- | --- |
| 1 | [`ch01-fake-model`](ch01-fake-model/) — a weight matrix the size of a small model, decoded at rising batch sizes | Rust |
| 2 | [`ch02-count-a-model`](ch02-count-a-model/) — a GGUF header reader that prices one decode step, per operator, for any local model | Python |
| 3 | [`ch03-roofline`](ch03-roofline/) — measures the CPU's bandwidth and compute roofs and predicts a decode sweep from them | Rust |
| 4 | [`ch04-prefill-decode`](ch04-prefill-decode/) — the capstone model reads a prompt in one step and token by token, bit-identical, at very different costs | Rust |
| 5 | [`ch05-kv-cache`](ch05-kv-cache/) — generation with a KV cache against recomputation from scratch, bit for bit, and what one erased entry does | Rust |
| 6 | [`ch06-napkin`](ch06-napkin/) — the step-time model as a calculator for any model and machine; its tests reproduce the chapter's tables | Python |
