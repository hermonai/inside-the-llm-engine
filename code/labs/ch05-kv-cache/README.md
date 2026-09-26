# Chapter 5 lab: a cache you can prove correct

Generation with a KV cache must be invisible: it must produce exactly what the
model would produce by recomputing every earlier token at every step. This lab
generates 128 tokens from a 32-token prompt with a small model built from the
capstone engine's code (4 layers, width 256, 3.1 million parameters, seeded
random weights) two ways, and compares them.

**Oracle.** The recomputation is the capstone's oracle, which shares only the
arithmetic kernels with the engine. At every step the cached path must choose
the same token and produce the same logits, bit for bit, before any time is
printed.

**Predict.** Count the token positions each way projects through the model
(159 against 12,224 here) and predict the ratio of the total times.

**Run.** From `code/labs`:

```bash
cargo run --release -p ch05-kv-cache -- --evict 5 --csv steps.csv
```

Options: `--new N` (tokens to generate), `--evict P` (zero prompt position P's
cached keys and values after the prefill), `--csv PATH` (per-step times).

**Explain.** The cached cost per token is nearly flat --- it grows only by one
more cached token to read each step --- while the recomputation's cost grows in
a straight line with the position, so the total ratio keeps growing with the
reply's length.

**Break.** `--evict P` makes the first decode step's logits differ from the
oracle's; when the chosen token first differs depends on which position was
erased (step 10 for position 5, step 2 for position 30, step 45 for position 0
on the author's run). A damaged cache produces plausible, wrong output.
