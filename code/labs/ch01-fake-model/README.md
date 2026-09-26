# Chapter 1 lab: a fake model with real costs

A decode step multiplies every weight matrix by one activation vector per
sequence. This lab keeps one matrix the size of a small model's weights
(512 MiB of f32 by default, 32,768 rows of 4,096) and runs decode steps for
batch sizes 1 to 64. The weights are random, so the outputs mean nothing; the
bytes are real, so the costs are.

**Oracle.** Before timing a batch, a sample of 64 rows is recomputed one
output at a time (must agree bit for bit) and as a float64 sum (must agree to
1e-5 of the row's scale).

**Predict.** The program prints the bytes one step reads. Divide your
machine's memory bandwidth by them: that is the batch-one rate. Then guess the
batch size at which the step time starts to grow.

**Run.**

```bash
cargo run --release -p ch01-fake-model -- --threads 4
cargo run --release -p ch01-fake-model -- --threads 1
```

Options: `--mib N` (weight size), `--threads T`, `--steps K` (best of K).

**Explain.** While the step reads memory faster than the cores can multiply,
the step time is flat and tokens per second rise with the batch; once the
arithmetic takes longer than the read, the step time grows with the batch and
tokens per second stop rising. Chapter 1 gives the M1's numbers; Chapter 3
turns them into the roofline.

**Break.** `--threads 1` makes one core the limit even at batch 1. `--mib 8`
fits the matrix in cache and removes the memory regime altogether.
