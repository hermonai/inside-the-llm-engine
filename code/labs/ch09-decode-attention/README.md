# Chapter 9 lab: split-K decode attention with a log-sum-exp merge

One decode step of attention for one layer: a single query per head against a
cache of S tokens, in a grouped-query layout (by default 16 query heads sharing
2 key/value heads of width 128, the shape of Qwen2.5-Coder 3B, with a 32-bit
cache). The work is cut into tasks and run on a pool of threads; every task
returns a partial (maximum, denominator, unnormalized output), written into a
slot indexed by head and chunk, and each head's partials are merged in chunk
order. Four plans:

- `heads`: one task per query head over the whole cache;
- `packed`: one task per key/value head, its query heads sharing one pass;
- `split`: one task per (query head, chunk of `--chunk` tokens);
- `packed+split`: one task per (key/value head, chunk).

Every output is checked against float64 attention before it is timed. The
program prints the best of `--reps` times, the effective bandwidth (cache bytes
counted once), each plan's error, whether its bits are identical on every
thread count, and how many outputs match the `heads` plan to the bit.

**Predict.** How many threads can `packed` use at the default shape? Will
packing help one CPU core, whose loop computes rather than waits on memory?
Will the split plans agree with `heads` to the bit?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch09-decode-attention
cargo run --release -p ch09-decode-attention -- --contexts 256,32768 --threads 1,4 --chunk 16
cargo run --release -p ch09-decode-attention -- --q-heads 8 --kv-heads 8
```

**Explain.** Splitting gives all cores work when the heads alone do not; the
`heads` plan's repeated reads of a group's keys are served by the shared cache
when threads work on the same group at once; fixed chunks merged in index order
give the same bits on any number of threads, and a chunked sum is more accurate
than one long running sum.

**Break.** Tiny chunks (`--chunk 16`) drown the gain in task overhead; short
contexts (`--contexts 256`) cost more to start threads for than to compute;
multi-head attention (`--kv-heads` equal to `--q-heads`) leaves packing nothing
to share.
