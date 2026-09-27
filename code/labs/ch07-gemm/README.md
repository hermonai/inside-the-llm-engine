# Chapter 7 lab: tile a GEMM and watch it climb

Two experiments on the CPU, every product checked against a float64 sum on
sampled entries before it is timed.

**Part 1: the ladder.** C = A B for 512 x 512 f32 matrices, five ways that
perform the same multiply-adds in different orders: naive (i, j, k), reordered
(i, k, j), cache-blocked (64), a register tile of 4 x 16 accumulators, and the
register tile split by rows across threads.

**Part 2: skinny shapes.** A 4,096 x 4,096 f32 weight matrix (64 MiB) times N
vectors, N = 1 to 256, with the capstone engine's row-streaming decode kernel
and with the register-tiled GEMM, whose tile is 16 columns wide.

**Predict.** From Chapter 3's lab, what is your multiply-add peak, and what
fraction should the tiled kernel reach? Where should the two kernels of part 2
cross? (Chapter 7's third worked problem predicts about 11 on the M1.)

**Run.** From `code/labs`:

```bash
cargo run --release -p ch07-gemm -- --threads 4
cargo run --release -p ch07-gemm -- --size 1024 --threads 4
```

**Explain.** Each rung saves loads per multiply-add: contiguous rows, then
reuse from cache, then reuse from registers (64 multiply-adds per 20 loads),
then more cores. In part 2 the streaming kernel reads the weights once and is
memory-bound for a few vectors; the tiled kernel pads N up to 16 and costs the
same for one vector as for sixteen.

**Break.** Change `TR` and `TC` in `src/main.rs`: a 2 x 8 tile loses much of the
reuse, an 8 x 32 tile needs more accumulators than the registers hold. Try
`--size 1024` and see whether blocking starts to pay.
