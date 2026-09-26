# Chapter 3 lab: draw your own roofline

A roofline needs two measured ceilings and a workload to place between them.
This lab measures both on your CPU and then tests them on decode-like steps.

1. **Bandwidth roof.** Sum (read) and copy arrays of 256 MiB, far larger than a
   laptop's caches, with one to `--threads` threads. For contrast, one thread
   also sums with a single running total, which is limited by the adder's
   latency rather than by memory.
2. **Compute roofs.** A loop of independent multiply-adds shows what the
   arithmetic units can do. The capstone engine's matrix kernel, run on weights
   that stay in cache, shows what this book's code does, at every batch size.
3. **The sweep.** Decode steps multiply 256 MiB of f32 weights by B vectors,
   B = 1 to 256, after five untimed warm-up passes. For each step the lab prints
   its prediction, max(weight bytes / read roof, FLOPs / in-cache rate at B),
   beside the measurement.

**Oracle.** Every batch is checked before it is timed: a sample of rows is
recomputed one output at a time (bit for bit) and as a float64 sum (relative
1e-5 of the row's scale).

**Predict.** With 4-byte weights a step does B/2 FLOPs per byte of weights, so
the sweep should stop being free near B* = 2 x (kernel roof / read roof).
Write your guess down before running.

**Run.** From `code/labs`:

```bash
cargo run --release -p ch03-roofline -- --threads 4 --csv roofline.csv
cargo run --release -p ch03-roofline -- --threads 1
```

Options: `--threads T`, `--mib N` (array and weight size), `--csv PATH`.

**Explain.** Below B* the step time is set by the weight bytes and stays flat;
above it the kernel's arithmetic sets it and it grows with B. The kernel's own
rate is lower at large batches than at four vectors, because the activations
stop fitting in the first-level cache: the memory hierarchy has a roof per
level. Chapter 3 gives the M1's numbers.

**Break.** `--threads 1` makes one core's arithmetic, not memory, the limit
even at B = 1. `--mib 8` puts the weights in the second-level cache; the steps
become so short that the prediction misses costs neither roof contains.
