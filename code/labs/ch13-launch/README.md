# Chapter 13 lab: what a launch costs, what a graph saves, what fusion saves

A worker thread stands in for the GPU. It owns a buffer and runs the commands
the host sends through a queue, strictly in order, as a CUDA stream or a Metal
command queue does; the host never touches the buffer.

**Part 1** applies a chain of `--ops` dependent additions, x += c_k, in four
ways:

- `sync`: send one addition, wait for it, send the next --- a launch followed
  by a synchronization, every time;
- `stream`: send them all, then wait once --- asynchronous launches;
- `graph`: the chain recorded once, before timing, and sent as one command ---
  a CUDA graph, or one Metal command buffer;
- `fused`: one command whose kernel applies every addition to a tile of 64
  elements while they sit in registers.

Each method must reproduce the host oracle's bits exactly, since each performs
the same float additions in the same order, before it is timed. The table gives
microseconds per addition (median of `--reps`) at several buffer sizes.

**Part 2** computes y = rms_norm(x) * w + b --- three graph nodes that
llama.cpp's Metal backend fuses into one kernel --- over 4,096 rows of 3,072 as
three passes or one, checks both against float64 and against each other bit for
bit, and reports bandwidth by a byte model that counts one full read and one
full write per pass.

**Part 3** derives how much work padding wastes when every batch runs at the
next captured graph size, for vLLM's default capture sizes (commit `bcdacfc`)
and for powers of two.

**Predict.** How many times more does a synchronous round trip cost than
queuing a command? At what buffer size does the per-command cost vanish into
the work? If every pass of Part 2 is limited by memory, how much faster is one
pass than three? With 51 captured sizes, what fraction of the work is padding?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch13-launch
cargo run --release -p ch13-launch -- --ops 100 --sizes 64,1024,16384 --reps 15
cargo run --release -p ch13-launch -- --tokens 1024 --threads 1,2,4,8
```

**GPU extension (macOS, optional).** `metal/launch.m` runs the same chain on
the Apple GPU five ways: `sync` (a command buffer per addition, each waited
for), `queued` (a command buffer per addition, one wait), `batched` (all
additions as dispatches in one command buffer), `barriers` (the same in a
concurrent encoder with a memory barrier between dependent dispatches, as
llama.cpp encodes a token's graph) and `fused` (one dispatch). The shaders are
compiled with safe math, every result is compared bit for bit with the CPU, and
the command buffers' GPU timestamps separate time on the GPU from time spent
getting work to it:

```bash
cd code/labs/ch13-launch
xcrun clang -O2 -fobjc-arc -framework Foundation -framework Metal metal/launch.m -o ../target/ch13-metal
../target/ch13-metal
../target/ch13-metal --ops 200 --sizes 256,4096 --reps 9
```

**Explain.** A round trip pays for waking the worker and waking the host, each
time; a queued command pays only for passing a message; a recorded graph pays
for one message whatever its length; a fused kernel also stops writing the
buffer out and reading it back between operations. Once each operation does
enough work, the fixed costs disappear into it --- and only fusion still pays,
because it removes memory traffic rather than overhead. Padding costs little
when the captured sizes are dense and a lot when they double.

**Break.** Replace `TILE` with 8 and watch the fused kernel become slower than
the graph: its additions form too few independent chains to keep the adders
busy, and fusion has spent latency to save bandwidth. Run Part 2 with
`--tokens 64`, so that x and y fit in cache: the bandwidth figures collapse,
because each pass now spends most of its time starting and joining its threads
--- a launch cost of its own --- and one pass still beats three, by saving two
launches rather than two trips through memory. On the GPU, run `--sizes 256`:
the times hardly change from 1,024 elements, because every method is overhead.
