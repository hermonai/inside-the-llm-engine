# Chapter 8 lab: a tiled attention checked against a naive oracle

Causal attention for one head of width 64, three ways, every result checked
against float64 attention on eight sampled rows (the last row, which attends to
every key, always among them) before it is timed:

- **naive**: writes all S x S scores, softmaxes each row, multiplies by V;
- **online**: one pass over the keys per query row, keeping a running maximum,
  denominator and output (the online softmax);
- **tiled**: blocks of 64 queries against blocks of 64 keys, each block's
  scores held in a 64 x 64 buffer and every row's running state updated once
  per block, as FlashAttention does.

The program prints each version's time and temporary memory for S = 256 to
`--max` (default 8,192), then two demonstrations at S = 2,048: FlashAttention-4's
conditional rescaling (keep the old maximum unless a block raises it by more
than `--threshold`, default ln 256), on ordinary keys and on "ramp" keys whose
scores grow along the sequence; and, with `--forget-rescale`, the tiled kernel
with the rescale removed.

**Predict.** How much memory will the naive version ask for at 16,384 tokens?
On one CPU core, will the naive or the tiled version be faster? (Chapter 8's
napkin math says neither: a CPU core's ridge is far below attention's
arithmetic intensity.)

**Run.** From `code/labs`:

```bash
cargo run --release -p ch08-flashattention -- --forget-rescale
cargo run --release -p ch08-flashattention -- --max 16384 --threshold 0
```

**Explain.** The naive version's temporaries quadruple with every doubling of
S; the others stay constant. The online version is the slowest because it
rescales its output after every key rather than every block. The tiled kernel
rescales a row only when a block holds a new record score, about ln n times for
a row that meets n blocks (Chapter 8's second worked problem).

**Break.** `--forget-rescale` gives an error of about 0.7 of the output's scale
at 2,048 tokens, yet is exactly right for any sequence of 64 tokens or fewer.
Try `--threshold 100` and see that it is still exact on these inputs; then
construct scores that make it overflow.
