# Chapter 12 lab: quantize a real KV cache and watch attention drift

Layer 0 of Llama 3.2 3B computes its queries, keys and values from the token
embeddings alone, so this lab builds them from the model file (with Chapter
10's lab as a library) for the 3,217 tokens of the book's `AUTHORING.md`: 24
query heads sharing 8 key/value heads of width 128, with RoPE applied as
llama.cpp applies it for this model (adjacent channel pairs, base 500,000, the
file's frequency factors). The last 256 tokens attend causally to everything
before them.

The keys and values are stored in twelve formats — 16-bit, FP8 E4M3 with scale
1 and with a calibrated scale (Chapter 11's lab), llama.cpp's `q8_0` and `q4_0`
per-token blocks, `q4_0` with llama.cpp's Hadamard rotation, per-token and
per-channel (KIVI-style) asymmetric 4- and 2-bit formats, and per-channel keys
quantized before RoPE — and each format's attention is compared with full
precision: the output error, how often the most-attended token of a (query,
head) pair changes, and the KL divergence of the attention distributions. The
lab also quantizes keys and values separately, and sweeps the context from 256
tokens to the whole text.

**Predict.** Which half of the cache is fragile? Does the damage grow with
context, and by which measure?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch12-kv-quant -- --gguf ~/.ollama/models/blobs/sha256-dde5aa3f...
```

**Explain.** Keys have outlier channels, so per-token scales starve their quiet
neighbours; per-channel scales do not. Value errors never change which tokens a
query attends to. A longer context puts more keys near the top, so the winner
changes more often even when the distribution moves no further.

**Break.** Use per-channel groups of 256 tokens instead of 32; run the queries
over only the first 512 tokens and see whether a short context hides the damage.
