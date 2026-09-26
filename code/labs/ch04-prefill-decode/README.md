# Chapter 4 lab: one model, read and written

A prompt can be processed in one step (prefill) or one token at a time
(decode). Both must leave the same keys and values in the cache and give the
same next-token logits; they cost very different amounts.

This lab builds the capstone engine's 97-million-parameter model with seeded
random weights (388 MB of f32, so a decode step streams its weights from
memory) and runs prompts of 1 to 512 tokens both ways.

**Oracle.** Before timing, a 64-token prompt is processed both ways and the
logits and every cached key and value must agree bit for bit. The capstone's
kernels fix their summation order, so they can; production kernels usually
agree only to rounding (Chapter 38).

**Predict.** From Chapter 3's lab on the same machine, at what prompt length
should the prefill rate stop rising, and what is the largest prefill-to-decode
ratio you expect?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch04-prefill-decode -- --threads 4
cargo run --release -p ch04-prefill-decode -- --threads 1 --max 128
```

**Explain.** Decode costs one weight stream per token. Prefill costs one weight
stream per step until the arithmetic catches up, so its rate rises with the
prompt to the kernel's compute roof, then falls as the capstone's simple
attention grows with the square of the prompt.

**Break.** Make the dot product split its sum differently when more tokens are
in the batch, and watch the oracle fail in the last bits of the logits. Then
compare your GPU with llama.cpp: `llama-bench -p 1,2,4,8,16,32,64,128,256,512 -n 0`
and `llama-bench -p 0 -n 64 -d 0,1024,4096 -fa 0,1`.
