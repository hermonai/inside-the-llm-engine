# Chapter 2 lab: count a real model from its file

A model file's header says everything a decode step will cost, before a single
weight is loaded. `count_model.py` reads a GGUF header (metadata and tensor
descriptions only), groups the tensors by the operator that uses them, and
prints one decode step's parameters, FLOPs and bytes read, at the context
length and batch size you choose. For a mixture of experts it separates the
parameters a token uses from the ones it only stores, and estimates the share
of experts a batch touches.

**Oracle.** Before pricing anything, the program recomputes every layer's
attention and FFN parameters from the metadata's shape keys with Chapter 2's
equations (params-attn and params-ffn, or the expert count and router for a
mixture) and stops if any layer's tensors disagree.

`worked_numbers.py` computes every small example the chapter works by hand
(RMSNorm, one attention head, rotary embedding, SwiGLU, a router, temperature),
the way the capstone engine's kernels do.

**Predict.** For each model, write down its parameter count and guess whether
its vocabulary tables are tied. Then guess the share of its bytes one token
reads, and its KV cache per token.

**Run.** Ollama prints a model's weights path on the `FROM` line of
`ollama show <model> --modelfile`. From the repository root:

```bash
python3 code/labs/ch02-count-a-model/count_model.py <path> --context 4096 --batch 16
python3 code/labs/ch02-count-a-model/count_model.py <path> --dump rope_freqs.weight
python3 code/labs/ch02-count-a-model/worked_numbers.py
python3 -m unittest discover -s code/labs/ch02-count-a-model
```

The tests write tiny GGUF files, so they need no model.

**Explain.** Tied models read every byte once per token (the table is the
output projection); untied ones skip all but one row of the input table. A
mixture of experts reads its active experts at batch one, and nearly all of
them once the batch is large enough. Attention over the cache is the only line
that grows with `--context`.

**Break.** Run it on Gemma 4 E4B (`gemma4` in Ollama). The oracle stops at
layer 0: its sliding-window layers use 256-wide heads, and the metadata's
`key_length` describes only the global layers. Extending the oracle to two
layer shapes is Chapter 2's implementation exercise.
