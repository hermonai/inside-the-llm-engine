# Two books, one model-artifact boundary

**Training Large Language Models From Scratch** — *From tokens and gradients
to industrial distributed training.* [Training repository](https://github.com/mapleaiorg/llm-train-book).

**Inside the LLM Engine** — *From model weights and KV memory to industrial
inference serving.* This repository.

The training book owns data provenance, forward/loss/backward, optimizer state,
precision, distributed gradient/parameter ownership and resumable experiments.
This book owns loading, execution, inference-state memory, request lifecycle,
scheduling, batching, admission, serving latency and distributed decode.
GPU kernels and model mathematics recur with different questions; do not copy
whole chapters to manufacture parallel coverage.

## The handoff is a contract, not just weights

| Producer supplies | Consumer validates | Required failure test |
| --- | --- | --- |
| Architecture/configuration | Dimensions, layer order, RMSNorm epsilon, activation and positional convention | Same shapes but different epsilon must not silently pass |
| Named weights and buffers | Dtype, axis order, tying and supported representation | Detect missing tensor or wrong transpose |
| Tokenizer identity and specials | Vocabulary, template and stop-token policy | Reject incompatible tokenizer despite equal vocabulary size |
| Tensor/artifact manifest | Version, integrity and intended model revision | Reject truncated or mismatched artifact |
| Small reference input/output | Logits at a named stage, precision and tolerance | Compare all logits, not only the sampled token |

Optimizer moments, data cursor and RNG state are training-resume state, not
inputs to an inference-only graph. A weight export cannot promise exact training
resume. A Python checkpoint format is not automatically portable to GGUF.
Format conversion requires model-semantic and numerical equivalence tests.

## Co-design example and limit

For a uniform dense-attention decoder, unquantized KV payload is
`2 × layers × retained_tokens × kv_heads × head_width × bytes_per_element`.
Reducing KV heads through GQA changes training architecture and later cache
capacity; it does not by itself prove throughput or latency improvement.
MLA, recurrent state and sliding-window retention require other accounting.
The training book derives architecture choices; this book measures their
runtime consequences on an explicitly supported model and workload.

## Current milestone boundary

The teaching inference engine currently has tensors, checked linear kernels,
embedding and RMSNorm primitives, not a complete Transformer loader. The training
companion now has a CPU-tested modern core with 17 correctness tests and local
optimizer-boundary resume. Cross-book export/run
equivalence remains a future integration gate until both sides can execute the
same model. A shared subtitle or diagram is not proof of interoperability.
