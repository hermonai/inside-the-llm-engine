# Part II — Build a Transformer Inference Engine

**Goal:** derive and implement a decoder-only Transformer without hiding tensor
semantics behind a framework.
**Chapters:** 5–13, tensors and matmul through embeddings, RMSNorm, Q/K/V, RoPE,
causal attention, FFN, one layer, and the decoder stack.
**Prerequisites:** Part I and comfort reading array code.
**Code milestone:** ENGINE-2 produces verified next-token logits.
**Conceptual milestone:** account for every tensor shape and operation.
**Later parts:** provides the semantics that formats, caches, and kernels preserve.
