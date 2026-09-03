# Part II — Build a Transformer Inference Engine

**Goal:** derive and implement a decoder-only Transformer without hiding tensor
semantics behind a framework.
**Chapters:** 5–13, tensors and matmul through embeddings, RMSNorm, Q/K/V, RoPE,
causal attention, FFN, one layer, and the decoder stack.
**Prerequisites:** Part I and comfort reading array code.
**Code milestone:** ENGINE-2 provides a verified linear-algebra kernel layer;
later Part II chapters extend it toward a tiny Transformer.
**Conceptual milestone:** account for every tensor shape and operation.
**Later parts:** provides the semantics that formats, caches, and kernels preserve.

| Chapter | Question | Status |
| ---: | --- | --- |
| [5. Tensors Without Magic](chapter-05-tensors-without-magic.md) | How are logical tensors represented safely in physical memory? | COMPLETE |
| [6. Matrix Multiplication: The Engine Room](chapter-06-matrix-multiplication-the-engine-room.md) | How do we multiply them correctly and efficiently? | COMPLETE |
| 7. Embeddings and RMSNorm | How do lookup and normalization transform them? | NEXT |
| 8. Queries, Keys, and Values | How do projections create head-shaped activations? | PLANNED |
| 9. Position: RoPE From First Principles | How does position enter Q/K geometry? | PLANNED |
| 10. Causal Self-Attention | How do scores, masking, softmax, and values combine? | PLANNED |
| 11. The Feed-Forward Network | How does the token-wise nonlinear path work? | PLANNED |
| 12. One Complete Transformer Layer | How do the operations compose safely? | PLANNED |
| 13. The Decoder Stack and Next-Token Generation | How do layers produce verified logits? | PLANNED |
