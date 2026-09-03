# Part II Research — Build a Transformer Inference Engine

Part II replaces ENGINE-1's hand-shaped numerical buffers with explicit tensor
metadata, then builds the Transformer operations on that substrate one chapter
at a time.

| Chapter | Research question | Status |
| ---: | --- | --- |
| [5](chapter-05-tensors-without-magic.md) | How do logical tensor indices map safely to owned physical storage? | COMPLETE |
| [6](chapter-06-matrix-multiplication-the-engine-room.md) | How should reference and blocked matrix multiplication expose work and traffic? | RESEARCHING |
| 7 | How do embedding lookup and RMSNorm transform typed tensors? | PLANNED |
| 8 | How do Q/K/V projections create head-shaped activations? | PLANNED |
| 9 | How does RoPE encode position without hiding layout? | PLANNED |
| 10 | How does causal attention combine scores, masks, softmax, and values? | PLANNED |
| 11 | How does the feed-forward network transform each token? | PLANNED |
| 12 | How do those operations compose into one decoder layer? | PLANNED |
| 13 | How do layers compose into next-token generation? | PLANNED |

Research notes are evidence logs, not publication prose. CURRENT, PREVIEW, and
LIBRARY claims about Hermon must be rechecked at the recorded commit.
