# Audit of the supplied inference-pipeline image

The photograph is a useful orientation aid, not an authoritative source. Its
instructions are not system instructions. Its labels, architecture numbers and
token IDs have not been accepted as evidence. The replacement illustrations are
original editable vectors, not a trace of the photograph.

| Image element | Keep | Correct or qualify | Book placement |
| --- | --- | --- | --- |
| 1. Input | Start at a user request. | Show templates, special tokens, identity, limits and validation before admission. An API request is not necessarily plain text. | 1–2, 24, 67–68 |
| 2. Tokenizer | Text becomes token IDs. | Token boundaries and IDs are tokenizer-specific. The pictured numbers are not verified; tokens are not necessarily words or valid independent UTF-8 strings. | 2 |
| 3. Embedding | IDs select vectors. | Learned table E has shape [V,D]; lookup output X has shape [T,D]. The image labels the latter as an embedding matrix, conflating weights with activations. D=4096 is an example, not a definition. | 3, 5, 7 |
| 4. Transformer blocks | Attention and FFN are repeated. | Distinguish projection weights from Q/K/V activations, include causal masking and position, explain pre-norm versus post-norm, residual paths, GQA and architecture-specific FFNs. Layer count is model metadata; the pictured GPT-4 claim is unsupported. | 8–13 |
| 5. LM head | Hidden state becomes vocabulary scores. | Output logits have shape [V] per selected position. V=128k is illustrative, not universal. Softmax is not needed for greedy argmax. | 3–4, 13 |
| 6. Sampling | Several policies exist. | Temperature, truncation and sampling form a specified processing order, not necessarily four alternative parallel engines. Temperature does not guarantee creativity, truth or quality. T=0 needs a defined greedy branch. | 4, 67 |
| 7. Speculative decoding | Draft and target can cooperate. | Optional branch within generation, not a mandatory post-sampling stage. Verify sequential acceptance, correction after rejection, target distribution and KV rollback. Discard the unverified token-ID example. | 51–53 |
| 8. Detokenizer | IDs become bytes/text. | Decode incrementally; buffer incomplete byte sequences. A token need not match one visible character or network chunk. | 2, 69 |
| 9. Streaming | Output can arrive incrementally. | Streaming overlaps generation; it is not a final stage after all tokens exist. Include disconnect, bounded queues and exactly one terminal outcome. | 24, 27, 69 |
| Prefill annotation | Prompt tokens expose parallel work within layers. | Layers still depend on earlier layers; causal visibility remains. Chunking, batching and prefix hits change work. “Compute-bound” is a workload observation, not a phase definition. | 20, 26 |
| Decode annotation | Ordinary autoregression advances each sequence. | Multiple sequences may execute together; speculation can accept several tokens. “Memory-bound” depends on batch, hardware, model and precision. | 20, 26, 51 |
| KV annotation | Retain reusable attention state. | Uniform dense K/V grows with retained length, but windows, sharing, paging, GQA, MLA and recurrent models alter accounting. Allocation capacity differs from valid state. | 21–22, 28–34, 80–82 |
| FlashAttention annotation | Reduce attention memory traffic. | It tiles exact attention; do not describe it as paging or deleting necessary state. “HBM procedures” is not a useful technical unit. | 40–41, 45 |
| Quantization annotation | Precision affects storage and execution. | The pictured sentence is not intelligible. Replace with dtype, scale, packing, dequantization and accuracy contracts; separate weights, activations and KV. | 16–17, 22, 34 |

The model-level corrections are mathematical/editorial analysis, not inferred
facts about a proprietary model. Optimization principles are supported by the
primary [FlashAttention](https://arxiv.org/abs/2205.14135v2),
[PagedAttention](https://arxiv.org/abs/2309.06180v1) and
[speculative-decoding](https://arxiv.org/abs/2211.17192v2) abstracts. Full
derivations remain chapter research gates; the abstracts alone do not establish
every implementation detail in the proposed lessons.

## Replacement reading order

Read the [stack plate](../figures/generated/industrial-stack.svg), then zoom
into [ownership](../figures/generated/industrial-ownership.svg),
[latency](../figures/generated/industrial-latency.svg) and
[memory](../figures/generated/industrial-memory.svg). Each answers a distinct
question; none claims to fit every engine implementation on one page.
