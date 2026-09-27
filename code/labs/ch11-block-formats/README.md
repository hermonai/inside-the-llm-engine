# Chapter 11 lab: round-trip real weights through every format

Small floating-point formats written from their specifications — FP8 E4M3 (the
OCP "FN" variant) and E5M2, FP6 E2M3 and E3M2, FP4 E2M1, and the E8M0 scale —
with round-to-nearest-even and saturation, and the tensor formats built on
them: FP8 with a scale per tensor, the OCP MX family (32 elements share a
power-of-two scale chosen by Algorithm 1 of the MX paper) and NVFP4 (16
elements share an E4M3 scale under an FP32 tensor scale, by the equations of
NVIDIA's NVFP4 pretraining paper). The tests decode and re-encode every code of
every format and pin the chapter's hand examples.

With `--gguf <Llama 3.2 3B file>`, the lab uses Chapter 10's lab as a library to
load layer 0's value projection and its real inputs, and measures every format's
bits per value, weight error and output error — first on the weights, then on
the inputs with the weights exact. Part 4 puts one outlier in a block.

**Predict.** Which clamps more MX blocks, E2M3 or E3M2? Will 8-bit integers with
a scale per 32 beat FP8 on weights? On activations?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch11-block-formats -- --gguf ~/.ollama/models/blobs/sha256-dde5aa3f...
cargo run --release -p ch11-block-formats -- --outlier 100
```

**Explain.** Mantissa bits set the error once a block is scaled; MX's
power-of-two scales clamp the top of the binade; activations' outliers need the
exponent bits that weights do not.

**Break.** Give the MX scale one more bit of headroom (subtract `emax() + 1`) and
compare the clamp's cost with the coarser grid's.
