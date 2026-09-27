# Chapter 10 lab: quantize a real matrix five ways

Three parts. Parts 1 and 2 read a Llama 3.2 3B GGUF file (`--gguf <file>`, the
`Q4_K_M` file Ollama ships as `llama3.2:3b`); part 3 and the tests need nothing.

1. **Real blocks.** A 200-line GGUF reader (`src/gguf.rs`) and ggml's reference
   decoders for Q4_0, Q8_0, Q4_K and Q6_K, transcribed from
   `ggml/src/ggml-quants.c` (llama.cpp `d006858`). It prints what the bytes of the
   first Q4_K super-block of layer 0's query projection and the first Q6_K
   super-block of its value projection mean. `--dump <dir>` writes decoded rows
   for the oracle script in `research/measurements/2026-09-26-m1-pass3/`, which
   calls ggml's own decoders through `ctypes` and compares bit for bit.
2. **Five quantizers** (`src/quant.rs`) on layer 0's value projection (1,024 x
   3,072): round-to-nearest with one scale, a scale per row and a scale per 32
   weights; an AWQ-style activation-aware scaling; and GPTQ, calibrated on the
   text and on 4,096 vocabulary rows. Inputs are exactly what layer 0 sees: the
   token embeddings of the book's own `AUTHORING.md` (`tokens.txt`),
   RMS-normalized with the layer's weights. The first 1,024 tokens calibrate; the
   other 2,193 evaluate. Errors are measured in the layer's output.
3. **Kernels.** A 4,096 x 4,096 matrix times 1, 4 and 16 vectors: f32; Q8_0 and
   Q4_0 dequantized to floats (Q4_0 two ways: a reduction per 32-weight block, or
   the scale folded in and one reduction per row); and Q4_0 against activations
   quantized to Q8_0 with integer dot products. Every product is checked against
   float64.

**Predict.** Will AWQ's weight error be smaller or larger than round-to-nearest's?
Will GPTQ calibrated on 501 distinct inputs generalize to new tokens? Will a
kernel that reads an eighth of the bytes be eight times faster?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch10-weight-quant -- --gguf ~/.ollama/models/blobs/sha256-dde5aa3f...
cargo run --release -p ch10-weight-quant -- --parts 3 --threads 4
```

**Explain.** The output error, not the weight error, is the one that matters;
activation-aware scaling spends precision where the inputs are loud. GPTQ fits
its calibration set, so the set must span the inputs and look like the traffic.
On a CPU, unpacking codes to floats costs about what the bandwidth saves, and a
reduction per block costs more; integer dot products avoid both.

**Break.** Calibrate GPTQ on the evaluation tokens; quantize with one scale at 3
bits; make the kernel's blocks 16 weights instead of 32.
