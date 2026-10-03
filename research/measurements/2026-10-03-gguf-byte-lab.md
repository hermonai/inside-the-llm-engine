# GGUF byte-path correctness record — 2026-10-03

Scope: printed Chapter 12, stable source `f12-model-formats`. These are
numerical observations, not timing measurements. No model, training,
quality benchmark, GPU or remote machine was used.

## Reproduction

- Book base: `2c0474d` plus the source additions committed with this record.
- Upstream ggml: `353b63b439f27ab2cc19dac97ab1681ba6d2d084`; tracked source clean.
- Machine: Apple M1, macOS 26.6.2 (25G83), arm64.
- Compiler: Apple clang 17.0.0 (clang-1700.6.4.2); CMake 4.3.1; Python 3.9.6.
- Build: Release; CPU backend; Metal/CUDA/BLAS/OpenMP off; GGML_NATIVE off.
  CPU Accelerate support on (the separate BLAS backend is off).
- Execution: one CPU graph thread. Absolute graph tolerance `1e-5` against
  the conversion-aware prediction; exact equality for direct weight decoding.
- Fixture SHA-256: `4944ba6c0907263024b13c8b1efaf5e2605e5d4438f483edc6fb6a4b8919f851`.
- Commands: [reference README](../../code/foundations/ggml-reference/README.md).
  No latency statistics are reported; the final numerical check was run twice
  with identical printed outputs.

## Preserved failed expectation

The initial graph assertion compared the Q4_0 all-ones dot product to
the exact real-arithmetic value -8 at absolute tolerance 1e-5. It failed:

```text
F32 projection: 49.000000000 (expected 49.000000000) 19.000000000 (expected 19.000000000)
Q4_0 projection: -7.999511719 (expected -8.000000000)
projection differs from hand calculation
```

The direct Q4_0 weight decoder agrees exactly on all 32 weights. The pinned
CPU type table selects Q8_0 for the other operand. A unit input becomes
`127 * round_binary16(1/127) = 0.99993896484375`. Thus the predicted sum is
`-8 * 0.99993896484375 = -7.99951171875`. The finished test changes the
prediction for that documented arithmetic path, not the tolerance to hide
the failure. The scalar prediction has an independent Python test.

## Final raw output

```text
Weight decoder: all 32 values match exactly
F32 projection: 49.000000000 (expected 49.000000000) 19.000000000 (expected 19.000000000)
Q4_0 projection: -7.999511719 (expected -7.999511719)
basis[00]: -3.999755859 (expected -3.999755859)
basis[01]: -3.499786377 (expected -3.499786377)
basis[02]: -2.999816895 (expected -2.999816895)
basis[03]: -2.499847412 (expected -2.499847412)
basis[04]: -1.999877930 (expected -1.999877930)
basis[05]: -1.499908447 (expected -1.499908447)
basis[06]: -0.999938965 (expected -0.999938965)
basis[07]: -0.499969482 (expected -0.499969482)
basis[08]: 0.000000000 (expected 0.000000000)
basis[09]: 0.499969482 (expected 0.499969482)
basis[10]: 0.999938965 (expected 0.999938965)
basis[11]: 1.499908447 (expected 1.499908447)
basis[12]: 1.999877930 (expected 1.999877930)
basis[13]: 2.499847412 (expected 2.499847412)
basis[14]: 2.999816895 (expected 2.999816895)
basis[15]: 3.499786377 (expected 3.499786377)
basis[16]: 3.499786377 (expected 3.499786377)
basis[17]: 2.999816895 (expected 2.999816895)
basis[18]: 2.499847412 (expected 2.499847412)
basis[19]: 1.999877930 (expected 1.999877930)
basis[20]: 1.499908447 (expected 1.499908447)
basis[21]: 0.999938965 (expected 0.999938965)
basis[22]: 0.499969482 (expected 0.499969482)
basis[23]: 0.000000000 (expected 0.000000000)
basis[24]: -0.499969482 (expected -0.499969482)
basis[25]: -0.999938965 (expected -0.999938965)
basis[26]: -1.499908447 (expected -1.499908447)
basis[27]: -1.999877930 (expected -1.999877930)
basis[28]: -2.499847412 (expected -2.499847412)
basis[29]: -2.999816895 (expected -2.999816895)
basis[30]: -3.499786377 (expected -3.499786377)
basis[31]: -3.999755859 (expected -3.999755859)
PASS: ggml container, orientation and all 32 packed coordinates
```

Limit: a small fixture cannot establish whole-model compatibility or broad backend correctness.

