# Independent GGUF and ggml check

This optional CPU check complements the standard-library Python lesson.
It loads the same 256-byte fixture through upstream ggml, decodes all 32
Q4_0 weights exactly, and executes F32 and Q4_0 matrix-vector graphs.
It is not an LLM benchmark, general model loader, or GPU test.

Prerequisites: Git, CMake >= 3.19 and a C++17 compiler. From the book root,
use a fresh scratch checkout at the exact recorded revision:

```bash
git clone https://github.com/ggml-org/ggml.git build/ggml-format-reference
git -C build/ggml-format-reference checkout --detach 353b63b439f27ab2cc19dac97ab1681ba6d2d084
cmake -S code/foundations/ggml-reference -B build/gguf-reference \
  -DGGML_SOURCE="$(pwd)/build/ggml-format-reference" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/gguf-reference -j 4
python3 code/foundations/gguf_bytes.py --write build/byte-lab.gguf
build/gguf-reference/gguf-reference build/byte-lab.gguf
```

If the clone or fixture already exists, reuse it after checking its revision
or contents; do not overwrite it. The fixture writer refuses replacement.
CMake rejects a different revision or tracked upstream changes. It disables
CUDA, Metal, BLAS and OpenMP backends for this exercise; CPU acceleration
available to the compiler may still be used. No library is installed globally.

Predict before running:

- The loaded data origin is 192; tensor payloads occupy 32 and 18 bytes.
- Dense weights `ne=[4,2]` times input `[1,2,4,8]` produce `[49,19]`.
- The decoded packed weights sum to -8, with positions 0 and 16 equal to
  -4 and 3.5. A direct decode must agree exactly.
- This pinned CPU graph path converts activations to Q8_0. Its binary16
  scale rounds `1/127` to `0.00787353515625`, making a reconstructed one
  equal to `0.99993896484375`. The packed graph result is therefore
  `-7.99951171875`, not exactly -8.

The first run's incorrect exact-mathematics expectation is retained in
[the numerical record](../../../research/measurements/2026-10-03-gguf-byte-lab.md).
The finished check compares graph outputs to the conversion-aware prediction
with absolute tolerance `1e-5`. Thirty-two basis probes catch permutation
errors that a sum alone would miss.

Break a scratch copy: interleave low/high nibble outputs, read offsets as
absolute file positions, or reinterpret `ne=[4,2]` as two columns. Explain
which independent check catches each error. Do not weaken the assertions to
accept a failure before tracing its cause.

The Python reader is intentionally restricted to little-endian v3,
UINT32/STRING/ARRAY-of-STRING metadata, F32/Q4_0 positive 1-4D tensors,
and 1 MiB total input. Its strict zero-padding, nonoverlap and trailing-byte
rules describe the supported teaching subset. It rejects some files that
other GGUF readers may support. Feed the C++ route only the trusted fixture.
