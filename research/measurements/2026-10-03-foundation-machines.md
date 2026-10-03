# Foundation numerics and execution contracts — 2026-10-03

This record covers derived numerical examples and observed correctness checks.
There are **no CPU throughput measurements or GPU runs** in this pass.
Printed Chapters 13–15 retain their stable `f13`–`f15` source IDs.

## Reproduction and environment

Apple M1, macOS 26.6.2 (25G83), Python 3.9.6, Apple Clang 17.0.0
(clang-1700.6.4.2), target `arm64-apple-darwin25.6.0`. The PDF build record
hashes the committed Python/C lesson sources. Inputs are deterministic toy
values; no model weights, inference service, accelerator or download required.

```bash
python3 code/foundations/numerical_path.py
python3 code/foundations/memory_execution.py
python3 -m unittest discover -s code/foundations -p 'test_*.py'
mkdir -p build/foundations-machine
clang -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  code/foundations/cpu-dot.c -o build/foundations-machine/cpu-dot
build/foundations-machine/cpu-dot
clang -std=c11 -O1 -g -ffp-contract=off -fsanitize=undefined \
  -fno-omit-frame-pointer code/foundations/cpu-dot.c \
  -o build/foundations-machine/cpu-dot-ubsan
build/foundations-machine/cpu-dot-ubsan
clang -std=c11 -O2 -ffp-contract=off -S code/foundations/cpu-dot.c \
  -o build/foundations-machine/cpu-dot.s
```

## Numerical output

```json
{
  "f16_1_5_hex": "0x3e00",
  "bf16_1_5_hex": "0x3fc0",
  "f16_min_subnormal": "1/16777216",
  "sum_4096_ones_f16": "2048",
  "sum_4096_ones_f32": "4096",
  "serial_cancellation": "0",
  "regrouped_cancellation": "1",
  "separate_multiply_add": "0",
  "fused_multiply_add": "-1/67108864",
  "outlier_codes": [0, 0, 0, 7],
  "outlier_reconstruction": ["0", "0", "0", "28"]
}
```

The rational decoder checks all 65,536 FP16 encodings against `struct`.
Finite nonzero values also round-trip through the exact encoder. Sampled
neighbour midpoints and points to either side compare conversion decisions
against `struct`. BF16 tests use exact midpoint mathematics and field
fixtures, **not an independent native BF16 instruction**. The model does not
emulate NaN propagation, signed-zero operations, flags or hardware FTZ.

## Address and tile output

```json
{
  "f32_16x16_row_scan_lines": 16,
  "f32_16x16_one_column_lines": 16,
  "aligned_warp_sectors": 4,
  "offset_warp_sectors": 5,
  "strided_warp_sectors": 32,
  "shared_column_conflict": 32,
  "shared_padded_column_conflict": 1,
  "shared_broadcast_conflict": 1,
  "tile_5x3_by_3x7": {
    "blocks": 12,
    "logical_workers": 48,
    "valid_outputs": 35,
    "global_value_loads": 123
  },
  "storage_only_blocks": 4,
  "all_limits_blocks": 3
}
```

The cooperative matrix loop agrees with an independent integer loop for 210
shapes and four tile sizes (840 combinations). Invalid early reads, overwrite,
duplicate consumers and retirement with missing readers are rejected.
These are a serial teaching object's invariants, not CUDA race detection.
Sector/bank counts have explicit address-unit assumptions; they do not count
actual cache misses, DRAM traffic or device cycles.

## Native CPU result and inspected assembly

The optimized build returned:

```text
path=AArch64 NEON, four F32 lanes; checked=516 length/offset cases; exact dyadic oracle passed
```

The undefined-behaviour-sanitized build returned the same line, exit zero.
Both normal and UBSan runs were repeated with identical results. A separate
build using `-U__ARM_NEON` exercised the fallback preprocessor branch and
returned `path=scalar fallback` with the same 516 successful cases. This
does not substitute for compiling on an actual x86 Linux CI runner.
Clang's static analyzer also completed without diagnostics. Neither static
analysis nor UBSan is a substitute for the unavailable AddressSanitizer check.
The fixture's signed integer numerator oracle makes equality exact for its
bounded dyadic inputs; ordinary model dot products need a different contract.
The `lane_dot` function in this build's assembly includes:

```text
ldr      q1, [x11], #16
ldr      q2, [x10], #16
fmul.4s  v1, v1, v2
fadd.4s  v0, v0, v1
...
faddp.4s v0, v0, v0
faddp.2s s0, v0
```

This establishes generated vector instructions in the observed path, not a
speedup. The compiler also vectorizes some scalar-loop products while keeping
ordered additions. A source-level scalar loop is not proof of scalar assembly.

## Negative result and unavailable sanitizer boundary

A scratch-only mutation removed the scalar tail from `lane_dot`, leaving all
other code intact. The mutated build exited one:

```text
mismatch n=1 offset=0
```

There are no complete vectors at length one. The true product is
`(-6/4)*(-3/2) = 2.25`, whereas the tail-free vector path returns zero.
Divisible-by-four test lengths alone can hide this defect.

The combined AddressSanitizer/UBSan build compiled, but did not reach `main`.
A process sample showed `__asan::InitializeShadowMemory` re-entering sanitizer
malloc initialization and waiting in `StaticSpinMutex::LockSlow`. After about
seven minutes without output it was terminated. A bounded second attempt
with `MallocNanoZone=0` also timed out (exit 142 after ten seconds).
**AddressSanitizer is unverified locally**, not passed. UBSan alone passed.
CI contains the combined check for its Linux environment; no CI run is
claimed in this record. The blocked macOS runtime is not evidence of a fault
or of memory safety in the lab itself.
