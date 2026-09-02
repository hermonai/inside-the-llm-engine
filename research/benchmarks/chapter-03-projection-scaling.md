# Chapter 3 Scalar Projection Scaling Probe

Date: 2026-09-02.

This is a pedagogical scaling observation, not an LLM throughput benchmark. It
uses plain Python scalar loops to make the `V*D` work visible. It neither runs
ENGINE-1 Rust nor predicts optimized CPU, GPU, or billion-parameter behavior.

## Environment

- repository starting commit: `133cd181c6378eec3ad4b0411ebc87f120e9e39a`
- harness: `code/experiments/chapter-03-projection-scaling.py`
- build/profile: Python interpreter, no compilation flags
- Python: 3.9.6
- machine: MacBook Pro (MacBookPro17,1), Apple M1, 8 cores, 16 GiB RAM
- operating system: macOS 26.6.2, Darwin 25.6.0
- model/quantization: synthetic row-major `f32`-conceptual values represented
  by Python floats; no model artifact and no quantization
- workload: one scalar output projection per sample; fixed deterministic
  hidden/weight values; sizes listed below
- concurrency: 1
- warmup: one unrecorded forward per shape
- repetitions: 21 per shape
- statistic: median wall-clock nanoseconds from `perf_counter_ns`
- cache state: uncontrolled warm process/OS state; arrays allocated before
  measurement; not suitable for cross-system comparison
- correctness control: a deterministic checksum is printed for each shape;
  Chapter 3's exact numerical correctness is established separately by the
  Python oracle and Rust full-vector test

Command:

```sh
python3 code/experiments/chapter-03-projection-scaling.py
```

## Raw result

```text
warning=pedagogical Python scalar probe; do not extrapolate to LLM throughput
V=    4 D=   3 parameters=       28 parameter_bytes=       112 median_forward_ns=      1666 checksum=0.054688
V=  100 D=  16 parameters=     3300 parameter_bytes=     13200 median_forward_ns=    122209 checksum=-0.015625
V= 1000 D=  64 parameters=   129000 parameter_bytes=    516000 median_forward_ns=   4474916 checksum=-0.132812
V= 2000 D= 128 parameters=   514000 parameter_bytes=   2056000 median_forward_ns=  18534791 checksum=-0.070312
```

## Interpretation and limits

Increasing `V` or `D` increases stored parameters and the number of scalar dot
product terms. The observations show longer execution in this specific
interpreter probe as the loop count grows. They do not isolate memory
bandwidth, arithmetic throughput, interpreter overhead, cache effects, or CPU
frequency. They must not be extrapolated linearly to a real model or compared
with Hermon, llama.cpp, BLAS, SIMD, or accelerators.
