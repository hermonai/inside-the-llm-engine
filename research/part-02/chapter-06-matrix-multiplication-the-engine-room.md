# Chapter 6 Research — Matrix Multiplication: The Engine Room

Research date: 2026-09-03.

Starting book commit: `5ae1f87cfd79d4f8df6916d6bce8b42abafde0f4`.

Hermon commit: `472a44cdb511b2dae6c9569e59543db8f8350b25`.

Pinned llama.cpp commit:
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

## Research question

What is the smallest explicit linear-algebra layer that can define dot product,
matrix-vector multiplication, and matrix-matrix multiplication over Chapter
5's checked tensors; retain a transparent reference computation; demonstrate
cache-aware scalar blocking; and prove an optimized substitution correct before
measuring it?

## Scope

Chapter 6 establishes ENGINE-2 / Linear Algebra Kernel Layer v1. It implements
`f32` dot, GEMV, and GEMM reference kernels plus one configurable blocked scalar
GEMM. It covers shape contracts, accumulation order, layout policy, output
ownership, loop order, cache locality, reuse, idealized byte traffic,
arithmetic intensity, Roofline reasoning, equivalence, and controlled timing.

It excludes RMSNorm, attention, Q/K/V, RoPE, KV state, model formats,
quantization, packing implementations, SIMD intrinsics, BLAS calls, threading,
accelerators, autograd, and general graph execution. Embedding remains a row
lookup. Only ENGINE-1's output projection moves to GEMV.

## Primary sources

| Source | Direct evidence used |
| --- | --- |
| Netlib, [BLAS overview](https://www.netlib.org/blas/) and [quick reference](https://www.netlib.org/lapack/lug/node145.html) | Level 1 covers vector-vector work, Level 2 matrix-vector work, and Level 3 matrix-matrix work; GEMV/GEMM include explicit dimensions, transpose choices, increments/leading dimensions, and alpha/beta accumulation contracts |
| Netlib, [`SGEMV`](https://netlib.org/lapack/explore-html/d7/dda/group__gemv_ga0d35d880b663ad18204bb23bd186e380.html) and [`SGEMM`](https://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f_source.html) reference sources | Concrete shape/error/empty-dimension behavior and the distinction between a portable reference implementation and tuned vendor implementations |
| Williams, Waterman, and Patterson, [original Roofline report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf) | Attainable performance is bounded by peak compute and bandwidth times operational intensity; the analyzed memory boundary must be named |
| Lawrence Berkeley National Laboratory, [Roofline model](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/) | Current authoritative summary connecting locality, bytes moved, arithmetic intensity, bandwidth, and compute ceilings |
| Intel, [loop interchange and cache blocking](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/2023-0/optimize-memory-access-patterns.html) | For row-major matrix multiplication, interchanging inner loops can replace a large constant stride with unit-stride access; cache blocking reduces the active working set and can increase reuse |
| Intel, [Optimization Reference Manual](https://cdrdv2-public.intel.com/821612/248966-Optimization-Reference-Manual-V1-050.pdf) | Production matrix multiplication uses multi-level blocking and accumulator/register strategies; optimal extents are workload and architecture dependent |
| Oracle, [Associative Operations](https://docs.oracle.com/cd/E37069_01/html/E39019/gnabm.html) | Real-number associativity does not survive floating-point roundoff; evaluation and reduction order can change results |

These sources define vocabulary and constraints. The teaching implementation
does not reproduce BLAS, Intel's tuned example, or a measured Roofline. Its
arithmetic-intensity calculations are declared lower-bound models rather than
hardware-counter measurements.

## From scalar multiplication to dot product

One scalar product becomes useful to a kernel when accumulated:

$$
s \leftarrow s + ab.
$$

For rank-1 tensors $a,b:[K]$, the dot contract is

$$
\operatorname{dot}(a,b)=\sum_{k=0}^{K-1}a_kb_k.
$$

Both operands must have rank 1 and equal length. The reference loop validates
both facts and accumulates in increasing $k$ order. It must not use a truncating
`zip` without a prior length check. For $K=0$, the result is the additive
identity `0.0f32`.

## Matrix-vector multiplication

For $A:[M,K]$, $x:[K]$, and $y:[M]$:

$$
y_i=\sum_{k=0}^{K-1}A_{ik}x_k,
\qquad 0\le i<M.
$$

GEMV is one dot product per matrix row. It requires a rank-2 matrix, rank-1
vector, and equal inner dimensions. The API allocates and returns an
`OwnedTensor` with shape `[M]`; output never aliases either input. The reference
path accepts valid non-contiguous immutable views and uses Chapter 5 checked
logical indexing.

The approximate HPC operation-count convention is $2MK$ FLOPs, counting a
multiplication and addition as two floating-point operations. More exactly a
nonempty row uses $K$ multiplications and $K-1$ additions when initialized from
its first product; the teaching loop instead performs $K$ of each because it
starts from zero. `FLOP` counts work; `FLOP/s` measures a rate.

## Matrix-matrix multiplication and shape contract

For $A:[M,K]$, $B:[K,N]$, and newly owned $C:[M,N]$:

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

Both inputs must have rank 2 and their inner dimensions must agree. Output size
$M\times N$ is checked before allocation. Under the declared convention GEMM
performs approximately

$$
F_{\mathrm{GEMM}}=2MNK\ \text{FLOPs}.
$$

The reference order is `i-j-k`. It accepts valid strided views and produces a
canonical row-major owner. Asymmetric rectangular fixtures are mandatory;
square or symmetric-only tests can hide an accidental `B[j,k]` access.

## Empty dimensions

Chapter 5 permits zero dimensions, and kernels define them intentionally:

- `[0,K] × [K,N]` produces canonical `[0,N]` with empty storage;
- `[M,0] × [0,N]` produces `[M,N]` filled with `0.0`, the additive identity;
- a length-zero dot produces scalar `0.0`;
- GEMV `[0,K] × [K]` produces `[0]`, while `[M,0] × [0]` produces `M` zeros.

These semantics are tested rather than left to incidental loop behavior.

## Row-major access and loop permutations

Six permutations of `i`, `j`, and `k` exist. With row-major `A:[M,K]`,
`B:[K,N]`, and `C:[M,N]`, `i-j-k` holds `j` fixed while `k` changes, so
`B[k,j]` jumps by $N$ elements. `i-k-j` holds `A[i,k]` in a scalar and walks
`B[k,j]` and `C[i,j]` in unit-stride order as $j$ changes. Both perform the same
mathematical contractions, but their cache-line use and compiler opportunities
can differ substantially.

Loop-order variants belong in the benchmark example rather than the public
kernel surface. The public reference stays `i-j-k` because it mirrors the
equation. The blocked candidate uses `ii-kk-jj-i-k-j`, preserving increasing
$k$ accumulation order for each output while exposing contiguous inner rows.

## Locality, reuse, and working set

Spatial locality means adjacent accesses can benefit from nearby data already
fetched in a cache line. Temporal locality means a value is reused while it is
still resident in a faster level. A working set is the data that must remain
actively accessible for a computation interval. Blocking bounds subranges of
`A`, `B`, and `C` so useful values can be consumed again before a larger matrix
walk displaces them.

Blocking does not reduce ordinary GEMM's mathematical FLOP count. It changes
access time and reuse. Tile sizes are explicit `BlockSize { m, k, n }`; zero is
rejected before any `step_by`, and `min`-bounded tails support dimensions not
divisible by the tile. No size is called universally optimal.

## Memory traffic and arithmetic intensity

An idealized unique-payload lower bound for canonical `f32` GEMM is

$$
Q_{\min}=4(MK+KN+MN)\ \text{bytes},
$$

counting one read of each input and one output write. Actual traffic across a
named cache or DRAM boundary may be larger because of repeated loads, write
allocation, eviction, and imperfect reuse. The benchmark reports this formula
as an estimate; it does not claim hardware-counter traffic.

Arithmetic intensity is work divided by bytes moved across a named boundary:

$$
I=\frac{F}{Q}\quad\text{FLOPs/byte}.
$$

For a large `f32` GEMV, matrix weights alone give the back-of-envelope ratio
$2MK/(4MK)\approx0.5$ FLOP/byte. GEMM with more output columns can reuse each
weight for more work. That distinction previews why single-token decode and
many-token prompt processing can stress hardware differently without yet
teaching their runtime architectures.

The conceptual Roofline bound is

$$
P\le\min(P_{\text{peak}},\,B_{\text{memory}}I).
$$

It is a reasoning tool here, not a fitted model for the test Mac. `performance
≈ max(...)` is dimensionally wrong if written as time; the equivalent time
model is the maximum of compute time and data-movement time.

## Numerical accumulation and equivalence

Inputs, accumulator, and outputs are `f32`. Source loops use `acc += a*b` in
increasing-$k$ order and do not explicitly request fused multiply-add. A future
compiler, ISA, vector reduction, or GPU kernel may round differently. The scalar
reference is deterministic within the stated implementation/toolchain, not a
cross-platform bitwise standard.

Integer fixtures require exact equality. Candidate/reference floating results
use the per-element contract

$$
|c-r|\le 10^{-5}+10^{-5}|r|.
$$

The benchmark also reports maximum absolute error. This tolerance is small
enough to expose indexing/tail failures in bounded Chapter 6 fixtures while
allowing low-order accumulation differences. No tolerance can rescue NaN or an
infinite candidate when its reference is finite.

## Chosen ENGINE-2 API

Use a focused `engine0::linear` module rather than methods on tensors or a new
crate:

```text
dot_reference(&TensorView, &TensorView) -> Result<f32, KernelError>
gemv_reference(&TensorView, &TensorView) -> Result<OwnedTensor, KernelError>
matmul_reference(&TensorView, &TensorView) -> Result<OwnedTensor, KernelError>
matmul_blocked(&TensorView, &TensorView, BlockSize)
    -> Result<OwnedTensor, KernelError>
```

`KernelError` describes valid tensors that cannot participate in an operation:
rank mismatch, inner/length mismatch, unsupported optimized layout, invalid
block size, or output-shape overflow. It wraps `TensorError` for representation
failures. Public functions allocate new canonical outputs, so output aliasing is
impossible.

Reference kernels accept any validated non-negative-stride immutable view.
Blocked GEMM validates canonical row-major layout once and obtains checked
contiguous slices for its safe hot loop. It never calls `to_contiguous`
implicitly. The API remains functions rather than a trait because there is no
runtime provider selection yet.

Rejected alternatives: methods on `Tensor`, six public loop-order functions,
generic dtypes, caller-provided outputs, hidden copies, transposed/packed
weights, BLAS, SIMD, threading, const-generic matrices, and an operator graph.

## Reference and blocked kernels

The reference GEMM directly mirrors $C_{ij}$ with `i-j-k` and checked `get2`.
It is intentionally not optimized away after the candidate exists: it remains
the specification, debugging oracle, portability fallback, and validation
target.

The blocked scalar candidate validates rank, dimensions, layout, block sizes,
and output count at entry. Its trusted internal loop uses safe slices and
checked canonical offset expressions. `C` starts at zero because each $K$ tile
adds a partial sum. The block boundaries use `min(start + block, dimension)`
with checked/saturating endpoint construction so tails never escape. “Trusted”
means the safe loop executes after validated invariants, not that it uses
`unsafe`.

## ENGINE-1 migration

Keep embedding as row selection. Construct a rank-1 borrowed view over the
request-local hidden vector, call `gemv_reference(output_weight, hidden)`, then
add bias explicitly. The known hidden vector, full logits, `Rust` token, and EOS
must remain unchanged. This substitution changes the computation owner without
creating an elementwise framework.

## Independent oracle and tests

`code/reference/python/chapter06_matmul_oracle.py` independently implements dot,
GEMV, and GEMM with Python loops. It records the hand product

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}
=
\begin{bmatrix}58&64\\139&154\end{bmatrix},
$$

plus a non-integer fixture and a transpose-view logical product. It does not
import Rust or a numerical framework.

Rust tests cover dot/GEMV/GEMM hand cases, strided and transposed reference
inputs, exact empty-shape semantics, all typed failures, output overflow,
deterministic pseudo-random shape grids, odd rectangular tails, blocks larger
than dimensions, and ENGINE-1 regression. Every blocked candidate crosses the
equivalence gate before timing.

## Benchmark methodology

A dependency-free release example records three independently interpretable
experiments:

1. `i-j-k` versus `i-k-j` on identical canonical square matrices, alternating
   measurement order after warmup;
2. reference, `i-k-j`, and blocked scalar GEMM plus bounded block sizes and a
   small-matrix crossover sweep;
3. one fixed weight matrix multiplied by 1, 8, and 64 state columns, reporting
   elapsed median, the $2MNK$ convention, effective GFLOP/s, estimated ideal
   intensity, checksum, and maximum candidate/reference error.

The primary timing is end-to-end public/helper call time including equal output
allocation and zero initialization. Input construction and correctness checks
are outside timing. Results use one warmup, seven repetitions, median, consumed
checksums, one thread, release mode, and the exact machine/toolchain record.
They are exploratory and do not claim production GEMM or a hardware peak.

## Hermon paths inspected

All classifications are reverified at Hermon `472a44c` on 2026-09-03.

- **CURRENT:** the default batched runtime delegates the model graph and its
  numerical work to the pinned llama.cpp context. Hermon does not substitute
  the teaching kernel.
- **CURRENT boundary:** `hermon-llamacpp/src/linked.rs:255-455` exposes checked
  packed-weight matvec, matrix-by-batch, and bundled projection wrappers. It
  validates ranks, shared widths, input/output counts, `usize`/`i64` conversion,
  host residency, and errors before/after the FFI boundary.
- **PREVIEW:** `hermon-runtime/src/paged.rs:821-856,1050-1130` keeps quantized
  matrices packed and invokes `matmul_bundle_with_session`; runtime selection
  and real GGUF execution remain explicitly gated in `dispatch.rs`.
- **PREVIEW oracle:** `paged.rs:547-573` has a simple row-major `f32` projection
  helper used by the test/oracle forward, not the default production path.
- **LIBRARY/PREVIEW:** `hermon-kernels` contains native packed expert matvec and
  other kernel components. File presence does not make them the default dense
  model GEMM.

The case-study lesson is measured substitution: shape, dtype, layout,
residency, scratch/session ownership, thread count, and failure semantics all
surround the arithmetic.

## Pinned llama.cpp/GGML paths inspected

- `ggml/include/ggml.h:1414-1438` declares `ggml_mul_mat` with explicit tensor
  dimension conventions.
- `ggml-cpu/ggml-cpu.cpp:423-470` gates CPU operation support by operand type.
- `ggml-cpu/ggml-cpu.c:1151-1438` chooses type-specific vector-dot functions,
  honors stride/contiguity conditions, converts input types when needed, tiles
  work, and distributes chunks across threads; an optional llamafile SGEMM path
  may intercept supported cases.
- `ggml-cpu/ggml-cpu.c:207-235` maps stored types to conversion and vector-dot
  traits; `ggml-cpu/vec.cpp` contains architecture-conditioned vectorized dot
  implementations.
- Metal and CUDA backends dispatch `GGML_OP_MUL_MAT` separately, including
  shape/type-specific matvec, matrix, and library paths.

This is the full progression from mathematical op to graph node, type/layout
dispatch, and hardware-specific implementation. Chapter 6 stops at safe scalar
cache blocking and makes no equivalence or speed claim against GGML.

## Potential diagrams

Canonical Unicode diagrams should cover multiply-accumulate, GEMV/GEMM shapes,
one output cell, row-major access, `ijk` versus `ikj`, tiling/reuse, reference
versus candidate, optimization ladder, GEMV/GEMM intensity, Roofline, ENGINE-2
stack, weight orientation, and the FLOP/byte/reuse journeys.

## Chapter 7 handoff

Chapter 7 may assume a checked tensor representation and explicit reference
linear-algebra layer with a measured, equivalence-gated blocked candidate. It
will revisit embedding lookup and implement RMS/RMSNorm, epsilon, learned
scale, residual-stream shapes, accumulation stability, and out-of-place
reference behavior. Chapter 6 must not implement any of those operators.
