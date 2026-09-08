# Chapter 6 industrial regeneration

Status: COMPLETE. Started and completed 2026-09-08.
Baseline: `a5b540d9ee1c4845b73ff1d708dfee4c3a2004e0`, branch
`astra-visual-rewrite`; local and remote tracking heads agree and Chapter 5
is an ancestor. Preserve the dirty Chapter 1/runtime diagram and Chapter 8
research/status work. This pass does not begin Chapter 7 or 8.

## Untouched baseline

PASS: rustfmt, all-target check/test, 167 unit/integration tests, two compile-fail
doctests, strict Clippy, all five Python oracles, all five executable examples,
Chapter 5 parity, structure, Markdown/image links, 115 display equations,
31 real-valued shape declarations, 78 Unicode diagrams, 17 scenes/38 deterministic
artifacts, full publication build and offline/browser checks at 1024/768/390px.
PDF marker was recorded before the baseline publication rebuild.

The user's all-example gate necessitated executing timing examples as smoke
tests. Their outputs are not new scientific evidence and do not replace any
historical result. No kernel implementation changes are planned; final example
execution has the same verification-only purpose.

## Audit and dispositions

| Artifact | Disposition | Reason |
| --- | --- | --- |
| Chapter 6 manuscript | EXPAND / REPAIR | Strong contracts and negative evidence; rebuild numerical spine and add hardware/source depth |
| linear.rs dot/GEMV/GEMM/blocked | KEEP | Safe checked API, correct general-reference/canonical-candidate split, increasing-k reductions |
| 24 linear tests | KEEP / EXPAND | Shape grids, zeros, strides and tails already strong; add fixture and rounding proofs |
| chapter06_bench.rs | KEEP / HISTORICAL EVIDENCE | Fixed-order exploratory harness; no kernel tuning in this pass |
| Three benchmark records | HISTORICAL EVIDENCE | Preserve all raw values and environment fields, including losses |
| Existing Python oracle | KEEP | Independent F32 reference and old hand fixtures remain regression evidence |
| Labs 22-29 | KEEP / EXPAND | Preserve numbering and original exercises; add exact visual checkpoints |
| Historical research note | KEEP / REPAIR | Append dated corrections, never silently re-date original source inspection |
| gemm-shape-contract (D049) | REDRAW | A does not become B; both are independent inputs |
| gemm-one-output-cell (D050) | REDRAW | Existing connectors pair wrong reduction indices |
| cache-reuse-and-tiling (D054) | REDRAW | Replace false serial A-to-B-to-C flow with multiply/accumulate inputs |
| dot-product-multiply-accumulate | KEEP | Correct paired products and empty identity |
| gemv-shape-contract | REPAIR | Make two inputs and result explicit without an ambiguous crossing |
| gemv-vs-gemm-reuse | REPAIR | Independent weight/activation inputs, not W becoming a vector |
| row-major-access | KEEP | Correct B stride-N versus stride-1 contrast |
| loop-order-ijk-vs-ikj | KEEP | Correct coordinate progression |
| reference-vs-blocked-kernel | KEEP | Honest layout split and new owner |
| optimization-ladder | KEEP | Explicit milestone boundary |
| roofline-concept | KEEP | Conceptual bound, not fitted data |
| engine-2-kernel-stack | KEEP | Correct model/kernel/substrate boundary |
| weight-orientation | KEEP | Candidate rows, hidden width, explicit bias |
| follow-the-flop | KEEP | Two operands enter one multiply |
| follow-the-byte | KEEP | Canonical input slices and owned output |
| follow-the-reuse | KEEP | One A scalar reused across j |
| reference-candidate-gates | KEEP | Independent candidate/reference, equivalence before timing |
| Existing GEMV prototype | HISTORICAL prototype | Retain as early atlas design; new coherent fixture gets new chapter IDs |
| Figure/publication infrastructure | EXPAND | Shared numerical trace, two animations, Chapter 6 standalone output |

No artifact needs deletion from Git or removal from the active edition.

## Corrections established by code inspection

The research note's proposed alternating benchmark order is not implemented.
The actual harness runs fixed candidate order and uses different repetition
counts by case. Unit tests use absolute/relative 1e-5/1e-5; the historical
benchmark helper uses 1e-4/1e-5. These are distinct contracts, not one tolerance.
The current blocked kernel preserves increasing k for each cell despite changing
inter-cell visitation. It does not explicitly use FMA or SIMD. Finiteness is not
scanned by generic kernels; finite model parameters/logits are a different layer.

## Canonical storyboard

Use W [3,4], x [4], and B [4,2] whose first column is x. One fixture drives
dot, one-row accumulation, full GEMV, addresses, GEMM cells, repeated GEMV,
outer products and loop traversal. Keep the historical [2,3] example and
[5,7] tail stress as explicitly separate regression/generalization cases.

Fourteen plates: engine map; paired dot; one-row accumulation; whole GEMV;
physical GEMV; GEMM cell/three views; loop order; tiling/tails; memory hierarchy;
SIMD/CPU/GPU; measured crossover; measured throughput; teaching/production;
source-to-kernel/equivalence synthesis. Two four-step animations: accumulation
and loop order. All motion has a complete static explanation.

## Fresh source inspection — 2026-09-08

Hermon local HEAD: `2a3fd5214e17ca7283656108847d02dcd6cdf7c5`.
Pinned `vendor/llama.cpp`: `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.
Inspection is read-only; no industrial benchmark, build, or deployment is claimed.

| Status | Pinned source / symbol | Finding and limit |
| --- | --- | --- |
| CURRENT | Hermon `crates/hermon-runtime/src/dispatch.rs:78`, mode selection | Absent/unrecognized mode selects Batched; paged selection warns and has a separate GGUF preview gate |
| CURRENT | `crates/hermon-runtime/src/batched.rs:829` | Worker calls `Context::decode_batch` |
| CURRENT | `crates/hermon-llamacpp/src/linked.rs:761`; `csrc/shim.c:463` | Decode FFI wrapper reaches `llama_decode`; not the tensor-bundle helper |
| LIBRARY / PREVIEW caller | `crates/hermon-llamacpp/src/linked.rs:255` through bundle helpers | Matvec uses one row; one-to-eight tensors, positive dimensions/shared K, checked counts/conversions, owned Vec outputs; exclusive mutable session |
| PREVIEW | `crates/hermon-runtime/src/paged.rs:1070` and `:1117` | Bundle calls on the gated paged path; availability is not default use |
| LIBRARY | `crates/hermon-llamacpp/csrc/tensor_bridge.cpp:285` through `:345` | F32 row-batched input copied into GGML; weight operands remain packed; graph expands outputs, session computes on CPU, output copied back; requires host weights |
| LIBRARY | llama.cpp `src/llama-graph.cpp:978`, `build_lora_mm` | Model graph builds a `ggml_mul_mat` node; node construction is not kernel execution |
| LIBRARY | `ggml/src/ggml.c:3240`, `ggml_mul_mat` | Validates contraction/non-transposed A; F32 result ne=[a.ne1,b.ne1,b.ne2,b.ne3], op and two sources |
| LIBRARY | `ggml/src/ggml-backend.cpp:1864` through `:1901` | Scheduler splits/allocates a graph, computes splits; synchronous wrapper synchronizes backends; distinct from the host CPU bridge |
| LIBRARY | `ggml/src/ggml-cpu/ggml-cpu.c:207`, `:1151` | CPU type traits select vec_dot and vec_dot_type; row conversion/work buffer and chunk/tile traversal |
| LIBRARY | `ggml/src/ggml-cpu/vec.cpp:11` onwards | F32 vector dot has SIMD loads, FMA macro, multiple partial accumulators and reduction, selected by build/target |
| LIBRARY | `ggml/src/ggml-cuda/ggml-cuda.cu:2537` | Eligibility branches for float/quantized matvec/matmul and cuBLAS; no claim this inspection executed CUDA |
| LIBRARY | `ggml/src/ggml-metal/ggml-metal-ops.cpp:2025` | Type/shape-dependent matrix operation and small-batch paths; source thresholds are not the historical CPU benchmark crossover |

Pinned repositories: [Hermon](https://github.com/hermonai/hermon/tree/2a3fd5214e17ca7283656108847d02dcd6cdf7c5),
[llama.cpp/GGML](https://github.com/ggml-org/llama.cpp/tree/389ff61d77b5c71cec0cf92fe4e5d01ace80b797).
Paths in the table are relative to the indicated repository, never links to a
private machine checkout.

### External primary sources actually inspected

- [Netlib BLAS](https://www.netlib.org/blas/): operation families and the distinction between a BLAS family and the smaller teaching API.
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab): matrix tiling, coalesced access and redundant-load reduction; not imported benchmark numbers.
- [NVIDIA CUTLASS explanation](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/): historical primary explanation of block/warp/thread tiling, register accumulation and matrix instructions, not a current release or performance endorsement.
- [BLIS threading documentation](https://github.com/flame/blis/blob/master/docs/Multithreading.md): loop partitioning around microkernels, runtime versus build-time threading and affinity.

The Goto paper and Berkeley Roofline PDF endpoints did not fetch successfully
during this refresh. They are not counted as newly read primary evidence.
The retained Roofline equation remains a conceptual model supported by the
historical research bibliography, not a calibrated measurement on this machine.

## Numerical and representation ledger

Input-only JSON: W=[1,-2,3,0; 0,1,-1,2; 2,0,1,-1], x=[2,1,-1,0.5],
B=[2,1; 1,-1; -1,2; 0.5,0].
Oracle and actual Rust result: y=[-3,3,2.5], C=[-3,9; 3,-3; 2.5,4].
Row-0 products [2,-2,-3,0], partial sums [2,0,-3,-3].
Canonical strides W=[4,1], B/C=[2,1]; W[1,2] element 6, byte 24, value -1.
Reference columns of B are valid stride-2 vector views.

GGML translation: teaching B[K,N] must be prepared as row-batched X=Bᵀ[N,K];
Y=XWᵀ[N,M]=Cᵀ. GGML ne reverses these two row-major dimensions. This is an
axis/value correspondence, not proof that a raw teaching B buffer can be
passed unchanged. Packed ne describes logical elements, nb describes bytes;
type-specific block metadata prevents naive F32 pointer arithmetic.

Tail stress [5,7] × [7,3], blocks [4,4,2]: eight combinations; final
i=4..5,k=4..7,j=2..3 contributes three products to the last C cell.
Six tests add repeated-GEMV, outer-product/candidate equality, physical offsets,
tail/layout rejection, binary32 reassociation, and explicit FMA rounding.
No public API or implementation file changed.

## Historical performance preserved

All three original records remain byte-for-byte unchanged, dated 2026-09-03,
code `03e08a877be445d70a211996a8eb735a982e5c0f`, Apple M1, arm64,
16 GiB, macOS 26.6.2 build 25G83, rustc 1.92/LLVM 21.1.3, release/default
flags, one CPU thread, no GPU. Allocation/zeroing included; input creation and
correctness outside timing. Warm process; cache/frequency/load not controlled;
fixed order, workload-specific repetition counts, medians, deterministic F32.
Read the raw records for individual repetitions and all fields.

Retained losses: tile 8 at size 192; tile 32 at square sizes 8 and 16;
N=8 effective throughput below N=1 GEMV. Tile 64 won the recorded 192 sweep,
not the default tile 32. Charts parse CSV medians directly. Causes such as
bookkeeping overhead and cache behavior are hypotheses, not measured counters.
Operation counts are conventional 2MNK; ideal intensity is modeled compulsory
traffic, never actual cache/DRAM traffic or a count of CPU instructions.

## Open boundaries

No SIMD, packing, quantization, BLAS, threads, GPU, fusion, attention, QKV,
RoPE, KV cache, or Chapter 7/8 regeneration was implemented. No industrial
performance or universal crossover is established. GPU figure is conceptual.
All reviewer roles below are self-review lenses by one agent, not claims of
independent human review or external approval.

## Five parity reviews — PASS

| Gate | Evidence |
| --- | --- |
| Prose / mathematics | Same W/x/B throughout new numerical plates; all six C values, columns and outer products agree; dimensions, K bounds and byte units explicit |
| Mathematics / code | Four quoted Rust loop bodies match the actual source through an automated whitespace-normalized check; zero/layout/ownership and generic finiteness policies unchanged |
| Code / oracle | Real dot/GEMV/reference/blocked outputs and metadata match the independent F32 oracle; six added tests cover additional representation/rounding cases |
| Figures / semantics | Fourteen deterministic SVG/TXT plates embedded once each; all paired indices, row/column highlights, offset 6/byte 24, tile edges and measured CSV values checked; source arrows no longer imply production calls ENGINE-2 |
| Production / source | Reverified clean inspected files at the same Hermon/GGML pins; default decode distinguished from optional host CPU bundle; GGML row-batched orientation and byte/block semantics explicit |

## Seven scientific reader reviews — self-review, PASS

| Review lens | Finding and disposition |
| --- | --- |
| Numerical linear algebra | Corrected the primary spine to y=[-3,3,2.5]; six-cell GEMM, outer products and repeated GEMV agree. Real arithmetic is separated from the tested binary32 reassociation/FMA examples. |
| CPU kernel engineering | Address formulas apply only after their layout checks. The manuscript quotes real blocked code, including saturating tile endpoints. No cache-hit counters or universal tile/crossover claims. |
| GPU kernel engineering | Separate CUDA/Metal eligibility paths are source-backed. Group/lane diagrams are conceptual, not a claim of simultaneous whole-matrix execution or a new accelerator implementation. |
| ML systems engineering | CURRENT batched decode is not the host tensor bridge. PREVIEW caller status, data copies, session ownership and axis translation are distinguished; no industrial model benchmark claimed. |
| Rust systems programming | No public API, production implementation, dependency or unsafe addition. Six new tests and actual API trace pass strict Clippy, shape/layout failures and regression gates. |
| Beginner CS reader | One rectangular signed fixture connects products, accumulators, full outputs and storage. Short independent-input diagrams replace crossing/misleading edges; every animation has a complete static plate. |
| Skeptical technical editor | Retained all three historical records and negative cases. Corrected false alternating-order/seven-repetition research claims, chart ratio labeling and unsupported causal language. Added explicit misconceptions and a bounded Chapter 7 handoff. |

## Separate editorial and publication review — PASS

The chapter is 9,984 whitespace-delimited words (the repository's `wc -w`
convention, including code/math). Original strong shape, overflow, zero-size,
ownership, benchmark methodology, and failure sections remain. New material is
integrated at its teaching point, not appended as an unrelated appendix.
Labs 22–29 preserve numbering and old fixtures, with explicit visual checkpoints.

Every new plate was inspected in color in its chapter context and in a separate
grayscale atlas render. Labels, underlines, frames, indices and arrow direction
carry meaning without color. Fourteen vector figures appear once in the
standalone chapter; they are not raster screenshots. All states remain present
in static row/loop plates; animation controls are keyboard reachable, honor
reduced motion and never autoplay. At phone width the page reflows and figures
scale without page overflow; dense vector labels remain zoomable rather than
being claimed readable at every unzoomed phone scale. This is visual/functional
QA, not an external accessibility certification or a screen-reader user study.

The render cycle caught and repaired a split reference/blocked table, misleading
GEMV/selected-row connectors, long inline pins, and two pre-existing off-page
source paths in the complete book. The PDF builder wraps long inline paths
without altering manuscript source or code blocks. Standalone title metadata is
set once and the contents fit a single page; no stranded three-line second TOC
page remains. Math uses native offline MathML in HTML and vector TeX in PDF.

Final publications: 137-page full book; 33-page Chapter 6; 23-page Chapter 5;
31-page vector atlas. Poppler word geometry verifies every extracted word stays
inside all four PDF page boundaries. All current Chapter 6 pages are rendered
for review; the unchanged Chapter 5 chapter and the repaired full-book pages
also passed inspection. The complete book is still seven written chapters.

## Final validation ledger

- PASS `cargo fmt --all -- --check`, `cargo check --workspace --all-targets`,
  `cargo test --workspace --all-targets`, `cargo test --workspace --doc`,
  `cargo clippy --workspace --all-targets -- -D warnings`.
- PASS 173 unit/integration tests plus two compile-fail doctests.
- PASS all six Python oracles: five historical chapter oracles and the new
  independent visual oracle; both Chapter 5/6 real-Rust visual parity gates.
- PASS all six release examples. Timing example executions are verification
  smoke runs only; their newer timing output was not substituted for evidence.
- PASS original tiny-model trace: logits [-0.7,0.1,0.40000004,2.2] in actual
  F32, intended oracle [-0.7,0.1,0.4,2.2], text Rust then EOS, one terminal.
- PASS structure (15 parts / 94 specifications), Markdown links, 121 display
  equation blocks / 31 real-valued shape declarations, all 78 Unicode diagrams
  (17 Chapter 6 retained), and diff whitespace.
- PASS 31 figure scenes / 68 deterministic artifacts; 22 canonical plates and
  nine prototypes, two new animations / five total.
- PASS all four vector PDF builds, standalone/offline HTML, all five animations,
  keyboard controls, reduced motion, SVG canvas text bounds and viewport widths
  1024/768/390. Both chapter publication structure checks and all-PDF bounds pass.
- PASS credential guard across repository remotes; no engine/library dependency
  added, no lockfile change, no unsafe added, no numerical implementation change.
  Publication now explicitly requires the already installed TeX `seqsplit`
  package; Python package requirements are unchanged.
- PASS three raw benchmark records and Chapter 7 manuscript/implementation remain
  unchanged. Preserved user-owned Chapter 1/runtime-diagram and Chapter 8
  research/status work remain outside this commit.

No material blocker remains. The next bounded regeneration is Chapter 7
embedding and normalization, using the existing oracle and scale/magnitude
stress evidence. It has not begun in this milestone.
