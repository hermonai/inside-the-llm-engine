# Chapter 5 Research — Tensors Without Magic

Research date: 2026-09-03.

Starting book commit: `326f268b87fc537e7b8e3feed24c590253dd28ee`.

Hermon commit: `472a44cdb511b2dae6c9569e59543db8f8350b25`.

Pinned llama.cpp commit:
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

## Research question

What minimum representation lets an inference engineer answer, for every
logical element, which `f32` storage location it occupies, who owns that
storage, which views may borrow it, whether access is contiguous, and which
malformed metadata must fail before indexing or allocation?

## Scope

Chapter 5 establishes Tensor Substrate v1 and migrates ENGINE-1's embedding and
output projection without changing its equations or generated tokens. It covers
rank, shape, element count, one implemented dtype, row-major layout,
element-strides, non-negative-stride read-only views, base offsets, strict canonical
contiguity, reshape, rank-2 transpose, bounded slices, explicit contiguous
copies, Rust ownership, bounds, and checked arithmetic.

It excludes broadcasting, negative strides, fancy indexing, autograd, device
abstractions, lazy graphs, quantized storage, matrix-multiplication APIs, SIMD,
GPU execution, GGUF parsing, and Transformer operations.

## Primary sources

| Source | Direct evidence used |
| --- | --- |
| NumPy, [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) and [`ndarray.strides`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html) | Shape, dtype, buffer, offset, and strides are distinct metadata; NumPy strides are byte steps and indexing is a stride dot product |
| PyTorch, [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html), [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html), and [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html) | Views share storage, transpose can be non-contiguous, and no-copy reshape depends on compatible size/stride relationships |
| Rust Reference, [slice types](https://doc.rust-lang.org/stable/reference/types/slice.html) | `&[T]` and `&mut [T]` borrow rather than own; safe slice access is bounds checked |
| The Rust Book, [References and Borrowing](https://doc.rust-lang.org/stable/book/ch04-02-references-and-borrowing.html) | References remain valid, and mutation requires exclusive borrowing rather than overlapping shared access |
| Rust standard library, [`usize::checked_mul`](https://doc.rust-lang.org/std/primitive.usize.html#method.checked_mul) | Overflow can be represented as `None` instead of wrapping size arithmetic |
| Pinned GGML, [`ggml.h`](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/include/ggml.h) | `ggml_tensor` records type, dimensions `ne`, byte strides `nb`, data, and view/source metadata because operations cannot assume contiguity |

Framework behavior verifies terminology; Tensor Substrate v1 deliberately has
a narrower contract. In particular, NumPy records byte strides while v1 uses
element strides, and PyTorch accepts more layout-compatible reshapes while v1
only reshapes a strictly canonical row-major view.

## Tensor terminology

A tensor is a typed multidimensional array with explicit shape and layout
semantics. A scalar has rank 0 and shape `[]`; a vector has rank 1; a matrix has
rank 2; higher-rank tensors have more axes. Tensor rank is the number of axes,
not the independent-row/column rank from linear algebra.

For shape `d = [d_0, ..., d_(r-1)]`, rank is `r` and element count is:

```text
N = product(d_i), with product([]) = 1.
```

The empty product makes shape `[]` one scalar. Any zero-sized dimension makes
`N = 0`. Shape `[0,4]` is therefore valid, owns no elements, has no valid
logical index, and may be reshaped only to another zero-element shape.

An axis is one logical direction and a dimension is that axis's length. Dtype
describes interpretation and arithmetic, not merely byte width. V1 implements
only `F32`; later `f16`, `bf16`, integer, and packed quantized formats require
different numerical and storage contracts.

## Logical layout and physical storage

An owned v1 tensor contains a contiguous `Vec<f32>` plus dynamic shape and
validated canonical-stride metadata. A view contains an
`&[f32]`, its own dynamic shape and strides, and a base offset into that borrowed
storage. The view owns metadata but not elements.

For logical index `i = [i_0, ..., i_(r-1)]`, element strides
`s = [s_0, ..., s_(r-1)]`, and base offset `b`, the physical element offset is:

```text
offset(i) = b + sum(i_a * s_a), for axes a in 0..r.
```

Each `i_a` must be less than `d_a`. Every multiply and add is checked before
the final storage lookup. Units are elements; for `f32`, a stride of 3 elements
corresponds to 12 bytes. This differs intentionally from NumPy and GGML's
published byte-stride metadata.

Canonical row-major strides are derived right to left:

```text
s_(r-1) = 1
s_a     = product(d_(a+1)..d_(r-1))
```

Thus shape `[2,3,4]` has strides `[12,4,1]`. V1 defines contiguous narrowly:
the view's strides must equal these canonical strides exactly. This is stricter
than mature frameworks' treatment of size-one axes, but makes kernel
preconditions auditable. Row-major is a chosen convention, not a universal
performance claim; column-major is a valid different layout.

## Storage extent and zero dimensions

For nonempty non-negative-stride metadata, the greatest reachable offset is:

```text
max_offset = b + sum((d_a - 1) * s_a).
required_storage_len = max_offset + 1.
```

Construction rejects rank mismatch, arithmetic overflow, or required extent
beyond the borrowed slice. For a zero-element view, no offset is reachable;
`b <= storage.len()` is sufficient. A base offset equal to storage length is
therefore valid only for an empty view.

Zero strides are accepted for immutable views because overlapping reads are
safe and the extent formula remains valid. Negative strides are not
representable: all v1 dimensions, strides, indices, and offsets are `usize`.
Mutable arbitrary-stride views are rejected by API design; the only mutable
view is the owner's full canonical, non-overlapping storage.

## Views, copies, aliasing, and ownership

`OwnedTensor::view` borrows without allocation. Rank-2 `transpose` swaps shape
and stride metadata and moves no values. A bounded axis slice changes its base
offset and one dimension. Multiple immutable views may overlap.

`reshape` is intentionally named `reshape_view`: it succeeds only when the
source is strictly canonical contiguous and the new checked element count
matches. It borrows the same storage with canonical strides and never silently
copies. A transpose is not a reshape: it changes which logical index reaches
which original value.

`to_contiguous` is the explicit materialization boundary. It allocates a new
owned tensor and copies values in row-major logical iteration order. The copy
has independent ownership and canonical layout. For an already contiguous
view, this method still creates a new owner; allocation is never disguised.

Aliasing means two logical objects can reach overlapping storage. A transpose,
slice, and source view may alias. Rust lifetimes prevent a view from outliving
the `OwnedTensor`, while `&mut` exclusivity prevents a caller from retaining an
immutable view during mutation. V1 exposes `view_mut` only for the complete
canonical owner and provides no safe constructor for an arbitrary overlapping
mutable layout.

## Chosen mini-engine API

The chosen module is `engine0::tensor`, not a new crate:

```text
OwnedTensor
  shape: Vec<usize>
  strides: Vec<usize>
  data: Vec<f32>

TensorView<'a>
  storage: &'a [f32]
  shape: Vec<usize>
  strides: Vec<usize>
  base_offset: usize

TensorViewMut<'a>
  storage: &'a mut [f32]
  shape: Vec<usize>
  canonical strides only
```

Core public operations are `from_vec`, `zeros`, `view`, `view_mut`, `get`,
`get2`, `transpose`, `slice_axis`, `reshape_view`, and `to_contiguous`.
`canonical_row_major_strides`, `checked_element_count`, `checked_byte_count`,
and the one authoritative checked offset routine remain visible teaching
functions. Debug output reports metadata but does not dump elements.

Dynamic rank was selected over const generics because later decoder tensors
change rank across operators and the book should discuss runtime metadata
directly. `OwnedTensor` stores its canonical strides even though they are
derivable, so repeated checked access does not allocate/recompute them and
debugging exposes one complete layout contract. Constructors validate that
redundancy once. General non-negative-stride immutable views were selected over
transpose-only types because one extent/indexing implementation remains small
and makes the real contract visible.

Rejected complexity includes generics over dtype/device/layout/allocator,
operator overloading, `Index` panic semantics, advanced slicing, implicit
materialization, arbitrary mutable striding, and tensor-owned computation.
Tensor is data plus metadata; Chapter 6 kernels are separate operators.

## ENGINE-1 migration plan

The Chapter 3 model currently stores embedding and projection matrices in bare
`Vec<f32>` fields and manually computes `row * hidden_dim + column`. Migrate
those two immutable parameter matrices to `OwnedTensor` with shapes `[V,D]`.
Keep bias and request-local hidden/logit vectors simple where tensor wrapping
would obscure their already explicit one-dimensional meaning.

The scalar equation remains:

```text
z_i = b_i + sum_j W[i,j] h_j
```

Only address calculation changes: all embedding and weight access flows through
checked `get2`. Existing full-logit and autoregressive tests remain the primary
semantic regression gate; add explicit parameter shape/stride tests.

## Error and allocation contracts

Typed errors cover shape and byte-count overflow, storage-length mismatch,
shape/stride rank mismatch, invalid base offset/extent, index rank mismatch,
index out of bounds, offset overflow, non-contiguous reshape, incompatible
reshape, non-matrix transpose, and invalid slice range. Public checked
operations do not panic for malformed metadata or indices.

Allocation is explicit:

| Operation | Allocates element storage? | Copies elements? |
| --- | --- | --- |
| `from_vec` | caller allocated; ownership moves | no |
| `zeros` | yes | initializes zeros |
| `view`, `view_mut` | no | no |
| `transpose`, `slice_axis`, `reshape_view` | no | no |
| `to_contiguous` | yes | yes |

Metadata vectors may allocate when views are constructed. The table concerns
the potentially large element payload, which is the important inference cost.

## Correctness oracle and tests

The independent plain-Python oracle will derive canonical strides, enumerate
row-major indices, compute physical offsets, transpose a `[2,3]` matrix,
reshape it, and materialize the transposed logical order. Integer/index results
use exact equality and import no Rust code.

Rust tests cover scalar through rank 4, deterministic small-shape enumeration,
zero dimensions, count/byte/extent overflow without allocation, malformed
storage, all indexing failures, transpose metadata and values, reshape gates,
slices/base offsets, strict contiguity, logical-order copies, alias visibility,
copy independence, mutable canonical access, and ENGINE-1 numerical/generation
equivalence. `#![forbid(unsafe_code)]` remains crate-wide.

## Hermon source paths inspected

All claims below are pinned to Hermon `472a44c` and inspected 2026-09-03.

- **CURRENT / boundary:** `crates/hermon-llamacpp/src/lib.rs` declares itself
  the only Hermon crate permitted to use unsafe. `ModelTensorInfo` carries four
  GGML dimensions, rank, and serialized type across the safe wrapper.
- **CURRENT / owner:** `crates/hermon-llamacpp/src/linked.rs:148-247` performs
  tensor lookup through a borrowed `&Model`; returned metadata is copied, and a
  requested row is materialized as owned `Vec<f32>`. The model handle remains
  the lifetime owner of packed tensors.
- **CURRENT / FFI:** the wrapper's unsafe calls are narrow and documented; the
  C shim validates model/tensor existence, shape, CPU residency, layout, and
  conversion support before writing the caller-sized output.
- **LIBRARY:** `crates/hermon-gguf/src/lib.rs:298-419,758-780` exposes tensor
  dimensions, per-tensor dtype, checked offsets/byte lengths, and exactly
  bounded readers without loading all weight bytes. This proves a parser
  capability, not default runtime use.
- **PREVIEW:** `crates/hermon-runtime/src/paged.rs:821-930` combines validated
  GGUF metadata with the llama.cpp tensor bridge for the gated paged model.
  `TensorSession` owns mutable reusable GGML scratch state behind a mutex. This
  is not the default batched execution path.
- **LIBRARY/PREVIEW:** `hermon-kernels` exports explicit dtype and geometry
  fields across a C ABI, but its presence does not make every model tensor a
  Hermon-native view.

The bounded lesson is that industrial tensor boundaries must carry shape,
type, layout/residency preconditions, and lifetime ownership. Chapter 5 does
not teach GGUF or quantization mechanics.

## Pinned llama.cpp/GGML findings

Pinned `ggml/include/ggml.h:112-148` states that tensors record size, type,
buffer, dimensions `ne`, and byte strides `nb`, and that operations must honor
strides rather than assume contiguous storage. The `ggml_tensor` structure at
the same commit also includes an owning buffer/data pointer and view/source
metadata. GGML uses fixed maximum rank and byte strides; v1 uses dynamic rank
and element strides. The conceptual correspondence is evidence, not API
compatibility.

## Diagrams and presentation decision

Eleven Chapter 5 diagrams cover logical versus physical storage, row-major
offsets, shape/strides, transpose, view/copy, ownership, lifetime, contiguity,
and the three Part II journeys. At the user's direction, Chapter 5 changes the
book convention from plain ASCII punctuation to polished Unicode box-drawing
text. All thirty-six current canonical diagrams now follow that convention and
pass a Unicode-aware 100-column gate; no renderer-specific dependency is used.

Important equations in manuscript Markdown will use display-math delimiters,
with a nearby plain-text/code form where executability or terminal portability
matters. Each equation defines symbols, units, shapes, and checked-code mapping.

## Recorded experiment

The committed harness traverses one contiguous `[2048,2048]` `f32` tensor in
row-major and column-wise logical order while keeping the arithmetic and exact
`f64` checksum equal. Seven warmed release repetitions on the recorded Apple M1
produced medians of 4,163,875 ns and 13,692,583 ns respectively. The durable
benchmark record contains the complete environment and controls. This local
observation is not a universal ratio or a matrix-multiplication benchmark.

## Failure cases

- shape product or byte count wraps before allocation;
- shape says `[2,3]` while storage contains five values;
- indices have the wrong rank or exceed one axis;
- arbitrary strides reach past the borrowed slice;
- code treats transpose metadata as canonical rows;
- reshape silently copies or accepts non-contiguous input;
- two mutable overlapping views race and produce plausible wrong values;
- a view outlives its owner;
- a kernel receives shape but not its layout precondition.

## Chapter 6 handoff

No material Chapter 5 question remains open once the implementation, oracle,
regressions, and experiment pass. Chapter 6 may assume checked `f32` tensors,
canonical 2-D owners, strided immutable views, explicit materialization, and
reliable `get2`. It will derive dot products, matrix-vector and matrix-matrix
multiplication, loop order, FLOPs, memory traffic, arithmetic intensity,
tiling, and blocked scalar kernels. Chapter 5 must not implement those
operators.
