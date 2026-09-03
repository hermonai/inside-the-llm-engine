# Chapter 5 — Tensors Without Magic

Part I ended with a complete loop. Text became token IDs. A tiny language model
turned the last ID into an embedding, multiplied that activation by an output
projection, produced logits, and fed a selected token back into the next step.
Every number was small enough to print. Every array was short enough to count.

That convenience hid a systems problem. The model stored its embedding in a
`Vec<f32>`, yet the vector itself did not say that its logical shape was
`[vocabulary, hidden]`. The projection used another `Vec<f32>`. A comment and a
pair of loop bounds carried the missing meaning. If those facts ever disagreed,
the bytes could remain finite and the program could still compute the wrong
answer.

This chapter gives those bytes a contract. It does not build a tensor framework
or perform matrix multiplication. It builds the smallest substrate on which the
rest of Part II can reason honestly about memory.

> **FIRST PRINCIPLE**
> Mathematics tells us *what* value we need. Tensor layout tells the machine
> *where* that value lives.

## The vector that could not tell the truth

Suppose a loader hands us twelve `f32` values. What tensor did it load?

```text
[12]       [3, 4]       [4, 3]       [2, 2, 3]       [1, 3, 4]
```

Every listed shape has twelve elements. They do not have the same axes or
indexing semantics. Even shape `[3,4]` does not tell us whether adjacent bytes
move across a row, down a column, or through some strided view of a larger
allocation. Twelve scalar payloads are evidence about storage length, not a
complete tensor description.

For every tensor in this book, ask five questions:

1. What is its shape?
2. What is its dtype?
3. What is its physical layout?
4. Who owns the underlying bytes?
5. How long may a view reference those bytes?

Later we will add location—CPU memory, device memory, mapped file, or another
tier. Chapter 5 stays in local CPU memory so we can settle the first five
without hiding them behind a device abstraction.

## From scalar to tensor

A **scalar** is one value and has rank 0. A **vector** has one logical axis and
rank 1. A **matrix** has two axes and rank 2. We use **tensor** for the general
case: a typed multidimensional array with explicit shape and layout semantics.

| Object | Example | Shape | Tensor rank |
| --- | --- | --- | ---: |
| Scalar | `42.0` | `[]` | 0 |
| Vector | `[1,2,3]` | `[3]` | 1 |
| Matrix | three rows, two columns | `[3,2]` | 2 |
| Higher-rank tensor | two batches of three-by-four values | `[2,3,4]` | 3 |

Here **rank** means number of logical axes. Linear algebra also uses *matrix
rank* for the dimension of a matrix's row or column space. These are different
concepts. A shape `[3,2]` tensor always has tensor rank 2, even if all its rows
are linearly dependent.

If a rank-$r$ tensor has shape

$$
\mathbf{d} = [d_0,d_1,\ldots,d_{r-1}],
$$

then $d_a$ is the length of axis $a$. A later activation might use named axes
such as `[batch, token, hidden]`; the names explain model meaning while the
integer dimensions define valid indices.

## Element count is checked arithmetic

The number of logical elements is the product of all dimensions:

$$
N(\mathbf{d}) = \prod_{a=0}^{r-1} d_a.
$$

For shape `[3,4]`, $N=12$. For rank 0, the empty product is one, so shape `[]`
describes one scalar. This definition is mathematically compact, but a systems
implementation must ask whether the product fits its integer type.

```rust
shape.iter().try_fold(1_usize, |count, &dimension| {
    count.checked_mul(dimension).ok_or(TensorError::ShapeOverflow)
})
```

The real implementation handles zero dimensions first, then uses checked
multiplication. It also checks the byte count:

$$
B = N(\mathbf{d}) \times \operatorname{bytes\_per\_element}(\text{dtype}).
$$

Here $B$ is bytes, not elements. For `[2,3,4]` in `f32`, $N=24$ elements and
$B=24\times4=96$ bytes. Both products can overflow independently.

> **ENGINEERING FAILURE — WRAPPED SIZE**
> External metadata claims shape `[usize::MAX,2]`. Unchecked multiplication
> wraps to a small value, code allocates too little storage, and later address
> arithmetic trusts the original dimensions. Checked size arithmetic is not
> defensive decoration; it is part of model-loading security.

## Zero-sized dimensions are deliberate

Tensor Substrate v1 permits zero-sized dimensions. Shape `[0,4]` has zero
elements and owns an empty vector. It has rank 2, canonical strides `[4,1]`,
and no valid logical index because index 0 is already outside axis 0.

This decision keeps shape algebra honest. Empty batches and empty slices can be
represented without a sentinel. A zero-element view may have its base offset
at the end of storage, because it reaches no element. It may reshape only to a
shape whose checked element count is also zero.

Allowing empty tensors does not disable all validation. Canonical stride
products still have to fit `usize`; malformed metadata cannot smuggle an
unrepresentable layout through a zero dimension.

## Dtype is interpretation, not width

A **dtype** specifies how stored bits represent values and participate in
arithmetic. Tensor Substrate v1 implements only `F32`. Each element is one
IEEE-754 binary32 value and occupies four bytes, while reductions in later
operators may choose a different accumulation type.

Other systems use `f16`, `bf16`, integers, and packed quantized formats. Two
types can occupy the same byte width yet interpret bits differently. A packed
quantized block may represent many logical weights with shared scale metadata,
so “one scalar equals one fixed byte range” eventually stops being a useful
model. Chapter 16 will face that complexity. Adding a generic dtype hierarchy
now would create machinery without a chapter requirement.

`DType::F32` is still explicit in v1. That apparently redundant name prevents
future code from treating `size_of::<f32>()` as the whole semantic contract.

## Logical values and physical storage

Consider this logical matrix:

$$
A =
\begin{bmatrix}
A & B & C \\
D & E & F
\end{bmatrix},
\qquad \operatorname{shape}(A)=[2,3].
$$

A canonical row-major representation places each row consecutively:

```text
         ┌─────┬─────┬─────┬─────┬─────┬─────┐
storage  │  A  │  B  │  C  │  D  │  E  │  F  │
         └─────┴─────┴─────┴─────┴─────┴─────┘
offset      0     1     2     3     4     5
```

The complete canonical diagram is
[logical tensor versus physical storage](../../diagrams/tensor/logical-vs-physical.txt).
The logical coordinate `[1,1]` names $E$; the layout maps it to physical offset
4. **Shape** determines whether `[1,1]` is a legal coordinate. **Strides**
determine how that coordinate moves through storage.

Column-major storage is another valid convention. It would place $A,D,B,E,C,F$
consecutively for the same logical matrix. Row-major is v1's chosen owner
layout, not a claim that one convention is universally faster.

## Strides turn indices into offsets

A stride is the distance in storage caused by incrementing one logical axis by
one. Tensor Substrate v1 measures strides in **elements**. NumPy and GGML expose
byte strides, so source code crossing those boundaries must convert units
explicitly.

Let a view have shape $\mathbf d$, strides $\mathbf s$, base element offset
$b$, and logical index $\mathbf i$. Its physical element offset is

$$
\operatorname{offset}(\mathbf i)
= b + \sum_{a=0}^{r-1} i_a s_a,
\qquad 0 \le i_a < d_a.
$$

Every symbol has a physical interpretation:

- $b$ is an element index into the borrowed slice;
- $i_a$ is a logical coordinate with no byte unit;
- $s_a$ is elements per one-axis step;
- the result is an element offset, not a pointer and not a byte address.

For contiguous row-major shape `[2,3,4]`, calculate strides from right to left:

$$
s_{r-1}=1,
\qquad
s_a=\prod_{k=a+1}^{r-1}d_k.
$$

The result is `[12,4,1]`. Index `[1,2,3]` maps to

$$
1\times12 + 2\times4 + 3\times1 = 23\text{ elements}.
$$

Because the dtype is `f32`, the illustrative byte displacement is
$23\times4=92$ bytes. See the full
[offset derivation](../../diagrams/tensor/row-major-offsets.txt).

## One authoritative checked offset

V1 sends every checked read through `checked_offset`. The operation follows a
fixed order:

1. Validate that shape and stride ranks match.
2. Validate that the metadata's reachable extent fits storage.
3. Validate that index rank equals tensor rank.
4. Validate each index against its dimension.
5. Multiply index by stride with `checked_mul`.
6. Add each contribution and the base with `checked_add`.
7. Perform the final safe slice lookup.

Validating indices before accepting their contribution matters. Computing an
offset first and checking only the final number can let a malformed coordinate
alias an apparently valid physical location. Logical bounds and physical
bounds are separate invariants.

The public API returns `Result<&f32, TensorError>` rather than implementing
Rust's `Index` trait. `Index` conventionally panics on failure; a typed result
makes model-derived metadata and indices inspectable without a process-level
surprise. `get2(row,column)` is a convenience that calls the same general path.

## Storage extent is not element count

An arbitrary strided view can touch more storage than its logical element
count suggests. Shape `[2,2]` has four logical values. With strides `[100,1]`,
the view reaches offsets 0, 1, 100, and 101. Four backing elements are nowhere
near enough.

For nonempty v1 views with non-negative strides, the largest reachable offset
is

$$
o_{\max}=b+\sum_{a=0}^{r-1}(d_a-1)s_a.
$$

The required storage length is

$$
L_{\min}=o_{\max}+1.
$$

Every multiply and add is checked. The constructor requires
$L_{\min}\le L_{\text{storage}}$. For a zero-element view there is no reachable
offset, so the rule becomes $b\le L_{\text{storage}}$.

This formula relies on non-negative strides. V1 uses `usize` and does not
support reverse traversal through negative strides. It accepts zero strides for
immutable views: several logical indices may then read the same physical value.
That is safe for reading but not canonical contiguous storage.

## Contiguous means one strict thing here

A tensor is often called contiguous when logical iteration visits a dense
region in an expected order. Mature frameworks account for memory formats and
size-one dimensions in more flexible ways. Tensor Substrate v1 intentionally
uses a narrower definition:

> A view is canonical row-major contiguous exactly when its stride vector
> equals `canonical_row_major_strides(shape)`.

Thus `[2,3] / [3,1]` is contiguous. Its transpose `[3,2] / [1,3]` is not.
Shape `[1,3,1]` has canonical strides `[3,1,1]`; a different stride on a
length-one axis is non-canonical even if it reaches the same values. This strict
rule gives Chapter 6 a simple kernel precondition. We describe it as *v1
canonical contiguity*, not universal framework semantics.

The [contiguous versus strided](../../diagrams/tensor/contiguous-vs-strided.txt)
diagram shows why a valid logical walk need not be physical-order traversal.

## A view owns metadata, not elements

`OwnedTensor` owns `Vec<f32>`, shape, and validated canonical strides.
`TensorView<'a>` owns shape/stride/base metadata but borrows `&'a [f32]`.
Creating a view may allocate those small metadata vectors; it never allocates or
copies the element payload.

```rust
pub struct TensorView<'a> {
    storage: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
    base_offset: usize,
}
```

The lifetime `'a` ties the view to valid storage. It cannot outlive its owner.
The compiler rejects returning a view after a locally created owner has been
dropped. This is not merely a language lesson: a dangling model-weight view is
a use-after-free at the heart of numerical execution.

Multiple immutable views may overlap. A source view, transpose, and row slice
can all read some of the same values. They **alias** because their reachable
storage regions overlap. The
[ownership](../../diagrams/tensor/tensor-ownership.txt) and
[lifetime](../../diagrams/tensor/tensor-memory-lifetime.txt) diagrams make those
relationships explicit.

## Copy means a new owner

A copy creates new element storage. It has a separate lifetime, separate
mutation behavior, and a memory-traffic cost proportional to copied payload.
V1 names that boundary `to_contiguous`.

| Operation | Element allocation | Element copy |
| --- | --- | --- |
| `from_vec` | performed by caller; ownership moves | no |
| `zeros` | yes | zero initialization |
| `view` / `view_mut` | no | no |
| `reshape_view` | no | no |
| `transpose` | no | no |
| `slice_axis` | no | no |
| `to_contiguous` | yes | yes |

The distinction is shown in [view versus copy](../../diagrams/tensor/view-vs-copy.txt).
Metadata allocation is intentionally separated from element allocation in this
table because model payload dominates later memory costs.

> **FIRST PRINCIPLE**
> Allocation and copying must be visible operations. A convenient-looking
> metadata transform must not secretly materialize a model-sized payload.

## Reshape changes grouping

A no-copy reshape changes how one contiguous sequence is grouped into axes. The
six physical values `A,B,C,D,E,F` can be viewed as `[2,3]`, `[3,2]`, `[6]`, or
`[1,6]`. Logical row-major iteration remains physical offsets
`0,1,2,3,4,5`.

V1's `reshape_view` requires two proofs:

1. The source has exact canonical row-major strides.
2. The requested checked element count equals the current count.

It then derives canonical strides for the new shape and borrows the same
storage. Shape `[4,2]` fails because it asks six values to become eight. A
non-contiguous transpose fails with `NonContiguous`; v1 does not copy as a
fallback.

PyTorch's `view` supports a broader compatibility condition based on adjacent
stride relationships, while `reshape` may choose a view or copy. That is useful
framework behavior. It is too implicit for our first substrate, where the
method name itself should reveal allocation.

## Transpose changes logical movement

Transpose is not reshape. For the original `[2,3]` matrix:

```text
┌───┬───┬───┐
│ A │ B │ C │
├───┼───┼───┤
│ D │ E │ F │
└───┴───┴───┘
```

the rank-2 transpose is logically:

```text
┌───┬───┐
│ A │ D │
├───┼───┤
│ B │ E │
├───┼───┤
│ C │ F │
└───┴───┘
```

No bytes move. Shape `[2,3]` becomes `[3,2]`; strides `[3,1]` become `[1,3]`.
Logical row-major traversal now visits physical offsets `[0,3,1,4,2,5]`.
The canonical [transpose view](../../diagrams/tensor/transpose-view.txt) shows
both interpretations over one owner.

> **ENGINEERING FAILURE — TRANSPOSE ASSUMED CONTIGUOUS**
> A kernel receives the transposed shape but ignores its strides. It reads
> physical offsets `[0,1,2,3,4,5]`, producing finite, plausible, wrong output.
> Layout is part of the kernel's semantics, not an optional optimization hint.

## Slices introduce a base offset

Take rows `1..3` from a canonical `[4,3]` tensor. The new shape is `[2,3]` and
the strides remain `[3,1]`, but the first view element is three elements into
the owner. Its base offset is 3.

`slice_axis(axis,start,end)` implements only bounded half-open ranges. It does
not implement Python slicing syntax, steps, reverse views, new axes, or fancy
indices. The new base is

$$
b' = b + \text{start}\times s_{\text{axis}},
$$

computed with checked arithmetic. The constructor then validates the complete
new extent. A zero-length slice may begin at the end of an axis and reach no
storage.

The representation keeps the full borrowed storage plus `base_offset` rather
than slicing the Rust reference itself. That makes the address formula and
diagnostics explicit for later model-file views. A subslice representation
could encode part of the bound in Rust's slice length, but would hide the
original storage-relative offset we want to teach.

## Mutable aliasing is a numerical hazard

Two overlapping immutable views are safe. Two overlapping writable views can
race or make one operation observe another's partial output. The result may be
nondeterministic without ever becoming NaN or crashing. Ownership is therefore
a numerical correctness concern.

V1 does not offer a constructor for arbitrary mutable strides. An owner can
produce one `TensorViewMut` over its complete canonical storage. That operation
borrows the owner exclusively. While the mutable view is live, Rust prevents
another shared or mutable view from coexisting.

```rust
let mut tensor = OwnedTensor::zeros(vec![2, 3])?;
{
    let mut writable = tensor.view_mut();
    *writable.get_mut(&[1, 2])? = 7.0;
}
assert_eq!(*tensor.view().get(&[1, 2])?, 7.0);
```

A future safe split operation could prove that two canonical ranges do not
overlap before returning two mutable borrows. Chapter 5 does not need it. It
also needs no unsafe code: `engine0` retains `#![forbid(unsafe_code)]`.

## Physical addresses, alignment, and locality

`Vec<f32>` stores initialized `f32` elements contiguously. If an illustrative
base address were `0x1000`, consecutive elements would begin at `0x1000`,
`0x1004`, `0x1008`, and so on. Real allocator addresses vary; the point is the
four-byte displacement, not those invented addresses.

The allocation is aligned suitably for `f32`. That does not promise alignment
for a future SIMD instruction, GPU buffer, stable C ABI, or direct I/O request.
Part VII will make arena and native alignment explicit.

Layout also affects locality. CPU caches fetch regions larger than one scalar,
so visiting neighboring physical elements often benefits from already-fetched
data. For a row-major `[N,N]` matrix, row-then-column iteration visits offsets
`0,1,2,...`. Column-then-row iteration visits `0,N,2N,...`, then
`1,N+1,2N+1,...`. The mathematical checksum can be identical while the memory
access pattern differs.

This is a preview, not a cache-architecture chapter. Associativity, cache
levels, hardware prefetching, NUMA, and kernel tiling remain ahead.

## Build Tensor Substrate v1

The implementation lives in `engine0::tensor`. A module is enough: a separate
crate would add a dependency boundary before another consumer exists.

```rust
pub struct OwnedTensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}
```

The owner stores canonical strides although they are derivable. Constructors
validate that redundancy once; repeated indexing and debug output can then use
the complete layout without recomputing metadata. Dynamic rank was chosen over
const generics because later operations naturally move among vectors, matrices,
and higher-rank activations. Our purpose is to inspect runtime tensor metadata,
not encode every rank into a different Rust type.

The v1 public surface fits on one conceptual page:

- construct with `from_vec` or `zeros`;
- inspect dtype, rank, shape, strides, length, and physical slice;
- create canonical immutable or exclusive mutable views;
- create a validated general immutable view from parts;
- read through `get` or `get2`;
- create rank-2 transpose, bounded slice, or canonical reshape views;
- explicitly materialize with `to_contiguous`;
- call checked count, byte-count, stride, and offset helpers.

There is no broadcasting, autograd, device method, negative stride, fancy
indexing, generic operator graph, quantized dtype, BLAS dispatch, SIMD, or GPU
path. A tensor describes data. It does not own execution.

## Migrate ENGINE-1 without changing it

Before this chapter, `TinyLanguageModel` stored embedding and projection as bare
vectors and indexed both manually. They are now immutable `OwnedTensor` values
with shape `[V,D]` and strides `[D,1]`. The scalar forward loop remains visible:

$$
z_i=b_i+\sum_{j=0}^{D-1}W_{i,j}h_j,
\qquad z\in\mathbb{R}^{V}.
$$

The model owns $W$ with shape `[V,D]`, the request owns $h$ with shape `[D]`,
and the forward result owns logits $z$ with shape `[V]`. Storage and
accumulation are `f32` in this tiny implementation. Each `W[i,j]` and embedding
element is now reached through checked `get2`; no matrix multiplication operator
has been smuggled into `Tensor`.

Bias, hidden activation, and `Logits` remain specialized one-dimensional
vectors. Wrapping every vector would increase abstraction without eliminating
an ambiguous multi-axis indexing assumption. Chapter 6 can revisit activation
representations when operators need common tensor inputs.

This refactor is our first internal substitution gate. The representation
changed; model semantics did not. The full hand oracle still gives
`[-0.7,0.1,0.4,2.2]` for token `like`, and greedy generation still emits
`Rust` followed by EOS.

## Prove the memory model independently

The plain-Python oracle does not import Rust code. It derives `[12,4,1]` for
shape `[2,3,4]`, maps `[1,2,3]` to 23, enumerates the `[2,3]` matrix at physical
offsets `[0,1,2,3,4,5]`, and proves two different `[3,2]` interpretations:

- reshape offsets: `[0,1,2,3,4,5]`;
- transpose offsets: `[0,3,1,4,2,5]`.

The transposed logical copy is exactly `[A,D,B,E,C,F]`. These are integer and
symbolic results, so no floating-point tolerance is needed.

Run it from the repository root:

```sh
python3 code/reference/python/chapter05_tensor_oracle.py
```

The Rust suite adds deterministic generated shapes of ranks one through four.
Every canonical logical index must map to its flat row-major index, and the
largest index must map to `element_count - 1`. Boundary tests exercise shape,
byte, stride, offset, and extent overflow without attempting giant allocations.

## Performance lab: same sum, different walk

The Chapter 5 harness allocates one canonical `[2048,2048]` tensor—4,194,304
`f32` values, or 16 MiB—and accumulates every value as `f64`. It alternates
row-major and column-wise traversal across seven warmed release repetitions.
Both orders produce the exact checksum `524280621.0`.

On the recorded Apple M1 system, medians were:

| Order | Median |
| --- | ---: |
| Physical row-major | 4,163,875 ns |
| Column-wise logical | 13,692,583 ns |

The durable [benchmark record](../../research/benchmarks/chapter-05-traversal-order.md)
contains the commit, command, hardware, software, shape, repetitions, checksum,
and raw result. It supports one narrow observation: on that run, different
access order had different cost. It does not establish a universal ratio,
isolate a cache level, or measure matrix multiplication or inference.

## Follow the element, byte, and owner

One logical element now has a complete route:

```text
index [1,2,3] ──▶ bounds ──▶ strides ──▶ checked offset 23 ──▶ storage[23]
```

See [Follow the Element](../../diagrams/tensor/follow-the-element.txt). The
route is where mathematical coordinates meet memory-safe code.

The byte journey begins at an owned `Vec<f32>`. Views, slices, reshapes, and
transposes change metadata while borrowing the same payload. Only
`to_contiguous` creates a new allocation and moves logical values into canonical
order. [Follow the Byte](../../diagrams/tensor/follow-the-byte.txt) records that
fork.

The ownership journey begins with `OwnedTensor`, fans out into immutable
borrowed interpretations, and either ends those views before the owner or
creates an explicit independent copy. [Follow the Owner](../../diagrams/tensor/follow-the-owner.txt)
shows the lifetime boundary.

## A complete worked view

Let us run one allocation through every v1 metadata operation. Begin with
twelve values in canonical shape `[4,3]`:

$$
X=
\begin{bmatrix}
0&1&2\\
3&4&5\\
6&7&8\\
9&10&11
\end{bmatrix}.
$$

`OwnedTensor::from_vec([4,3], data)` first calculates 12 elements and 48 bytes,
checks the data length, derives strides `[3,1]`, and takes ownership of the
caller's vector. The logical coordinate `[2,1]` passes bounds
$2<4$ and $1<3$, then maps to

$$
0 + 2\times3 + 1\times1 = 7.
$$

Storage element 7 contains value 7. No pointer arithmetic is exposed to the
caller.

Now take rows `1..3`. The slice has shape `[2,3]`, preserves strides `[3,1]`,
and moves the base to

$$
b'=0+1\times3=3.
$$

Its logical `[0,0]` reaches owner offset 3; `[1,2]` reaches
$3+1\times3+2\times1=8$. The largest reachable offset is 8, so a storage length
of at least 9 is sufficient for this view even though its owner contains 12
values. The view remains canonical under v1's stride definition: shape `[2,3]`
expects `[3,1]`. It occupies one dense subregion from offsets 3 through 8.

Transpose that slice. Shape becomes `[3,2]`, strides become `[1,3]`, and base
remains 3. Logical `[2,1]` still reaches offset

$$
3+2\times1+1\times3=8,
$$

but logical iteration visits offsets `[3,6,4,7,5,8]`. This is valid and
non-contiguous. `reshape_view([6])` refuses it because v1 cannot reinterpret
that logical walk as canonical physical order without moving values.

Finally, `to_contiguous` enumerates those six logical coordinates, reads
`[3,6,4,7,5,8]`, and creates a new `[3,2]` owner whose physical storage holds
exactly that sequence with strides `[2,1]`. Source and destination have equal
logical values but different ownership and physical order. Mutating the new
owner cannot affect $X$.

This worked path separates four statements that are easy to collapse:

- the original owner is contiguous;
- the row slice is also contiguous but begins at a nonzero base;
- the transpose is a valid non-contiguous alias;
- the materialized result is a separate contiguous owner.

No single `shape` or `element_count` field can prove all four.

## Error taxonomy is part of the API

A checked tensor API needs errors precise enough to identify which invariant
failed. V1 keeps the list small while preserving that distinction.

| Error | Rejected condition | Why it is separate |
| --- | --- | --- |
| `ShapeOverflow` | element or canonical-stride product does not fit | prevents undersized or unrepresentable owners |
| `ByteCountOverflow` | elements times dtype width does not fit | element count can fit while bytes do not |
| `StorageLengthMismatch` | owner shape and supplied vector length disagree | owner construction requires exact storage |
| `ShapeStrideRankMismatch` | view metadata vectors have different lengths | no axis-to-stride mapping exists |
| `RankMismatch` | an index has too few or too many axes | logical coordinate is malformed |
| `IndexOutOfBounds` | one coordinate is outside its dimension | logical bounds failed before addressing |
| `OffsetOverflow` | extent or index arithmetic cannot be represented | prevents wrapped physical addresses |
| `InvalidViewExtent` | reachable offsets exceed borrowed storage | arbitrary strides need more than count validation |
| `NonContiguous` | reshape source lacks canonical strides | no-copy contract cannot be met |
| `ReshapeElementMismatch` | requested count changes | reshape cannot create or discard values |
| `ExpectedMatrix` | transpose source is not rank 2 | v1 implements only the promised operation |
| `InvalidSlice` | axis or half-open range is malformed | slicing stays bounded and explicit |

The display text helps a human; callers can match the typed variant. No checked
operation converts a malformed request into a default tensor or clamps an
index. Silent recovery would turn structural corruption into plausible
numbers.

There are still internal invariants. An `OwnedTensor` has already proven that
its strides are canonical and its storage count is exact. Methods may rely on
that fact, but public inputs pass through checked construction. This boundary
lets code remain readable without repeating an entire model-file parser's
threat model at every loop iteration.

## Why the type system stops here

It is tempting to encode every fact in Rust types:

```text
Tensor<Element, Device, Layout, Rank, Shape, Allocator, Mutability, ...>
```

Such a design can be valuable in a specialized library. It would be harmful at
this point in the book. Transformer intermediates change rank and shape at
runtime. Model metadata eventually arrives from files. If the reader spends
the chapter decoding const-generic machinery, the physical memory model has
again disappeared behind an abstraction.

Dynamic `Vec<usize>` metadata makes each runtime check visible. `F32` makes one
dtype concrete. An immutable general view demonstrates real striding. A narrow
mutable view demonstrates exclusivity. This is enough structure to prevent the
current errors and enough simplicity to audit on one page.

The substrate is also intentionally not an operator object. Adding methods
such as `tensor.matmul(other)` would couple data representation to arithmetic,
temporary allocation, provider choice, and error policy before those contracts
have been derived. Chapter 6 will accept tensor views as inputs to separately
named kernels. Later a planner may choose a scalar CPU, SIMD, Metal, or CUDA
implementation while the input tensor remains data plus metadata.

## Correctness matrix

The test suite does more than exercise happy-path indexing. Each group protects
a later inference-engine boundary.

**Construction tests** cover rank 0 through rank 4, exact shape retrieval,
canonical strides, dtype width, storage mismatch, zero dimensions, element
overflow, byte overflow, and canonical-stride overflow. The overflow cases call
helpers directly near `usize::MAX`; they never attempt dangerous allocations.

**Indexing tests** cover first, middle, and final elements, the hand-computed
`[1,2,3] → 23` case, wrong-rank coordinates, per-axis bounds, base offsets,
extent overflow, and strides that require more storage than logical count. A
generated loop enumerates every coordinate for small ranks one through four and
requires the canonical physical offset sequence exactly.

**View tests** prove rank-2 transpose shape, strides, and values; compatible
reshape shapes; rejection of incompatible and non-contiguous reshape; bounded
axis slicing; zero-length slices; zero-stride immutable aliasing; and explicit
logical-order materialization. Reference identity proves a view reaches the
owner's element rather than a copied value.

**Ownership tests** mutate one complete canonical owner through
`TensorViewMut`, end the exclusive borrow, and read the new value through a
later immutable view. A separately materialized owner is mutated and the source
is proven unchanged. Compile-time borrow failures belong in Lab 21 rather than
normal test sources, because committed test targets must compile.

**Regression tests** retain every Part I contract: tokenizer boundaries, strict
UTF-8 streaming, full numerical logits, sampling distributions, seeded state,
token feedback, cancellation, EOS, budgets, failure paths, and exactly one
terminal outcome. The tensor refactor is invalid if any of those observable
semantics changes.

Together these are stronger evidence than testing one printed matrix. Shape,
layout, storage, and lifetime are orthogonal enough that each deserves its own
failure injection.

## Review from four engineering perspectives

A systems programmer should now be able to locate every allocation and copy.
The model owns parameter vectors. Views borrow them. Metadata transforms do not
move elements. `to_contiguous` does. Size, extent, and offset arithmetic are
checked at public boundaries.

An ML engineer should recognize standard rank, shape, stride, transpose, and
view concepts while seeing the declared differences from NumPy and PyTorch.
Model equations retain their shapes. No layout detail changes the mathematical
meaning of $z=Wh+b$.

A Rust engineer should find no dangling reference escape and no safe path to
arbitrary overlapping mutable strided views. Shared borrows can coexist;
exclusive mutation cannot overlap them. There is no `unsafe`, and malformed
public indexing returns typed errors.

A beginner should be able to answer the concrete question “where is `[1,2]`?”:
check two bounds, multiply each coordinate by its element stride, add the base,
then read that physical element. The term *tensor* now expands into inspectable
data instead of shrinking into framework vocabulary.

## Inside Hermon

The following findings are pinned to Hermon commit `472a44c` and its llama.cpp
submodule `389ff61d`; they are not claims about an uninspected future revision.

> **INSIDE HERMON — CURRENT**
> `hermon-llamacpp` is the project's single allowed unsafe crate. Its safe
> `Model::tensor_info` copies a tensor's GGML dimensions, rank, and type across
> the FFI boundary. `tensor_row_f32` borrows the model handle for the call and
> returns one newly owned decoded row. The shim validates tensor existence,
> shape, host residency, layout, conversion support, and caller output length.

The loaded model owns packed tensor lifetime; Rust callers do not receive a raw
borrow that can outlive it. Unsafe handle access is localized and each wrapper
block states its invariant. `TensorSession` similarly owns reusable mutable
GGML scratch state, is movable between threads, but is deliberately not `Sync`;
entry points require exclusive `&mut` access.

> **INSIDE HERMON — LIBRARY**
> `hermon-gguf` exposes names, dimensions, per-tensor serialized type, offsets,
> and checked byte lengths. It can open an exactly bounded reader or read a
> checked subrange without materializing the whole tensor. That proves parser
> behavior, not that the default inference path uses Hermon-native tensors.

> **INSIDE HERMON — PREVIEW**
> The gated paged GGUF runtime combines validated metadata with the llama.cpp
> packed tensor bridge. It keeps a `TensorSession` behind a mutex for exclusive
> graph/scratch use. The default production batched path remains llama.cpp
> managed, so PREVIEW tensor plumbing must not be flattened into a CURRENT
> end-to-end claim.

Pinned GGML reinforces the same lesson. Its `ggml_tensor` records type,
dimensions `ne`, byte strides `nb`, storage/buffer information, and view/source
metadata because operations cannot assume contiguity. GGML fixes a maximum rank
and has packed dtype concerns that v1 excludes. The shared principle is not API
identity: kernels need shape, layout, type, and lifetime facts together.

## Common mistakes

### “A tensor is a GPU object”

A tensor is a data/layout abstraction. V1 is entirely CPU-local. A later
provider may place tensor storage on a GPU, but device placement is not the
definition.

### “Tensor means matrix”

A matrix is the rank-2 case. Scalars, vectors, and higher-rank arrays are also
tensors under this book's definition.

### “Rank tells me linear independence”

Tensor rank counts axes. Matrix rank measures a different algebraic property.

### “Shape defines layout”

Shape defines logical dimensions. Strides and base offset map those dimensions
to storage. Equal shape does not imply equal storage order.

### “The element count validates a view”

Four strided elements can reach offset 101. Required extent, not count alone,
must fit storage.

### “Strides are always bytes”

Their unit is an API contract. V1 uses elements; NumPy and GGML expose bytes.
Mixing the two multiplies every address error by dtype width.

### “Reshape and transpose are the same”

Reshape regroups canonical order. Transpose changes which axis step moves where
and often becomes non-contiguous.

### “Transpose always copies”

V1 transpose only swaps metadata. An explicit `to_contiguous` performs a copy.

### “Views own their data”

A `TensorView<'a>` owns metadata and borrows storage. It cannot outlive the
owner. Two views may alias.

### “A tensor should perform its own matmul”

Tensor is data plus metadata. A matrix multiplication kernel is an operation
with layout, arithmetic, and execution contracts. Chapter 6 keeps them
separate.

## Labs

Labs 16–21 turn each contract into evidence:

- [Lab 16](../../labs/lab-16-offset-by-hand.md) derives offsets by hand.
- [Lab 17](../../labs/lab-17-transpose-without-copy.md) proves transpose aliases.
- [Lab 18](../../labs/lab-18-reshape-view.md) checks no-copy reshape gates.
- [Lab 19](../../labs/lab-19-non-contiguous-copy.md) materializes logical order.
- [Lab 20](../../labs/lab-20-break-shape-arithmetic.md) attacks size and extent arithmetic.
- [Lab 21](../../labs/lab-21-mutation-and-aliasing.md) uses exclusive mutation.

Each moves through CHECK, BUILD, BREAK, or EXTEND without adding later
Transformer operators.

## Exercises

1. Derive strides for `[5]`, `[2,5]`, `[3,2,5]`, and `[1,3,1]`. State units.
2. For shape `[4,3]`, slice rows `1..3`. Derive shape, strides, base, and largest
   reachable offset.
3. Give two different stride vectors for shape `[2,2]` over six storage values.
   List the logical visit order of each.
4. Explain why a zero-stride immutable view is safe to read but cannot be the
   basis of an arbitrary overlapping mutable API.
5. Calculate the minimum storage length for shape `[2,2]`, strides `[100,1]`,
   base 7. Identify every overflow check the implementation needs.
6. Compare `[3,2]` reshape and transpose views of `[A,B,C,D,E,F]`. Which logical
   index first reveals that they differ?
7. Explain why `from_vec` moves an allocation while `to_contiguous` copies
   elements, even though both return an owner.
8. Design a typed error for a kernel that accepts only canonical rank-2 input.
   Which checks belong to the tensor, and which belong to the kernel?
9. Sketch a safe API that splits a canonical matrix into two non-overlapping
   mutable row ranges. State the proof required before returning borrows.
10. Predict how element-stride APIs must adapt at an FFI boundary exposing byte
    strides and a packed quantized dtype.

## What we still have not built

Tensor Substrate v1 can describe and safely access data. It cannot multiply two
matrices, normalize an activation, form Q/K/V, encode position, apply causal
attention, run a feed-forward network, or stack decoder layers. It has no
broadcasting, training graph, packed dtype, model loader, SIMD kernel, or
accelerator provider.

Those exclusions are the point. A small substrate lets us audit every invariant
before performance work raises the stakes. We can now distinguish a data
structure error from an operator error and a mathematical result from its
physical traversal cost.

## Summary

A tensor is not a vector with a suggestive variable name. It is storage plus
shape, layout, dtype, ownership, and lifetime. Rank counts logical axes. Shape
defines their dimensions. Checked products establish element and byte counts.
Strides and a base offset map bounded logical indices to physical elements.

Owned tensors are canonical row-major `f32` storage in v1. Immutable views may
carry general non-negative strides and alias their owner. Reshape borrows only
canonical layouts with equal element counts. Transpose swaps rank-2 metadata.
Slices adjust one dimension and base. `to_contiguous` is the visible allocation
and copy boundary. Rust lifetimes prevent dangling views; exclusive borrowing
keeps arbitrary mutable aliasing out of the API. Every external size and offset
operation is checked.

ENGINE-1 now stores its embedding and projection as tensors, yet produces the
same logits and tokens. Representation changed without changing model
semantics. That is the substitution discipline the rest of the book will need.

Return to the five opening questions. For ENGINE-1's embedding, the answers are
now executable facts: shape `[V,D]`; dtype `F32`; canonical row-major strides
`[D,1]`; storage owned immutably by `TinyLanguageModel`; and views bounded by a
borrow of that model. For a transposed temporary, the shape and strides change,
the owner does not, and the borrow cannot escape. For a contiguous copy, the
logical values remain equal while ownership and physical ordering change.

That checklist scales. When later chapters introduce activations, packed model
weights, KV blocks, and device buffers, each new representation must answer the
same questions before a kernel touches it. A familiar equation is not a waiver.
Neither is a familiar framework name. Correct inference begins by making the
data contract complete enough that wrong layout, wrong lifetime, or wrong size
cannot masquerade as valid arithmetic.

## Chapter 6 preview: matrix multiplication

We can represent matrices correctly. We still need to compute

$$
C = AB
$$

without treating the equation as an execution plan. It does not say where
$A$, $B$, and $C$ live, which dimensions agree, which index changes fastest,
how many floating-point operations occur, how many bytes move, or whether
reuse fits a cache.

Chapter 6—*Matrix Multiplication: The Engine Room*—will derive dot product,
matrix-vector multiplication, matrix-matrix multiplication, loop ordering,
operation count, memory traffic, arithmetic intensity, tiling, reference
kernels, and blocked scalar kernels on top of Tensor Substrate v1. It will be
the first chapter that deliberately thinks like a performance engineer.

## References

- NumPy developers. [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
  and [`ndarray.strides`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html).
- PyTorch contributors. [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html),
  [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html),
  and [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html).
- The Rust Project. [Slice types](https://doc.rust-lang.org/stable/reference/types/slice.html)
  and [References and Borrowing](https://doc.rust-lang.org/stable/book/ch04-02-references-and-borrowing.html).
- The Rust Project. [`usize::checked_mul`](https://doc.rust-lang.org/std/primitive.usize.html#method.checked_mul).
- ggml-org. Pinned [`ggml.h`](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/include/ggml.h).
- Hermon source at commit
  [`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25),
  especially `hermon-llamacpp`, `hermon-gguf`, and the paged runtime boundary.
