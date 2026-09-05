# Chapter 7 — Embeddings and Normalization

Chapter 2 ended with integers. A tokenizer transformed UTF-8 bytes into token
identities under a model-specific vocabulary contract. Chapter 3 used one of
those identities to select a hidden vector, but its purpose was to make an
entire language-model forward pass small enough to calculate by hand. Now we
can inspect that first numerical step as an engine operation in its own right.

A token ID is not a small activation. Multiplying the integer `128` by a model
weight does not recover what token 128 “means.” The integer is an addressable
symbol. The model's **embedding table** stores one learned floating-point row
for each symbol. A checked lookup crosses from discrete token space into the
continuous model space in which the network computes.

Once there, scale matters. A vector and a hundred-times-larger copy point in
the same direction, but their components drive later arithmetic at very
different magnitudes. Transformer architectures commonly place normalization
around learned transformations to control that scale. This chapter builds
**RMSNorm** from squares, a mean, a square root, a reciprocal, and a learned
element-wise scale. There is no hidden framework call.

These operators are short. Their contracts are not. We need to know which axes
mean vocabulary and model width, which bytes a row uses, whether the result
aliases weights, what happens to an empty dimension, where epsilon appears,
which precision performs the reduction, and what finite values can still
overflow. Those decisions become assumptions of every later Transformer
chapter.

> **FIRST PRINCIPLE**
> Tokenization chooses a discrete model symbol. Embedding lookup materializes
> that symbol's learned model-space vector. Normalization then changes the
> vector's scale under an explicit numerical contract; it does not invent its
> meaning or clip its coordinates.

## The first learned vector

The complete boundary is visible in the
[token-to-model-space diagram](../../diagrams/transformer/token-to-model-space.txt):

```text
┌──────────────┐    encode     ┌──────────────┐    row lookup    (E [V,D])
│ UTF-8 text   │ ────────────▶ │ TokenId t    │ ──────────────▶ row t
└──────────────┘               └──────────────┘                       │
                                                                   copy
                                                                     ▼
                                                            [activation x [D]]
```

The tokenizer owns text segmentation and the mapping from pieces to integer
IDs. The numerical model owns the embedding table. An ID has meaning only
relative to that pair of artifacts: the same integer can select an unrelated
row in a different model.

Let $V_{\mathrm{vocab}}$ be the vocabulary size and $D$ be the model or
embedding dimension. The table is a learned parameter matrix

$$
\mathbf{E}\in\mathbb{R}^{V_{\mathrm{vocab}}\times D}.
$$

For a valid token ID $t$, lookup produces

$$
\mathbf{x}=\mathbf{E}_{t,:}\in\mathbb{R}^{D},
\qquad 0\le t<V_{\mathrm{vocab}}.
$$

The colon means every column in row $t$. Written element by element,

$$
x_j=E_{t,j},
\qquad 0\le j<D.
$$

This equation specifies selection, not arithmetic over every table row. A
one-hot vector times $\mathbf{E}$ is a mathematically equivalent description,
but constructing that mostly-zero vector and performing a dense product would
be a perverse implementation. The engine already has the row index.

The output is the first **residual activation**. In a decoder, the **residual
stream** is the persistent model-width state carried from one Transformer
sublayer to the next. Chapter 7 establishes its width and normalization; it
does not yet build any of the transformations that consume it.

## What is physically stored?

The table contains floating-point parameters learned during training. It does
not contain source strings, dictionary definitions, or human-readable facts.
Rows can acquire useful geometric relationships, but an inference engine needs
no story about their semantics to execute lookup. It needs a tensor with valid
metadata and a token in range.

The [logical layout](../../diagrams/transformer/embedding-logical-layout.txt)
separates the two axes:

- axis 0 has length $V_{\mathrm{vocab}}$ and is selected by token identity;
- axis 1 has length $D$ and becomes the model-space vector.

That is why the semantic shape is `[V,D]`, rather than `[D,V]`. The convention
matches the operator: selecting `table[t,:]` returns `D` values. Another system
can store axes differently, but then its metadata and kernel must make that
orientation explicit.

In Tensor Substrate v1, an `OwnedTensor` uses canonical row-major storage. An
embedding owner therefore reports

```text
shape   = [V,D]
strides = [D,1]        // measured in elements
dtype   = f32
```

For base offset zero, the canonical physical element offset is

$$
o(t,j)=tD+j.
$$

Because one `f32` occupies four bytes, the corresponding byte displacement
from the allocation's first element is

$$
b(t,j)=4(tD+j)\quad[\mathrm{bytes}].
$$

Those equations describe the canonical case. The reference operator accepts a
general validated `TensorView`, so its actual logical mapping remains Chapter
5's stride equation:

$$
o(t,j)=b_0+t s_0+j s_1,
$$

where $b_0$ is the view's base element offset and $s_0,s_1$ are nonnegative
element strides. The [physical-layout diagram](../../diagrams/transformer/embedding-physical-layout.txt)
puts the canonical shortcut beside the general rule. Code calls checked
`get2(t,j)` instead of reconstructing either formula in the operator. That
keeps extent validation and logical addressing in the tensor substrate.

For a canonical table, one selected row is physically contiguous: reading
columns `0..D` walks adjacent `f32` values. A strided table can place padding
or another logical arrangement between columns. A zero-stride view can repeat
one storage value across an axis. Tensor Substrate v1 admits those immutable
views, so the broad reference operator defines their behavior instead of
quietly assuming stride one.

## The checked lookup contract

`embedding_lookup_reference` accepts an immutable table view and a `TokenId`.
It requires:

1. table rank exactly two;
2. vocabulary size $V_{\mathrm{vocab}}>0$;
3. model dimension $D>0$;
4. token ID in the half-open interval $[0,V_{\mathrm{vocab}})$;
5. a view whose reachable extent was already validated by Tensor Substrate v1.

It returns a canonical `OwnedTensor` of shape `[D]`. Wrong rank, empty axes,
and an invalid ID produce structured `EmbeddingError` variants. A token equal
to $V_{\mathrm{vocab}}$ is already out of range; the last valid ID is
$V_{\mathrm{vocab}}-1$. Validation precedes element reads, so ordinary bad
input cannot become a panic or a speculative out-of-bounds access.

The core loop is deliberately unsurprising:

```rust
let mut output = Vec::with_capacity(model_dimension);
for column in 0..model_dimension {
    output.push(*table.get2(row, column)?);
}
OwnedTensor::from_vec(vec![model_dimension], output)
```

The important code is also around the loop: rank and dimension checks, checked
conversion from `TokenId`, output shape construction, and the owned return
type. A five-line numerical body does not imply a five-line operator contract.

> **BUILD IT**
> Use [Lab 30](../../labs/lab-30-inspect-embedding-row.md) to calculate
> canonical and strided offsets, then [Lab 31](../../labs/lab-31-checked-embedding-lookup.md)
> to audit every validation boundary. An off-by-one comparison must fail at
> token `V`, not during a later storage access.

## View or copy?

The table already owns the selected values. Why allocate another vector?

One possible interface returns a `TensorView<'weights>` of the row. That avoids
copying $D$ elements. It also aliases model parameters and carries a lifetime
tied to their owner. An immutable consumer could use it safely, but a residual
stream is not just an immutable observation: later sublayers produce additions
and other request-local state. If code expects to mutate the row as an
activation, aliasing model weights would be catastrophic.

The selected Chapter 7 policy is therefore explicit materialization:

```text
immutable parameter view ──▶ checked lookup and copy ──▶ owned activation
```

The fresh owner costs one read and one write per element. In the simplest
cold-payload accounting, a canonical `f32` lookup moves approximately

$$
Q_{\mathrm{lookup}}\approx 8D\quad[\mathrm{bytes}],
$$

counting $4D$ input bytes and $4D$ output bytes. This is not measured memory
traffic: cache lines, write allocation, allocator metadata, and warm data can
change physical transfers. It is a useful statement of the policy's payload
cost.

In exchange, downstream mutation cannot alter $\mathbf{E}$, the activation can
outlive the lookup borrow, and request cleanup has one obvious owner. The
[view-versus-copy diagram](../../diagrams/transformer/embedding-view-vs-copy.txt)
records both options rather than presenting the copy as free.

A database-table analogy can help for one moment: the embedding table is
long-lived storage, and an ID chooses a row. The analogy stops there. These are
learned numerical weights, not relational records; there is no SQL predicate,
transaction, schema evolution, or human field meaning. The actual mechanism
remains tensor indexing and copying.

> **PROVE IT**
> [Lab 32](../../labs/lab-32-embedding-view-vs-copy.md) mutates the returned
> `OwnedTensor` and verifies that the original table is unchanged. Ownership is
> tested behavior, not a comment.

## From one token to a sequence

Inference begins with more than one prompt token even though later generation
produces one new token at a time. The single-row rule generalizes without
introducing attention.

For the token sequence

$$
\mathbf{t}=[t_0,t_1,\ldots,t_{T-1}],
$$

sequence embedding produces

$$
\mathbf{X}\in\mathbb{R}^{T\times D},
\qquad X_{ij}=E_{t_i,j},
$$

for $0\le i<T$ and $0\le j<D$. The shapes are:

```text
token identities   [T]
embedding table    [V,D]
result             [T,D]
```

`embedding_sequence_reference` validates the same positive table dimensions,
checks prospective output element count $TD$, then checks each token before
copying its logical row. It preserves order and repetition. Tokens `[3,0,3]`
produce rows `[E3,E0,E3]`; it does not deduplicate them or sort for locality.

An empty token sequence has $T=0$ and returns an owned tensor with shape
`[0,D]`. That result has a well-defined shape and no values. This does not make
an empty model vocabulary valid: $V=0$ leaves no token identity selectable,
and $D=0$ would not create a model-space vector. The axes have independent
contracts.

The implementation is intentionally only a gather into `[T,D]`. It does not
create batches, positions, masks, Q/K/V, or a Transformer block. Those concepts
need separate semantic and ownership work.

## The residual-stream invariant

Once lookup produces $\mathbf{x}_0\in\mathbb{R}^{D}$ for a token, $D$ becomes
one of the decoder's central dimensional invariants. Transformer sublayers can
create internal tensors with other shapes, but a value added back into the
residual stream must return to model width $D$.

The [residual-width diagram](../../diagrams/transformer/residual-stream-width.txt)
states that bounded claim. “Persistent” means the state is carried through the
layer sequence for this request, not that it has model lifetime or durable
storage. It is activation data. The embedding table and normalization weights
are parameters; the residual vectors are request-owned values.

This distinction is summarized in the
[parameter/activation diagram](../../diagrams/transformer/parameters-vs-activations.txt):

```text
MODEL LIFETIME                          REQUEST LIFETIME
(embedding E [V,D]) ── read/copy ──▶   [residual x [D]]
(RMSNorm w [D]) ────── read ───────▶   [normalized y [D]]
```

Inference does not update the learned parameters. Multiple requests may read
them concurrently. Each request owns the activations it creates and eventually
releases them. Later chapters will give that distinction consequences for
loading, placement, caches, and scheduling; here it already prevents parameter
aliasing.

## Why scale needs a contract

Consider two vectors:

$$
\mathbf{x}_1=[1,2,-1,2],
\qquad
\mathbf{x}_2=[10,20,-10,20].
$$

The second is $10\mathbf{x}_1$. Their coordinate pattern and direction agree,
but every component of the second is ten times larger. Repeated learned
transformations and residual additions can produce activation scales that vary
across layers and inputs. Later matrix products multiply those values by many
weights, so scale affects representable range and numerical behavior.

Normalization provides a defined rescaling based on a vector statistic. It is
not clipping. A clipping operator might independently replace every value
above a threshold with that threshold, changing coordinates selectively.
RMSNorm computes one scalar from the entire vector, scales every coordinate by
that shared value, and then applies learned per-coordinate weights. Large
components remain large relative to small components unless learned scale
changes that relation.

The original RMSNorm paper by Biao Zhang and Rico Sennrich proposed removing
LayerNorm's recentering step while retaining RMS-based rescaling. That history
is useful, but the inference engine still needs the exact forward equation.

## Root mean square from first principles

Let

$$
\mathbf{x}=[x_0,x_1,\ldots,x_{D-1}]\in\mathbb{R}^{D},
\qquad D>0.
$$

First square each component. Squaring makes every contribution nonnegative,
so opposite signs cannot cancel. Next take the arithmetic mean:

$$
m_2(\mathbf{x})=\frac{1}{D}\sum_{i=0}^{D-1}x_i^2.
$$

The quantity $m_2$ has squared units relative to $\mathbf{x}$. Taking the
square root restores the original units:

$$
\operatorname{RMS}(\mathbf{x})
=\sqrt{\frac{1}{D}\sum_{i=0}^{D-1}x_i^2}.
$$

That sequence—square, mean, root—is the name **root mean square**. The
[RMS pipeline](../../diagrams/transformer/rms-calculation-pipeline.txt) makes
each lowering step visible.

Why take a mean rather than only a sum? If a vector repeated the same magnitude
at every coordinate, the sum of squares would grow with $D$. Dividing by $D$
makes its RMS equal that coordinate magnitude, independent of vector length.
For `[4,4,4,4]`, RMS is 4, not 8.

RMS alone has a zero-denominator problem. For the zero vector, every square and
their mean are zero. The reciprocal required for normalization would be
undefined. RMSNorm therefore defines an epsilon-stabilized denominator. In
this book and the checked operator, epsilon is *inside* the square root:

$$
r=\frac{1}{\sqrt{m_2(\mathbf{x})+\epsilon}},
\qquad \epsilon>0.
$$

The expression

$$
\frac{1}{\sqrt{m_2(\mathbf{x})}+\epsilon}
$$

is a different function. It cannot be substituted just because both contain a
square root and epsilon. PyTorch's official RMSNorm semantics and the inspected
Hermon/llama.cpp paths use the first convention.

## The RMSNorm operator

RMS rescaling alone would apply scalar $r$ to every coordinate. RMSNorm also
has a learned scale vector

$$
\mathbf{w}\in\mathbb{R}^{D}.
$$

The complete operator is

$$
\boxed{
y_i=x_i
\left(
\frac{1}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1}x_j^2+\epsilon}}
\right)
w_i
},
\qquad 0\le i<D.
$$

Equivalently, if $r$ names the scalar reciprocal RMS,

$$
\mathbf{y}=r(\mathbf{x}\odot\mathbf{w}),
\qquad
\mathbf{x},\mathbf{w},\mathbf{y}\in\mathbb{R}^{D}.
$$

The operator has two different kinds of multiplication. The reciprocal RMS
$r$ is one scalar derived from all of $\mathbf{x}$. The learned scale
$\mathbf{w}$ is element-wise: coordinate $i$ uses $w_i$. The symbol $\odot$
denotes that element-wise product, not a dot product.

Learned scale lets the trained model adjust normalized coordinates instead of
forcing every feature to use identical output scale. During inference,
$\mathbf{w}$ is immutable parameter data. No learning occurs inside this
function.

### A complete calculation by hand

Use

$$
\mathbf{x}=[1,-2,3,-4],qquad
\mathbf{w}=[1,0.5,2,-1],qquad
\epsilon=10^{-5}.
$$

The squares sum to

$$
1^2+(-2)^2+3^2+(-4)^2=1+4+9+16=30.
$$

With $D=4$,

$$
m_2=\frac{30}{4}=7.5,
$$

and

$$
r=\frac{1}{\sqrt{7.5+10^{-5}}}
\approx0.36514813.
$$

Applying both scales gives

$$
\begin{aligned}
y_0&= 1(r)(1)       &&\approx 0.36514813,\\
y_1&=-2(r)(0.5)     &&\approx-0.36514813,\\
y_2&= 3(r)(2)       &&\approx 2.1908888,\\
y_3&=-4(r)(-1)      &&\approx 1.4605925.
\end{aligned}
$$

The independent Python oracle and a focused Rust test calculate the entire
vector. The printed decimals are evidence from those programs, not unchecked
mental arithmetic.

> **BUILD IT**
> Complete [Lab 33](../../labs/lab-33-compute-rms-by-hand.md). Keep exact
> symbolic steps separate from approximate decimal output, then use
> [Lab 34](../../labs/lab-34-implement-rmsnorm.md) to trace those same stages
> through the reference operator.

## Equation to two loops

The direct lowering naturally has two logical passes:

```text
sum_squares = 0
for i in 0..D:
    sum_squares += x[i] * x[i]

mean_square = sum_squares / D
inverse_rms = 1 / sqrt(mean_square + epsilon)

for i in 0..D:
    y[i] = x[i] * inverse_rms * weight[i]
```

Pass one reduces $D$ values to the scalar $r$. Pass two broadcasts that scalar
while reading the input again and reading learned weights. The output for
coordinate zero cannot be finalized before pass one has seen the last input,
because the last square contributes to its denominator.

The [two-pass diagram](../../diagrams/transformer/rmsnorm-two-pass.txt) follows
the data, and the [equation-to-loop diagram](../../diagrams/transformer/equation-to-loop.txt)
connects symbols, tensor shapes, checked logical indices, and execution. The
reference implementation follows it closely:

```rust
let mut sum_squares = 0.0_f32;
for index in 0..dimension {
    let value = *input.get(&[index])?;
    let square = value * value;
    sum_squares += square;
}

let mean_square = sum_squares / dimension as f32;
let inverse_rms = 1.0_f32 / (mean_square + epsilon).sqrt();

for index in 0..dimension {
    output.push(*input.get(&[index])? * inverse_rms * *weight.get(&[index])?);
}
```

The source contains finite-value checks omitted from this condensed excerpt.
Those checks make the simple algorithm fail explicitly when its arithmetic
leaves the declared domain.

## Shape, layout, and ownership contracts

`rms_norm_reference` accepts two immutable `TensorView` values and a scalar
epsilon. Its complete preconditions are:

- input rank is one;
- learned-scale rank is one;
- their lengths agree;
- $D>0$;
- epsilon is finite and greater than zero;
- every logical input and weight value is finite.

The function accepts any valid nonnegative-stride rank-one view. A stride-two
input reads every other physical element. A zero-stride input repeats one
physical value $D$ times; a zero-stride weight repeats one learned scale. That
behavior is safe because views are immutable and the tensor constructor has
proved their reachable extent. The operator does not call `to_contiguous` and
does not disguise a layout repair as numerical work.

The output is a new canonical `[D]` `OwnedTensor`. It aliases neither input nor
weight. That policy matches embedding output: parameter owners are read-only,
and new activation data has request-local ownership.

Wrong rank produces `RankMismatch` naming the operand. Unequal lengths produce
`LengthMismatch` with both sizes. Empty dimension and invalid epsilon have
dedicated variants. `TensorError` remains wrapped for substrate failures.
There are also structured numerical errors for non-finite input, square,
accumulated reduction, inverse RMS, and output.

Ordinary invalid operator inputs do not panic. Resource exhaustion remains a
different boundary: after checked element count, a real allocator failure is
not converted into a tensor-shape error by this small teaching API.

## Epsilon is semantic metadata

For $\mathbf{x}=\mathbf{0}$ and positive finite epsilon,

$$
r=\frac{1}{\sqrt{\epsilon}}
$$

is finite. Each output still equals zero because $x_i=0$. The
[epsilon diagram](../../diagrams/transformer/epsilon-zero-vector.txt) shows the
valid and rejected paths.

The Chapter 7 API rejects:

- `epsilon == 0`, because the zero vector would have an undefined reciprocal;
- `epsilon < 0`, because the square-root input can become negative;
- NaN, because it poisons comparisons and arithmetic;
- positive or negative infinity, because the operator requires finite
  configuration.

Why not accept zero for nonzero vectors? An operator contract must cover every
valid input, not only the vector a caller happens to provide today. Positive
epsilon makes zero-vector behavior defined. Requiring positivity at entry is
more predictable than conditionally accepting a configuration based on input
values.

The number itself is model configuration, not a universal constant. The
educational examples use `1e-5`; current model families can specify other
values. PyTorch even defines a dtype-based default when its `eps` argument is
omitted. An inference runtime loading trained weights must use the model's
declared convention and value. Hermon's paged preview reads
`llama.attention.layer_norm_rms_epsilon` metadata, defaults only under its
current code path, and rejects nonpositive or non-finite results at model
construction. That is validation at a different lifecycle stage, not a license
to ignore epsilon.

> **ENGINEERING FAILURE**
> Treating epsilon as an arbitrary “small number” can create a numerically
> different model. Moving it outside the square root, changing it during a
> port, or accepting NaN metadata violates the operator contract even if many
> ordinary vectors produce superficially similar output.

Use [Lab 35](../../labs/lab-35-break-epsilon.md) to make every invalid case and
the zero-vector result executable.

## RMSNorm is not LayerNorm

Readers will encounter both names. They are related normalization operators,
not spelling variants.

LayerNorm computes a mean

$$
\mu=\frac{1}{D}\sum_{i=0}^{D-1}x_i,
$$

centers the vector with $x_i-\mu$, and scales by a variance-like statistic
before learned affine parameters. RMSNorm omits that recentering step. It uses
the root mean square of the uncentered coordinates and a learned scale.

The [LayerNorm/RMSNorm diagram](../../diagrams/transformer/layernorm-vs-rmsnorm.txt)
shows the difference in one glance. Removing the mean and subtraction changes
both mathematical invariances and execution. It is not correct to explain
RMSNorm as “LayerNorm but faster” without first naming the missing operation
and without measurement. The original RMSNorm paper reports experimental
efficiency and task results for its studied systems; that evidence does not
predict a universal inference-kernel speed ratio on every engine.

Chapter 7 stops at this contrast. Batch normalization, group normalization,
partial RMSNorm, training gradients, and convergence behavior are outside its
inference boundary.

## Precision is part of the operator

Real-number notation has unlimited range and precision. The Rust operator does
not. Transformer Primitives v1 makes one simple choice:

| Stage | Educational dtype |
| --- | --- |
| input activation storage | `f32` |
| learned scale storage | `f32` |
| `x*x` product | `f32` |
| sum-of-squares accumulator | `f32` |
| mean, square root, reciprocal | `f32` |
| element-wise output product | `f32` |
| output storage | `f32` |

The [precision-flow diagram](../../diagrams/transformer/normalization-precision-flow.txt)
places the finite-range gates along that path. There is no implicit promotion
to `f64` and no mixed-precision type machinery.

Production execution can store activations or weights in FP16 or BF16 while
using wider arithmetic for sensitive reductions. Meta's published Llama 4
reference, for example, casts the normalization input to float for square,
mean, and reciprocal-square-root work, then casts the normalized value back
before applying the learned weight. This is one source-verified model
implementation, not a claim that every backend follows the same sequence.
Backend kernels can fuse stages while preserving the model's accepted
precision contract.

The reduction order matters too. In finite precision,

$$
(a+b)+c
$$

can round differently from

$$
a+(b+c).
$$

The scalar reference visits logical indices in increasing order. A future SIMD
tree, threaded partition, or GPU reduction can combine partial sums in another
order. Equivalent real-number equations therefore need a numerical agreement
policy: finite outputs are compared with an absolute-plus-relative tolerance,
while NaN or infinity cannot pass merely because their neighboring values are
close. Bit identity remains appropriate for row selection and exact zero
fixtures, not as a universal cross-backend RMSNorm rule.

## Finite input can still fail

Checking `x.is_finite()` is necessary but insufficient. Squaring expands the
exponent. A finite `f32` around `1e20` has a mathematical square around `1e40`,
which exceeds the maximum finite binary32 magnitude. `x*x` becomes infinity.
Even when every individual square is finite, their sum can overflow.

At the other end, very small squares can become subnormal or underflow to zero.
For tiny activations, epsilon can dominate the mean square. The operator then
behaves approximately like

$$
y_i\approx\frac{x_iw_i}{\sqrt{\epsilon}},
$$

not like an exactly scale-invariant map.

The reference implementation remains the transparent naïve reduction. It
checks each square and the running sum and returns `NonFiniteSquare` or
`NonFiniteReduction` instead of silently continuing. Detection does not expand
the algorithm's numerical range; it makes the limit observable.

Stable norm algorithms exist. Netlib's `SLASSQ`, for example, represents a sum
of squares using a scale and a scaled sum so intermediate values can avoid
many overflow and underflow cases. That is valuable evidence and a possible
future candidate. It does not belong in the active Chapter 7 reference path:
the more complex invariant would obscure the direct equation, and there is no
measured model workload demanding substitution yet.

> **PROVE IT**
> [Lab 37](../../labs/lab-37-rmsnorm-magnitude-stress.md) distinguishes finite
> input from finite intermediate arithmetic. Remove the checks only in a
> disposable change and watch infinity turn a normalization into a misleading
> zero or NaN.

## Scale experiment: where epsilon becomes visible

Without epsilon and under exact real arithmetic, a positive scalar $\alpha$
cancels:

$$
\frac{\alpha\mathbf{x}}
{\sqrt{\operatorname{mean}((\alpha\mathbf{x})^2)}}
=
\frac{\mathbf{x}}
{\sqrt{\operatorname{mean}(\mathbf{x}^2)}}.
$$

With fixed positive epsilon, the denominator becomes

$$
\sqrt{\alpha^2m_2(\mathbf{x})+\epsilon},
$$

so exact cancellation no longer holds. It remains a good approximation when
$\alpha^2m_2\gg\epsilon$.

The committed Rust example and independent Python oracle use the hand vector,
learned scale, and `epsilon=1e-5`. Relative to `alpha=1`, the Python oracle
records:

| Positive scale $\alpha$ | Maximum absolute output difference |
| ---: | ---: |
| `1e-8` | `2.1908698` |
| `0.1` | `0.000144584` |
| `1` | `0` |
| `10` | `0.000001446` |
| `100` | `0.000001460` |

The tiny-scale counterexample matters more than the close large-scale rows. A
blanket sentence “RMSNorm is scale invariant” would be false for the operator
we actually execute. [Lab 36](../../labs/lab-36-rmsnorm-scale-experiment.md)
reproduces the sweep and asks where the epsilon crossover lies.

The magnitude sweep adds another view:

| Input magnitude | Teaching `f32` observation with alternating signs |
| ---: | --- |
| `1e-20` | finite subnormal square; output scale dominated by epsilon |
| `1e-10` | finite square; epsilon still dominates |
| `1` | output approximately `±0.999995` |
| `1e10` | finite square and output `±1` at printed precision |
| `1e20` | typed square-overflow error |

Four values of `1e19` separately square to finite values but overflow their
running `f32` sum, exercising a different error. These are bounded numerical
experiments, not a claim that real trained activations span this entire range.

## Work and memory behavior

Embedding lookup and RMSNorm differ from Chapter 6's matrix products.

A single embedding lookup selects one row, checks addresses, reads $D$ values,
and under our policy writes $D$ values. It performs almost no floating-point
arithmetic. Vocabulary size controls parameter-table capacity and ID bounds,
but a single valid lookup does not scan all $V_{\mathrm{vocab}}$ rows.

The RMSNorm reference performs a reduction and an element-wise pass. Under a
simple cold-payload `f32` model it:

1. reads input once for the reduction: $4D$ bytes;
2. reads input again for scaling: $4D$ bytes;
3. reads learned weight: $4D$ bytes;
4. writes output: $4D$ bytes.

Thus

$$
Q_{\mathrm{RMSNorm}}\approx16D\quad[\mathrm{bytes}].
$$

This estimate excludes tensor metadata, output allocation overhead, cache-line
effects, write allocation, and reuse from nearby operations. The computation
does roughly linear work in $D$, including a square and sum per input and two
multiplications per output, plus division, square root, and reciprocal work.
Assigning one universal FLOP count to square root or reciprocal is not useful
here.

The structural conclusion is narrower: this operation exposes much less data
reuse than a large GEMM, in which loaded matrix blocks can feed many output
cells. Its arithmetic intensity is low under the stated payload boundary, so
optimization questions will often concern data movement, reduction strategy,
and fusion. “Often” is architectural reasoning, not a measurement of the
current example.

No timing benchmark is published for Chapter 7. A nanosecond result for one
tiny vector would mostly expose call, allocation, compiler, and timer details.
The scale and magnitude experiments answer actual correctness questions. A
later optimization should first define the candidate, workload, cache state,
repetitions, hardware, compiler, and equivalence gate before measuring speed.

## Embedding is not output projection

ENGINE-1 contains both an embedding table and an output weight matrix with
shape `[V,D]`. Equal shape does not make the operations equal.

Embedding maps a token identity to a model-space row:

$$
t\longmapsto\mathbf{E}_{t,:}\in\mathbb{R}^{D}.
$$

Output projection maps a hidden vector to one score per vocabulary item:

$$
\mathbf{z}=\mathbf{W}_{\mathrm{out}}\mathbf{x}+\mathbf{b},
\qquad
\mathbf{W}_{\mathrm{out}}\in
\mathbb{R}^{V_{\mathrm{vocab}}\times D},
\quad
\mathbf{z}\in\mathbb{R}^{V_{\mathrm{vocab}}}.
$$

Lookup reads one selected row. Projection uses every output row in a GEMV. The
[comparison diagram](../../diagrams/transformer/embedding-vs-output-projection.txt)
shows where the two `[V,D]` tensors act.

Some architectures tie embedding and output-projection weights, sharing
parameter storage. That is a model architecture decision, not a consequence of
shape. ENGINE-1 keeps distinct owners and Chapter 7 does not redesign it.

## Integration without rewriting history

Before this chapter, `TinyLanguageModel::embedding_row` manually checked the
token and copied each tensor element. It now calls
`embedding_lookup_reference`, converts the owned tensor into its existing
hidden `Vec<f32>`, and maps an invalid token back to the established
`ModelError::TokenOutOfRange` variant. The output GEMV and bias addition are
unchanged.

RMSNorm is *not* inserted into ENGINE-1. That tiny model was defined without a
normalization weight or normalization step. Adding one merely to claim
integration would change its logits and invalidate the historical hand
fixture. The new operator is a tested primitive for the Transformer path that
later chapters will compose honestly.

The known regression remains:

```text
input token:       like
embedding/hidden: [1.0,-0.5,2.0]
logits:            [-0.7,0.1,0.4,2.2]
greedy output:     Rust, then EOS
```

This is a useful architectural lesson: extracting an operation should preserve
the model's semantics. Adding a new primitive to a library does not authorize
changing old model graphs.

## Independent correctness

The Rust suite adds 30 deterministic tests. Embedding coverage includes first,
middle, and last rows; dimensions one and greater; wrong rank; empty axes;
out-of-range IDs; strided and zero-stride views; sequence order and repetition;
empty token sequence; and ownership isolation.

RMSNorm coverage includes dimension one; the hand vector; mixed signs; zero
and uniform vectors; non-unit weights; strided and zero-stride views; wrong
ranks; length mismatch; empty dimension; invalid epsilon; non-finite operands;
large and small magnitudes; square, reduction, and output overflow; and
approximate positive-scale invariance where epsilon is negligible.

The Python oracle is independently expressed. It uses ordinary Python lists,
`math.fsum`, and Python's wider floating-point evaluation for the primary
mathematical result. A separate `struct`-based helper rounds every stress
operation to IEEE binary32 so it can classify the Rust reference's finite
range. It does not import Rust, NumPy, or another tensor framework.

Independent expression reduces correlated mistakes: the Rust operator follows
checked tensor strides, while Python works from mathematical lists. It does not
prove every possible vector. Boundary tests, deterministic fixtures, and the
unchanged previous oracles remain part of the gate.

> **PROVE IT**
> [Lab 38](../../labs/lab-38-rust-python-rmsnorm.md) compares exact lookup
> results and tolerant RMSNorm results. Deliberately permute learned weights;
> an RMS-only scalar check will miss the bug, while full-vector comparison will
> catch it.

## Inside Hermon

The following observations were verified against Hermon commit
`472a44cdb511b2dae6c9569e59543db8f8350b25` on 2026-09-03. Status labels matter
because Hermon has a release path and a separately gated engine preview.

> **INSIDE HERMON — CURRENT**
> With `HERMON_RUNTIME_MODE` unset, `dispatch.rs` selects `Batched`. That path
> constructs the llama.cpp-backed batched runtime. Embedding lookup,
> normalization, and the rest of the default model graph therefore execute in
> the pinned llama.cpp/GGML machinery rather than the Rust reference loop in
> `paged.rs`.

> **INSIDE HERMON — PREVIEW**
> Explicit `HERMON_RUNTIME_MODE=paged` selection is accompanied by source-level
> preview warnings and additional gates. Its `GgufLlamaForward` reads token
> embedding rows through a typed llama.cpp bridge, loads normalization vectors,
> validates model epsilon, and runs a visible Rust `rms_norm` function for its
> Hermon-owned forward path.

> **INSIDE HERMON — LIBRARY**
> `hermon-gguf::model_shape` verifies that `token_embd.weight` begins with GGML
> dimension `embedding_length` and derives vocabulary size from the next
> dimension. `hermon-llamacpp::tensor_row_f32` validates a requested row and
> materializes it as a Rust `Vec<f32>`, converting supported host-resident
> packed storage through the C++ bridge. These library abilities are consumed
> by the paged preview; their existence does not make that path the default.

The preview Rust function computes `sum(x*x)` in `f32`, divides by length,
adds epsilon inside `sqrt`, takes the reciprocal, and multiplies input and gain.
Its debug assertions rely on model construction to establish equal slice
lengths. The teaching API is more general and defensive because it is a public
checked operator over arbitrary valid views.

The [Hermon mapping diagram](../../diagrams/transformer/hermon-llamacpp-normalization-path.txt)
keeps CURRENT, PREVIEW, and LIBRARY boundaries on the same page. It also
prevents a common source-reading error: finding native code does not prove the
default API request reaches it.

## Inside llama.cpp and GGML

Hermon's submodule pins llama.cpp/GGML commit
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`. At that revision the architecture
has several layers.

The llama graph builder creates token embeddings with `ggml_get_rows` over the
token-embedding tensor and input-token tensor. Its `build_norm` function maps
the model's RMS normalization type to `ggml_rms_norm`, passing
`f_norm_rms_eps`. Learned multiplication is represented by subsequent graph
work and can participate in fused execution.

`ggml_get_rows` first creates an operator node and result tensor. The result's
fastest physical dimension is the source row width; remaining dimensions track
the index tensor. At CPU execution, backend dispatch selects by source storage
type:

- F32 rows are copied;
- F16 and BF16 rows are converted to F32;
- supported quantized rows are dequantized to F32.

Thus a “row lookup” can include conversion and materialization in a production
engine. The graph knows the semantic operation; the backend kernel knows the
physical type.

The inspected CPU F32 RMSNorm kernel requires the first tensor dimension to be
contiguous. It distributes outer rows across threads and visits each contiguous
inner row. At this commit `ggml-cpu/vec.h` defines the `ggml_float` accumulator
as `double`, but `x[i]*x[i]` is formed from two F32 operands before that product
is cast into the wider sum. It computes `mean = sum / row_width`, then
`scale = 1/sqrtf(mean + eps)`. The plain path copies and scales the row. A fused
RMSNorm-plus-multiply path applies the learned weight during the output loop.

GGML stores epsilon in operator parameters and asserts a nonnegative value in
that kernel. It asserts row indices and layout invariants after graph and model
construction. Those assertions are appropriate inside its validated execution
graph; the public educational functions instead return typed errors for
ordinary malformed calls.

The source flow is:

```text
llama model graph
        │
        ▼
GGML operator and tensor metadata
        │
        ▼
backend dispatch
        │
        ▼
typed CPU or accelerator implementation
```

This chapter describes the pinned CPU path, not every backend. Accelerator
kernels can partition reductions, fuse operations, and impose other layout or
dtype contracts. Source at one commit also cannot justify “llama.cpp always”
claims across revisions.

## Educational and production boundaries

The teaching path is intentionally narrow:

```text
f32 TensorView ──▶ checked scalar reference ──▶ f32 OwnedTensor
```

A production engine may need lower-precision activation storage, packed or
quantized parameter rows, wider accumulation, vector or thread reductions,
device-local addresses, graph scheduling, batch dimensions, and fused
normalization with neighboring work. Each complication changes a contract:
conversion precision, work partition, ownership, layout, or failure placement.

The reference remains valuable precisely because it does not absorb all those
concerns. A future optimized candidate can be compared against one named
semantic path. The permanent substitution order remains:

```text
specification ──▶ reference ──▶ independent oracle ──▶ correctness
                                                           │
                                                           ▼
candidate ──▶ equivalence gate ──▶ performance gate ──▶ substitution
```

Chapter 7 has no candidate because correctness infrastructure should precede
optimization pressure. That is not unfinished work. It is a deliberate engine
state with an inspectable active implementation.

## Common mistakes

Several short statements conceal major errors:

- **“The token ID is fed into the neural network as a number.”** The ID selects
  a learned row; its scalar magnitude is not an activation feature.
- **“Embedding is a matrix multiply.”** One-hot multiplication is an
  equivalence, while the implemented operation is indexed row materialization.
- **“Shape `[V,D]` guarantees contiguous rows.”** Only shape plus strides and
  storage metadata establish layout.
- **“Returning a view is always faster.”** It avoids one copy but constrains
  lifetime and preserves aliasing; safety and later mutation requirements
  decide the interface.
- **“RMSNorm makes every component unit length.”** It uses one vector-wide RMS
  factor and then learned per-coordinate scale. Individual coordinates need
  not have magnitude one.
- **“RMSNorm subtracts the mean.”** That describes the centering found in
  LayerNorm, not this operator.
- **“Epsilon can go anywhere near the denominator.”** Placement defines the
  function.
- **“Finite input means finite squares.”** A finite `f32` can overflow when
  squared, and finite squares can overflow their sum.
- **“Equivalent reductions must match bit-for-bit.”** Different valid
  reduction trees can round differently; equivalence needs a justified
  tolerance and finite-value gate.
- **“A Rust RMSNorm function proves Hermon uses it in production.”** Runtime
  selection proves the current default path; at the inspected commit it is
  llama.cpp-backed batched execution.

## Transformer Primitives v1

Chapter 7 completes a named extension of ENGINE-2 rather than inventing
ENGINE-3. **Transformer Primitives v1** means the repository can now execute:

- checked single-token embedding `[V,D] + TokenId -> [D]`;
- checked sequence embedding `[V,D] + [T] -> [T,D]`;
- explicit parameter-to-activation copy ownership;
- checked scalar RMSNorm `[D] + [D] + epsilon -> [D]`;
- valid strided and zero-stride reference layouts;
- typed shape, epsilon, ID, and finite-range failures;
- independent Python and Rust numerical verification.

ENGINE-2's dot, GEMV, reference GEMM, and blocked GEMM remain intact. The
[milestone architecture](../../diagrams/transformer/chapter07-engine-architecture.txt)
places all three ingredients—embedding, normalization, and linear algebra—on
the checked tensor substrate without composing a fake Transformer layer.

The labs form a complete evidence path:

- [Labs 30–32](../../labs/README.md) follow embedding offsets, checked lookup,
  and ownership;
- [Labs 33–35](../../labs/README.md) derive RMSNorm and break its epsilon
  contract;
- [Labs 36–37](../../labs/README.md) expose scale and finite-range behavior;
- [Lab 38](../../labs/lab-38-rust-python-rmsnorm.md) closes the independent
  oracle gate.

## What we can now account for

For embedding lookup, we can name the long-lived parameter owner, both tensor
axes, the selected logical coordinates, canonical and strided offsets, copied
bytes, activation owner, valid ID interval, and every empty-axis decision.

For RMSNorm, we can name input and learned-scale shapes, the reduced dimension,
epsilon convention, storage and reduction dtype, loop order, two-pass traffic,
output owner, zero-vector result, invalid configuration, overflow boundary,
oracle, and tolerance policy.

That is the level at which later kernels must be understood. The equation,
memory, ownership, and production mapping agree. Short code is no excuse for a
vague system contract.

## The boundary to Chapter 8

We now have model-space vectors, normalization, and matrix multiplication. How
does a Transformer turn one residual vector into three different
representations that determine what information it asks for, what information
it offers, and what information is ultimately transported?

That is the bounded question for **Chapter 8 — Queries, Keys, and Values**.
Chapter 8 may build the three linear projections and their shape/ownership
contracts. It must not smuggle in attention, position rotation, or KV caching:
those remain separate later chapters.

## Primary references

- Biao Zhang and Rico Sennrich, [“Root Mean Square Layer Normalization”](https://arxiv.org/abs/1910.07467), 2019; [NeurIPS proceedings version](https://papers.neurips.cc/paper_files/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf).
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, [“Layer Normalization”](https://arxiv.org/abs/1607.06450), 2016.
- PyTorch, [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) and [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), accessed 2026-09-03.
- Meta, [Llama 4 reference `model.py`](https://github.com/meta-llama/llama-models/blob/main/models/llama4/model.py), accessed 2026-09-03.
- Netlib LAPACK, [`SLASSQ`: scaled sum of squares](https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq_ga0596b4bfa745d0d1c5817d4790921cda.html), accessed 2026-09-03.
- Rust standard library, [`f32`](https://doc.rust-lang.org/std/primitive.f32.html), accessed 2026-09-03.
- Hermon source at `472a44cdb511b2dae6c9569e59543db8f8350b25` and its pinned llama.cpp/GGML source at `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`, inspected 2026-09-03; exact paths and classifications are recorded in the [research note](../../research/part-02/chapter-07-embeddings-and-normalization.md).
