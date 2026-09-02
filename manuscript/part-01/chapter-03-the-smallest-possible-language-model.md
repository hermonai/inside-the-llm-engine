# Chapter 3 — The Smallest Possible Language Model

## A token ID is still not an answer

Chapter 2 ended at a precise boundary. Human text became UTF-8 bytes. A
model-specific tokenizer turned those bytes into token IDs. The request owned
the resulting sequence, and generated IDs could travel back through token
bytes and a strict UTF-8 framer to become streamed text.

But ENGINE-0 still cheated in the middle. Its “model” returned a table of
handwritten candidate scores:

```text
blue   9
green  4
EOS    1
```

Those integers were useful in Chapter 1 because they let us study admission,
selection, stopping, streaming, cancellation, failure, and terminal ownership
without pretending that lifecycle code was a neural network. They have now
served their purpose. A real model does not decide that `blue` should receive
9 by consulting a source-code answer table.

What can a model actually do with a token ID?

Suppose the tokenizer emits `TokenId(2)` for `like`. The integer `2` does not
mean “moderately positive.” It is not twice as semantic as token 1. It does not
contain a direction toward `Rust`, a probability, or a grammatical role. It is
an index in one model vocabulary. The next boundary must turn that discrete
identity into numerical data on which the model can compute.

This chapter builds that boundary with the smallest useful numerical language
model we can inspect completely:

```text
text -> token IDs -> embedding lookup -> hidden vector
     -> output projection -> vocabulary logits
```

There is no neural-network framework, general tensor library, BLAS call,
Transformer, attention mechanism, training loop, or model file. The parameters
are 28 visible `f32` values. The kernel is two nested Rust loops. We will
calculate every multiplication independently and compare the whole result.

The model will produce **logits**: one unnormalized score for every vocabulary
token. It will not produce probabilities. A temporary deterministic argmax
selector keeps the existing end-to-end lifecycle executable, but selection is
outside the model. Chapter 4 will ask how logits become a distribution and how
an engine selects and feeds back the next token. We stop at correct logits.

> **FIRST PRINCIPLE**
> A token ID is a categorical identity, not a meaningful scalar quantity.
> Numerical model semantics begin when that identity selects learned
> parameters.

## What makes this a language model?

A language model assigns scores or probabilities to linguistic sequences. For
next-token inference, we ask for a distribution over the next vocabulary token
given an observed history. If the history is

```text
x_0, x_1, ..., x_t
```

the general target is often written as:

```text
P(x_(t+1) | x_0, x_1, ..., x_t)
```

The output space matters: the possible outcomes are all `V` entries in the
model vocabulary. The model must therefore produce `V` scores before a policy
can choose one token.

Our first numerical model makes a severe approximation:

```text
P(next | complete history) is approximated by P(next | final token)
```

It accepts a full sequence but uses only `x_t`, the last token. This is a
first-order, neural bigram-like language model. It is still a language model:
its input represents token history, its task is next-token prediction, and its
output has one position per vocabulary token. It is not a Transformer and
cannot represent long-range context.

That simplicity is an advantage for the present question. If we began with
attention, normalization, residual connections, positional encoding, dozens
of layers, and billions of parameters, the first numerical boundary would
disappear inside machinery we could not yet audit. Here, every output score
will have a short arithmetic trail.

Neural language models predate Transformers. Bengio, Ducharme, Vincent, and
Jauvin described a neural probabilistic language model that learned distributed
representations of words along with a probability function over sequences in
2003. Their model uses richer context and training machinery than ENGINE-1.
The relevant historical point is narrow: a language model need not be a
Transformer, and discrete vocabulary identities can enter computation through
learned vector representations.

## From a discrete ID to a vector

Let the vocabulary contain `V` token identities and let each identity have a
vector with `D` components. Store those vectors as rows of an **embedding
matrix**:

```text
E: [V, D]
```

`V` is the number of rows. `D`, the **hidden dimension**, is the number of
values in each row. ENGINE-1 stores every value as `f32` in contiguous
row-major order.

If the input token is `x`, the embedding operation is:

```text
h = E[x]
```

`x` is a scalar `TokenId`. `h` is a vector of shape `[D]`. The
[embedding lookup diagram](../../diagrams/model/embedding-row-lookup.txt)
makes the access concrete:

```text
                 D = 3 contiguous f32 values per row
                      column 0   column 1   column 2
                   +----------+----------+----------+
TokenId(0) <eos>   |    0.0   |    0.0   |    0.0   |
                   +----------+----------+----------+
TokenId(1) I       |    0.0   |    1.0   |    0.0   |
                   +----------+----------+----------+
TokenId(2) like -->|    1.0   |   -0.5   |    2.0   |--> h [3]
                   +----------+----------+----------+
TokenId(3) Rust    |   -1.0   |    0.0   |    0.0   |
                   +----------+----------+----------+
```

For `x = 2`, the result is:

```text
h = [1.0, -0.5, 2.0]
```

This is a row selection, sometimes called a gather. ENGINE-1 computes the row
start as:

```text
row_start = token_id * D
```

and copies the next `D` values. The scalar value 2 is used only in address
calculation. We do not multiply the vector by 2.

### Why not feed the integer directly into arithmetic?

Imagine that vocabulary revision A assigns:

```text
2 -> like
3 -> Rust
```

and revision B assigns:

```text
2 -> Rust
3 -> like
```

The language meanings swapped while the integers did not. Any rule that treats
3 as inherently larger or closer to something than 2 would change meaning
when the vocabulary was renumbered. Embedding rows attach numerical parameters
to identities explicitly.

An embedding does not guarantee that each dimension has a clean human label.
We must not say that dimension 0 is “programming,” dimension 1 is “affection,”
or dimension 2 is “grammar” merely because a tiny hand fixture makes a story
tempting. In a trained model, useful behavior can be distributed across many
dimensions. For ENGINE-1, `h` is exactly this: the numerical representation of
the current input token consumed by the next computation.

### Is lookup secretly matrix multiplication?

We could represent token 2 as a one-hot vector:

```text
[0, 0, 1, 0]
```

and multiply it by `E`. The result would select row 2. That equivalence can be
useful mathematically, but it is a poor description of ENGINE-1's execution.
Materializing `V` values, almost all zero, and multiplying through every row
would hide the actual access pattern. The code reads one contiguous row.

> **FIRST PRINCIPLE**
> An embedding lookup reads the row named by a token ID. It does not assign
> meaning to the ID's integer magnitude, and it need not execute as dense
> matrix multiplication.

## One score for every possible next token

The hidden vector represents the input. The model now needs one score for each
candidate next token. With vocabulary size `V`, the result must have shape
`[V]`.

Introduce an **output projection** matrix `W` and bias vector `b`:

```text
W: [V, D]
b: [V]
```

Each row `W[i]` belongs to candidate token `i`. Compute:

```text
z = W h + b
```

where `z` has shape `[V]`. For one candidate `i`:

```text
z_i = b_i + sum from j=0 to D-1 of W[i,j] * h_j
```

This operation takes the **dot product** of row `i` and `h`, then adds one
bias. The [one-logit diagram](../../diagrams/model/one-logit-dot-product.txt)
connects vector notation to scalar arithmetic:

```text
hidden h              candidate row W[3]          products
[1.0, -0.5, 2.0]      [1.0, -0.4, 0.25]          1.0 * 1.0
                                                + -0.4 * -0.5
                                                + 0.25 * 2.0
                                                + bias[3] 0.5
                                                        |
                                                        v
                                               logit[3] = 2.2
```

Repeat that operation for all `V` rows. Matrix notation compresses the idea;
the implementation exposes the work:

```rust
for output in 0..vocab_size {
    let mut accumulator = output_bias[output];
    let row_start = output * hidden_dim;
    for dimension in 0..hidden_dim {
        accumulator +=
            output_weight[row_start + dimension] * hidden[dimension];
    }
    logits[output] = accumulator;
}
```

The inner loop calculates one candidate score. The outer loop repeats it for
the vocabulary. There is no answer table in this code: changing an embedding,
projection weight, or bias changes the arithmetic that produces the vector.

### “Linear” layers usually include bias

Strictly, `W h` is a linear map and `W h + b` is affine when `b` is nonzero.
Machine-learning libraries commonly call the combined operation a linear
layer. We will use **output projection** for the role and write the bias
explicitly so neither terminology nor code can hide it.

## What a logit is—and is not

Each `z_i` is a **logit**: an unnormalized numerical score for candidate token
`i`. ENGINE-1's complete result for `like` will be:

```text
token       logit
<eos>       -0.7
I            0.1
like         0.4
Rust         2.2
```

The values do not sum to 1. They can be negative. A logit of 2.2 is not a
probability of 2.2, 220 percent, or “2.2 times likely.” A difference of 1.8 is
not directly a probability difference. We have not yet defined the
transformation needed for those interpretations.

For a deterministic argmax policy, the larger finite score wins, so `Rust`
would be selected here. That statement is about ordering only. It does not
make the largest score probability 1, and it does not make argmax part of the
model.

The boundary remains:

```text
Model::forward(history) -> Logits
Selector::select(logits) -> TokenId
```

ENGINE-0 blurred this distinction because the fake model returned candidates
already carrying token kinds and integer scores. ENGINE-1 returns a typed
`Logits` vector. The selector scans it. The runtime uses the model/tokenizer
contract to recognize EOS and otherwise asks the tokenizer for output bytes.

> **FIRST PRINCIPLE**
> The model produces logits. Token selection is a separate engine policy.

Chapter 4 owns softmax, numerical stability, temperature, randomness, seeds,
top-k, top-p, and the complete feedback loop. We will not smuggle those topics
into `forward` merely to make this chapter appear more complete.

## The complete hand-calculated forward pass

Our vocabulary has four identities:

```text
0 <eos>
1 I
2 like
3 Rust
```

The hidden dimension is three. Therefore `V=4` and `D=3`. The parameters are:

```text
E = [
  [ 0.0,  0.0,  0.0],
  [ 0.0,  1.0,  0.0],
  [ 1.0, -0.5,  2.0],
  [-1.0,  0.0,  0.0],
]

W = [
  [-0.5,  0.4,  0.1],
  [ 0.2,  0.2,  0.0],
  [ 0.3,  0.2,  0.1],
  [ 1.0, -0.4,  0.25],
]

b = [-0.2, 0.0, 0.0, 0.5]
```

These values are supplied directly. In a real neural language model, training
would normally produce parameters by optimizing an objective over data. This
book is about inference, so we hold training outside the boundary. “Learned
parameters” names their usual origin; it does not mean ENGINE-1 trains them.

Input `like` has ID 2. First select the embedding row:

```text
h = E[2] = [1.0, -0.5, 2.0]
```

Now calculate every candidate.

For EOS, row 0:

```text
z_0 = b_0 + W[0,0]h_0 + W[0,1]h_1 + W[0,2]h_2
    = -0.2 + (-0.5 * 1.0) + (0.4 * -0.5) + (0.1 * 2.0)
    = -0.2 + -0.5 + -0.2 + 0.2
    = -0.7
```

For `I`, row 1:

```text
z_1 = 0.0 + (0.2 * 1.0) + (0.2 * -0.5) + (0.0 * 2.0)
    = 0.0 + 0.2 + -0.1 + 0.0
    = 0.1
```

For `like`, row 2:

```text
z_2 = 0.0 + (0.3 * 1.0) + (0.2 * -0.5) + (0.1 * 2.0)
    = 0.0 + 0.3 + -0.1 + 0.2
    = 0.4
```

For `Rust`, row 3:

```text
z_3 = 0.5 + (1.0 * 1.0) + (-0.4 * -0.5) + (0.25 * 2.0)
    = 0.5 + 1.0 + 0.2 + 0.5
    = 2.2
```

The full oracle is:

```text
z = [-0.7, 0.1, 0.4, 2.2]
```

Notice what the oracle does not say. It does not say merely “the answer is
token 3.” Many wrong vectors can share the same largest index. A transposed
matrix, missing bias, wrong embedding row, or misaligned vocabulary can still
produce argmax 3 by accident.

> **FIRST PRINCIPLE**
> A correct selected token does not prove that the model produced correct
> logits.

## An independent numerical oracle

The Rust model and its test could share the same indexing bug. Rewriting the
same function under a different name would not provide much independence. The
repository therefore includes a plain-Python oracle in
`code/reference/python/chapter03_oracle.py`.

The script defines `E`, `W`, and `b` separately, selects the row, implements
the equations with its own loops, and asserts the full expected vector. It
does not import ENGINE-1, call the Rust binary, or use NumPy. Run it with:

```sh
python3 code/reference/python/chapter03_oracle.py
```

Expected output includes:

```text
input=2:like
hidden=[1.0, -0.5, 2.0]
logits=[-0.7, 0.1, 0.4, 2.2]
oracle=PASS
```

The Rust differential uses a small absolute-plus-relative tolerance:

```text
|actual - expected| <= abs_epsilon + rel_epsilon * |expected|
```

Why not exact equality? Several fixture values happen to have exact binary
representations, and this short reduction is deterministic on the current
scalar path. But floating-point arithmetic generally rounds, and later
execution providers may change reduction order. An absolute tolerance covers
small values near zero; a relative tolerance scales with larger expected
values. The tolerance must be tight enough to catch errors and documented
instead of chosen after seeing a failure.

The model also rejects non-finite parameters and non-finite output logits.
`NaN` is especially dangerous because comparisons with it are false; a naive
argmax scan can silently keep an unrelated candidate. Some future operations
may deliberately use infinities to represent masks. ENGINE-1 has no mask or
logits processor, so `NaN`, positive infinity, and negative infinity indicate
invalid numerical state at this boundary.

## A tensor, before a tensor framework

Chapter 5 will treat tensors systematically. We need a smaller working
definition now. A **tensor** is numerical data interpreted with metadata and
contracts such as:

- shape;
- element type, or **dtype**;
- physical data;
- layout and strides;
- owner and lifetime;
- device or location.

ENGINE-1 uses only one location, the CPU, and one storage type, `f32`. All
arrays are contiguous. Matrices are row-major. There is no dynamic rank,
broadcasting, autograd, GPU abstraction, or general stride machinery.

The canonical [shape table](../../diagrams/model/tiny-model-tensor-shapes.txt)
is:

| Name | Symbol | Shape | Lifetime | Inference-mutable? |
| --- | --- | --- | --- | --- |
| Embedding | `E` | `[V,D]` | model | no |
| Projection | `W` | `[V,D]` | model | no |
| Bias | `b` | `[V]` | model | no |
| Input token | `x` | scalar | request/step | new |
| Hidden | `h` | `[D]` | forward | new |
| Logits | `z` | `[V]` | forward/result | new |

“Mutable?” in this table describes inference semantics. A loader may fill an
allocation during construction, and Rust's `Vec<f32>` is mechanically capable
of mutation. After construction, ordinary forward inference does not change
parameters. `TinyLanguageModel::forward` takes `&self`, making that rule visible
in the API.

> **FIRST PRINCIPLE**
> Every tensor needs an explicit shape, dtype, layout, owner, lifetime, and
> access pattern. Shape is an executable contract, not decoration.

## Logical shape and physical row-major memory

`W` has logical shape `[V,D]`. For `V=4`, `D=3`, picture four rows and three
columns:

```text
W[0,0] W[0,1] W[0,2]
W[1,0] W[1,1] W[1,2]
W[2,0] W[2,1] W[2,2]
W[3,0] W[3,1] W[3,2]
```

The contiguous physical vector is:

```text
[W00, W01, W02, W10, W11, W12, W20, W21, W22, W30, W31, W32]
```

The offset for logical element `(row, column)` is:

```text
offset = row * D + column
```

This convention is not forced by the equation. We could store columns
contiguously, pad rows, tile blocks, quantize groups, or use provider-specific
layouts. Those choices would change access and kernels without changing the
model's mathematical meaning. ENGINE-1 chooses row-major because one output
candidate consumes one complete row, so the inner loop reads contiguous
weights.

Matrix notation alone cannot tell us whether bytes are contiguous, transposed,
padded, packed, resident on a device, or available through a view. An engine
must connect logical indices to physical addresses.

## Parameters and activations are different memory

The embedding, projection, and bias are **parameters**: persistent values that
define model behavior. The selected hidden row and logits are **activations**:
values produced while executing the model for a particular input.

The [ownership diagram](../../diagrams/model/parameters-vs-activations.txt)
shows why the distinction matters:

```text
                         MODEL LIFETIME
              immutable parameters shared by requests
        +-----------------------------------------------+
        | E [V,D]        W [V,D]          b [V]         |
        +-----------------------+-----------------------+
                                |
                   +------------+------------+
                   |                         |
                   v                         v
        Request A / forward A      Request B / forward B
        [input x_A, h_A, z_A]      [input x_B, h_B, z_B]
```

ENGINE-1 does not implement concurrency, but it chooses an ownership model
that can support it later. Multiple requests may read the same immutable
parameters. Each forward call creates its own hidden and logit vectors. If
activations were accidental mutable fields on the shared model, concurrent
requests could overwrite one another even though the weights were correct.

This distinction will recur in memory planning, GPU execution, batching, and
the KV cache. Parameters often live as long as the loaded model. Activations
may live for one operation, one token step, one request, or longer if the model
needs persistent state.

The [memory diagram](../../diagrams/model/parameters-vs-activations.txt) is
small now:

```text
MODEL MEMORY
+------------------------------+
| E: embedding weights         |
+------------------------------+
| W: output projection weights |
+------------------------------+
| b: output biases             |
+------------------------------+

FORWARD ACTIVATIONS
+------------------------------+
| h: hidden vector             |
+------------------------------+
| z: logits                    |
+------------------------------+
```

No “hidden reasoning” is stored here. The adjective *hidden* distinguishes an
internal representation from observable input and output. It does not imply a
human-readable chain of thought.

## Follow the token, byte, and owner

The [complete Chapter 3 path](../../diagrams/model/token-id-to-logits.txt)
extends our first recurring journey:

```text
"I like"
    |
    v
UTF-8 bytes
    |
    v
TinyLmTokenizer
    |
    v
[TokenId(1), TokenId(2)]
    |
    v
E[2] = [1.0, -0.5, 2.0]
    |
    v
W h + b
    |
    v
[-0.7, 0.1, 0.4, 2.2]
```

Follow one parameter byte conceptually. It begins as a fixture literal. Model
construction places it in a contiguous `Vec<f32>`. A forward call calculates
its row-major offset, the CPU loads it, multiplies it by one hidden value, and
adds the product to an `f32` accumulator. Later chapters will replace source
literals with model-file bytes and may replace scalar loads with packed SIMD
or device operations. The ownership and interpretation trail must remain
auditable.

Follow the owner. `TinyLanguageModel` owns `E`, `W`, `b`, `V`, and `D`. A
`Request` owns its input token sequence and generation budget. The forward
result owns `h` and typed `Logits`. The runtime owns generated token history,
the UTF-8 buffer, ordered stream emissions, timings, and the only terminal
transition. The selector borrows logits and returns an identity; it owns
neither parameters nor activations.

## Count the parameters before counting performance

The embedding contains `V*D` parameters. The untied output projection contains
another `V*D`. The bias contains `V`. Therefore:

```text
parameter_count = V*D + V*D + V
                = 2VD + V
```

With `f32`, each parameter occupies four bytes:

```text
parameter_bytes = (2VD + V) * 4 bytes
```

For ENGINE-1, `V=4` and `D=3`:

```text
E: 4 * 3 = 12 parameters
W: 4 * 3 = 12 parameters
b: 4     =  4 parameters
total          = 28 parameters
f32 bytes      = 28 * 4 = 112 bytes
```

This count excludes `Vec` metadata, allocator bookkeeping, dimensions, code,
and forward activations. It describes parameter payload only. The hidden
activation uses `D*4 = 12` bytes of `f32` payload; the logits use `V*4 = 16`.

The same formulas produce a useful scale table:

| `V` | `D` | `E` parameters | `W` parameters | Total `2VD+V` | `f32` bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 12 | 12 | 28 | 112 |
| 100 | 16 | 1,600 | 1,600 | 3,300 | 13,200 |
| 50,000 | 4,096 | 204,800,000 | 204,800,000 | 409,650,000 | 1,638,600,000 |

The formula scales quickly. Consider a hypothetical vocabulary of 50,000 and
hidden dimension 4,096 under this same simplified, untied architecture:

```text
one V-by-D matrix = 50,000 * 4,096
                  = 204,800,000 parameters

two matrices plus bias
                  = 409,650,000 parameters

f32 payload       = 1,638,600,000 bytes
                  = about 1.526 GiB
```

That is not the size of a complete real language model and not a measurement.
It isolates vocabulary embedding and projection under stated assumptions. It
does explain why output heads are not negligible and why real designs care
about lower-precision storage, quantization, and optimized multiplication.

Some models use **weight tying**: the input embedding and output projection
reuse the same parameter storage, with the orientation interpreted according
to the chosen convention. That can eliminate one `V*D` parameter collection.
It is an architectural option, not a universal law. ENGINE-1 leaves `E` and
`W` separate because changing an input representation and changing a candidate
scoring row should be visibly different experiments.

## Count the work and the bytes read

Embedding lookup performs little arithmetic. It validates `x`, computes an
offset, and reads one row: `D` parameter values. It does not scan all `V` rows.

Output projection is different. For each of `V` output rows, it performs `D`
multiplications and approximately `D` additions including bias accumulation.
The work grows as:

```text
O(V * D)
```

For one forward call, the scalar kernel reads essentially all `V*D` projection
weights and `V` biases and writes `V` logits. It reuses the small hidden vector
across all rows. Even in this tiny model, two distinct systems patterns appear:

- embedding lookup: sparse row access, little arithmetic;
- output projection: broad weight access plus many dot-product operations.

Inference performance depends on computation and data movement. A processor
cannot multiply a weight it has not obtained through its memory hierarchy.
Later chapters will ask whether a kernel is limited by arithmetic, weight
bandwidth, cache reuse, conversion, launch cost, synchronization, or another
resource. Chapter 3 establishes the accounting habit without pretending that
a 4-by-3 loop predicts a production workload.

The repository includes
`code/experiments/chapter-03-projection-scaling.py`. It varies modest `V` and
`D`, reports parameter payload and median time for a plain-Python scalar loop,
and prints a checksum. Its result record is in
`research/benchmarks/chapter-03-projection-scaling.md`.

> **PERFORMANCE LAB**
> The scaling probe is pedagogical. It shows that larger shapes create more
> stored values and loop iterations in that harness. Python interpreter timing
> does not predict Rust, BLAS, SIMD, GPU, or real-model throughput, and it must
> not be compared with Hermon or llama.cpp.

## Model semantics are not one implementation

The equations say what the model computes:

```text
h = E[x]
z = W h + b
```

The scalar Rust loops say how ENGINE-1 executes those semantics today. The
[semantics/execution diagram](../../diagrams/model/semantics-vs-execution.txt)
keeps that line visible:

```text
                 MODEL SEMANTICS
TokenId -> embedding -> hidden -> projection -> logits
=======================================================
              EXECUTION IMPLEMENTATION
today: scalar row-major Rust loops
later: CPU SIMD, Metal, CUDA, other native kernels
```

An optimized implementation may load several weights into vector registers,
tile a matrix, use a quantized packed layout, fuse operations, or dispatch to a
device. It can allocate scratch differently. None of those choices may change
which token owns a row, transpose the mathematical convention accidentally,
drop the bias, or produce errors outside the declared numerical tolerance.

> **FIRST PRINCIPLE**
> An engine may change execution while preserving model semantics. The oracle
> defines what must remain invariant.

This separation is the foundation of later differential testing. The clarity
path need not be fast. The fast path must agree with an independent clarity
path before its timing is meaningful.

## BUILD IT: ENGINE-1

ENGINE-1 evolves the existing Rust crate instead of maintaining parallel
copies of an entire runtime. Git history preserves the exact ENGINE-0
milestone. The current code keeps Chapter 2's tokenizer trait, chat/template
types, request, streaming events, UTF-8 framer, cancellation contract, timings,
and terminal lifecycle. It replaces the fake numerical center.

### The model structure

`TinyLanguageModel` owns:

```rust
pub struct TinyLanguageModel {
    vocab_size: usize,
    hidden_dim: usize,
    embedding: Vec<f32>,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
}
```

The deliberately small structure is not a general tensor framework. Field
names carry the semantic rank and role. Construction checks every length
before the model can execute.

The model trait accepts the full history:

```rust
pub trait Model {
    fn vocabulary_size(&self) -> usize;
    fn forward(&self, input: &[TokenId])
        -> Result<ForwardPass, ModelError>;
}
```

Why accept a sequence if ENGINE-1 uses only its last element? Passing only
`last_token` would be locally narrower, but it would make the runtime decide
which history information matters to the model. The full borrowed slice keeps
next-token history at the model boundary, introduces no input allocation, and
makes the limitation testable. `forward` calls `input.last()` and says so in
code. Future models may consume more positions without changing what a request
means.

This is a documented teaching choice, not a claim that every production model
API should use the same signature. Batched engines need shapes, positions,
sequence identifiers, cache handles, and output-row requests that this method
does not express.

### A typed forward result

`ForwardPass` contains:

```text
input_token: TokenId
hidden:      Vec<f32>
logits:      Logits
```

Keeping the hidden activation in the teaching result makes trace and oracle
inspection straightforward. A production API might retain it only in debug
mode or place scratch in a reusable context. `Logits` wraps its vector and
exposes a borrowed slice. This prevents unrelated code from treating every
`Vec<f32>` as interchangeable.

The model owns `Logits` construction and guarantees finite values. It returns
a typed `ModelError` for an empty sequence or out-of-range final token. The
selector receives only `&Logits`, so it cannot reach into parameters or alter
the hidden activation.

### Construction is a correctness boundary

`TinyLanguageModel::try_new` rejects:

- zero vocabulary size;
- zero hidden dimension;
- overflow in `V*D`;
- an embedding length other than `V*D`;
- a projection length other than `V*D`;
- a bias length other than `V`;
- any non-finite parameter.

These are load-time properties. Paying once during construction lets the hot
inner loop index validated storage without rechecking every row length.
Forward still checks the request-dependent token ID.

Shape bugs are not cosmetic. If `W` has 11 values instead of 12, choosing to
truncate would omit a term, choosing to pad would invent a parameter, and
indexing blindly could panic. None preserves the declared model.

> **PROVE IT**
> Model shape validation is inference correctness. Malformed parameter counts
> and vocabulary mismatches must fail before plausible output can hide them.

### Vocabulary coupling becomes numerical

Chapter 2 established that a tokenizer revision and chat template belong to a
model contract. Chapter 3 adds a numeric equality:

```text
model vocabulary size
    == tokenizer vocabulary size
    == ModelContract vocabulary size
```

Embedding row `i`, logit position `i`, tokenizer identity `i`, and EOS
recognition must refer to the same token. A vector of length four cannot safely
consume Chapter 2's sparse byte-BPE IDs, which extend through 1007.

ENGINE-1 therefore has a deliberately tiny paired tokenizer:

```text
0 <eos>   special, no ordinary output bytes
1 I       bytes "I"
2 like    bytes " like"
3 Rust    bytes " Rust"
```

Leading spaces belong to two pieces, allowing `I like Rust` to round-trip. The
tokenizer rejects all other text with a byte-offset error. That narrow domain
is honest: four embeddings cannot represent a thousand-ID vocabulary.
Chapter 2's byte tokenizer and BPE remain in the repository for their own
labs; they are not silently remapped into ENGINE-1.

`Runtime::try_new` validates the model, tokenizer, and contract sizes and
requires an EOS identity. It cannot admit a request when the model's fourth
logit would mean one token to the model and another to the decoder.

### The inherited lifecycle

For prompt `I like`, the tokenizer yields `[1,2]`. The first forward uses the
last token 2 and produces the oracle logits. Temporary argmax selects token 3,
which decodes to bytes for ` Rust`. The runtime emits a token event and then a
valid text event.

The generated token is appended to history. The next forward therefore ends
in token 3. With the fixture's `Rust` embedding `[-1.0,0.0,0.0]`, the logits
are:

```text
[0.3, -0.2, -0.3, -0.5]
```

EOS at position 0 is the unique maximum. The runtime recognizes it as a
terminal identity, does not ask for ordinary text bytes, checks that the UTF-8
buffer is complete, and emits exactly one completed terminal event.

This two-step behavior is not a disguised step table. Both logit vectors come
from the same immutable `E`, `W`, and `b`; only the input embedding row changes.

The lifecycle retains its earlier failure properties:

- empty input or zero generation budget fails before admission;
- cancellation stops before another model call or token emission;
- model, tokenizer, and UTF-8 failures produce one failed terminal;
- EOS produces no text token event;
- maximum-token termination checks pending UTF-8 bytes;
- no event or trace can occur after terminal;
- one request cannot both complete and fail.

Chapter 4 will explain the loop as autoregressive generation. Here it remains
minimal infrastructure needed to prove that the new logit boundary integrates
with established ownership.

### An educational numerical trace

Run:

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace 'I like'
```

The trace names admission and execution, then includes the input token, hidden
shape and values, logit shape and values, selected token, decoded byte count,
streamed text, and terminal reason. ENGINE-1 vectors are small enough to show
completely. This must not become a habit of dumping future tensors with
millions of elements. The trace already names shapes separately from values;
later milestones can retain shapes, dtypes, selected values, and statistics
while bounding payload size.

## Test the arithmetic, not the story

The Rust suite checks more than one happy result:

- valid construction and exact parameter/byte counts;
- zero `V`, zero `D`, and multiplication overflow;
- embedding, projection, and bias count mismatches;
- non-finite parameters;
- empty model input and an ID outside `0..V`;
- exact embedding-row selection;
- the complete oracle logit vector within tolerance;
- bias behavior with a zero embedding;
- negative and zero arithmetic;
- repeatability of the complete forward result;
- identical logits for different histories with the same final token;
- expected different logits for different embedding rows;
- EOS as an ordinary position in the output vector;
- one changed projection weight affecting only its row's logit;
- model/tokenizer/contract vocabulary mismatch;
- runtime cancellation, failure, EOS, max-token, UTF-8, event ordering, and
  exactly-one-terminal behavior.

Changing one projection weight is a particularly useful causality test. For
input `like`, change `W[3,0]` from 1.0 to 1.5. Since `h_0=1.0`, only `z_3`
increases, by 0.5:

```text
before: [-0.7, 0.1, 0.4, 2.2]
after:  [-0.7, 0.1, 0.4, 2.7]
```

The embedding does not change. Rows 0 through 2 do not read `W[3,0]`. By
contrast, changing one embedding component can affect every logit because all
projection rows consume the shared hidden component.

Labs 5–8 turn these invariants into exercises: calculate the full pass, change
one weight, prove the context limitation, and break the shape deliberately.

## Inside Hermon: the same boundary, industrial machinery

> **INSIDE HERMON — CURRENT**
> At inspected commit `472a44c`, Hermon's default runtime is the continuous-
> batched path, and real-model numerical execution is delegated through
> `hermon-llamacpp` to pinned llama.cpp commit `389ff61`.

Hermon's default path is not an embedding-plus-one-projection model. It runs a
real model graph with layers, context state, batching, and hardware backends.
The useful comparison is the input/output contract.

`crates/hermon-runtime/src/dispatch.rs` selects `RuntimeMode::Batched` when
`HERMON_RUNTIME_MODE` is absent. `crates/hermon-runtime/src/batched.rs` owns
request scheduling and assembles token positions from multiple active
sequences into a `Batch`. Each entry can request that its output logits be kept.
The worker calls one context decode and samples the requested logit row for
each sequence.

The simpler blocking path in `crates/hermon-engine/src/engine.rs` makes the
journey easy to follow: clear per-request context state, tokenize the prompt,
construct a sampler, call context generation, turn returned token IDs into
pieces, frame UTF-8, and emit text. Inside `hermon-llamacpp`, `Context::decode`
passes token IDs through a C shim to `llama_decode`.

The pinned llama.cpp header defines the numerical output contract. After
`llama_decode`, logits requested by the batch are stored as rows; each row has
`n_vocab` columns. `llama_get_logits_ith` retrieves one output row. The context
implementation synchronizes pending computation before returning that result.

Hermon therefore owns substantial systems machinery around the current
forward call—model resolution, runtime choice, batch construction, scheduling,
streaming, and lifecycle—while llama.cpp owns the default real-model numerical
graph and its output-logit storage. A sampler consumes those logits through
the llama.cpp context. ENGINE-1 exposes `Logits` directly because the teaching
goal is to make that otherwise industrial boundary visible.

> **INSIDE HERMON — PREVIEW**
> Hermon's own paged GGUF forward is present behind
> `HERMON_RUNTIME_MODE=paged` and `HERMON_PAGED_GGUF=1`, with a currently stated
> greedy CPU limit. Its real-model differential test compares next-token
> behavior with the pinned llama.cpp graph. The gate means this path is not the
> default CURRENT numerical engine.

Hermon also contains tensor, backend, paged-KV, and native-kernel components.
Their existence is **LIBRARY** evidence unless the request path selects them.
This distinction prevents a common source-reading error: finding a sophisticated
kernel and describing every production request as if it executes there.

ENGINE-1 resembles none of Hermon's production model architecture. It does
establish the abstract contract both systems need:

```text
token identities -> model execution -> one vocabulary score per output row
```

## The fatal context limitation

Return to the approximation:

```text
P(next | history) is approximated by P(next | final token)
```

Consider two histories:

```text
History A: I -> like    -> Rust
History B: I -> dislike -> Rust
```

Both end in the same `TokenId` for `Rust`. ENGINE-1 selects the same `E[Rust]`,
produces the same hidden vector, and calculates the same logits. The
[context limitation diagram](../../diagrams/model/context-limitation.txt)
shows information disappearing at the model boundary.

This is not merely theoretical. The test
`same_last_token_produces_same_logits_despite_different_history` passes two
different-length sequences with different prefixes and the same final token.
It asserts equality of both hidden vectors and the complete logit vectors. A
control with a different final token produces a different embedding and the
expected different logits.

No choice of argmax, softmax, temperature, or sampling can recover history the
model never used. Selection policies operate on the logits they receive. The
missing capability must arise before logits:

```text
We need a mechanism that makes the representation at the current position
depend on context.
```

Then stop. Part II will build that mechanism progressively. Jumping directly
to attention here would skip the tensor, multiplication, normalization,
projection, position, and ownership foundations needed to implement it
correctly.

## Common mistakes

**Treating token IDs as ordinal measurements.** IDs are vocabulary identities.
Only their mapping to parameter rows supplies numerical semantics.

**Calling embedding lookup a required dense multiplication.** One-hot
multiplication is an equivalent equation, not ENGINE-1's physical execution.
The implementation reads one row.

**Giving hidden dimensions human definitions.** A hidden vector is a learned
activation consumed by later computation. Individual coordinates need not
have isolated, stable meanings.

**Calling the hidden vector a probability distribution.** Its values can be
arbitrary finite model activations. They need not be positive or sum to one.

**Calling logits probabilities.** Logits are unnormalized scores. Negative
values and totals other than one are normal.

**Assuming the largest logit has probability one.** Argmax returns an index. It
does not define probability or uncertainty.

**Calling argmax the model.** The model produces the score vector. Argmax is
one selection policy consuming it.

**Putting selection inside `forward`.** Doing so prevents independent testing
of logits and entangles immutable model semantics with generation policy.

**Confusing parameters and activations.** `E`, `W`, and `b` persist across
requests. `h` and `z` belong to a forward execution.

**Mutating weights during inference.** Ordinary forward execution reads the
loaded model. Training or adaptation is a separate operation and ownership
contract.

**Testing only the selected token.** Wrong logit values can preserve the same
argmax. Compare the whole vector with an independent oracle.

**Passing anonymous vectors without shape meaning.** `Vec<f32>` says neither
whether values are hidden activations or logits nor which vocabulary owns
their positions. Lightweight types and construction checks carry the contract.

**Treating shape as prose.** A dimension mismatch changes or invalidates the
computation. Reject it before indexing.

**Assuming matrix notation specifies memory.** `z = W h + b` does not tell us
row-major versus column-major storage, padding, dtype, location, or ownership.

## Exercises

### CHECK

1. For `V=8`, `D=5`, give the shapes of `E`, `W`, `b`, `h`, and `z`.
2. Calculate the untied parameter count and `f32` payload bytes.
3. If input token 6 is valid, how many embedding values does lookup read?
4. Explain why logit position 6 and embedding row 6 must share vocabulary
   identity even though they serve different computations.

### BUILD

Complete [Lab 5](../../labs/lab-05-forward-pass-by-hand.md). Write all four
dot products before running either oracle. Then add a hand oracle for input
`Rust` and verify `[0.3,-0.2,-0.3,-0.5]`.

### BREAK

Complete [Lab 8](../../labs/lab-08-break-the-shape.md). Try malformed lengths,
zero dimensions, a non-finite parameter, an out-of-range ID, and a vocabulary
mismatch. Identify which checks occur once at model construction and which
remain request-dependent.

### EXTEND

Complete [Lab 6](../../labs/lab-06-change-one-weight.md) and
[Lab 7](../../labs/lab-07-same-last-token-same-output.md). Then add a second
small valid fixture with `V=3`, `D=2`. Keep all values hand-computable and
write the expected full vector independently before implementing the Rust
case.

## Summary

A token ID contains identity, not geometry. An embedding matrix gives that
identity a `D`-component numerical representation by selecting one row. An
output projection compares the hidden vector with one row per vocabulary
candidate and adds a bias:

```text
h = E[x]
z = W h + b
```

ENGINE-1 makes each shape, dtype, layout, owner, lifetime, and scalar operation
explicit. Its `V=4`, `D=3` fixture has 28 parameters occupying 112 bytes of
`f32` payload. For input `like`, it produces the independently verified vector:

```text
[-0.7, 0.1, 0.4, 2.2]
```

The fake candidate table is gone from the current model path. Model execution
returns typed finite logits; a separate temporary selector chooses an ID. The
request lifecycle, EOS handling, byte decoding, strict UTF-8 framing,
cancellation, failure, and exactly-one-terminal invariant remain executable.

The model is intentionally inadequate. It uses only the final token, so any
two histories ending in that token produce identical logits. That failure
creates the need for context-dependent representations without hiding the
first numerical inference step behind Transformer complexity.

## What comes next

We now have arbitrary-looking but correct scores:

```text
[-0.7, 0.1, 0.4, 2.2]
```

How should an engine turn them into the next token? Chapter 4—**Logits,
Sampling, and the Autoregressive Loop**—will derive numerically stable softmax,
distinguish greedy and categorical selection, introduce temperature, seeds,
top-k and top-p, append the selected identity to history, execute the model
again, and reconcile EOS and token limits with terminal ownership.

The separation established here must survive:

```text
model semantics -> logits -> logits processing -> sampler -> runtime stopping
```

Chapter 4 will not have to invent logits or guess what they mean. It can begin
at a typed, oracle-verified full vocabulary vector.

## References

- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin,
  [“A Neural Probabilistic Language Model”](https://jmlr.org/papers/v3/bengio03a.html),
  *Journal of Machine Learning Research* 3, 2003.
- PyTorch,
  [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding),
  official lookup-table contract, inspected 2026-09-02.
- PyTorch,
  [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html),
  official affine-layer contract, inspected 2026-09-02.
- Hermon source at commit
  `472a44cdb511b2dae6c9569e59543db8f8350b25`, especially
  `crates/hermon-engine/src/engine.rs`,
  `crates/hermon-runtime/src/dispatch.rs`,
  `crates/hermon-runtime/src/batched.rs`, and
  `crates/hermon-llamacpp/src/linked.rs`.
- llama.cpp source pinned by Hermon at commit
  `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`, especially
  `include/llama.h`, `src/llama-context.cpp`, and `src/llama-sampler.cpp`.
