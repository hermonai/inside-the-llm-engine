# Chapter 4 Research — Logits, Sampling, and the Autoregressive Loop

Status: COMPLETE research basis for manuscript and ENGINE-1 generation.

Inspected: 2026-09-03.

Starting book commit: `d5ac4e8bc12f3fb8dda02ea8c6f6d3fff4084e62`.

ENGINE-1 sampling commit: `5e38b8b612394e56df6aa240a6fe660e0331b51e`.

Hermon commit: `472a44cdb511b2dae6c9569e59543db8f8350b25`.

Pinned llama.cpp commit:
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

## Question

Chapter 3 ends with one finite raw score per vocabulary token. What exact,
owned, testable machinery turns that vector into a token, feeds it back into
the model, and terminates the request once?

The chapter's boundary is:

```text
model.forward(history) -> raw Logits -> processing -> selection -> TokenId
        ^                                                        |
        +---------------- append non-EOS token ------------------+
```

It does not introduce Transformer layers, attention, position, KV caching,
GGUF loading, batching, or accelerators.

## Primary sources inspected

| Subject | Primary source | Use |
| --- | --- | --- |
| Autoregressive language probability | Bengio et al., [A Neural Probabilistic Language Model](https://jmlr.org/papers/v3/bengio03a.html), JMLR 2003 | Sequence probability and next-item conditional framing |
| Nucleus/top-p and top-k comparison | Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), ICLR 2020 | Dynamic probability-mass nucleus and decoding-policy evidence |
| Stable softmax analysis | Blanchard, Higham, and Higham, [Accurately computing the log-sum-exp and softmax functions](https://doi.org/10.1093/imanum/draa038), IMA Journal of Numerical Analysis 2021 | Shifted softmax and floating-point analysis |
| Softmax definition | PyTorch, [`Softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) | Official formula and distribution properties |
| SplitMix design | Steele, Lea, and Flood, [Fast Splittable Pseudorandom Number Generators](https://doi.org/10.1145/2660193.2660195), OOPSLA 2014 | Small deterministic non-cryptographic PRNG provenance |
| llama.cpp public sampler API | Pinned `vendor/llama.cpp/include/llama.h:1174-1321` | Chain API, greedy/dist/top-k/top-p/temperature contracts |
| llama.cpp implementation | Pinned `vendor/llama.cpp/src/llama-sampler.cpp:218-326,920-1420,1798-1888` | Max-shift softmax, seeded distribution, tie behavior, filters |

External sources support general semantics. Current Hermon and pinned library
claims below come from the recorded local source, not from changing web pages.

## Semantics fixed before manuscript drafting

### General autoregressive model

For token variables `x_1 ... x_n`, the chain rule is:

```text
P(x_1, ..., x_n)
= P(x_1) P(x_2 | x_1) ... P(x_n | x_1, ..., x_{n-1})
```

At generation step `t`, a causal language model produces one score vector for
`x_(t+1)` conditioned on the visible history. A decoding policy chooses one
token, the runtime adds that token to the history, and the next forward call
uses the enlarged history.

ENGINE-1 implements the loop but not general context sensitivity. Its model
reads only `history.last()`. Therefore two histories ending in the same token
produce the same logits. That limitation is an explicit Chapter 5 handoff, not
a property of autoregressive models in general.

### Boundary and ownership

The model does not sample:

```text
Model::forward(&history) -> ForwardPass { raw Logits, ... }
SamplerState::sample(&raw_logits) -> SamplingStep { TokenId, ... }
Runtime -> sequence mutation, decode, stream, stop, terminal transition
```

`Logits` remains a finite immutable model artifact. Stochastic processing uses
a separate `Vec<Option<f64>>` workspace, so masking never inserts `-infinity`
into the raw `f32` model boundary. The request owns immutable
`SamplingConfig`; `generate` constructs a fresh mutable `SamplerState` with an
RNG and sample count. No mutable global RNG exists.

### ENGINE-1 modes

```rust
enum SamplingConfig {
    Greedy,
    Stochastic(StochasticConfig {
        temperature,
        top_k,
        top_p,
        seed,
    }),
}
```

Greedy is a separate mode. ENGINE-1 does not call it “temperature-zero
softmax,” never divides by zero, and does not run softmax for argmax.

Stochastic validation:

- `temperature` must be finite and greater than zero;
- `top_k=None` disables top-k;
- `top_k=0` is invalid;
- `top_k >= V` changes no candidates;
- `top_p=None` disables top-p;
- `top_p` must be finite and in `(0,1]`;
- `top_p=1` changes no candidates.

Invalid configuration is a typed failure. There is no silent greedy fallback.

### Fixed stochastic pipeline

ENGINE-1 chooses and tests this order:

```text
finite raw f32 logits
        |
        v
copy to f64 workspace and divide by temperature
        |
        v
top-k mask in logit space
        |
        v
stable softmax over retained scores
        |
        v
top-p filter in probability space, crossing token included
        |
        v
renormalize retained probability mass
        |
        v
SplitMix64 draw in [0,1) -> cumulative categorical selection
```

Filter order is policy. Other engines may expose another order. The chapter
must not imply mathematical commutativity or universal convention.

Top-k can operate on logits because positive-temperature scaling and softmax
preserve ordering. Top-p needs normalized probability mass. Equal logits or
probabilities sort by token ID ascending after score descending.

## Softmax derivation and numerical behavior

For finite score vector `z` with vocabulary length `V`:

```text
p_i = exp(z_i) / sum_(j=0)^(V-1) exp(z_j)
```

Every numerator is positive, so `p_i >= 0`; dividing by the common positive
sum makes `sum_i p_i = 1` in exact arithmetic.

Naively computing `exp(1000)` overflows ordinary floating-point formats. Let
`m = max_j z_j` and compute:

```text
p_i = exp(z_i - m) / sum_j exp(z_j - m)
```

Proof of shift invariance for any constant `c`:

```text
exp(z_i + c) / sum_j exp(z_j + c)
= exp(z_i) exp(c) / (exp(c) sum_j exp(z_j))
= exp(z_i) / sum_j exp(z_j)
```

Choosing `c=-m` makes every exponential argument non-positive and one exactly
zero. The largest numerator is therefore `exp(0)=1`; overflow is avoided and
the denominator is positive.

Independent Python and Rust values for `[1,2,3]`:

```text
m = 3
shifted = [-2,-1,0]
exp = [0.1353352832, 0.3678794412, 1.0]
sum = 1.5032147244
p = [0.0900305732, 0.2447284711, 0.6652409558]
```

`[1000,999,998]` produces the reverse probability order with finite values:

```text
[0.6652409558, 0.2447284711, 0.0900305732]
```

Greedy equivalence follows because both division by positive temperature and
`exp` are strictly increasing. Therefore:

```text
argmax(z) = argmax(softmax(z))
```

Softmax is unnecessary for greedy selection.

## Temperature

For stochastic mode and `T > 0`:

```text
z'_i = z_i / T
```

`T<1` magnifies score gaps and sharpens the distribution. `T=1` preserves the
scores. `T>1` reduces gaps and flattens the distribution. It does not change
ordering, and it does not itself add randomness; the RNG-backed categorical
draw does that.

Oracle for `[1,2,3]`:

| T | Probabilities |
| ---: | --- |
| 0.5 | `[0.0158762400, 0.1173104278, 0.8668133322]` |
| 1.0 | `[0.0900305732, 0.2447284711, 0.6652409558]` |
| 2.0 | `[0.1863237232, 0.3071958857, 0.5064803911]` |

Token 2 remains highest in all three cases.

## Greedy selection

Greedy scans once and replaces the current best only for strict `>`.
Iteration is in vocabulary order, so equal maximum logits choose the lowest
token ID. This is explicit, tested behavior.

Greedy is deterministic and cheap. It is a local next-token choice, not proof
of the globally highest-probability complete sequence, and it can make output
repetitive. The chapter must avoid calling it universally best.

## Categorical sampling

For normalized `[p_0,...,p_(V-1)]`, draw `r` in `[0,1)` and select the first
index whose cumulative probability is strictly greater than `r`:

```text
token 0 interval: [0, p_0)
token 1 interval: [p_0, p_0+p_1)
...
```

ENGINE-1's pure `categorical_select(probabilities, draw)` separates this logic
from its RNG. Tests use artificial boundary draws, and the Python oracle can
accept the same artificial draw. If floating rounding leaves the final sum a
few ulps below one, a surviving draw maps to the final positive-probability
candidate only after the distribution has passed finite/non-negative/sum
validation.

## Top-k and top-p

Top-k retains at most a fixed number of candidates. `k=1` leaves one candidate
with probability one after normalization and is effectively greedy for this
pipeline, though it still takes the stochastic code path.

Top-p (nucleus) sorts candidates by probability descending and retains the
smallest prefix whose cumulative probability reaches or exceeds `p`. The token
that crosses the threshold is included. For:

```text
A=.40 B=.30 C=.15 D=.10 E=.05, p=.80
```

ENGINE-1 retains A, B, and C (mass .85), then renormalizes them to roughly:

```text
[.470588, .352941, .176471, 0, 0]
```

Top-k has fixed cardinality; top-p has distribution-dependent cardinality.
Neither is declared generally superior.

## Randomness and reproduction contract

ENGINE-1 uses a local SplitMix64 implementation with wrapping `u64` arithmetic.
The seed initializes one request's state. Each successful stochastic sample
consumes one value. The high 53 bits are divided by `2^53` to create a binary64
draw in `[0,1)`. A pinned three-value output vector tests the algorithm.

The implementation is pedagogical and non-cryptographic. It must never be
described as secure randomness.

Promise:

> For the same ENGINE-1 commit, executable/toolchain/target, model parameters,
> tokenizer, prompt tokens, sampling configuration, and seed, the scalar path
> repeats the same token sequence.

This does not promise identical text across engines, providers, versions,
quantizations, floating-point libraries, or models. Seed is one input to
reproduction, not a universal determinism switch.

## Autoregressive runtime order

The implemented request loop is:

1. Validate prompt, output budget, and sampling configuration.
2. Construct request-local `SamplerState`.
3. Admit and start execution.
4. Before each forward, check cancellation and `max_new_tokens`.
5. Build history from immutable prompt tokens plus committed generated tokens.
6. Call `model.forward(history)` and validate logit vocabulary length.
7. Record raw forward trace without mutating raw logits.
8. Check cancellation again before sampling.
9. Sample a token and trace processed probabilities/draw when stochastic.
10. If token is EOS, do not append, decode, or stream it; request completes.
11. Otherwise commit token to generation state, emit token identity, decode its
    bytes, feed strict UTF-8 framing, and emit complete text.
12. Repeat.
13. One `Lifecycle::finish` transition emits the sole terminal event.

The commit point is step 11: a non-EOS token enters `GenerationState` before
tokenizer/UTF-8 streaming. If decoding later fails, the result accurately
reports the committed token and one failed terminal outcome. EOS is a control
decision, not user-visible text, and is not counted in generated text tokens.

`max_new_tokens` counts committed non-EOS output tokens only; prompt length is
separate. Stop-string matching is deferred.

Cancellation checks before forward and after forward prevent starting more
work or committing a sampled token after cancellation is observed. The current
engine is synchronous; later asynchronous cancellation needs stronger
coordination without changing this ownership rule.

## Failure model

Typed sampling errors cover:

- empty scores;
- non-finite processed scores or probabilities;
- invalid temperature, top-k, top-p, or artificial draw;
- negative probabilities;
- invalid probability sum;
- all candidates filtered;
- token index overflow.

Runtime failures also retain typed category boundaries for request, model,
tokenizer, sampling, and UTF-8 errors. No failure branch emits `Done`; all call
the same terminal owner exactly once.

## Independent oracle

`code/reference/python/chapter04_sampling_oracle.py` independently implements:

- stable softmax;
- temperature;
- deterministic top-k tie order;
- top-p crossing-token inclusion;
- renormalization;
- fixed-draw categorical selection.

It imports no Rust code and runs internal numerical assertions before printing
JSON. Verified command:

```text
python3 code/reference/python/chapter04_sampling_oracle.py \
  --logits -0.7 0.1 0.4 2.2 --temperature 1 \
  --top-k 3 --top-p 0.9 --draw 0.63
```

Result: retained tokens `[2,3]`, probabilities approximately
`[0,0,.1418510649,.8581489351]`, selected token `3`, PASS.

## Test matrix and results

At sampling commit `5e38b8b`, `cargo test --workspace` passes 83 tests:

- 20 dedicated sampling tests;
- stable and large-logit softmax;
- finite/non-negative/sum and shift invariants;
- greedy argmax, ties, and empty input;
- three temperatures;
- top-k selection, ties, `k=1`, and `k>V`;
- top-p nucleus, crossing token, `p=1`, renormalization, and combined order;
- fixed-draw boundaries near zero and one;
- pinned PRNG vector and unit interval;
- same-seed repetition, different-seed divergence, per-request isolation;
- invalid configurations and all-filtered distribution;
- raw-logit preservation and a deterministic property grid;
- EOS, budget, cancellation before commit, model/sampler/tokenizer/UTF-8
  failure, no post-terminal output, and exactly one terminal.

## Performance experiment

Methodology and raw output:
`research/benchmarks/chapter-04-sampling-cost.md`.

The Apple M1 exploratory medians at `V=4096` were 5,137 ns greedy, 41,467 ns
softmax plus categorical, 107,582 ns top-k 40, and 127,452 ns top-p .9. The
measurement excludes model forward and cannot support a production latency
claim. It illustrates the operations: greedy scans, stochastic processing adds
exponentials and cumulative work, and clarity-first filters sort.

## Current Hermon findings

### CURRENT — default batched path

- `crates/hermon-runtime/src/batched.rs:82-135` stores sampler configuration in
  a submitted request and a concrete mutable llama.cpp `Sampler` in each
  `ActiveSeq`.
- `batched.rs:523-609` constructs a new sampler at admission. Concurrent
  sequences do not share one sampler object.
- `batched.rs:618-1050` batches model work but samples each eligible sequence
  from its own logit row and sampler state.
- `batched.rs:1004-1025` checks EOG and the completion-token limit before
  streaming the candidate; it maintains a pending token for the next decode.
- `batched.rs:1113-1127` sends `StreamItem::Done(EngineMetrics)` on finalize.
  Error branches use `Err(EngineError)` instead. Chapter prose should describe
  this as a stream contract, not claim Hermon shares ENGINE-1's exact enum.

### CURRENT — engine/llama.cpp facade

- `crates/hermon-engine/src/sampler.rs` defines `SamplerConfig` and a separate
  `SamplerState` facade/history type. The executing default batched path lowers
  configuration to `hermon_llamacpp::SamplerConfig` and uses the linked
  sampler object.
- `crates/hermon-llamacpp/src/linked.rs:800-874` owns the native sampler chain
  in `Sampler`; `sample(&mut self, &mut Context, i)` reads one logit row.
- `linked.rs:1040-1080` shows the simpler iterator path: sample, stop on EOG,
  decode the selected token to advance KV, decrement the budget, return token.
- `csrc/shim.c:352-395` builds greedy for `temperature<=0`; otherwise its
  Hermon-specific chain order at this commit is top-k, top-p, min-p,
  temperature, distribution. Seed zero becomes time-derived in the shim.

### PREVIEW — Hermon-owned paged GGUF path

- `crates/hermon-runtime/src/paged.rs:1733-1818` currently uses a local
  deterministic `argmax` for the gated paged path.
- `docs/CORE_ENGINE_ARCHITECTURE.md:263-268` and `docs/ROADMAP.md:120` describe
  CPU plus greedy sampling and the unfinished temperature-zero equivalence
  corpus as release gates.

The manuscript must not flatten CURRENT llama.cpp-backed stochastic sampling
and PREVIEW Hermon-owned greedy paged execution into one claim.

## Pinned llama.cpp findings

- `include/llama.h:1174-1194` documents a sampler chain ending in a selection
  sampler such as greedy or distribution.
- `llama-sampler.cpp:956-970` greedy uses strict `>` while scanning, so its
  current CPU tie behavior keeps the first candidate in array order.
- `llama-sampler.cpp:1036-1090` distribution computes a maximum-shifted
  exponential sum and draws from request-owned `std::mt19937` state.
- `llama-sampler.cpp:1255-1260` top-k delegates to score ordering and truncation.
- `llama-sampler.cpp:1351-1398` top-p computes probabilities, sorts, and
  includes the candidate that crosses cumulative `p`.
- `llama-sampler.cpp:1807-1811` temperature delegates to logit scaling.

These facts describe the pinned library version. ENGINE-1 intentionally does
not copy its implementation or promise RNG identity with it.

## Planned canonical diagrams

1. Sampling pipeline and control/data arrows.
2. Autoregressive generation state.
3. Categorical cumulative intervals.
4. Follow the token: text to logits to feedback.
5. Follow the byte: parameters to output stream.
6. Follow the owner: model/request/step/stream lifetimes.
7. Reusable model with two independent request samplers.

## Labs

Register Labs 9–15:

1. Stable softmax by hand.
2. Temperature distributions.
3. Fixed categorical draw.
4. Top-k versus top-p.
5. Trace the complete autoregressive loop.
6. Change the seed and state the reproduction boundary.
7. Break sampler configuration and require typed failure.

Existing later planned labs move to 16–28 without changing their chapter
scope.

## Misconceptions to correct

- logits are probabilities;
- softmax belongs inside every model forward;
- highest logit must always be sampled;
- temperature creates randomness or changes ordering;
- top-p keeps `p` percent of tokens;
- top-k and top-p are interchangeable;
- a fixed seed guarantees identical output everywhere;
- sampling is stateless;
- EOS is visible output;
- `max_new_tokens` includes prompt tokens;
- greedy needs softmax;
- stochastic generation is random noise.

## Material open questions

No Chapter 4 gate remains open. Later chapters must decide:

- stop-string matching across token/UTF-8 boundaries;
- logit bias, repetition/frequency penalties, and constrained grammars;
- sampling-state behavior in continuous batching and speculation;
- optimized selection algorithms and hardware placement;
- broader reproducibility modes and deterministic math.

Those questions must not expand Chapter 4.

## Part II handoff

Part I now has a real, complete loop. Its remaining model defect is clear:
ENGINE-1's logits depend only on the final token. Chapter 5 begins tensor rank,
shape, dtype, row-major layout, stride, contiguity, views, copies, aliasing,
ownership, bounds, and overflow-safe element counts. It must not jump directly
to attention.
