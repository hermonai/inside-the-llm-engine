# Chapter 4 — Logits, Sampling, and the Autoregressive Loop

Chapter 3 ended with a vector:

```text
[-0.7, 0.1, 0.4, 2.2]
```

Those four numbers are the complete output of one ENGINE-1 model forward for
the token `like`. They say that candidate token 3, `Rust`, has the largest raw
score. They do not themselves say whether the runtime must choose `Rust`, how
randomness enters, what temperature changes, or what happens after a token is
chosen.

That missing machinery is the subject of this chapter.

We will turn one forward result into a complete generation loop:

```text
text -> token IDs -> model -> logits -> sampling policy -> next TokenId
          ^                                              |
          +---------------- feedback --------------------+
```

The result is small, but it is real. ENGINE-1 will support a separate greedy
path, numerically stable softmax, categorical sampling, temperature, top-k,
top-p, seeded request-local randomness, EOS, output-token budgets,
cancellation, failure, byte-safe streaming, and exactly one terminal outcome.
Nothing outside the model will manufacture a candidate table. Every generated
token will come from actual model logits.

## The model has spoken—only in numbers

A language model and an inference engine have different responsibilities.
The model maps token history to scores. The engine decides how to use those
scores, owns mutable generation state, returns bytes, and stops the request.

> **FIRST PRINCIPLE**
> The model predicts scores; the sampler chooses an output token.

Keeping this separation seems fussy in a four-token engine. It becomes vital
when one model serves concurrent requests with different sampling policies,
when a grammar masks candidates, when speculative decoding verifies proposed
tokens, or when an operator needs to compare raw model output with the policy
that transformed it.

ENGINE-1 therefore keeps this boundary:

```rust
Model::forward(&history) -> ForwardPass { logits, ... }
SamplerState::sample(&logits) -> SamplingStep { token_id, ... }
```

`Logits` is evidence about what the model predicted. It is finite, typed, and
unchanged by sampling. A separate workspace holds temperature-adjusted and
filtered candidates.

## What autoregressive generation means

Suppose a token sequence contains random variables `x_1,...,x_n`. The chain
rule factors its joint probability:

```text
P(x_1,...,x_n)
= P(x_1) P(x_2|x_1) ... P(x_n|x_1,...,x_(n-1))
```

An autoregressive language model repeatedly predicts the next token given the
tokens already visible. At step `t`, it produces scores for `x_(t+1)`. The
engine chooses one candidate, appends that token to the sequence, and runs the
model again. Bengio and colleagues' early [neural probabilistic language
model](https://jmlr.org/papers/v3/bengio03a.html) is one primary source for
this conditional sequence framing.

The general statement is about full visible history. ENGINE-1 has a deliberate
limitation: its `TinyLanguageModel` reads only `history.last()`. Two histories
with the same final token therefore produce identical logits, even if every
earlier token differs. ENGINE-1 implements genuine autoregressive feedback;
its representation is not yet context-sensitive.

That distinction matters. “Autoregressive” describes how outputs become
future inputs. It does not guarantee that a particular model uses all history
well—or at all.

## From logits to a probability distribution

A logit is an unnormalized score. It may be negative, zero, or positive. Its
absolute value is not a probability, and the vector does not need to sum to
anything meaningful.

For a finite score vector `z` of length `V`, softmax defines:

```text
                  exp(z_i)
p_i = --------------------------------
      sum_(j=0)^(V-1) exp(z_j)
```

Here:

- `V` is vocabulary size;
- `z_i` is the raw score for token ID `i`;
- `p_i` is that token's normalized probability;
- `z` is `[V]` model output;
- `p` is a `[V]` request-local distribution.

Every exponential is positive. The denominator is the sum of all numerators.
Therefore every `p_i` is non-negative and the probabilities sum to one in
exact arithmetic. PyTorch's official [`Softmax`
documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
states the same mathematical contract.

For logits `[1,2,3]`, the direct calculation is approximately:

```text
exp values = [2.718282, 7.389056, 20.085537]
sum        = 30.192875
p          = [0.090031, 0.244728, 0.665241]
```

Token 2 remains the largest, but now the values describe a distribution from
which a categorical outcome can be drawn.

### Why naive softmax fails

Try logits `[1000,999,998]`. The ordering and gaps are harmless. The direct
implementation is not: `exp(1000)` overflows ordinary floating-point storage.
An infinite numerator can produce infinite sums and undefined divisions.

The mathematical formula needs an equivalent numerical form. Let:

```text
m = max(z)
```

Then compute:

```text
                      exp(z_i - m)
p_i = ---------------------------------------------
      sum_(j=0)^(V-1) exp(z_j - m)
```

Subtracting one constant from every score does not alter softmax. For any
constant `c`:

```text
exp(z_i+c) / sum_j exp(z_j+c)
= exp(z_i)exp(c) / (exp(c)sum_j exp(z_j))
= exp(z_i) / sum_j exp(z_j)
```

With `c=-m`, every exponential argument is at most zero. At least one is
exactly zero, whose exponential is one. Numerators cannot overflow, and the
denominator cannot be zero.

This is **stable softmax**. Blanchard, Higham, and Higham analyze shifted
formulas in [Accurately computing the log-sum-exp and softmax
functions](https://doi.org/10.1093/imanum/draa038). ENGINE-1 uses the
clarity-first maximum, exponential, sum, and normalization passes in `f64` for
the sampling workspace while preserving the model's raw `f32` logits.

For `[1000,999,998]`:

```text
m       = 1000
shifted = [0,-1,-2]
p       = [0.6652409558,0.2447284711,0.0900305732]
```

All results are finite.

> **ENGINEERING FAILURE — NAIVE SOFTMAX**
> A mathematically recognizable expression can still be an invalid numerical
> program. Algebraic equivalence does not imply equal floating-point behavior.

## Greedy decoding needs no softmax

Softmax preserves ordering because exponentiation is strictly increasing and
every candidate shares the same positive denominator:

```text
argmax(z) = argmax(softmax(z))
```

Greedy decoding can scan raw logits and choose the maximum. It does not need
exponentials or normalized probabilities.

ENGINE-1 treats greedy as a separate `SamplingConfig::Greedy` mode. It never
divides by temperature zero and does not approximate greedy with a tiny
positive temperature. A strict `>` comparison replaces the current winner;
iteration proceeds in token-ID order. Equal maxima therefore choose the lowest
token ID, deliberately and reproducibly.

Greedy is deterministic and inexpensive: one `O(V)` scan. It is not globally
optimal sequence search. Choosing the locally highest next-token score does
not prove that the resulting complete sequence has the greatest joint
probability. Greedy can also produce dull or repetitive text. Those are policy
tradeoffs, not reasons to hide its simple semantics.

## Randomness is request state

Random sampling is not a command sent to the model. It is stateful computation
in the engine.

```text
REQUEST
  prompt and generated history
  immutable sampler configuration
  mutable PRNG state
  successful sample count
  remaining output budget
  terminal state
```

A pseudorandom number generator, or PRNG, advances deterministic internal
state. A seed chooses its starting state. Give the same algorithm the same
seed and consume values in the same order, and it produces the same number
sequence.

ENGINE-1 implements SplitMix64, derived from the design described by Steele,
Lea, and Flood in [Fast Splittable Pseudorandom Number
Generators](https://doi.org/10.1145/2660193.2660195). It uses wrapping `u64`
arithmetic and converts the high 53 output bits into a binary64 draw in
`[0,1)`.

This generator is compact, deterministic, and testable. It is not
cryptographically secure, and the engine never describes it that way.

Each admitted request constructs a fresh `SamplerState`. Two concurrent
requests do not share one mutable RNG. Otherwise scheduling would change which
request consumes which draw:

```text
global draws: r0 r1 r2 r3 ...

schedule A,B,A,B -> A gets r0,r2; B gets r1,r3
schedule A,A,B,B -> A gets r0,r1; B gets r2,r3
```

The model and seeds could be identical while outputs changed with scheduling.
Per-request state removes that coupling.

> **FIRST PRINCIPLE**
> Randomness is mutable request state, not a hidden global service.

### What a seed guarantees

ENGINE-1's promise is intentionally bounded:

> For the same ENGINE-1 commit, executable/toolchain/target, model parameters,
> tokenizer, prompt tokens, sampling configuration, and seed, the scalar path
> repeats the same token sequence.

A fixed seed alone does not guarantee identical output across engine versions,
models, providers, quantizations, RNG algorithms, floating-point libraries, or
sampler orders. Even tiny logit or cumulative-boundary changes can map a draw
to another token.

“Seed equals determinism everywhere” is therefore false. A seed is one member
of a reproduction contract.

## Categorical sampling, one interval at a time

Suppose probabilities are:

```text
[0.20,0.30,0.50]
```

Their cumulative boundaries divide `[0,1)`:

```text
0.0                 0.2                         0.5                    1.0
 |------ token 0 -----|----------- token 1 -------|------- token 2 -----|
```

Draw `r` uniformly in `[0,1)`. Select the first cumulative boundary strictly
greater than `r`:

```text
r=0.19 -> token 0
r=0.20 -> token 1
r=0.49 -> token 1
r=0.50 -> token 2
r=0.99 -> token 2
```

ENGINE-1 exposes the pure boundary as:

```rust
categorical_select(probabilities, draw) -> Result<TokenId, SamplingError>
```

The RNG is only one producer of `draw`. Tests and the independent Python
oracle can supply an artificial value. This separation prevents one failing
test from becoming an unsolved mixture of softmax, filtering, PRNG, and
cumulative-selection bugs.

The function validates finite, non-negative probabilities and a sum within
tolerance of one. Floating-point rounding can leave the last cumulative sum a
few ulps below one. If a valid draw survives every earlier interval, the
implementation returns the final positive-probability candidate. It never uses
that fallback to rescue an invalid or empty distribution.

## Temperature changes shape, not randomness

For stochastic mode, ENGINE-1 requires finite `T>0` and transforms scores:

```text
z'_i = z_i / T
```

Then it computes the distribution. Temperature affects score gaps:

- `T<1` magnifies gaps and sharpens the distribution;
- `T=1` leaves scores unchanged;
- `T>1` shrinks gaps and flattens the distribution.

Positive division preserves ordering. For logits `[1,2,3]`:

| Temperature | Probabilities |
| ---: | --- |
| 0.5 | `[0.015876,0.117310,0.866813]` |
| 1.0 | `[0.090031,0.244728,0.665241]` |
| 2.0 | `[0.186324,0.307196,0.506480]` |

Token 2 stays highest. The cold distribution concentrates more mass on it;
the hot distribution spreads more mass to alternatives.

Temperature does not produce a draw. If selection remains greedy, changing a
positive temperature cannot change the winner. Randomness comes from
categorical sampling.

ENGINE-1 rejects zero, negative, NaN, and infinite stochastic temperatures.
Greedy remains a different mode. This makes the API say what it means.

## Top-k: a fixed candidate count

A production vocabulary may contain tens of thousands of candidates. Many
receive tiny probability. Top-k keeps the `k` highest scores and removes the
rest before sampling:

```text
scores -> order/select highest k -> mask others -> normalize -> sample
```

Because softmax preserves ordering, ENGINE-1 applies top-k in logit space. Its
simple teaching implementation sorts indexes by score descending and token ID
ascending. It stores removed entries as `None` in a workspace rather than
putting negative infinity into raw `Logits`.

The edge contract is explicit:

- `None`: disabled;
- `k=0`: invalid;
- `k=1`: one survivor, hence probability one;
- `k>=V`: no filtering.

Sorting costs `O(V log V)` in this implementation. A production sampler may
use partial selection, but optimization would not change the survivor
semantics or deterministic tie rule.

## Top-p: an adaptive probability mass

Top-p, or nucleus sampling, keeps a variable-size set. Sort candidates by
probability descending and retain the smallest prefix whose cumulative mass
reaches threshold `p`. Holtzman and colleagues introduced and evaluated this
policy in [The Curious Case of Neural Text
Degeneration](https://arxiv.org/abs/1904.09751).

Consider:

| Token | Probability |
| --- | ---: |
| A | 0.40 |
| B | 0.30 |
| C | 0.15 |
| D | 0.10 |
| E | 0.05 |

For `p=0.80`, A and B provide only `0.70`. C crosses the threshold, so C is
included. The nucleus is A/B/C with mass `0.85`. ENGINE-1 renormalizes:

```text
[0.470588,0.352941,0.176471,0,0]
```

Top-k retains a fixed count. Top-p retains an adaptive count determined by the
distribution's concentration. Neither is universally superior.

ENGINE-1 accepts `p` only in `(0,1]`; `p=1` is a deliberate no-op. Equal
probabilities use token ID ascending as the secondary order.

## Filter order is inference behavior

Temperature, top-k, and top-p are not a bag of commutative options. Their order
can change the candidate set and final probabilities.

ENGINE-1 fixes this pipeline:

```text
raw finite logits
  -> copy and divide by temperature
  -> top-k mask in logit space
  -> stable softmax over survivors
  -> top-p filter in probability space
  -> renormalize
  -> categorical draw
```

The implementation and prose use this exact order. Configuration fields do not
silently determine order.

Other engines can choose differently. In particular, the current Hermon shim
examined later builds a llama.cpp-backed chain with a different order. That is
not a contradiction. Sampling order is policy, and reproducibility requires
identifying the policy.

[The canonical sampling pipeline](../../diagrams/sampling/sampling-pipeline.txt)
keeps raw logits separate from processed candidates.

> **ENGINEERING FAILURE — DESTROYED EVIDENCE**
> If top-k mutates raw logits in place, a later debugger sees masks rather than
> model output. Keep the model prediction and the sampler decision as separate
> artifacts.

## The complete ENGINE-1 loop

The runtime's operation order now has a precise commit point:

1. Validate input tokens, `max_new_tokens`, and sampling configuration.
2. Construct a fresh request-owned `SamplerState`.
3. Admit the request and begin execution.
4. Check cancellation and output budget.
5. Build history from prompt plus committed output tokens.
6. Run `Model::forward` and validate exactly `V` logits.
7. Preserve and optionally trace the raw forward result.
8. Check cancellation again before sampling.
9. Select a `TokenId`; trace probabilities and draw for stochastic mode.
10. If it is EOS, stop without decoding or streaming it.
11. Otherwise append it to `GenerationState`. This is the commit point.
12. Emit token identity, decode bytes, frame strict UTF-8, and emit text.
13. Repeat or transition exactly once to terminal state.

The [autoregressive state
diagram](../../diagrams/sampling/autoregressive-state.txt) distinguishes data
movement from stop checks.

### A complete greedy trace

Prompt text:

```text
I like
```

Tokenizer output:

```text
[1,2]
```

The last token is `like`. Chapter 3 computed:

```text
hidden = [1,-0.5,2]
logits = [-0.7,0.1,0.4,2.2]
argmax = 3 = Rust
```

The runtime appends token 3:

```text
[1,2,3]
```

It decodes token 3 to the byte piece for ` Rust`, emits valid text, and runs
the model again. ENGINE-1 reads the final `Rust` token:

```text
hidden = [-1,0,0]
logits = [0.3,-0.2,-0.3,-0.5]
argmax = 0 = <eos>
```

EOS completes the request. It is not added to visible output, decoded, or
streamed. The complete model-derived path is:

```text
"I like" -> [1,2] -> Rust -> [1,2,3] -> <eos> -> completed
```

No fake candidate table produces `Rust`.

### A seeded stochastic trace

Run:

```sh
cd code/mini-engine
cargo run -p engine0 -- --trace --sample --temperature 1 \
  --top-k 3 --top-p .9 --seed 42 --max-tokens 3 'I like'
```

At the first step, ENGINE-1 records raw logits, then:

```text
probabilities = [0,0,0.1418510634,0.8581489366]
draw          = 0.7415648788
selected      = 3
```

At the next step:

```text
probabilities = [0.4639634327,0.2814080427,0.2546285245,0]
draw          = 0.1599103929
selected      = 0 = <eos>
```

The same bounded reproduction environment repeats those draws and tokens.
Timing fields are not part of semantic equality.

## EOS, budgets, and visible output

EOS is a vocabulary token with a model score. It is also semantic control.
ENGINE-1 chooses this policy:

- sampled EOS ends generation;
- EOS is not committed to the visible generated-token list;
- EOS is not decoded to bytes;
- EOS is not streamed as ordinary text;
- terminal reason is `EndOfSequence`.

`max_new_tokens` counts committed non-EOS output tokens. A prompt of ten tokens
and `max_new_tokens=5` allows at most five new visible tokens; it does not cap
the total sequence at five.

Application stop strings are more complicated. A string can span tokens and
UTF-8 pieces, and policy must decide whether matching bytes are included or
removed. Chapter 4 defers that design. Its core stop set is EOS, output budget,
cancellation, and failure.

## Cancellation and failure without partial ambiguity

Cancellation is checked before a forward and again after forward before
sampling. If cancellation is observed at either point, no later token is
committed. The current engine is synchronous, but this placement establishes
the invariant an asynchronous runtime must preserve.

Sampling can fail. Invalid temperature, top-k, or top-p does not silently fall
back to greedy. Empty logits, non-finite processed values, negative
probabilities, invalid sums, invalid draws, and empty candidate sets have typed
errors.

For a non-EOS token, ENGINE-1 commits it before tokenizer/UTF-8 processing. If
decoding fails afterward, the generation result contains the committed token
and a failed terminal outcome. This is more truthful than pretending the model
never selected it. Stream consumers may already have received token identity,
so rollback would be dishonest without a richer transactional protocol.

Terminal delivery still has one owner: `Lifecycle::finish`. Every successful,
cancelled, or failed branch returns an outcome to that owner. It emits one
terminal stream event, records one terminal trace event, and closes future
emission. A second transition returns `AlreadyTerminal`.

> **ENGINEERING FAILURE — DOUBLE TERMINATION**
> If the EOS branch emits `Done` and loop cleanup emits another `Done`, the
> client sees two endings. Terminal success is a state transition, not an
> informal callback.

> **ENGINEERING FAILURE — POST-EOS OUTPUT**
> If the loop appends or decodes another token before checking EOS, its sequence
> and stream disagree with model control. Stop ordering must be explicit.

## Follow the token, byte, and owner

Part I can now complete all three recurring journeys.

### Follow the token

```text
human text
 -> tokenizer
 -> TokenId history
 -> embedding row
 -> hidden activation
 -> output projection
 -> raw logits
 -> temperature/filtering
 -> greedy or categorical selection
 -> new TokenId
 -> append
 -> model again
```

The reusable [Part I token
diagram](../../diagrams/sampling/part1-follow-token.txt) also shows the decode
and stream branch.

### Follow the byte

Model parameter bytes are read by scalar `f32` arithmetic. The result is a
finite raw `f32` logit vector. Stochastic processing copies values to a
request-local `f64` workspace. Selection yields an integer token ID. The
tokenizer maps that ID to bytes, the UTF-8 decoder holds incomplete suffixes,
and only valid complete text reaches the stream.

See [Part I follow the
byte](../../diagrams/sampling/part1-follow-byte.txt).

### Follow the owner

The model lifetime owns immutable parameters and the tokenizer/model contract.
A request owns prompt tokens, committed output, sampling configuration, RNG
state, budget, UTF-8 buffer, and terminal state. A forward step owns temporary
hidden values, raw logits, processed candidates, and one selection.

See [Part I follow the
owner](../../diagrams/sampling/part1-follow-owner.txt).

This separation prepares the engine for concurrency:

```text
                         shared MODEL
                      immutable weights
                       /             \
              Request A               Request B
             history/RNG A           history/RNG B
```

The model can be shared. Mutable sequence and sampler state cannot. A future
continuous batch may evaluate several sequences together, but sampling remains
logically per sequence. The [two-request ownership
diagram](../../diagrams/sampling/model-two-requests.txt) makes that future
constraint visible.

## Build it: the sampler and runtime

The Chapter 4 implementation lives in:

```text
code/mini-engine/crates/engine0/src/sampling.rs
code/mini-engine/crates/engine0/src/lib.rs
```

The public design separates immutable and mutable pieces:

```rust
SamplingConfig       // mode and parameters, carried by Request
SamplerState         // one RNG and sample count per generate call
ProbabilityDistribution
SamplingStep         // token, optional probabilities/draw, sample index
SamplingError        // typed validation and numerical failures
```

The model still returns `Logits`. It has no RNG, no temperature, and no stop
policy. The runtime no longer carries the old temporary `GreedySelector`.
Instead it creates a sampler from each request's configuration.

Trace mode exposes tiny educational fixtures:

```text
[model] raw logits
[sampler] immutable configuration
[sampler] processed probabilities, when stochastic
[sampler] request-local draw, when stochastic
[sampler] selected TokenId
[sequence] commit non-EOS token
[tokenizer/stream] bytes and valid text
```

Production tracing should not log sensitive prompts or huge tensors by
default. ENGINE-1's `TraceSink` already requires an explicit opt-in for full
values.

## Prove it: an independent oracle

`code/reference/python/chapter04_sampling_oracle.py` independently implements
stable softmax, temperature, top-k, top-p, renormalization, and fixed-draw
categorical selection. It imports no Rust engine code.

Run:

```sh
python3 code/reference/python/chapter04_sampling_oracle.py \
  --logits -0.7 0.1 0.4 2.2 \
  --temperature 1 --top-k 3 --top-p .9 --draw .63
```

Expected core output:

```text
probabilities = [0,0,0.1418510649,0.8581489351]
retained      = [2,3]
selected      = 3
PASS
```

The fixed draw is important. A numerical oracle should not need to duplicate
ENGINE-1's PRNG before it can prove the cumulative boundary. RNG output vectors
are tested separately.

Rust tests cover 83 cases across the workspace. Twenty dedicated sampling
tests cover numerical values, large-logit stability, shift invariance,
probability invariants, greedy ties, temperature, top-k/top-p edges and order,
fixed draws, pinned SplitMix64 output, same/different seeds, raw-logit
preservation, and a deterministic property grid. Lifecycle tests cover EOS,
budget, cancellation at both checkpoints, model/sampler/tokenizer/UTF-8
failure, per-request seed isolation, no post-terminal output, and exactly one
terminal event.

> **PROVE IT**
> A sampler is correct only when its stages can be checked independently and
> its full loop preserves lifecycle invariants.

## Performance lab: sampling is work

Sampling over vocabulary size `V` can require:

- a maximum scan;
- temperature scaling;
- exponentiation and summation;
- sorting or selection for filters;
- renormalization;
- a cumulative scan.

Greedy needs only the maximum scan. ENGINE-1's teaching top-k and top-p use
straightforward sorting for clarity. Their asymptotic and constant costs are
visible.

The reproducible probe in
`code/mini-engine/crates/engine0/examples/chapter04_sampling_cost.rs` measures
sampling separately from model forward on an Apple M1. The full record is
[`research/benchmarks/chapter-04-sampling-cost.md`](../../research/benchmarks/chapter-04-sampling-cost.md).

Exploratory medians in nanoseconds per call were:

| V | Greedy | Softmax + categorical | Top-k 40 | Top-p .9 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 26 | 341 | 472 | 597 |
| 256 | 284 | 2,832 | 5,295 | 6,817 |
| 4,096 | 5,137 | 41,467 | 107,582 | 127,452 |

These numbers are not end-to-end token latency and cannot be extrapolated to
production models. They exclude the model entirely. They show only that the
extra passes and clarity-first sorting have measurable cost in ENGINE-1.
For large language models, forward execution is often much more expensive, but
that proportion depends on model, hardware, provider, vocabulary, batching,
and implementation. Measure the actual system before making a latency claim.

## Inside Hermon

Hermon is the production case study, not the teaching implementation. The
following behavior was inspected at commit
`472a44cdb511b2dae6c9569e59543db8f8350b25`; its llama.cpp submodule was pinned
at `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

> **INSIDE HERMON — CURRENT**
> The default batched runtime carries sampler configuration in each submitted
> request and constructs one mutable llama.cpp sampler for each `ActiveSeq`.
> Model work may be batched, but eligible logit rows are sampled through their
> owning sequences' sampler objects.

`crates/hermon-runtime/src/batched.rs` stores prompt tokens, pending token,
completion count, maximum completion count, UTF-8 buffer, generated history,
and concrete sampler in `ActiveSeq`. Admission constructs that sampler. The
worker requests logits for eligible sequence positions, samples per sequence,
checks end-of-generation and budget, updates pending state, streams pieces, and
eventually sends `StreamItem::Done(EngineMetrics)`. Error branches send an
error rather than `Done`.

The linked facade in `crates/hermon-llamacpp/src/linked.rs` owns a native
`llama_sampler*`. Sampling takes mutable sampler and context references plus
the relevant logit-row index. The simpler iterator path makes the feedback
loop visible: sample from the previous decode, stop on EOG, feed the selected
token through decode so KV state advances, decrement remaining budget, and
return the token.

Hermon's C shim at this commit chooses a greedy llama.cpp sampler when
`temperature<=0`. Otherwise it constructs top-k → top-p → min-p → temperature
→ distribution. Seed zero becomes time-derived in the shim. That chain is
Hermon's current wrapper policy; it is not ENGINE-1's order.

> **INSIDE HERMON — PREVIEW**
> The gated Hermon-owned paged GGUF runtime is still CPU and greedy-only. Its
> source uses local argmax in the paged loop, and the documented 1,000-prompt
> temperature-zero equivalence corpus remains a release gate.

This status distinction prevents two misleading claims: the existence of
stochastic sampling on Hermon's default llama.cpp-backed path does not mean the
Hermon-owned paged path supports it, and an implemented paged path does not
mean it is the default release path.

### The pinned llama.cpp library

The pinned public `llama.h` documents sampler chains ending in a selection
stage such as greedy or distribution. Its current CPU greedy implementation
uses a strict greater-than scan. Its distribution sampler owns seeded
`std::mt19937` state, performs a maximum-shifted exponential computation, and
draws cumulatively. Top-k truncates score-ordered candidates; top-p computes
probabilities, orders them, and includes the threshold-crossing candidate.

ENGINE-1 borrows none of that code and promises no RNG identity with it. The
comparison validates boundaries: logits exist before selection, sampler state
is mutable, chain order matters, and end-of-generation belongs to the runtime.

## Why raw logits are durable evidence

It is tempting to treat a logit vector as disposable scratch. After all, the
request needs only one token. That shortcut destroys information at exactly the
boundary where inference behavior is hardest to debug.

Suppose a user reports that token 17 appeared unexpectedly. Several different
causes can lead there:

- the model genuinely assigned token 17 a large raw score;
- temperature flattened a concentrated distribution;
- top-k removed a token that would otherwise dominate;
- top-p's retained set changed at a cumulative boundary;
- a grammar or future constraint masked alternatives;
- the RNG draw landed in token 17's interval;
- token IDs and tokenizer bytes did not match the intended model revision.

If the sampler overwrote raw logits, these explanations collapse into one
processed vector. If trace data retains raw logits, processed probabilities,
configuration, draw, and selected ID as distinct artifacts, each stage can be
audited.

This is also a correctness requirement for future speculative decoding. A
draft model may propose several tokens, but a target model must verify them
under the target sampling semantics. Constrained generation may add grammar,
JSON-schema, tool-syntax, repetition, or forbidden-token masks. Those
processors answer “which outputs are currently allowed?” Raw logits answer
“what did the model predict before policy?” The questions are related but not
identical.

ENGINE-1 does not implement those future processors. It establishes the seam
they require:

```text
raw model evidence -> ordered logit processors -> candidate distribution
                   -> stateful selection -> committed token
```

> **FIRST PRINCIPLE**
> Sampling policy is inference behavior, not model weights. Preserve the
> evidence on both sides of that policy boundary.

## Renormalization is not optional

Softmax initially distributes one unit of mass over active candidates. Top-p
then removes some candidates. If the retained probabilities sum to `0.85`,
sampling directly against intervals ending at `0.85` leaves `[0.85,1)` without
a mathematically owned token.

One could scale a random draw by retained mass, but that is renormalization in
another form. ENGINE-1 performs it explicitly:

```text
q_i = p_i / sum_(j retained) p_j    when i is retained
q_i = 0                              otherwise
```

Now retained `q_i` values sum approximately to one. The categorical selector
has one contract regardless of which filters ran. Stage tests compose cleanly:
distribution construction proves normalization, and categorical selection
assumes normalized input.

The distinction clarifies the all-filtered failure. If candidate mass is zero,
there is no denominator and no distribution. ENGINE-1 returns
`AllCandidatesFiltered`; it does not unmask a convenient token or switch to
greedy. A future policy may require “always keep at least one,” but that rule
must be explicit in the processor that owns it.

## A lifecycle matrix for the real loop

The happy path alone cannot establish runtime correctness. Consider what has
been committed and what may be emitted at each terminal cause:

| Cause | Token selected? | Token committed? | Text emitted? | Terminal |
| --- | --- | --- | --- | --- |
| EOS | yes, EOS | no | no EOS text | completed: EOS |
| Budget before next step | previous only | previous only | previous text | completed: max tokens |
| Cancel before forward | no new token | no | no | cancelled |
| Cancel after forward | no new sample | no | no | cancelled |
| Model failure | no | no | no new text | failed: model |
| Sampling failure | no token | no | no new text | failed: sampling |
| Tokenizer failure | yes, non-EOS | yes | token event may exist | failed: tokenizer |
| UTF-8 failure | yes, non-EOS | yes | only prior valid text | failed: UTF-8 |

The table exposes why “stop the loop” is insufficient as a design. A stop has
a point in the state transition. An EOS decision occurs before commit.
Tokenizer failure occurs after commit. Cancellation can occur on either side
of expensive model work but before selection. The result must reflect those
facts.

ENGINE-1's stream is ordered:

```text
zero or more token/text events -> exactly one Terminal event -> nothing
```

Text events need not correspond one-to-one with token events because a token
piece may end in an incomplete UTF-8 scalar. The decoder buffers bytes until a
valid prefix exists. Sampling is token-level; user-visible streaming is
byte/text-level. Chapter 2's boundary remains active inside Chapter 4's loop.

## Selection state under future batching

Continuous batching will eventually assemble model work from several
sequences:

```text
Request A tokens --+
Request B tokens ---+--> model batch --> logits A, logits B, logits C
Request C tokens --+
```

That physical batch does not merge logical sequences. Row A must be processed
with A's configuration, RNG state, token history, and remaining budget. A
worker may change the order in which sequences enter batches without changing
which RNG stream each sequence consumes.

This is why request-owned randomness is a systems rule, not a detail of random
number generation. State ownership makes scheduling freedom possible. A global
RNG would turn an optimization—reordering batch work—into a semantic change.

Speculative decoding creates another version of the same constraint. A draft
path may propose multiple candidates in one step, but accepted output still
has to obey the target model and chosen sampling contract. Rollback must restore
model/KV state and stateful processors consistently. Chapter 4 does not
implement speculation; it supplies the raw-logit, sampler-state, and commit
boundaries later chapters will need.

## Seven first principles, connected

The chapter's individual rules form one chain:

1. **The model predicts scores; the sampler chooses a token.** This separates
   learned model semantics from request policy.
2. **Equivalent mathematics can have different numerical safety.** Stable
   softmax preserves semantics while preventing exponential overflow.
3. **Randomness is owned mutable state.** Seeds and consumption order belong
   to one request.
4. **Sampling policy is part of inference behavior.** Temperature, filters,
   tie rules, order, and RNG algorithm belong in a reproduction record.
5. **Raw and processed scores are different artifacts.** Debugging and future
   constraints need both.
6. **Selected output becomes future input.** Feedback is the operational heart
   of autoregressive generation.
7. **Termination has one owner and defined commit points.** EOS, budget,
   cancellation, and failure must produce one coherent final outcome.

Remove any one rule and the tiny engine develops a production-shaped bug. Put
softmax inside the model, and sampling policy contaminates model semantics.
Use naive exponentials, and ordinary logits can yield NaNs. Share an RNG, and
scheduling changes text. Mutate raw scores, and debugging loses evidence.
Append before checking EOS, and sequence state lies. Emit terminal events in
several branches, and clients can receive contradictory endings.

That is why ENGINE-1 spends more code on contracts and tests than on the few
lines of exponential arithmetic. The arithmetic chooses a token; the contracts
make that choice meaningful inside a system.

## Common mistakes

### “Logits are probabilities”

Logits are unnormalized scores. They can be negative and do not sum to one.
Softmax creates probabilities when a policy needs them.

### “Softmax belongs in every forward”

The model boundary returns logits. Greedy can consume them directly; other
processors may need raw scores. Making softmax mandatory would conflate model
semantics with selection policy.

### “The highest logit must be selected”

Greedy selects it. Categorical sampling can choose any positive-probability
candidate retained by policy.

### “Temperature adds randomness”

Temperature changes distribution shape. The RNG draw supplies randomness.

### “Temperature changes ordering”

Positive temperature divides every score by the same positive number, so
ordering stays fixed.

### “Top-p keeps p percent of tokens”

Top-p keeps the smallest probability-ordered prefix whose mass reaches `p`.
Its candidate count varies.

### “Top-k and top-p are the same”

Top-k fixes a count. Top-p fixes retained mass. They can keep different sets.

### “A fixed seed guarantees the same text everywhere”

Seed controls one PRNG start. Model logits, engine version, sampler order,
numeric behavior, and tokenization also affect output.

### “Sampling is stateless”

Every draw advances RNG state. Future repetition and grammar processors add
more mutable state.

### “EOS is visible text”

ENGINE-1 treats EOS as control. It is neither decoded nor streamed.

### “max_new_tokens includes the prompt”

It counts committed output tokens. Prompt length is a different quantity.

### “Random sampling means random noise”

The model determines probability mass. Sampling chooses from that structured
distribution after explicit filtering.

## Labs

Chapter 4 completes the following progression:

- [Lab 9 — Stable Softmax by Hand](../../labs/lab-09-stable-softmax-by-hand.md)
- [Lab 10 — Change Temperature](../../labs/lab-10-temperature.md)
- [Lab 11 — Select With a Fixed Random Draw](../../labs/lab-11-fixed-categorical-draw.md)
- [Lab 12 — Top-k Versus Top-p](../../labs/lab-12-top-k-vs-top-p.md)
- [Lab 13 — Trace the Autoregressive Loop](../../labs/lab-13-build-the-autoregressive-loop.md)
- [Lab 14 — Change the Seed](../../labs/lab-14-change-the-seed.md)
- [Lab 15 — Break the Sampler](../../labs/lab-15-break-the-sampler.md)

Each moves through CHECK, BUILD, BREAK, and EXTEND. The numerical labs compare
independent Python and Rust boundaries. The lifecycle labs deliberately inject
failure rather than proving only the happy path.

## Exercises

1. Prove algebraically that adding any constant to all logits preserves
   softmax. Then explain why subtracting the maximum improves floating-point
   behavior.
2. For logits `[2,2,1]`, state ENGINE-1's greedy result and its top-k order for
   `k=2`. Identify the tie rule.
3. Compute the temperature-one softmax of `[0,1]`. Without recomputing exact
   exponentials, predict how `T=.5` and `T=2` change the gap.
4. Draw cumulative intervals for `[.1,.2,.3,.4]`. Select tokens for draws
   `0`, `.1`, `.2999`, `.3`, and the largest value below one.
5. For `[.40,.30,.15,.10,.05]`, compare top-k 2 with top-p .8. Renormalize
   both survivor sets.
6. Give an example where applying top-p before top-k could differ from
   ENGINE-1's order. Explain why a sampler version belongs in a reproduction
   record.
7. Trace `I like` through both model forwards, including hidden vectors,
   logits, selected IDs, history, byte piece, and terminal reason.
8. Explain what happens if token decoding fails after ENGINE-1's commit point.
   Which state and events are truthful, and why is there still one terminal?
9. Design a test that would expose shared global RNG state between two
   requests.
10. Explain why a future batched model forward may combine several sequences
    while their sampler states remain separate.

## What Part I has built

Chapter 1 defined the machine and its request-to-terminal lifecycle. Chapter 2
crossed the text/token/byte boundary. Chapter 3 replaced fake candidates with a
real numerical model. Chapter 4 turns numerical prediction into generation.

```text
CHAPTER 1                    What machine are we building?
    |
CHAPTER 2                    How does text become token IDs?
    |
CHAPTER 3                    How do token IDs become logits?
    |
CHAPTER 4                    How do logits become generation?
```

The Part I engine is now:

```text
Human text
    |
Tokenizer
    |
Token IDs
    |
Tiny numerical language model
    |
Embedding lookup + output projection
    |
Raw logits
    |
Sampling policy
    |
Next TokenId --------+
    |                 |
Decode bytes          |
    |                 |
UTF-8 stream          |
    |                 |
Human output          |
                      |
append to history ----+
```

It is tiny. It is scalar. It has only four tokens. It depends only on the final
token. It does no training, attention, position handling, KV caching, GGUF
loading, quantization, batching, paging, native kernels, or acceleration.

Those limitations do not make it fake. Every parameter is visible. Every
forward score is real. Every selection follows a stated policy. Every random
draw has an owner. Every output token feeds back. Every request terminates
once. A reader can inspect and run the entire path.

> **FIRST PRINCIPLE**
> Autoregressive generation feeds selected outputs back into future model
> inputs, and every such loop needs explicit state and termination ownership.

## Summary

- A model produces raw logits; a sampler chooses a token.
- Softmax converts finite scores to non-negative probability mass summing
  approximately to one.
- Subtracting the maximum preserves softmax and prevents exponential overflow.
- Greedy selection uses one argmax scan and needs no softmax.
- Categorical selection maps a draw in `[0,1)` through cumulative intervals.
- Temperature changes distribution sharpness; it does not add randomness or
  change ordering when positive.
- Top-k keeps a fixed number of high scores; top-p keeps adaptive probability
  mass and includes the threshold-crossing candidate.
- Operation order is part of inference behavior.
- Raw logits and processed candidates are separate artifacts.
- RNG state belongs to one request. A seed has a bounded reproduction
  contract, not a universal guarantee.
- ENGINE-1 checks EOS before commit, counts only new visible tokens against the
  output budget, and checks cancellation at defined boundaries.
- One lifecycle owner emits exactly one completed, cancelled, or failed
  terminal outcome.
- Part I now contains the smallest complete inference engine.

## Part II preview: tensors without magic

ENGINE-1's remaining intellectual defect is now impossible to miss. It feeds
tokens back, but its model still reads only the final token. A modern decoder
needs internal representations spanning positions and hidden dimensions.

Before attention, queries, keys, values, or Transformer layers, we need to know
how those numbers occupy memory. Chapter 5 begins Part II with scalars,
vectors, matrices, tensors, rank, shape, dtype, row-major layout, stride,
contiguity, offset calculation, views, copies, aliases, ownership, bounds, and
overflow-safe element counts.

Attention comes later. First we build a tensor substrate whose semantics can
survive it.

## References

- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. [A
  Neural Probabilistic Language Model](https://jmlr.org/papers/v3/bengio03a.html).
  *Journal of Machine Learning Research* 3, 2003.
- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. [The Curious
  Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR
  2020.
- Pierre Blanchard, Desmond J. Higham, and Nicholas J. Higham. [Accurately
  computing the log-sum-exp and softmax
  functions](https://doi.org/10.1093/imanum/draa038). *IMA Journal of Numerical
  Analysis* 41(4), 2021.
- Guy L. Steele Jr., Doug Lea, and Christine H. Flood. [Fast Splittable
  Pseudorandom Number Generators](https://doi.org/10.1145/2660193.2660195).
  OOPSLA 2014.
- PyTorch. [`torch.nn.Softmax`
  documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
- Hermon source at `472a44cdb511b2dae6c9569e59543db8f8350b25`.
- llama.cpp source at `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.
