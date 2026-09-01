# Chapter 1 — The Missing Half of AI

## From a trained artifact to one terminal stream

You send a sentence to a language model. A moment later, text begins to arrive.
What happened between those two events?

One familiar answer is “a Transformer predicted the next token.” That answer is
important, but it leaves out the system that made the prediction usable. It
does not say who found and loaded the model, admitted the request, allocated its
state, chose where computation ran, selected a token, converted that token into
bytes, preserved output order, stopped generation, reported failure, or cleaned
up after cancellation. It does not say why two requests can share model weights
but must not accidentally share mutable generation state. It does not tell an
operator whether a slow answer waited in a queue, loaded a model, evaluated a
long prompt, stalled on a device, or waited behind a slow client.

That missing half is the **inference engine**.

This chapter builds a map of that engine before we study its numerical center.
We will distinguish training from inference, an artifact from a running model,
a model from an engine, and an engine from a server or operated service. We
will follow one token, one byte, and one owner. Then we will build ENGINE-0: a
small Rust runtime that uses a fake, hand-computable model source but implements
a real request lifecycle with streaming, stopping, cancellation, failure, and
named timing points.

The fake model is a boundary, not a shortcut disguised as a model. Chapter 2
will put real tokenization behind that boundary. Chapter 3 will put a tiny
numerical language model behind it. Chapter 4 will complete logits, sampling,
and the autoregressive loop. The runtime contract we establish here should not
need to be discarded when those parts become real.

## The answer is a running system, not a file

Start with a deliberately small request:

```text
prompt: "What color is the sky?"
maximum new tokens: 8
selection policy: greedy
```

At rest, a model may be a collection of files. Those files can contain
configuration, tokenizer data, and serialized tensors. They cannot admit a
request or emit a token. They are **model artifacts**: persistent bytes with an
interpretation.

Before generation, software must validate enough metadata to understand those
bytes, arrange or map weights into usable memory, create execution resources,
and choose implementations for model operations. The resulting **running
model** is a live computational object. It combines model semantics, resident
or addressable weights, and the resources needed to perform a forward step.

Even a running model is not the whole answer. A generation request needs
mutable state: its prompt representation, current position, generated history,
selection state, stop conditions, cancellation state, timing, output sink, and
eventually a terminal outcome. Several requests may borrow the same immutable
weights. They must retain distinct logical histories and lifecycle ownership.

The canonical [model-versus-engine diagram](../../diagrams/runtime/model-vs-engine.txt)
makes that split explicit:

```text
       persistent storage                    process / device memory
  +--------------------------+            +---------------------------+
  | (MODEL ARTIFACT)         | == load ==> | (RUNNING MODEL)           |
  | configuration            |            | validated model semantics |
  | tokenizer data           |            | resident weight bytes     |
  | serialized weights       |            | execution resources       |
  +--------------------------+            +-------------+-------------+
                                                        |
                                                        | candidates/logits
                                                        v
  +--------------------------------------------------------------------+
  | INFERENCE ENGINE                                                   |
  |  resolve -> admit -> schedule -> model step -> select -> stop      |
  |                |                                      |            |
  |                v                                      v            |
  |      [request state / timing]              [ordered output stream]  |
  +--------------------------------------------------------------------+
```

Parentheses denote data treated as immutable during a request. Brackets denote
mutable state with an owner. The distinction is logical: an implementation may
map weights from disk, copy them to RAM, mirror them on a device, or keep more
than one packed representation. Whatever the placement, request code must know
which bytes are shared, which bytes may be mutated, and who releases each
resource.

> **FIRST PRINCIPLE**
> A model defines a computation. An inference engine owns the process of using
> that computation to advance requests to observable terminal outcomes.

This definition is intentionally about responsibility, not process topology.
An engine can live inside a command-line program, a Python process, a mobile
app, or a fleet of services. Moving a boundary across processes changes
communication and failure costs; it does not remove the responsibility.

## Training and inference solve different problems

Training changes model parameters. Given examples, an objective, and an
optimization procedure, training repeatedly computes predictions and gradients
and updates weights. Its central state includes parameters, gradients,
optimizer statistics, training examples, and checkpoints. A training system
cares deeply about distributed gradient communication, update correctness,
checkpoint recovery, and sample throughput.

Inference uses already-chosen parameters to answer requests. In the ordinary
case, it must not update the model weights. Its central state is instead the
state needed to evaluate inputs and continue sequences: loaded weights,
temporary activations, per-request history, cached intermediate state,
selection state, queues, and streams. An inference service also has to handle
arrival bursts, deadlines, cancellation, failures, resource limits, and
protocol behavior.

The same mathematical model may appear in both systems, and both perform
forward computation, but their ownership and economics differ:

| Concern | Training | Inference |
| --- | --- | --- |
| Weight state | Deliberately updated | Normally immutable during a request |
| Primary unit | Optimization step / sample batch | Request / sequence / emitted token |
| Repeated state | Gradients and optimizer state | Generation history and reusable inference state |
| Scheduling goal | Efficient, correct parameter updates | Latency, throughput, fairness, and bounded memory |
| External contract | Checkpoint and training progress | Ordered output and terminal request behavior |
| Common failure question | Can the update recover? | Which requests fail, and is their state released? |

This book is about inference. Later we may use an engine during evaluation or
post-training workflows, but we will keep the runtime contract visible. Calling
the same model function does not make the surrounding system the same.

## Four layers that are often collapsed into “the model”

The word *model* is overloaded in ordinary conversation. A precise system map
needs at least four layers.

### The mathematical model

The mathematical model maps inputs and state to outputs. For a causal language
model, a useful future view is that a token history leads to a score for every
possible next token. We will derive that computation later. In Chapter 1, its
contract is only “produce candidates for the next token.”

### Model representation and execution

The representation layer interprets artifacts: tensor names, shapes, dtypes,
layouts, tokenizer configuration, and architecture metadata. Execution
implementations perform operations on some substrate. That may be scalar CPU
code, vectorized CPU code, Metal, CUDA, or another backend. A compiled backend
is not proof that it supports this model, this shape, or this request.

### The inference runtime or engine

The runtime owns live request progress and resource coordination. It resolves a
running model, admits work, decides when it may execute, invokes model steps,
applies selection and stopping policy, emits ordered events, accounts for work,
and releases state. Some projects use *runtime* for a narrower execution layer
and *engine* for the full coordinator. We will qualify external names, but our
book uses **inference engine** for the combined request-to-outcome
responsibility.

### The serving and operational surface

A server accepts requests through a protocol and converts them to the engine's
internal form. It may parse HTTP, authenticate, validate schemas, format
Server-Sent Events, enforce body limits, and map internal terminal causes to
wire finish reasons. A service adds deployment concerns: replicas, load
balancing, quotas, health, rollout, rollback, observability, and service-level
objectives.

The [canonical inference-stack diagram](../../diagrams/runtime/inference-stack.txt)
shows the roles:

```text
application
    |
    v
library API  ---------------- direct/offline generation
    |
    v
inference engine
    +--> provider choice  --> local | remote | managed capability
    +--> hardware backend --> scalar CPU | SIMD CPU | Metal | CUDA | ...
    |
    v
server  - - > protocol validation, HTTP/gRPC/SSE, multi-client boundary
    |
    v
service - - > deployment, replicas, auth, quotas, SLOs, operations
```

The diagram is not a mandatory call graph. An offline library may embed the
engine without a server. An online system may split API, engine core, and
device workers into separate processes. A managed provider may hide every
layer below an API. The roles remain useful because they identify contracts and
owners.

### Provider is not backend

Projects use these words differently, so we need a book convention.

A **provider** offers or selects an inference capability: for example, local,
remote, or managed execution, possibly associated with credentials, placement,
or a model namespace. A **backend** implements model operations on a hardware
and runtime substrate. A local provider might choose a Metal backend on one
machine and a CPU backend on another. A remote provider may hide its backend
entirely.

This distinction prevents two common reasoning errors. First, routing a request
to “local” does not tell us whether it runs on CPU or GPU. Second, compiling a
CUDA backend does not tell us that a request was routed to it. We must trace
both the provider decision and the backend selection before making a current
execution claim.

## The engine coordinates compute, memory, and requests

An inference engine sits where three systems meet.

**Compute** asks which operations must run, in what order, with what shapes,
dtypes, and dependencies. A model step eventually produces candidate scores.
Selection policy turns those scores into a token decision.

**Memory** asks where artifact bytes, resident weights, temporary activations,
and per-request state live; who may mutate them; whether they can be shared;
and when they can be released. The weight file may be much larger than one
request's state, but mutable request state is often the harder concurrency
problem because a small ownership error can produce plausible, wrong text.

**Requests** ask who may enter, when work runs, how progress is streamed, what
cancellation means, which limits stop generation, how failures are isolated,
and when capacity becomes reusable. A mathematically correct forward pass can
still belong to a broken engine if output is reordered, cancellation leaks
state, or two terminals are reported.

These domains interact. Admitting another request increases memory demand.
Combining work can improve hardware utilization but delay a particular token.
Retaining computed state may reduce future compute while consuming capacity.
Moving work to a device can accelerate a large operation while adding transfer,
launch, and synchronization costs. The engine is where those tradeoffs become
policy rather than slogans.

## Control plane and data plane

We will frequently separate two kinds of work.

The **control plane** makes decisions and maintains coordination metadata. It
validates and resolves requests, admits them, chooses models and backends,
schedules steps, applies stopping and cancellation, records metrics, and
decides when to release resources.

The **data plane** moves and transforms the bytes that implement inference. It
loads weight data, reads token and state arrays, runs tensor operations,
produces candidate scores, selects or transfers token identities, and sends
output bytes.

```text
control plane: validate -> resolve -> admit -> schedule -> stop/fail -> account
                                      |
                                      v
data plane:    artifact -> weights -> model step -> candidates -> token -> bytes
                                      ^                         |
                                      |                         v
                                request state <----------- ordered stream
```

This is a reasoning tool, not a claim that the planes require separate
machines. ENGINE-0 puts both in one synchronous Rust call. vLLM V1, by
contrast, documents an API-server process, an engine-core process, and worker
processes, making some of the split physical as well as conceptual. The API
process performs input processing and streaming; the engine core schedules and
manages KV state; workers perform model execution on devices ([vLLM architecture
overview](https://docs.vllm.ai/en/latest/design/arch_overview/)).

The distinction helps during failure analysis. If a model step is fast but
admission waits, the bottleneck is not model arithmetic. If a device finishes
but bytes cannot reach a slow client, producing more tokens can make the system
worse. If a provider route is wrong, optimizing a backend kernel will not fix
the request.

## Follow the token, the byte, and the owner

The rest of this book returns to three journeys. They are three views of the
same generation, and a trustworthy engine makes them agree. The complete
Chapter 1 version is stored in the
[token-byte-owner diagram](../../diagrams/runtime/token-byte-owner.txt).

### Follow the token

Input begins as text or some other modality. A tokenizer will turn text into
vocabulary identifiers. The running model will use the current token history to
produce a candidate score for each possible next identifier. Selection policy
will choose one. A decoder will turn identifiers into output bytes or pieces.
The engine will emit those pieces in order, append the new identity to request
history, and repeat until a stop condition or non-success terminal.

Chapter 1 fakes the tokenizer, decoder, and numerical model. It does not fake
the identity flow: the selected token has a stable ID, is appended once, is
emitted once if it is output text, and is associated with one request.

### Follow the byte

Artifact bytes begin on persistent storage or arrive through some model store.
Some become metadata; some represent packed weights. The runtime may map,
copy, transform, or mirror them into execution-ready forms. Model operations
read those bytes and request-state bytes to produce candidate-score bytes. A
selected token identity eventually becomes text bytes delivered to a sink.

Later chapters will make this path physical: GGUF offsets, packed
quantization, RAM and device residency, KV writes, attention reads, and output
buffers. For now, remember that every arrow has a cost and an owner. “The GPU
has the model” is incomplete if host code still owns admission, tokenization,
sampling, streaming, or a required copy.

### Follow the owner

The application owns the input before submission. Once accepted, the runtime
owns the mutable lifecycle state or holds an explicit lease for it. A running
model borrows the current inputs and state needed for one model step. A sink
receives immutable event values in order. At terminal, the runtime releases
request state and any capacity exactly once.

Ownership is broader than language-level memory safety. Rust can prevent many
use-after-free defects, but the program must still choose the right logical
owner. Two `Arc` values can safely point to state that should never have been
shared. An atomic counter can be race-free and still count the wrong lifecycle
transition. We will therefore write ownership invariants in prose, types, and
tests.

## A request is a state machine

An API payload is not the whole request. It is the initial intent from which
the runtime constructs a lifecycle.

A useful minimum state machine is:

```text
submitted -> validated -> admitted -> executing -> streaming -> terminal
                |            |            |            |
                +------------+------------+------------+
                                  |
                                  v
                  completed | cancelled | failed
```

Real engines add queued, blocked, preempted, draining, or model-loading states.
The minimum invariant is more important than the exact labels:

> **FIRST PRINCIPLE**
> Every submitted generation reaches one observable terminal outcome, and an
> admitted generation releases its owned state exactly once.

ENGINE-0 uses three terminal outcomes:

- `Completed(StopReason)` means generation stopped according to a successful
  rule. Its reason is `EndOfSequence` or `MaxTokens`.
- `Cancelled` means the caller or control policy withdrew the work.
- `Failed(GenerationError)` means invalid input, model failure, or an
  equivalent error prevented successful completion.

Cancellation is not successful completion. Failure is not cancellation. A
token limit and a model end marker are both successful stops but have different
causes. Keeping these distinctions internally allows a server to map them to a
wire protocol honestly.

The canonical [request-to-token diagram](../../diagrams/runtime/request-to-token.txt)
shows every terminal branch. Its most important line is at the bottom: after a
terminal event, no token or trace event may be emitted. That sounds small. In a
concurrent server it prevents duplicated close chunks, usage reported twice,
state released twice, and tokens arriving after a client believes the answer
is over.

### A stream is an ordered contract

A stream is not “some callbacks happen.” It is a sequence of events with
ordering and termination semantics. ENGINE-0's public stream contains zero or
more token events followed by one terminal event. Its sink is deliberately
infallible so Chapter 1 can isolate the producer's lifecycle. Later we will add
bounded queues, backpressure, network departure, and sink failure.

Token identity and output piece are also different. A tokenizer's vocabulary
token can correspond to a fragment whose bytes do not form a complete Unicode
string alone. Production runtimes may buffer bytes across token boundaries
before emitting valid text. Therefore, a wire *piece* need not correspond
one-to-one with a token. ENGINE-0 prints one teaching piece per text token and
states that limitation explicitly.

## Latency is a distance between named events

Saying “latency is 120 milliseconds” is incomplete. From which event to which
event? Under what concurrency? Does it include queueing, model load,
tokenization, network transfer, or client rendering?

Name lifecycle timestamps first:

```text
t0  request enters the measured runtime boundary
ta  request is admitted
ts  execution starts
tr  first output token is ready
te  first output token/piece is externally emitted
ti  each later output event is emitted
tt  terminal outcome is emitted
```

Then define intervals:

```text
queue delay              = ts - ta
first-token compute span = tr - ts
ready-to-emit delay      = te - tr
time to first token      = te - ta
inter-token latency i    = ti - t(i-1)
runtime request latency  = tt - t0
```

Some systems measure time to first token from client arrival rather than
admission. Some report the first token identity before complete text bytes are
available. Neither choice is universally wrong, but the endpoints must travel
with the number.

A useful decomposition is:

```text
T_request = T_validate + T_queue + T_prepare + T_model
          + T_select + T_emit + T_between_tokens + T_terminal
```

This is a sum of measured categories only if the implementation actually
records compatible boundaries. It is not permission to invent an end-to-end
number by adding unrelated benchmarks. Some terms may overlap in a pipelined
engine; in that case a timeline, not a simple sum, is the truthful model.

### TTFT, ITL, and request latency answer different questions

**Time to first token (TTFT)** describes how soon useful output begins.
**Inter-token latency (ITL)** describes the spacing of later output. **Request
latency** describes how long until the request is terminal. A system can improve
one and worsen another. Waiting briefly to combine work may increase TTFT while
improving aggregate throughput. A long prompt can dominate TTFT even when later
decode steps are quick. A slow client can increase terminal latency after the
device has produced tokens.

Report distributions when many requests matter. An average can hide a long
tail caused by queueing, model cold starts, uneven output length, or scheduling
unfairness. Part XII will turn these names into reproducible production
measurement.

### Throughput is not the inverse of one request's latency

Throughput is completed work per unit time:

```text
request throughput = completed requests / elapsed seconds
token throughput   = emitted output tokens / elapsed seconds
```

Both require a population and a workload definition. Requests with different
prompt and output lengths are not interchangeable work units. Token throughput
can hide poor fairness. Request throughput can hide enormous differences in
token count.

With concurrency greater than one, a system can increase total tokens per
second even if each request's token spacing gets slightly worse. Conversely, a
single request can have excellent latency while leaving most hardware idle.
That is why “fast” must be split into latency, throughput, fairness, memory,
and correctness rather than compressed into one number.

**Concurrency** is the number of requests simultaneously admitted or active.
It is not automatically the size of one physical execution batch. A scheduler
may have ten admitted requests while executing positions from only some of them
in the next device call.

## Prefill and decode: a preview, not yet an explanation

Generation work changes shape over a request.

During **prefill**, the engine evaluates prompt positions and creates reusable
per-position state. There may be many prompt positions available at once, so
the numerical work can have substantial parallel structure.

During **decode**, the engine advances active sequences with newly generated
positions. For an ordinary autoregressive sequence, each step contributes a
small number of new query positions while consulting an expanding history.
That changes arithmetic intensity, memory traffic, batching opportunities, and
latency sensitivity.

This distinction will become mathematical in Part IV. For Chapter 1, retain
only three facts:

1. tokenization is not prefill;
2. prefill and decode use the same model parameters but present different work
   shapes to the runtime;
3. a scheduler may mix prompt and decode work from different requests in one
   physical execution step.

ENGINE-0 has neither phase. Its `Model::candidates` call is a lifecycle marker,
not a claim to implement prefill or decode.

## “GPU inference” still has a CPU and a runtime

A GPU can perform large numerical operations very effectively. That does not
mean the request bypasses host software.

Someone must accept and validate the request, tokenize or otherwise prepare
inputs, choose a device and execution plan, allocate or find state, submit
work, handle completion, apply some selection and stopping policies, convert
output to protocol events, and react to cancellation. Different systems move
more of these tasks onto devices or overlap them with device work, but the
control path remains.

NVIDIA's current TensorRT-LLM architecture, for example, documents a high-level
`LLM` interface that manages tokenization and creates executor workers. Its
`PyExecutor` coordinates a scheduler, model engine, and decoder; its overlap
scheduler can launch GPU work for one step while processing previous results on
the CPU ([TensorRT-LLM architecture overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/overview.html)).
The important lesson is not a claim that every engine should copy that design.
It is that CPU orchestration and GPU execution can overlap and cooperate. The
device is a backend participant, not the owner of the entire request.

A GPU can also lose for a particular operation. Transfers, command encoding,
kernel launch, synchronization, small shapes, unsupported layouts, and idle
gaps can dominate useful arithmetic. We will measure those crossovers in Part
VIII. For now, reject two symmetrical myths: “GPU means the whole request is on
the GPU” and “CPU work means GPU acceleration failed.”

## Current engines expose different boundaries

The system map is not invented from one codebase. Current official sources
show several valid decompositions.

**llama.cpp** exposes both a library and `llama-server`. Its server developer
documentation describes HTTP routes, thread-safe request/response queues, a
`server_context` that owns primary inference state, and per-sequence
`server_slot` objects. The server advertises parallel decoding and continuous
batching ([llama-server developer documentation](https://github.com/ggml-org/llama.cpp/blob/3466812d1f06728effe7c0f3c0671117f461672d/tools/server/README-dev.md)).

**vLLM** offers an offline `LLM` entry point and online serving. Its documented
V1 topology separates API input/streaming, engine-core scheduling and KV
management, and device workers. This is a clear example of the same engine role
crossing process boundaries rather than disappearing.

**SGLang** describes its runtime as combining continuous batching, paged
attention, prefix reuse, chunked prefill, and several parallel execution forms.
Its current scheduler source initializes model workers, memory pools, request
receivers, output streaming, scheduling policy, and metrics. The safe Chapter 1
conclusion is bounded: a model worker is one component inside a larger
request-and-resource runtime. Feature support for a particular model or device
requires a later, narrower audit ([SGLang scheduler at the inspected
commit](https://github.com/sgl-project/sglang/blob/221a6273ce3212c79483df233b4511fdf8fbe6d0/python/sglang/srt/managers/scheduler.py)).

**Hugging Face Transformers** provides `GenerationMixin.generate`, generation
configuration, logits processors, stopping criteria, and streamer hooks. That
is a powerful library-level generation surface, but it is not by itself a
multi-client service ([Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)).

**Text Generation Inference (TGI)** shows a serving decomposition: a Rust
router/webserver receives and batches requests, a launcher starts components,
and one or more model servers perform inference behind gRPC. Its official docs
now mark TGI as maintenance mode and recommend current engines including vLLM,
SGLang, and local options such as llama.cpp. The architecture is still useful,
but the maintenance status belongs beside any present-day recommendation
([TGI architecture](https://huggingface.co/docs/text-generation-inference/architecture),
[status notice](https://huggingface.co/docs/text-generation-inference/index)).

These projects do not prove one universal class hierarchy. They support a more
durable claim: input processing, request control, model execution, memory
ownership, selection, and output streaming are separable responsibilities even
when a particular implementation combines them.

## Inside Hermon: one verified current request path

Hermon is this book's recurring production case study. The following account is
**CURRENT** only for commit
[`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25),
inspected on 2026-09-02. The durable source record is in the
[Chapter 1 research note](../../research/part-01/chapter-01-the-missing-half-of-ai.md).

> **INSIDE HERMON — CURRENT**
> For an OpenAI-compatible request, `hermon-api` first resolves the model name
> through its provider router. In an engine-enabled binary, if the name also
> resolves to a local GGUF and the native engine is linked, the handler calls
> `engine_route::stream_with_options`, then
> `hermon_runtime::Dispatcher::stream_with_options`. Otherwise the inspected
> handler falls through to its Ollama client path.

This is why provider and backend must remain separate in our vocabulary. The
provider router classifies a destination and connection information. The local
engine gate establishes whether this specific request can use the in-process
execution path. Tracing one without the other would produce an incomplete
claim.

The dispatcher reads `HERMON_RUNTIME_MODE` when it is constructed. Unset and
unrecognized values choose `batched`. The dispatcher canonicalizes the model
path, caches one per-model runtime, and clones a short-lived `Arc` without
holding the map lock across inference.

> **INSIDE HERMON — CURRENT**
> The default `BatchedRuntime` owns one shared llama.cpp model, one
> multi-sequence context, a submission channel, one dedicated OS worker thread,
> and atomic metrics. The worker alone mutates the context, shared batch,
> active-sequence table, per-request samplers, and sticky-slot prefix metadata.

The worker admits normalized sequence requests into logical slots. On each
iteration it assembles prompt chunks or pending decode positions from active
requests into one physical `llama_batch`, calls `decode_batch`, samples the
correct logit row for each eligible sequence, buffers token bytes until it can
emit valid UTF-8, and either continues or finalizes. Sticky slots may retain a
matching prompt prefix in the same sequence slot. That is reuse, but it is not
cross-request physical page sharing.

At the lower runtime boundary, a successful stream is zero or more
`Piece(String)` values followed by exactly one `Done(EngineMetrics)`, then
channel close. An `EngineError` receives no `Done`. The per-request output
channel is bounded, so a slow consumer eventually delays producer sends instead
of creating an unbounded output queue. Runtime metrics count admissions,
completions, failures, prefill/decode work, cache behavior, speculation, active
slots, and warm slots. The dispatcher's snapshot endpoint includes batched
runtimes; it does not pretend pool and paged modes expose the same current
metrics surface.

> **INSIDE HERMON — PREVIEW**
> Selecting `HERMON_RUNTIME_MODE=paged` is explicit preview behavior. Real
> packed-GGUF execution requires the additional `HERMON_PAGED_GGUF=1` gate and
> remains CPU, greedy, and serialized per model at the inspected commit.

> **INSIDE HERMON — LIBRARY**
> Hermon's native kernel and expert-storage components exist as usable,
> tested libraries, but their existence does not mean they own the default
> request path. Default execution still uses the pinned llama.cpp route through
> `BatchedRuntime`.

This status vocabulary prevents a common documentation failure: finding a
source file named `paged` or `cuda` and silently describing it as the default
runtime. Current means reached by the verified default path. Preview means
implemented behind a gate. Library means a component exists without end-to-end
default integration.

The source trace also preserved two caveats. The opening comment in
`dispatch.rs` still speaks from the older pool-era perspective even though the
executable default is batched. More importantly, the lower runtime contract
distinguishes error from successful `Done`, while the inspected OpenAI SSE
adapter logs an engine-stream error, exits its receive loop, and then attempts a
generic `finish_reason: stop` close and `[DONE]`. Chapter 1 does not repair
Hermon, and this task did not authorize doing so. It records the boundary so
the later protocol chapter can audit whether terminal cause survives every
adapter.

## Build ENGINE-0

We now need an executable system small enough to understand completely. A real
model would force us to explain tokenization, tensor math, model files, and
sampling before we could test the request lifecycle. Instead, ENGINE-0 makes
the model boundary artificial and the runtime boundary real.

The workspace lives at [`code/mini-engine`](../../code/mini-engine/README.md).
It is Rust 2021, forbids unsafe code, and has no external dependencies.

> **BUILD IT**
> Run ENGINE-0 with its trace enabled:
>
> ```sh
> cd code/mini-engine
> cargo run -p engine0 -- --trace 'What color is the sky?'
> ```

The model-facing oracle has two tables. At step 0:

```text
blue=9, green=4, <eos>=1
```

At step 1:

```text
<eos>=10, blue=1
```

The greedy selector chooses the highest score. Therefore the semantic output is
one text token, `blue`, followed by successful end-of-sequence. These integers
are **not logits**. They are candidate scores designed to be checked without
neural-network knowledge. The independent expected result is stored in
[`code/reference/engine-0-oracle.md`](../../code/reference/engine-0-oracle.md).

### The public concepts

`Request` contains a stable ID, opaque prompt text, and a positive maximum
number of new tokens. Validation rejects a blank prompt or zero token budget.
ENGINE-0 treats prompt text as opaque because Chapter 2 owns tokenization.

`GenerationState` contains generated tokens and exposes the current step. The
runtime creates and mutates it. The model borrows it for one candidate call but
cannot emit events or finalize the request.

`Model` is a trait:

```rust
pub trait Model {
    fn candidates(
        &self,
        request: &Request,
        state: &GenerationState,
    ) -> Result<Vec<Candidate>, ModelError>;
}
```

This is the substitution seam for Chapters 2–3. A later model can interpret
tokenized input and produce genuine numerical output without owning admission,
streaming, or terminal behavior.

`Selector` chooses one `Token` from candidates. ENGINE-0's `GreedySelector`
keeps the highest score and uses the first candidate as a deterministic tie
break. Chapter 4 will replace this small policy with complete logit transforms,
random-number ownership, stochastic sampling, and stopping rules.

`TokenSink` receives ordered `StreamEvent` values:

```rust
pub enum StreamEvent {
    Token { request_id: u64, index: usize, token: Token },
    Terminal { request_id: u64, outcome: TerminalOutcome },
}
```

The terminal event carries completed, cancelled, or failed. End-of-sequence is
selected but not emitted as text. That keeps a model control token from
appearing in the user-visible answer.

### The runtime loop

`Runtime::generate` owns the transition order:

1. create lifecycle and generation state;
2. validate the request, failing terminally if invalid;
3. record admission and execution start;
4. check cancellation;
5. check the maximum-token limit;
6. ask the model for candidates;
7. ask the selector for one token;
8. complete on end-of-sequence, or append and emit a text token;
9. repeat until one terminal outcome;
10. return `GenerationResult` with tokens, outcome, and timings.

The model never receives the sink. The selector never mutates request state.
The sink never decides whether generation continues. These restrictions make
ownership visible.

An internal `Lifecycle` object is the only component allowed to produce the
terminal transition. Its `finish` method rejects a second finish with
`AlreadyTerminal`. Token and trace emission check the terminal flag and do
nothing after it is set. This is defense in depth: normal control flow consumes
the lifecycle into a result, while focused unit tests deliberately attempt the
forbidden transitions.

### Trace mode is an experiment, not a benchmark

Trace events include admission, execution start, model invocation, token
selection, token emission, and terminal. Each carries an elapsed duration from
entry to `generate`. A typical run resembles:

```text
Admitted
ExecutionStarted
ModelInvoked(step=0)
TokenSelected(step=0, token_id=1)
TokenEmitted(index=0, token_id=1)
ModelInvoked(step=1)
TokenSelected(step=1, token_id=0)
Terminal(Completed(EndOfSequence))
```

Your microseconds will differ. ENGINE-0 runs synchronously, prints to a terminal,
and performs almost no model work. The values are useful for identifying
boundaries, not comparing machines or engines. A terminal print can itself
dominate the fake computation.

> **PERFORMANCE LAB**
> Use trace mode only to name `queue_delay`, `time_to_first_token`,
> `ready_to_emit_delay`, and total runtime. Do not publish the resulting
> microseconds as an inference benchmark.

Try the explicit non-success paths:

```sh
cargo run -p engine0 -- --trace --cancel-at 0 'cancel this request'
cargo run -p engine0 -- --trace --fail-at 1 'fail after one token'
```

The injected failure exits nonzero. Both runs still produce one terminal
event. The cancellation emits no token when checked at step 0. The step-1 model
failure may follow the valid `blue` token, but it cannot also report completed.

## Prove the lifecycle before adding a model

ENGINE-0 has eleven tests: two focused unit tests for the internal terminal
guard and nine integration tests for public behavior.

The proof obligations are:

1. the hand-computable oracle selects `blue`, then end-of-sequence;
2. the trace order matches admission, execution, model, selection, emission,
   and terminal causality;
3. a maximum-token stop is explicit;
4. cancellation produces one cancelled terminal;
5. blank input fails without admission or token emission;
6. an injected model failure cannot also complete;
7. an empty candidate set fails explicitly;
8. repeated identical runs have identical semantic streams;
9. a terminal event is last;
10. a second internal terminal transition is rejected;
11. token and trace emission are blocked after terminal.

Run the full gate:

```sh
cd code/mini-engine
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

> **PROVE IT**
> Correctness is the terminal state machine plus the candidate oracle. Timing
> output is not part of semantic equality. Repeated runs compare stream events,
> not elapsed durations.

This separation matters. Determinism does not mean identical wall-clock time.
It means that under identical request, model source, selection policy, and
cancellation configuration, the semantic outcome is stable.

## Where the naive engine fails

ENGINE-0 is production-shaped in lifecycle but intentionally naive in scope.

It serves one request synchronously. There is no admission queue, fairness
policy, physical batch, or concurrent mutation. Its queue delay is only the
distance between two local timestamps. A real server must handle work that
arrives while resources are occupied.

Its model source allocates a small vector of integer-scored candidates. There
is no vocabulary-scale score vector, tensor computation, tokenizer, model
artifact, or device. Treating those integers as real logits would confuse the
teaching boundary with the model we have not built.

Its sink is infallible. A real bounded channel can fill. A network client can
disconnect. A protocol adapter can fail after some pieces have been delivered.
The engine then needs a precise rule for whether the request is cancelled,
failed, drained, or allowed to finish off-wire.

Its cancellation source is a deterministic step predicate. Production
cancellation races with model execution, device queues, output emission, and
resource release. Exactly-once terminal behavior becomes harder when several
tasks can observe cancellation simultaneously.

It loads no artifact and selects no provider or backend. Those omissions keep
Chapter 1 honest, but they also mean ENGINE-0 cannot answer a real prompt. The
correct response is to extend its seams, not to stuff every future subsystem
into `Runtime::generate`.

## Common mistakes

### “The model answered the request”

The model contributed candidate computation. The engine owned the request,
selection, stream, stopping, and cleanup. Collapsing those owners makes failure
analysis vague.

### “Inference is the opposite of training”

They are different operational regimes, not mirror images. Both can use forward
computation. Their mutable state, scheduling, reliability, and external
contracts differ.

### “A token is the text chunk I saw”

A token is a vocabulary identifier. Output protocols carry bytes or strings,
which may combine or split token boundaries. ENGINE-0 makes them one-to-one for
teaching and labels the simplification.

### “The model is on the GPU, so the CPU is not involved”

Device kernels are part of a wider control and data path. Host work, transfer,
launch, synchronization, token handling, and protocol work still need owners.

### “More tokens per second means lower latency”

Aggregate throughput and per-request latency are different measurements.
Concurrency can increase one while worsening the other.

### “The stream ended, so it completed”

A channel can close after success, cancellation, producer failure, consumer
departure, or process loss. A robust contract carries terminal cause rather
than inferring success from silence.

### “The source file exists, so the feature is current”

Reachability matters. Hermon's paged and native components are valuable
evidence, but source presence does not make them the default request path.
Gates, runtime selection, fallbacks, and tests determine status.

> **ENGINEERING FAILURE**
> A protocol adapter receives a runtime error after streaming two pieces. It
> logs the error, breaks its loop, and emits its ordinary successful close
> chunk. The runtime failed correctly, but the wire contract lied. Terminal
> cause must be traced through every boundary, not checked only at the model
> worker.

## Exercises

### CHECK

1. For each noun—artifact, running model, request state, provider, backend,
   stream, terminal—write whether it is primarily persistent data, immutable
   runtime data, mutable runtime data, policy, or an interface. Explain any noun
   that spans categories.
2. An API reports “latency: 80 ms.” List at least five plausible endpoint pairs
   that number might describe. Which one would you call TTFT?
3. A service completes 20 requests in one second, each with a different output
   length. What can you conclude from “20 requests/s”? What can you not
   conclude about token throughput, fairness, or tail latency?
4. Why can two requests safely share model weights but not necessarily sampler
   or generation state?

### BUILD

Complete [Lab 1 — Generate One Token
Manually](../../labs/lab-01-generate-one-token-manually.md). Predict the token
and trace from the independent oracle before running ENGINE-0. Capture one
successful, one cancelled, and one failed stream.

Add a `RecordingSink` assertion that every token index is contiguous from zero.
Do not change the fake model to make the test easy; test the public stream.

### BREAK

Temporarily change the step-0 `green` score from 4 to 12. Predict the new token
before running. Which test fails, and why is that failure evidence that the
oracle is independent of the implementation?

In a temporary local branch, remove the terminal guard from `emit_token` and
make the unit test attempt emission after `finish`. Observe the failure, then
restore the invariant. Do not commit the broken state.

### EXTEND

Implement a second deterministic selector that chooses the lowest token ID,
regardless of score. Use it with the same `Runtime`, `Request`, `Model`, and
`TokenSink`. Explain which events change and which lifecycle invariants remain
unchanged.

Sketch—but do not yet implement—a `Tokenizer` boundary for Chapter 2. Identify
input bytes, output token IDs, error behavior, and ownership. Keep selection and
terminal behavior out of that interface.

## What this chapter has not explained

We have named stages that remain black boxes.

We have not explained how text becomes token IDs, why token boundaries can cut
across familiar words or Unicode boundaries, or how special tokens participate
in prompts. That is Chapter 2.

We have not explained embeddings, hidden states, tensor shapes, learned
weights, or how a numerical model produces a score for each vocabulary item.
Chapter 3 builds the smallest model that can do so.

We have not defined real logits, probability normalization, temperature,
top-k, top-p, random seeds, stop sequences, or the complete autoregressive
feedback loop. Chapter 4 owns those semantics.

We have only previewed prefill and decode. We have not derived KV caching,
continuous batching, paged memory, prefix reuse, speculative decoding,
quantization, SIMD, accelerators, distributed execution, or production
protocols. Later parts introduce each only after its simpler predecessor and
failure mode are visible.

Finally, we have not claimed that ENGINE-0 is fast. Its trace is instrumentation
for learning event boundaries. Performance claims begin only after correctness,
workload, hardware, and measurement controls are explicit.

## Summary

An LLM answer is not produced by weights alone. A model artifact must become a
running model. An inference engine must turn a request into owned mutable state,
coordinate compute and memory, invoke model work, select tokens, stream bytes,
and produce one terminal outcome. A server and service add protocol and
operational contracts without replacing the engine's responsibility.

The durable distinctions are:

- training updates model parameters; inference advances requests with normally
  immutable weights;
- an artifact is persistent representation; a running model is ready to
  execute; an engine owns request-to-outcome behavior;
- provider routing and hardware backend execution are separate decisions;
- the control plane decides and accounts; the data plane moves and transforms
  inference bytes;
- token identity, byte representation, and lifetime ownership are different
  journeys that must agree;
- completed, cancelled, and failed are distinct terminal outcomes;
- TTFT, ITL, request latency, throughput, and concurrency answer different
  questions;
- prefill and decode are different workload phases, even though we have not yet
  derived them;
- GPU execution remains part of a host-coordinated runtime;
- current, preview, and library status require call-path evidence.

ENGINE-0 makes the lifecycle executable. Its fake model chooses `blue` from a
hand-computable table. Its real contribution is the contract around that
choice: validation, admission, state ownership, deterministic selection,
ordered token emission, cancellation, failure, named timing points, and exactly
one terminal.

## Next: from text to tokens

Chapter 2 replaces the first fake boundary. We will begin with bytes, construct
a tokenizer contract, distinguish vocabulary identity from text display, and
prove round trips and failure behavior. The request state machine remains in
place. By the end of that chapter, ENGINE-0 will no longer treat the prompt as
opaque, but it will still use the same runtime, stream, and terminal ownership
established here.

## Primary references

- Hermon source and architecture at commit
  [`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25):
  [`dispatch.rs`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/dispatch.rs),
  [`batched.rs`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/batched.rs),
  [`metrics.rs`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/crates/hermon-runtime/src/metrics.rs), and
  [`CORE_ENGINE_ARCHITECTURE.md`](https://github.com/hermonai/hermon/blob/472a44cdb511b2dae6c9569e59543db8f8350b25/docs/CORE_ENGINE_ARCHITECTURE.md).
- llama.cpp, [`llama-server` architecture at
  `3466812`](https://github.com/ggml-org/llama.cpp/blob/3466812d1f06728effe7c0f3c0671117f461672d/tools/server/README-dev.md).
- vLLM, [current architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/),
  source snapshot [`80389cf`](https://github.com/vllm-project/vllm/commit/80389cfedd5040e382d64a64b1782f66de1a38bf).
- SGLang, [`Scheduler` at
  `221a627`](https://github.com/sgl-project/sglang/blob/221a6273ce3212c79483df233b4511fdf8fbe6d0/python/sglang/srt/managers/scheduler.py).
- NVIDIA TensorRT-LLM, [current architecture
  overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/overview.html),
  source snapshot [`fcc8454`](https://github.com/NVIDIA/TensorRT-LLM/commit/fcc84548ee6000530222600b33c4e733eaaf4de1).
- Hugging Face Transformers, [generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)
  and [generation strategies](https://huggingface.co/docs/transformers/generation_strategies),
  source snapshot [`ac32445`](https://github.com/huggingface/transformers/commit/ac3244569528944b9d5773cafea525cd8a8b63de).
- Hugging Face Text Generation Inference, [architecture](https://huggingface.co/docs/text-generation-inference/architecture)
  and [current maintenance notice](https://huggingface.co/docs/text-generation-inference/index),
  source snapshot [`b4adbf2`](https://github.com/huggingface/text-generation-inference/commit/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed).
