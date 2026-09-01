# Chapter 1 Research — The Missing Half of AI

Inspection date: 2026-09-02.

## Question

What machinery turns an already-trained language-model artifact and a request
into an ordered, observable, terminal stream, and which distinctions must a
reader learn before studying tokenization or transformer mathematics?

The chapter is conceptual, but it must leave an executable boundary. The
reader should be able to identify a model artifact, a running model, request
state, a model/token-source boundary, selection policy, an output sink, and one
terminal outcome without mistaking any of those for the others.

## Scope and truth categories

- **CURRENT** describes Hermon behavior traced at the pinned commit below.
- **EXTERNAL** describes current official project documentation or source at
  the recorded external repository heads.
- **INFERENCE** marks a cross-project synthesis made by this book.
- **ENGINE-0** describes the teaching implementation added with Chapter 1. It
  is not a real neural model or production server.

The chapter does not explain tokenizer algorithms, transformer layers, GGUF
layout, quantization, KV caches, continuous batching, or accelerator kernels.
It may name prefill and decode only to orient later work.

## Primary sources

All repository heads below were resolved with `git ls-remote ... HEAD` on the
inspection date. The documentation links are maintained, official project
surfaces; commit links pin the corresponding source snapshot.

| Project | Recorded source version | Primary evidence used |
| --- | --- | --- |
| Hermon | [`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25) | Local source: `hermon-api`, `hermon-runtime::{dispatch,batched,metrics}`, `hermon-core::provider`; canonical `CORE_ENGINE_ARCHITECTURE.md` and `INTERNALS.md` |
| llama.cpp | [`3466812`](https://github.com/ggml-org/llama.cpp/commit/3466812d1f06728effe7c0f3c0671117f461672d) | Official [`llama-server` developer architecture](https://github.com/ggml-org/llama.cpp/blob/3466812d1f06728effe7c0f3c0671117f461672d/tools/server/README-dev.md) and [server README](https://github.com/ggml-org/llama.cpp/blob/3466812d1f06728effe7c0f3c0671117f461672d/tools/server/README.md) |
| vLLM | [`80389cf`](https://github.com/vllm-project/vllm/commit/80389cfedd5040e382d64a64b1782f66de1a38bf) | Official [architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/) and linked V1 engine source |
| SGLang | [`221a627`](https://github.com/sgl-project/sglang/commit/221a6273ce3212c79483df233b4511fdf8fbe6d0) | Official [project architecture/features](https://github.com/sgl-project/sglang/tree/221a6273ce3212c79483df233b4511fdf8fbe6d0) and [`Scheduler` source](https://github.com/sgl-project/sglang/blob/221a6273ce3212c79483df233b4511fdf8fbe6d0/python/sglang/srt/managers/scheduler.py) |
| TensorRT-LLM | [`fcc8454`](https://github.com/NVIDIA/TensorRT-LLM/commit/fcc84548ee6000530222600b33c4e733eaaf4de1) | NVIDIA's current [architecture overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/overview.html) |
| Hugging Face Transformers | [`ac32445`](https://github.com/huggingface/transformers/commit/ac3244569528944b9d5773cafea525cd8a8b63de) | Official [`GenerationMixin.generate`](https://huggingface.co/docs/transformers/main_classes/text_generation) and [generation-strategy](https://huggingface.co/docs/transformers/generation_strategies) documentation |
| Hugging Face TGI | [`b4adbf2`](https://github.com/huggingface/text-generation-inference/commit/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed) | Official [TGI architecture](https://huggingface.co/docs/text-generation-inference/architecture) and [maintenance-status notice](https://huggingface.co/docs/text-generation-inference/index) |

No performance comparisons are used. Feature lists are architecture evidence,
not proof that a feature is selected, correct for every model, or faster on a
particular workload.

## Verified facts

### Cross-project boundaries

**[EXTERNAL] llama.cpp.** `llama-server` distinguishes an HTTP/routing layer,
thread-safe request and response queues, a `server_context` that holds primary
inference state, and per-sequence slots. Its documented server supports
parallel decoding and continuous batching. This is evidence that a server is
more than the model library and that request slots are mutable runtime state;
it does not make llama.cpp's exact slot design universal.

**[EXTERNAL] vLLM.** V1 separates an API-server process, an engine-core process,
and worker processes. The API server performs input processing and streaming;
the engine core schedules and manages KV state; workers load weights and run
model forward passes. The offline `LLM` interface reaches related engine
machinery without requiring the online HTTP service. This makes the
library/engine/server/service distinctions concrete.

**[EXTERNAL] SGLang.** The current SRT `Scheduler` is a large runtime owner: it
initializes a tensor-parallel model worker, cache/memory pools, scheduling
policy, request receiver, output streamer, metrics, and constrained-generation
support. The project documents continuous batching, paged attention,
RadixAttention prefix reuse, chunked prefill, and several parallel execution
forms. Chapter 1 should use only the boundary lesson: the runtime coordinates
resources and request state around model execution. Later chapters must verify
each named mechanism independently.

**[EXTERNAL] TensorRT-LLM.** The high-level `LLM` interface handles tokenization
and detokenization while creating executor workers. The documented `PyExecutor`
runs an asynchronous loop with scheduler, model engine, and decoder roles, and
can overlap GPU work for one step with CPU processing for the previous step.
This is evidence that “the model ran on a GPU” omits substantial host control
work.

**[EXTERNAL] Transformers and TGI.** Transformers exposes model loading and a
configurable `generate` loop with logits processors, stopping criteria, and
streamers. It is a library surface, not by itself a multi-user serving system.
TGI documents a router/webserver that validates and batches requests, a
launcher, and one or more model-server shards reached through gRPC. TGI is now
in maintenance mode and recommends current engines including vLLM, SGLang, and
local systems such as llama.cpp. The architecture remains useful evidence, but
the maintenance status must travel with any present-tense reference.

### Hermon request trace at `472a44c`

**[CURRENT] Provider and local-backend choice.** The OpenAI-compatible handler
first asks `ProviderRouter` to resolve the model name. In an engine-enabled
binary, a model name that also resolves to a local GGUF and passes
`hermon_engine::is_linked()` enters the in-process engine route. If that local
condition is not met, the shown handler falls through to its Ollama client.
Provider routing and execution backends are related layers, not synonyms: a
provider chooses where a request is directed; a backend implements model work
on some hardware/runtime substrate.

**[CURRENT] Dispatcher and runtime selection.** `Dispatcher::new` reads
`HERMON_RUNTIME_MODE` once. Unset or unrecognized values select `batched`;
`pool` selects the compatibility path; `paged` logs a preview warning. A
per-model runtime is cached by canonical model path. The dispatcher releases
its map lock before request execution.

**[CURRENT] Default runtime owner.** `BatchedRuntime` owns one shared llama.cpp
model, one multi-sequence context, a submission channel, one dedicated OS
worker thread, and atomic metrics. `BatchedWorker` is the sole mutator of the
context, batch, active-sequence table, samplers, and sticky-slot metadata. It
admits requests into logical sequence IDs, assembles prompt or decode work into
a shared physical batch, calls `decode_batch`, samples eligible logit rows,
buffers token bytes until valid UTF-8 can be emitted, and finalizes or
continues each request.

**[CURRENT] Stream and terminal contract.** The runtime's successful stream is
zero or more `StreamItem::Piece(String)` values followed by exactly one
`StreamItem::Done(EngineMetrics)`, then channel close. Runtime errors are
delivered as `Err(EngineError)` and do not receive a `Done`. Output channels
are bounded to 32 entries, so a slow consumer eventually backpressures the
producer. Dropping the receiver is treated as client departure when a send is
attempted, after which request state and counters are cleaned up.

**[CURRENT] Metrics boundary.** `EngineMetrics` carries per-request prompt
tokens, completion tokens, and decode time. The batched worker also owns atomic
runtime counters for admissions, completions, failures, cache activity,
prefill/decode work, speculative tokens, active slots, and warm slots. Only
batched runtimes appear in `Dispatcher::snapshot_metrics`; pool mode predates
that surface and paged persistent snapshots remain outside the current gate.

**[PREVIEW] Paged path.** `RuntimeMode::Paged` constructs a metadata-validated
runtime, but real packed-GGUF inference requires `HERMON_PAGED_GGUF=1`. The
current source describes CPU, greedy, serialized-per-model execution and keeps
the mode behind a release gate. The existence of paged-KV and native-kernel
crates is not evidence that they own the default path.

### Source/document discrepancies to preserve

1. The top comment of `dispatch.rs` still opens with the older pool design and
   says continuous batching is “what's next,” while the executable default in
   `RuntimeMode::from_env` is already `Batched`. The enum and call path outrank
   that stale introductory comment.
2. The lower-level runtime contract distinguishes error from successful
   `Done`. In the inspected OpenAI SSE adapter, an engine-stream error is logged
   and the loop breaks, after which the adapter still attempts a generic
   `finish_reason: stop` chunk and `[DONE]`. Chapter 1 states the verified
   runtime contract and names the wire-adapter caveat; it does not claim that
   every protocol currently preserves terminal cause perfectly.
3. Some canonical request-lifecycle prose lists stop sequence and cancellation
   alongside EOS/token-limit finalization. The specific OpenAI in-process call
   shown here does not thread `ChatRequest.stop` through
   `Dispatcher::stream_with_options`. Later protocol/lifecycle chapters must
   audit stop and cancellation end to end rather than inheriting the broad
   document wording.

## Derived system model

**[INFERENCE]** Across the inspected systems, an inference engine is the
stateful control-and-execution layer that turns normalized generation work and
a runnable model into ordered progress and one terminal outcome while owning
resource, concurrency, and failure policy. It may be embedded behind a library
call or deployed behind a server. It is not identical to the model artifact,
the mathematical forward function, the accelerator backend, or the network
service.

The minimum Chapter 1 map is:

```text
control plane: validate -> resolve -> admit -> schedule -> stop/fail -> account
                                      |
                                      v
data plane:    artifact -> weights -> model step -> candidates -> token -> bytes
                                      ^                         |
                                      |                         v
                                request state <----------- ordered stream
```

“Control plane” and “data plane” are teaching labels. Actual projects may put
both in one process or one worker. The distinction is about decisions versus
the bulk model/state data those decisions move and transform.

## Terminology decisions

- **Model artifact:** persistent serialized configuration, tokenizer data, and
  weights. It is inert bytes until loaded and interpreted.
- **Running model:** validated model semantics plus resident weights and
  execution resources ready to perform a forward step.
- **Model:** use carefully for either the mathematical mapping or a project
  object; qualify when ownership matters.
- **Inference runtime:** stateful machinery that advances one or more requests.
- **Inference engine:** runtime plus execution/control policy that produces
  generation outcomes. “Runtime” may name a narrower implementation unit.
- **Request:** bounded generation intent and its mutable lifecycle state; not a
  conversation and not a physical batch.
- **Provider:** routing/placement identity that offers an inference capability.
- **Backend:** implementation used to execute operations on a hardware/runtime
  substrate. Project vocabularies differ, so qualify external usage.
- **Stream:** ordered progress events; it must end in a defined terminal
  outcome rather than merely ceasing output.
- **Terminal outcome:** completed, cancelled, or failed. A completed request
  also records a stop reason such as end-of-sequence or token limit.
- **Latency:** elapsed time between named lifecycle points. Never report an
  unlabeled “latency.”
- **TTFT:** admission/arrival to first externally observable output token or
  piece, with the exact endpoints disclosed.
- **ITL:** elapsed time between consecutive observable output events/tokens,
  with aggregation disclosed.
- **Throughput:** completed work per unit time, with work unit and population
  defined.
- **Concurrency:** number of requests simultaneously admitted or active, not
  necessarily the number executed in one physical batch.
- **Prefill / decode:** prompt-state construction versus iterative generation
  work. Full treatment begins later.

## Chapter contract metadata

- **Purpose:** make the hidden inference half visible and establish the system
  nouns used by the rest of the book.
- **Prerequisites:** programming experience and basic client/API familiarity.
- **Key question:** what owns the journey from request to terminal stream?
- **Mathematics:** lifecycle timestamp decomposition and dimensional throughput
  definitions; no transformer math yet.
- **Systems concepts:** ownership, mutable request state, queueing, streaming,
  control/data planes, provider/backend boundaries, and terminal invariants.
- **Hardware concepts:** host control work and device/CPU execution are both
  part of inference; “GPU inference” does not eliminate CPU orchestration.
- **Implementation:** standard-library-only Rust ENGINE-0 with a deterministic
  candidate source, greedy selector, stream events, cancellation, errors,
  trace mode, and timing points.
- **Hermon connection:** bounded current request trace and status-labeled
  default/preview/library distinctions.
- **External connection:** compare boundaries, not performance, across the six
  projects above.
- **Deliverable:** Chapter 1, ENGINE-0, Lab 1, four canonical diagrams, updated
  glossary/status, and executable correctness evidence.
- **Next assumption:** Chapter 2 may replace the fake prompt handling with
  real byte-to-token semantics without rewriting request lifecycle.

## Planned diagrams

1. Request-to-token lifecycle with successful, cancelled, and failed terminals.
2. Model versus engine, showing immutable artifact/weights and mutable request
   state.
3. Inference stack from library through engine, server, service, provider, and
   hardware backend.
4. One combined follow-the-token / follow-the-byte / follow-the-owner trace.

## Experiment and independent oracle

ENGINE-0's deterministic fake model returns the following scored candidates:

```text
generation step 0: blue=9, green=4, <eos>=1  -> greedy token: blue
generation step 1: <eos>=10, blue=1         -> terminal: end-of-sequence
```

The oracle is written independently as this table rather than computed by the
runtime. Lab 1 asks the reader to predict the first token and event order before
running the program, change a score to break the prediction, and then restore
or extend the source. No timing claim is made: the trace timestamps are a
pedagogical decomposition on the reader's machine, not a benchmark.

## Correctness gates

- deterministic first token matches the hand-computable oracle;
- event order is admitted, started, zero or more tokens, one terminal;
- completed, cancelled, and failed paths each emit exactly one terminal;
- invalid input emits no token and fails explicitly;
- no event can be emitted after terminal state;
- a repeated run with identical input and configuration has identical semantic
  events (timestamps excluded);
- a model failure after one token ends as failed and cannot also complete;
- code formats, builds, tests, and passes Clippy with no external dependency.

## Open questions and later work

1. Chapter 2 must decide the exact tokenizer contract and byte round-trip
   fixtures; ENGINE-0 deliberately treats prompt text as opaque.
2. Chapter 3 must define a tiny numerical model output compatible with the
   candidate-selection seam without preserving fake “logits-like” scores as
   real model semantics.
3. Chapter 4 must add stochastic selection, RNG ownership, stop sequences, and
   the autoregressive feedback loop.
4. Parts V and XII must specify cancellation races, sink failures,
   backpressure, finish-reason mapping, and wire-terminal behavior completely.
5. Hermon's OpenAI SSE error-to-terminal mapping and stop-sequence propagation
   are bounded observations for later audit, not changes authorized by this
   book task.

## Completion review record — 2026-09-02

The technical and editorial passes were performed separately after the chapter,
code, oracle, lab, and diagrams existed.

### Technical pass

- Re-read the pinned Hermon handler, dispatcher, batched worker, stream, and
  metrics paths. Kept the default/preview/library classifications and the three
  source/document caveats visible.
- Checked every named ENGINE-0 public concept against the compiled API and every
  stated invariant against a test. Format, check, 11 tests, and Clippy passed.
- Rechecked external claims against official project documentation or source at
  the recorded heads. The chapter makes no comparative performance claim.
- Resolved one vocabulary defect found during review: *provider* now names a
  selectable capability/destination, while *backend* names the hardware/runtime
  operation implementation.

### Editorial pass

- **Beginner lens:** definitions precede comparisons; fake candidate scores are
  never presented as logits; prefill/decode remain a preview; the “not yet
  explained” section protects later prerequisites.
- **Systems lens:** all four diagrams expose mutation or ownership, and the
  chapter follows success, cancellation, and failure to a single terminal.
- **Inference-engineer lens:** logical request, concurrency, physical batch,
  model artifact, running model, provider, and backend remain distinct; Hermon
  reachability is not inferred from source-file presence.
- **Progression:** opening problem -> mental model -> boundaries -> lifecycle
  math -> current architectures -> Hermon evidence -> implementation -> proof
  -> limitations -> exercises -> Chapter 2 handoff.

### Cross-link and artifact pass

- The final chapter contains 6,874 words.
- All four canonical diagrams render at no more than 100 columns and define
  their arrows/ownership marks.
- Repository-relative links pass `scripts/check-links.py`; the structure and
  word-count contract pass `scripts/check-structure.sh`.
- Glossary, Part I index, book table of contents, Lab ledger, README, status,
  CI, and next-task handoff agree.
