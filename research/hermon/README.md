# Hermon Initial Architecture Inventory

## Snapshot and scope

Inspected 2026-09-02 against public repository
[`hermonai/hermon`](https://github.com/hermonai/hermon) commit
[`472a44c`](https://github.com/hermonai/hermon/commit/472a44cdb511b2dae6c9569e59543db8f8350b25),
workspace version 0.6.0. The local `origin/main` was fetched before inspection.
This is top-level reconnaissance, not a substitute for chapter-specific source
review.

Evidence inspected: root `Cargo.toml` and README; crate manifests; runtime
dispatcher, batched and paged sources; API engine route; GGUF, paged-KV,
kernel, and llama.cpp bridge sources/tests; and canonical architecture,
internals, kernel, storage, performance, benchmarking, innovation, and roadmap
documents.

## Architecture map

```text
wire client
    |
    v
hermon-api  -- protocol validation, SSE/NDJSON framing
    |
    +--> hermon-core -- config, persistence, cloud/provider routing
    |
    v
hermon-runtime::Dispatcher -- one selected runtime per model path
    |
    +--> pool     [compatibility] llama.cpp, one context per slot
    +--> batched  [CURRENT default] one multi-sequence llama.cpp context
    `--> paged    [PREVIEW] Hermon-owned blocks/radix + packed GGML math
                       |
                       +--> hermon-gguf      validated metadata/index
                       +--> hermon-paged-kv  F32 reference pool/oracle
                       +--> hermon-kernels   optional native attention A/B
                       `--> hermon-llamacpp  packed tensor bridge
```

### Workspace crates

| Crate | Verified responsibility | Status note |
| --- | --- | --- |
| `hermon-core` | Configuration, shared types, providers/routing, keystore, model/hub/Ollama integration, SQLite conversations | CURRENT support layer |
| `hermon-api` | OpenAI/Ollama/Anthropic/Hermon HTTP surfaces and stream framing | CURRENT; delegates local execution through Dispatcher when built with engine feature |
| `hermon-cli` | User commands and server entrypoint | CURRENT; backend feature forwards through API/runtime/engine |
| `hermon-runtime` | Dispatcher, production batched worker, prompt lookup, lifecycle/metrics, gated paged runtime | Mixed CURRENT and PREVIEW; inspect per module |
| `hermon-engine` | Safe tensor/model/context/sampler facade over `hermon-llamacpp` | CURRENT for native builds |
| `hermon-llamacpp` | Pinned upstream build, C/C++ shim, opaque handles, safe Rust wrapper, packed GGML bridge | CURRENT production model mechanism; unsafe allowed at FFI boundary |
| `hermon-gguf` | Bounds-checked header, metadata, tensor directory/ranges, common model shape extraction | CURRENT library; used by paged preview and model inspection |
| `hermon-paged-kv` | Safe CPU block pool, refcounts, prefix radix, COW, dense/reference paged attention | PREVIEW/reference component |
| `hermon-kernels` | C11 arena/pool, stable ABI, paged attention, ISA/provider dispatch, MoE expert storage/paging primitives | LIBRARY plus optional paged-preview attention path; second unsafe boundary |
| `hermon-tokenizer` | Tokenizer research interfaces and prefix cache | LIBRARY/research; real-model paths use pinned llama.cpp tokenizer today |
| `hermon-bench` | Latency/throughput client harness | Measurement tool, not production behavior |

## Runtime modes and default path

**[CURRENT]** `RuntimeMode::from_env` in
`crates/hermon-runtime/src/dispatch.rs` reads `HERMON_RUNTIME_MODE`. Unset or
unknown values select `batched`; `pool` is the v0.3 compatibility path; `paged`
is explicitly warned as preview. Runtime selection is fixed when the Dispatcher
is constructed, and per-model runtimes are cached by canonical path.

**[CURRENT]** The default `BatchedRuntime` owns one llama.cpp context with
multiple sequence IDs and one dedicated OS worker thread. API tasks submit
normalized requests; the worker alone mutates context/batch state, combines
prompt chunks and decode positions into shared `llama_batch` calls, streams over
bounded channels, and retains eligible sticky-slot prefixes. This is continuous
batching, but sticky slots are not cross-sequence physical page sharing.

**[CURRENT compatibility]** `pool` owns independent contexts per available
slot, sharing `Arc<Model>` weights. It remains selectable for fallback and A/B.

## Paged preview and KV ownership

**[PREVIEW]** Selecting `HERMON_RUNTIME_MODE=paged` does not silently enable real
model execution. `HERMON_PAGED_GGUF=1` is required for the current CPU/greedy
path; otherwise dispatch returns an actionable release-gate error.

The preview validates GGUF geometry, retains packed projection/MLP weights,
uses the pinned GGML bridge for packed matmuls, and owns explicit KV through
`CpuBlockPool` and `CpuPrefixRadix`. A request block table and radix entry both
hold references. A block returns to the free list iff its refcount reaches zero.
Shared partial-tail blocks are copied before continuation writes. A request
lease in `paged.rs` decrements tracked blocks on success or early failure.

Current documented limits: CPU and greedy real-model path, serialized paged
context per model, default 64 blocks unless configured, F32 reference pool as
the principal ownership structure, incomplete exported paged metrics, and an
open 1,000-prompt equivalence gate.

## GGUF and native kernel layer

**[CURRENT library / PREVIEW integration]** `hermon-gguf` validates file bounds,
metadata types, tensor offsets/lengths, quantization block geometry, and bounded
tensor reads. Its presence does not imply all model-family semantics.

**[LIBRARY]** `hermon-kernels` is a C11 library with a safe Rust facade. It owns
an arena, typed block pool, bulk operations, stable/versioned C ABI, and
plan/task/combine attention. It creates no threads: the host schedules immutable
indexed tasks, and combine reduces in deterministic split-index order.

**[PREVIEW]** `HERMON_PAGED_KERNELS=1` selects `KernelPagedAttention` on the
paged request path; otherwise the safe `CpuPagedAttention` oracle is used.
The backend abstraction includes writes, forward, and block copy after a real
partial-tail integration bug showed that state mutation outside the trait could
produce plausible but wrong text.

## Accelerator providers

The production llama.cpp build features expose CPU, Metal, CUDA, Vulkan, ROCm,
and SYCL backends. That means the pinned upstream model path can be built for
those providers; it is not evidence that Hermon's own C kernel layer implements
all of them.

For **Hermon-owned native paged attention**, source/canonical docs verify scalar
C plus ARM NEON and x86 AVX2 CPU dispatch, Metal, and CUDA. GPU selection is
shape-gated and falls back to CPU on unsupported/error cases. ROCm and SYCL are
named future providers for this native contract, not current implementations.
The exact thresholds are implementation/configuration details and must be
re-verified before a chapter cites them.

## MoE subsystem

**[LIBRARY]** The kernel/storage layer implements expert container addressing,
cache residency/pinning, acquire/prefetch/release, RAII Rust leases, direct-I/O
state reporting, Linux `io_uring` batch misses, and F32/F16/Q8_0 host matvec
coverage. It is not connected to `PagedRuntime` or the default request path.
The measured “tok/s ceiling” in kernel/storage docs is a storage movement bound,
not end-to-end generation. Router integration, K-quant compute, and real-model
token differential remain open gates. The queue-depth experiment did not yield
the predicted 2–3× improvement for multi-megabyte expert records.

## Correctness approach

- `hermon-paged-kv` covers allocation/refcounts, prefix ownership, COW, eviction,
  and paged attention against a dense reference.
- `hermon-kernels` has scalar-oracle differentials, ABI/allocator/dtype/task
  tests, task-order and thread-count determinism, provider conformance, and
  sanitizer scripts.
- `hermon-llamacpp/tests/tensor_bridge.rs` exercises real-model tensor metadata,
  conversion, packed matvec/matmul, and bundled projection equivalence when a
  model fixture is supplied.
- `hermon-runtime/tests/gguf_paged_differential.rs` compares greedy paged output
  with pinned llama.cpp when `HERMON_TEST_MODEL_PATH` is provided.
- Model-dependent tests are ignored without the external fixture. Ordinary CI
  proves compilation/unit contracts, not the open real-model release gates.

## Canonical documents and cautions

Start with `docs/CORE_ENGINE_ARCHITECTURE.md` (explicit default/preview/library
boundary) and `docs/INTERNALS.md` (code-first flow). Use `DESIGN.md` for
principles, `ENGINE_STRATEGY.md` for substitution gates,
`STORAGE_ARCHITECTURE.md` for expert/KV residency and I/O truthfulness,
`KERNEL_DESIGN.md` for the C contract, `PERFORMANCE.md` plus
`BENCHMARKING.md` for measured claims, and `ROADMAP.md` only for phase intent.

Some source comments and older design/innovation sections describe future
versions as though they are “next.” The canonical architecture dated
2026-08-31 and actual dispatch gates take priority. Re-trace the call path before
publication rather than quoting an old status line.

## Follow-up questions for chapter research

1. Re-run model-dependent equivalence on the exact fixture chosen for the book.
2. Trace all protocol handlers through normalization and cancellation for Parts
   V and XII.
3. Audit provider support by source and hardware before Part VIII measurements.
4. Revisit MoE runtime integration status immediately before Part X.
5. Compare canonical docs with source again before every Part XIII chapter.

## Chapter 1 refresh — 2026-09-02

The remote and local `main` branch still resolve to `472a44c`; the initial
inventory therefore remains pinned without a commit change. Chapter 1 retraced
the concrete OpenAI-compatible local request path rather than relying on this
summary:

```text
hermon-api::chat_completions_openai
  -> ProviderRouter::resolve
  -> resolve_local_gguf + hermon_engine::is_linked
  -> engine_route::stream_with_options
  -> Dispatcher::stream_with_options
  -> BatchedRuntime::stream_with_options            [CURRENT default]
  -> BatchedWorker admission / shared llama_batch
  -> Piece* -> Done(metrics) | EngineError
```

Verified boundaries:

- the in-process local path requires an engine-linked build and a resolvable
  local GGUF; the inspected OpenAI handler otherwise uses its Ollama path;
- runtime mode is chosen when `Dispatcher` is constructed and defaults to
  `batched`; `paged` remains explicit PREVIEW and real GGUF execution has a
  second `HERMON_PAGED_GGUF=1` gate;
- the batched worker alone mutates its llama.cpp context and active request
  state; bounded per-request channels carry UTF-8 pieces and terminal/error
  events;
- successful runtime streams have exactly one `Done`; error streams have no
  `Done`; batched snapshot metrics do not imply equivalent pool/paged metrics.

Three caveats were preserved in the Chapter 1 research note: the stale pool-era
opening comment in `dispatch.rs`, the OpenAI SSE adapter's generic stop close
after a logged engine error, and stop-sequence wording in architecture prose
that is broader than the inspected in-process call signature. See
[`research/part-01/chapter-01-the-missing-half-of-ai.md`](../part-01/chapter-01-the-missing-half-of-ai.md).

## Chapter 2 refresh — 2026-09-02

The Hermon repository remained at `472a44c`; its pinned llama.cpp submodule is
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`. The upstream llama.cpp head
observed during this pass was `b81c99b479d4c24e5eeca10de99032ebd343ef8f`.
The pin, rather than upstream HEAD, is the CURRENT code Hermon builds.

The default batched real-model path was traced through message rendering,
`Model::apply_chat_template`, `Context::tokenize(..., add_special=true,
parse_special=true)`, the `hermon-llamacpp` safe wrapper, Hermon's C shim, and
the pinned llama.cpp vocabulary APIs. Raw blocking generation uses
`parse_special=false`; the chat path enables it after trusted template
formatting.

Each `ActiveSeq` owns an output `utf8_buf`. Sampled IDs become bytes through
`token_to_piece`; the worker emits the longest valid UTF-8 prefix and retains a
partial suffix. A current caveat is now recorded: the path does not distinguish
a definite `Utf8Error::error_len()` from an incomplete suffix, and finalization
uses `String::from_utf8_lossy` for leftovers. The book's teaching runtime uses
a stricter typed-error policy; no Hermon change was made.

`hermon-tokenizer` remains **LIBRARY/research**: it contains interface and
prefix-cache skeletons but is not imported by the current real-model request
path. Full evidence and test boundaries are in
[`research/part-01/chapter-02-from-text-to-tokens.md`](../part-01/chapter-02-from-text-to-tokens.md).

## Chapter 3 refresh — 2026-09-02

Hermon remained at `472a44c`, with llama.cpp pinned at `389ff61d`. The Chapter
3 pass separated Hermon's CURRENT llama.cpp-backed logits from its PREVIEW
Hermon-owned paged model. In the default batched worker, `llama_decode` executes
the model and each active sequence samples from its own output row. The linked
facade exposes tensor metadata and packed operations, but ordinary tests without
an external GGUF fixture do not prove end-to-end model equivalence. The tiny
book model therefore uses an independent, fully visible embedding/projection
fixture rather than presenting Hermon's production model as simple. See
[`research/part-01/chapter-03-the-smallest-possible-language-model.md`](../part-01/chapter-03-the-smallest-possible-language-model.md).

## Chapter 4 refresh — 2026-09-03

Hermon still resolves to `472a44c`; its pinned llama.cpp remains `389ff61d`.
The CURRENT default batched path stores sampling configuration on the submitted
request and constructs one mutable llama.cpp sampler per admitted `ActiveSeq`.
Batching shares model execution, not RNG/sampler state: each sequence samples
from its own logit row, checks EOG and completion-token budget before streaming,
and carries the selected token into its next decode position.

The C shim selects greedy decoding when temperature is non-positive. Otherwise
its commit-specific chain is top-k, top-p, min-p, temperature, then categorical
distribution; seed zero is replaced with time-derived state. Finalization emits
one `StreamItem::Done`, while errors travel as `Err(EngineError)`. These are
Hermon's stream semantics, not ENGINE-1's exact terminal type.

The Hermon-owned paged GGUF path remains **PREVIEW** and locally uses greedy
argmax. It must not be described as having the CURRENT path's stochastic
sampler coverage. Exact paths, line ranges, and pinned llama.cpp behavior are in
[`research/part-01/chapter-04-logits-sampling-autoregressive-loop.md`](../part-01/chapter-04-logits-sampling-autoregressive-loop.md).
