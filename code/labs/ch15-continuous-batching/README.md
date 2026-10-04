# Chapter 32 lab — deterministic continuous batching

This lab isolates scheduler correctness from model and GPU performance. Time is
an integer tick. A launched ticket completes deterministically before the next
tick. No external crates are required.

## Predict

Before running the code, use `max_seqs=2` and `max_tokens=3`. Requests A and B
arrive at tick 0. A has a 3-token prompt and B a 1-token prompt. Prefill is not
chunked in this chapter. Predict which request is scheduled first and why the
unused sequence position cannot be used.

Expected reasoning: A consumes all three token positions, so the token budget,
not the sequence budget, blocks B.

## Run

```bash
cargo test
cargo run -- trace
```

The trace prints arrival, admission, schedule, terminal and retirement events.
The independent accounting oracle reconstructs sequence and token usage from
the trace instead of using scheduler-internal counters.

## Explain

The durable identity is `(request_id, generation)`. A physical slot is only a
location. Reuse changes its generation. Worker tickets snapshot both values.

The scheduler separates:

1. arrival/cancellation observation,
2. safe retirement,
3. admission into bounded slots,
4. per-iteration scheduling under token and sequence budgets,
5. result validation and commit.

This lab deliberately does **not** demonstrate Transformer kernels, GPU
throughput, paged KV allocation, prefix caching, speculative decoding, or
chunked prefill.

## Break

Three meaningful failures are covered by tests:

- **Retain finished slots:** omit retirement and queued requests cannot enter.
- **Over-admit:** ignore the slot limit and active state exceeds capacity.
- **Late result after reuse:** validate only a slot index and an old worker
  result can mutate a new request. The correct test requires request ID plus
  generation to match.

Run the focused stale-result test:

```bash
cargo test late_result_is_rejected_after_generation_changes -- --nocapture
```

The CLI command below is an instruction stub. It exits non-zero and points to
the focused test; it does **not** run a broken scheduler or an oracle:

```bash
cargo run -- broken late-result
```
