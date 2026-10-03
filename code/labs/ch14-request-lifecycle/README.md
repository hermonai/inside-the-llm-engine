# Printed Chapter 31 lab: terminal ownership and cancellation

Stable source/lab ID: `ch14-request-lifecycle`. The original ID is retained
to preserve its evidence history. This is a CPU-thread teaching model, not an
HTTP server, real-model benchmark or implementation of either named engine.

A small server in one process. An engine thread steps every running request
once per tick --- a decode step --- and delivers results to one queue that wakes
every waiting handler, as llama.cpp's server does (`server_response::send` and
`recv_with_timeout` in `tools/server`, commit `d006858`). Handler threads stand
in for HTTP threads: a streamed request gets a result per token and its handler
writes each one to the client; a non-streamed request gets one final result and
its handler waits for it.

**Part 1** replays the chapter's opening at one-tenth of its time scale (a
4 ms step for the server's 40 ms, 100 ms waits for its one-second
`HTTP_POLLING_SECONDS`). A client leaves a non-streamed request after 350 ms,
and the program counts the tokens the engine makes for nobody and the time
until it releases the request, for two handler designs:

- `poll`: wait for a result in slices and check the client only when a slice
  expires with no result. A result for any request wakes every waiter, and a
  woken waiter that finds nothing for itself starts a new slice;
- `event`: the disconnect wakes the handler, which cancels at once (vLLM races
  each request against the server's disconnect message);

with the rest of the server idle, streaming 500 tokens to another client, or
answering a monitor that asks for status every 2 ms (the chapter's measuring
script did that, every 20 ms). A streaming client that leaves after 20 tokens
closes the part.

**Part 2** checks the terminal contract without threads: 10,000 random
scenarios of one to eight requests with random arrivals, budgets, EOS
positions, injected failures and cancellations run through the engine, and
every request must end exactly once, with nothing after its end, its streamed
tokens numbered in order and complete, the outcome that an independent
token-by-token simulation predicts, and no token after its cancellation.

**Predict.** On an idle server, how many tokens can the poll design make after
the client leaves, at most? How many while another client streams 500 tokens?
Does the event design care what else the server is doing?

**Run.** From `code/labs`:

```bash
cargo run --release -p ch14-request-lifecycle
cargo run --release -p ch14-request-lifecycle -- --slice-ms 3
cargo run --release -p ch14-request-lifecycle -- --scenarios 100000 --seed 7
cargo test -p ch14-request-lifecycle
cargo run --release -p ch14-request-lifecycle -- --contracts-only --scenarios 100000 --seed 7
```

**Explain.** The poll design checks its client only when nothing at all has
happened for a whole slice, so its reaction time is set by the busiest other
request on the server, not by its own. Every token streamed to a neighbour,
and every status reply, restarts its wait. The event design reacts to the
disconnect itself, within a step. A streaming handler finds out at its next
write, so it wastes one token whatever the design.

**Break.** Run with `--slice-ms 3`, shorter than a step: the poll design now
catches the leaver, because slices expire between steps --- at the price of
waking every handler hundreds of times a second. In `Engine::step`, emit the
budget's `Final` and keep the request running (return `true`), and watch
Part 2 name the first request that ended twice.

## Protocol extension: publication is not retirement

`ownership.rs` is an executable single-owner protocol model. It holds a
bounded deque of text events, separate terminal metadata, and at most one
work ticket. Each ticket contains a slot, generation and serial. It neither
allocates GPU memory nor polls a socket.

Predict whether the lease is reusable after cancellation but before the
matching ticket returns. Run the program's initial protocol demonstration:
it delivers one terminal while reuse is false, then discards late text and
permits reuse after retirement. Fill both data slots: cancellation still
works because it does not need queue capacity. Natural EOS/budget completion
instead drains the prefix before terminal delivery.

The tests enumerate 78,125 seven-command sequences (submit, retire, cancel,
EOS, receive), 256 short notification schedules and 490 small natural-stop
specifications. Controlled cases reject stale generations, stale serials,
duplicate retirement, duplicate active admission, mixed-ID traces, status
events in a generation trace, post-terminal output and a natural completion
after observed cancellation. Seeded scenarios complement these tests; they
are not exhaustive production model checking.

Break `Owner::reusable` by ignoring pending work. The cancellation/retirement
test must fail. Break `check` by removing its observed-cancellation guard;
the dedicated rejection test must fail. Restore mutations before timing.

## Timing interpretation

The worker follows scheduled ticks, but OS sleeps and scheduling can
overshoot. The observer reads a published token-count snapshot immediately
before recording the close time; these operations are not an atomic cut
through the worker. Counts can differ at a boundary. Delays are printed
rounded to integer milliseconds, so `0` is not literally zero duration.
Results explain the mechanism, not a hard latency guarantee.

The historical llama.cpp model measurement is retained in
`research/measurements/2026-09-24-m1-part3.md`. The new scaled runs and
correctness checks are recorded separately in
`research/measurements/2026-10-03-request-lifecycle.md`.
