# Lab 3 — Stream UTF-8 Across Token Boundaries

## Purpose

Prove that token pieces are bytes and that valid text emission may wait across
multiple selected tokens.

## Prerequisites and artifact

Read Chapter 2's streaming-decode section and the
[`utf8-stream.txt`](../code/mini-engine/fixtures/tokenizer/utf8-stream.txt)
fixture. Produce an event trace showing token events, pending bytes, text
events, and the terminal outcome.

## CHECK

Write the UTF-8 bytes for `世` and `🚀`. For each possible split between bytes,
classify the left fragment as complete, valid-incomplete, or invalid.

## BUILD

Run the focused tests:

```sh
cd code/mini-engine
cargo test -p engine0 --test utf8_stream
cargo test -p engine0 two_token_byte_fragments_emit_one_valid_scalar
```

Confirm that the first partial token has an identity event but no text event,
and that the completing token emits one valid scalar.

## BREAK

Feed `C3 28`. Confirm that it fails rather than emitting U+FFFD. Then feed only
`F0 9F` and call `finish`; confirm that the incomplete suffix is a terminal
error. Do not use `from_utf8_lossy` to make the test pass.

## EXTEND

Add a fixture whose first token contains complete ASCII followed by an
incomplete emoji prefix. The ASCII prefix should emit immediately while at most
three bytes remain buffered. Add a test for several complete scalars in one
piece.

## Cleanup

Restore any injected malformed pieces and run the full workspace gate.

