# Lab 2 — Tokenize by Hand

## Purpose

Apply a fixed BPE merge table without treating tokens as words. This lab belongs
to Chapter 2 and uses the independent fixture in
[`code/reference/chapter-02-tokenizer-oracles.md`](../code/reference/chapter-02-tokenizer-oracles.md).

## Prerequisites and artifact

Read Chapter 2 through the BPE derivation. Produce a short trace containing the
symbol sequence after every merge for `lower`, `lolo`, and one input of your
choice. The oracle is the committed hand fixture; the Rust encoder is the
implementation under test, not the source of expected values.

## CHECK

Before running code, write the UTF-8 byte count, Unicode scalar count, and
expected final token IDs for `lower`. Explain why those three counts need not
equal a word count.

## BUILD

Run:

```sh
cd code/mini-engine
cargo run -p engine0 -- tokenize lower
cargo run -p engine0 -- decode 259
```

Compare the final ID and decoded bytes with your trace. Then test Chinese,
emoji, a combining sequence such as `é`, and text with leading whitespace.

## BREAK

In a temporary local change, swap the ranks of `(lo,w)` and `(e,r)`, or delete
one prerequisite rule. Predict the new intermediate sequence before running
tests. Restore the table afterward.

Also try `tokenize '<|assistant|>'`. Prove that ordinary input does not produce
the assistant control ID `1006`.

## EXTEND

Add one merge chain for a short word not already present. Record every new ID,
the rank order, the hand result, and a round-trip test. Reject a duplicate rank
or undefined prerequisite with a typed construction error.

## Cleanup

Restore the committed merge table and ensure `cargo test --workspace` passes.

