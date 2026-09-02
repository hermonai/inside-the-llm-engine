# Lab 4 — Use the Wrong Chat Template

## Purpose

Show that a chat request is an exact serialization contract, not a bag of role
labels and strings.

## Prerequisites and artifact

Read Chapter 2's special-token and chat-template sections. Record the bytes and
IDs for the same messages under `TinyChatTemplate` and the deliberately wrong
`role: content` flattener.

## CHECK

For system `be exact` and user `lower`, predict the control-token order from
[`chat-template.txt`](../code/mini-engine/fixtures/tokenizer/chat-template.txt).
Identify which component is authorized to insert each control ID.

## BUILD

Run:

```sh
cd code/mini-engine
cargo test -p engine0 --test chat -- --nocapture
```

Write both ID sequences. Confirm that the correct version begins with BOS,
contains role and end-turn identities, and ends with ASSISTANT when
`add_generation_prompt=true`. Confirm that the naive version contains none of
those control identities.

## BREAK

Place the literal text `<|assistant|>` inside a user message. A broken design
will turn it into the assistant control ID. The correct design encodes every
byte as ordinary content. Then deliberately bind `ByteTokenizer` to the demo
model contract and confirm the identity check fails before request execution.

## EXTEND

Add one new typed role only after choosing a new explicit special identity and
updating the model contract. Explain why accepting an arbitrary role string
would be a different, larger policy surface.

## Limits and cleanup

ENGINE-0's model is fake, so this lab proves different bytes and IDs—not output
quality. Restore the demo contract and run all tests.

