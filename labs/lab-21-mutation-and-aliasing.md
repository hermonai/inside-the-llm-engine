# Lab 21 — Mutation and Aliasing

**Chapter:** 5. **Level:** EXTEND.

## Prerequisites

Rust shared/exclusive borrowing and Chapter 5 ownership diagrams.

## Build

Create one `OwnedTensor`, borrow `view_mut`, change an element with `get_mut`,
end the mutable borrow, and prove a later immutable view observes the change.

## Oracle

`cargo test --workspace tensor` must retain the mutation/aliasing test. The
crate remains under `#![forbid(unsafe_code)]`.

## Break / prove

In a disposable snippet, try to hold an immutable view while requesting
`view_mut`, or request two mutable views simultaneously. Record the compiler
rejection; do not put non-compiling code in normal tests.

## Extend

Design—but do not implement—the proof needed to split one canonical tensor into
two non-overlapping mutable regions.

## Cleanup

Delete the deliberately non-compiling snippet.
