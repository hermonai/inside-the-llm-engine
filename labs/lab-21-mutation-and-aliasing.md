# Lab 21 — Mutation and Aliasing

**Chapter:** 5. **Level:** EXTEND.

## Visual contract checkpoint

Study the [canonical plate](../figures/generated/tensor-lifetime.svg) and its
[text equivalent](../figures/generated/tensor-lifetime.txt).
Read value 6 through a shared view, end its last use, write 60 through the exclusive view, and read 60 afterward. Run cargo test --doc from code/mini-engine to prove the two invalid borrow examples are rejected. OwnedTensor::clone copies payload; TensorView::clone copies metadata and a borrow.

From the repository root, run `python3 scripts/check-tensor-visual-parity.py`.
This supplements, rather than replaces, the original exercise and oracle.

## Reading prerequisite

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
