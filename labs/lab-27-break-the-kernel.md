# Lab 27 — Break the Kernel

**Chapter:** 6. **Level:** BREAK.

## Prerequisites

ENGINE-2's public error model and Labs 22–26.

## Build

Create a table mapping each malformed call to its expected `KernelError`:
wrong rank, wrong vector length, wrong matrix inner dimension, non-canonical
blocked input, zero block dimension, and overflowing output shape.

## Oracle

Run `cargo test -p engine0 --test linear`. Every malformed input must return the
specific typed error before allocation or hot-loop execution; no case may
panic, reshape, transpose, broadcast, or copy implicitly.

## Break / prove

Use `transpose()` to produce a valid but non-canonical view. Prove
`matmul_reference` computes it and `matmul_blocked` returns
`UnsupportedLayout`. Then call `to_contiguous` explicitly and show that the
blocked call becomes valid because the caller chose the copy.

## Extend

Add an error assertion for an empty shape whose prospective `[M,N]` output
overflows `usize`, without allocating a huge buffer.

## Cleanup

Keep failure tests deterministic and allocation-free.
