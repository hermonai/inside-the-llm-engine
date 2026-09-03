# Lab 30 — Inspect an Embedding Row in Memory

**Chapter:** 7. **Level:** CHECK.

## Objective

Connect one logical coordinate `E[t,j]` to tensor metadata and a physical
element offset without bypassing `TensorView`.

## Prerequisites

Labs 16–19 and the Chapter 7 embedding-table contract.

## Setup

Use a canonical `[4,3]` table containing row-distinct values. Record shape
`[V,D]`, element strides `[D,1]`, dtype `f32`, and token `t=2`.

## Build

Calculate every offset `t*D+j` for `0 <= j < D`, then verify the values by
calling `TensorView::get2(t,j)`. Repeat with the strided-table fixture in
`tests/transformer_primitives.rs` and use
`base + t*stride[0] + j*stride[1]`.

## Expected observation

The canonical row occupies three adjacent elements. The strided logical row
has the same shape but skips physical filler values. Shape alone does not
determine an address.

## Explanation

An embedding is learned parameter storage. Lookup is checked indexing followed
by data movement; it is not dense matrix multiplication.

## Verification

Run:

```bash
cd code/mini-engine
cargo test --test transformer_primitives embedding_reads_a_strided_table_logically
```

The selected values must be `[3.0,4.0]`, not the adjacent physical filler.

## Break / prove

Replace the view-aware offset with `t*D+j`. Show that the canonical case still
passes while the strided case fails. Restore logical indexing.

## Extend

Draw byte offsets by multiplying element offsets by `size_of::<f32>()`.

## Cleanup

Do not commit the intentionally wrong offset formula.
