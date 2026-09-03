# Lab 32 — Embedding View Versus Copy

**Chapter:** 7. **Level:** BREAK.

## Objective

Make the ownership cost and safety difference between a borrowed parameter row
and an owned residual activation observable.

## Prerequisites

Labs 19, 21, and 31.

## Setup

Use a `[2,2]` `OwnedTensor` as immutable model parameters. Select token one
with `embedding_lookup_reference`.

## Build

Mutate element zero through the returned activation's exclusive `view_mut`.
Then inspect the original table. Record both values and the output's canonical
shape/strides.

## Expected observation

The activation changes while the table remains byte-for-byte unchanged. The
lookup paid an `O(D)` copy to separate request-local mutable state from
long-lived immutable parameters.

## Explanation

A view can avoid the copy but carries a lifetime tied to weight storage and
aliases it. The teaching engine selects owned output because later residual
state needs independent mutation and lifetime.

## Verification

```bash
cd code/mini-engine
cargo test --test transformer_primitives embedding_result_owns_storage_independent_of_parameters
```

## Break / prove

Sketch a borrowed-row signature returning `TensorView<'weights>`. Ask the Rust
compiler to hold it after dropping the table, and preserve the compiler error
in your notes. Do not weaken lifetimes or add unsafe code.

## Extend

Estimate payload bytes for `D=4096` with `f32`: one row read plus one output
write under a cold-copy model. State that cache effects are not measured.

## Cleanup

Remove disposable compiler-failure snippets.
