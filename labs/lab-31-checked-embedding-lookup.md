# Lab 31 — Implement Checked Embedding Lookup

**Chapter:** 7. **Level:** BUILD.

## Objective

Implement and audit the complete `[V,D]` plus `TokenId` to owned `[D]`
operator contract.

## Prerequisites

Lab 30, `TokenId`, `OwnedTensor`, and `TensorView`.

## Setup

Open `engine0::embedding::embedding_lookup_reference` and list every precondition
before reading an element: rank two, positive `V`, positive `D`, and `t < V`.

## Build

Trace the validation order, row conversion, checked `get2` calls, and final
canonical allocation. Explain why the function accepts valid strided views but
never repairs or mutates the input layout.

## Expected observation

First, middle, and last valid IDs select exact rows. An ID equal to `V` returns
`EmbeddingError::TokenOutOfRange`; it never becomes an out-of-bounds access.

## Explanation

The table's vocabulary dimension defines ID validity. The model dimension
defines output width. They are different axes with different jobs.

## Verification

Run all embedding tests:

```bash
cd code/mini-engine
cargo test --test transformer_primitives embedding
```

## Break / prove

Temporarily change the bounds check from `row >= vocab_size` to `row >
vocab_size`. Confirm the equal-to-`V` test catches the off-by-one error before
any unsafe behavior is possible.

## Extend

Trace `embedding_sequence_reference` and prove that repeated tokens duplicate
values in output without duplicating or mutating parameter storage.

## Cleanup

Restore the checked comparison and retain only passing fixtures.
