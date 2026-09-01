# Mathematical Style

Mathematics in this book is rigorous, dimensional, and connected to memory and
code.

## Every important equation answers

- What does each symbol mean and who owns the represented data?
- What are the dimensions, dtype, layout, and units?
- What allocation and memory traffic does it imply?
- What arithmetic and asymptotic work does it imply?
- What changes between prefill and decode?
- How does the equation appear in the reference and optimized code?
- What numerical failure modes and tolerances apply?

Never present only
`Attention(Q,K,V) = softmax(QK^T / sqrt(d_head))V`. Establish, for example:

```text
Q: [n_query_tokens, n_query_heads, head_dim]
K: [visible_tokens, n_kv_heads, head_dim]
V: [visible_tokens, n_kv_heads, head_dim]
```

Then identify which query owns an output row, how GQA/MQA map query heads to KV
heads, which prior positions are causally visible, how scores are stabilized,
and how dense versus paged layouts alter access without altering semantics.

## Notation

- Define symbols at first use and maintain `GLOSSARY.md`/Appendix K.
- Put units on byte, bandwidth, time, and throughput formulas.
- Distinguish logical tokens from physical batch entries and logical blocks
  from physical block identifiers.
- State accumulation precision separately from storage dtype.
- Mark approximations and bounds; never use an equality for an estimate.

## Numerical examples

Begin with shapes small enough to compute by hand. Provide expected values and
tolerances. Explain overflow, underflow, cancellation, rounding, reduction
order, and nondeterminism when relevant. An optimized result is compared with
an independently implemented oracle, not with itself under another flag.

## Cost models

Separate FLOPs, bytes read/written, transfers, launch/synchronization overhead,
queue delay, and allocation. State when an asymptotic result hides a constant
that dominates real decode. Tie cost models to measurements without presenting
predictions as measured facts.
