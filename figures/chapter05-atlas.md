# Chapter 5: one allocation, several interpretations

These eight canonical plates are embedded at their teaching points in
[Tensors Without Magic](../manuscript/part-02/chapter-05-tensors-without-magic.md).
The chapter is complete without animation. Each SVG has a Unicode text
equivalent; shape, units, ownership and failures are explicit rather than
encoded only in color.

| Read | Question | Vector / text | Source |
| --- | --- | --- | --- |
| 1 | Which byte represents A at row 1, column 2? | [Storage](generated/tensor.svg) / [TXT](generated/tensor.txt) | [Scene](src/tensor.json) |
| 2 | What owns the allocation, and what only borrows it? | [Rust UML](generated/tensor-ownership.svg) / [TXT](generated/tensor-ownership.txt) | [Scene](src/tensor-ownership.json) |
| 3 | What changes when we materialize a view? | [Copy](generated/tensor-copy.svg) / [TXT](generated/tensor-copy.txt) | [Scene](src/tensor-copy.json) |
| 4 | Why can equal shapes contain different values? | [Reshape](generated/tensor-reshape.svg) / [TXT](generated/tensor-reshape.txt) | [Scene](src/tensor-reshape.json) |
| 5 | How does transpose change the coordinate map? | [Transpose](generated/tensor-transpose.svg) / [TXT](generated/tensor-transpose.txt) | [Scene](src/tensor-transpose.json) |
| 6 | Why do four logical values need six backing slots? | [Slice](generated/tensor-slice.svg) / [TXT](generated/tensor-slice.txt) | [Scene](src/tensor-slice.json) |
| 7 | When can an exclusive mutation begin? | [Lifetime](generated/tensor-lifetime.svg) / [TXT](generated/tensor-lifetime.txt) | [Scene](src/tensor-lifetime.json) |
| 8 | Which units and owners cross the production boundary? | [Production](generated/tensor-production.svg) / [TXT](generated/tensor-production.txt) | [Scene](src/tensor-production.json) |

## Reproduce the contract

From the repository root:

```sh
python3 scripts/check-tensor-visual-parity.py
python3 figures/build.py --check
cargo test --manifest-path code/mini-engine/Cargo.toml --workspace
```

The [Rust trace](../code/mini-engine/crates/engine0/examples/chapter05_visual_trace.rs)
runs the real tensor API. The [independent checker](../scripts/check-tensor-visual-parity.py)
compares it with the [shared fixture](../code/reference/fixtures/chapter05-visual.json).
[Contract tests](../code/mini-engine/crates/engine0/tests/tensor_visual_contract.rs)
check pointer identity, rejected layout, backing extent and independent mutation.
The implementation's two compile-fail doctests check invalid borrowing.

The [review record](../research/astra/chapter05-regeneration.md) documents fresh
production evidence, parity gates and publication limitations. Build the ebook,
standalone chapter and atlas with the [publication instructions](../docs/FIGURE_BUILD.md).
