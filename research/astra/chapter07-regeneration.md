# Chapter 7 visual regeneration

Started 2026-09-10 from book `463518dd12b05a0fe71af689a00181512a6ad0ff`.
The industrial architecture drafts, Chapter 1 formatting and Chapter 8 research
were already in the working tree. They are preserved, not claimed as this work.

## Storyboard and bounded scope

The reader follows identity → selected row → owned activation → square/reduce
→ reciprocal RMS → learned gain → owned output. Ten new canonical plates cover
the journey, logical/physical lookup, repeated tokens, ownership, two passes,
epsilon, finite range, analytical payloads, LayerNorm contrast and production.
One four-step sequence highlights the numerical stages without autoplay. Every
stage is also visible in the static plate. Existing fifteen Unicode diagrams,
the historical D=3 embedding fixture, hand RMSNorm values, scale/magnitude
evidence and ENGINE-1 logits remain intact. No QKV or attention implementation.

The new input-only fixture uses E [3,4] with row 1 equal to the existing
RMSNorm hand vector [1,-2,3,-4]. It connects previously separate lookup and
normalization explanations without replacing old regression fixtures. Python
independently computes the ideal equation and explicitly rounded F32 stages;
a Rust example calls the actual embedding and normalization operators. New
tests exercise this connection, copy isolation, zero input, non-unit output
RMS, and reduction overflow. No public numerical API changes.

## Source refresh

Inspected Hermon `2a3fd5214e17ca7283656108847d02dcd6cdf7c5` and its llama.cpp
submodule `389ff61d77b5c71cec0cf92fe4e5d01ace80b797` on 2026-09-10.

- CURRENT: `crates/hermon-runtime/src/dispatch.rs` defaults an unset mode to
  Batched. The default worker delegates through `Context::decode_batch`.
- PREVIEW: `crates/hermon-runtime/src/paged.rs` has explicit Rust `rms_norm`,
  epsilon validation during construction, and `tensor_row_f32` embedding reads.
- EXTERNAL: `src/llama-graph.cpp::build_norm` emits `ggml_rms_norm` with model
  epsilon; CPU `ggml/src/ggml-cpu/ops.cpp` forms F32 products before casting
  into `ggml_float`, defined as double in `vec.h`. This is not an assertion
  about all devices or storage types.
- [RMSNorm paper](https://arxiv.org/abs/1910.07467), reopened 2026-09-10:
  recentering is absent. Its reported speedups are not adopted as engine results.
- [PyTorch RMSNorm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html),
  accessed 2026-09-10: epsilon placement and per-coordinate affine gain are
  checked as API semantics, not copied as numerical implementation.

## Corrections and limitations

The hand-vector display had two literal `qquad` strings; restore TeX commands.
Distinguish signal scaling before normalization from learned gain afterward.
The latter may change direction and output RMS. Analytical 8D/16D byte counts
are payload models, not measured memory transfers. Typed finite-range failures
are detection, not a stabilized norm algorithm. Legacy source observations
remain historical; the visual edition refreshes their pins explicitly.

## Validation and review

2026-09-11: 178 unit/integration tests, two compile-fail doctests, Rust format,
workspace check and Clippy pass. All three chapter parity checks pass. The
analytical normalization curve is checked against the independent oracle.
All 45 semantic scenes regenerate exactly; industrial-plan and mutation gates pass.
Browser QA passes six step sequences, reduced motion, keyboard operation,
SVG label bounds and offline chapter layouts at 1024/768/390px.

The working PDF has 146 pages, Chapter 7 has 25 and the atlas 45 before the
separate native-TeX edition. Page-bound checks pass. Color plates and chapter
contact sheets were visually reviewed; reserve-space fixes keep the hand
calculation and reduction example together. Math, ownership, source status,
editorial continuity and accessibility were separate review lenses, not
independent agent reviews. No new accelerator or performance claim is introduced.

## Native LaTeX edition

The paired-book request adds a portable 150-page `.tex` edition, with all 45
semantic plate sources translated into native TikZ and 28 inline legacy diagrams
translated into vector strokes plus typeset labels. The source checker rejects
raster/image plates, machine paths and surviving character-box verbatim graphs.
Poppler confirms every extracted word is inside its page. Legacy migration
preserves topology; only the canonical chapter plates claim a full semantic
visual redesign. Rendered-page review is separate from source determinism.
