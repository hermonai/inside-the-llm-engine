# Chapter 9 research - Position: RoPE From First Principles

Status: COMPLETE. Inspected 2026-09-12. Book baseline `469e5dc`.
Hermon `2a3fd5214e17ca7283656108847d02dcd6cdf7c5`; pinned llama.cpp
`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.

## Question and bounded deliverable

How does rotating coordinate pairs introduce a relative-position dependence
without adding a positional vector? Prerequisites: Chapter 8 raw Q/K heads,
Chapter 5 ownership, scalar dot products. Deliver positioned Q/K, not scores,
masks, attention probabilities, value aggregation, cache or sequence runtime.

Selected API: RopeConfig validates head width, positive even rotary prefix R
no larger than head width, positive finite base theta, and explicit adjacent
or split-half pairing. Frequency omega_i=theta^(-2i/R), i=0..R/2. Signed i64
positions are diagnostic-friendly; reject magnitude greater than 2^53 before
F64 conversion. This is an integer-conversion boundary, not a quality/context
length guarantee. Frequencies, angles and pair arithmetic use F64, storage F32.

Expose immutable-view-to-fresh-owner and exclusive-owner in-place APIs. Both
validate rank [H,Dh], finite input including tail, representable frequencies,
and finite F32 outputs. In-place preflights every pair before the first write;
typed errors leave the whole owner unchanged. Tail coordinates remain exact.
Input striding is accepted by the fresh-output reference. No unsafe code.
Position zero preserves finite input bits, including signed zero.

## Primary sources and classifications

- EXTERNAL: Su et al., [RoFormer](https://arxiv.org/html/2104.09864v5),
  rotation and relative inner-product identity. Derive the elementary algebra
  independently in the chapter. Do not turn a long-distance envelope into a
  claim that each query/key similarity decreases monotonically.
- EXTERNAL: [Meta Llama model source](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/model.py),
  inspected 2026-09-12: adjacent real/imaginary pairs, complex phase multiply,
  Q/K rotation, no V rotation in that attention implementation. Its frequency
  scaling function is piecewise in wavelength, not merely a new base.
- EXTERNAL: [Position Interpolation](https://arxiv.org/abs/2306.15595) and
  [YaRN](https://arxiv.org/abs/2309.00071): context-extension methods involve
  modified positional geometry and empirical validation. Explain boundaries;
  do not implement them or inherit their reported quality/performance claims.
- CURRENT: Hermon `crates/hermon-runtime/src/dispatch.rs` defaults to Batched.
  `vendor/llama.cpp/src/models/llama.cpp` calls build_qkv then ggml_rope_ext
  for Q and K, with position input and model/runtime parameters; V is separate.
- PREVIEW: `crates/hermon-runtime/src/paged.rs::rope_inplace`, lines 578ff,
  uses adjacent pairs, F32 position/angle arithmetic, per-pair base frequency
  and optional rope_freqs.weight divisors. Model construction validates positive
  finite base, even head width and factor length/values. forward_layer rotates
  Q/K before later KV writes. The helper itself uses debug assertions; it is
  not the same error contract as our public checked teaching API.
- LIBRARY/EXTERNAL kernel: pinned GGML CPU ops.cpp supports NORMAL adjacent
  and NEOX split-half modes, caches phase terms across heads of a token, and
  preserves non-rotary trailing channels. Its extended scaling path has more
  parameters and precision differences than the teaching standard RoPE.
- Canonical docs: Hermon docs/CORE_ENGINE_ARCHITECTURE.md section 6 confirms
  normalize, QKV, RoPE(Q,K), then cache/attention. Source gates outrank diagrams.
- Tests inspected: Hermon paged.rs component/forward tests and
  tests/backend_equivalence.rs include position/block-boundary contexts. No
  real-model or production equivalence run is claimed in this milestone.

## Proof and experiment plan

Hand rotate [1,0] at positions 0 and 1 (one radian, not one degree), and use
R=4/theta=100 to demonstrate frequencies 1 and 0.1. Adjacent and split-half
are related by an explicit permutation; norm tests alone cannot detect a
wrong convention. Test zero, inverse, composition, same-position dot product,
relative offset n-m, common position shifts and intentional sign errors.

Test all rejected dimensions, bases, rank/width mismatch, empty heads,
non-finite inputs, output overflow, extreme positions, strided/zero-stride
views, F32 cast collision at 2^24, partial tails, late-pair failure atomicity,
and out-of-place/in-place parity. Use an independent Python complex oracle,
shared input-only fixtures and full QKV-to-positioned-QK composition with V
unchanged. Report fixture tolerances, not a universal large-position bound.

## Visual storyboard

Ten plates: raw-to-positioned boundary; actual plane rotation; two frequency
clocks; relative angle/inner product; adjacent versus split-half memory map;
partial prefix and preserved tail; validate/preflight/commit ownership;
logical positions versus batch/storage locations; standard/interpolated/base
change curves (analytical); source/default/preview execution synthesis.
Use native TikZ geometry via the deterministic SVG-to-TikZ pipeline. No text
graphs. Static print carries the whole explanation. No timing benchmark is
justified before a table/compute candidate exists.

## Deferred questions

Model-specific scaling, amplitude factors, multi-axis/vision RoPE, tables and
recurrence optimizations, fused kernels, position-aware KV compatibility and
long-context model evaluation remain explicitly unimplemented.

## Exact source trail

The relevant Hermon paths had no local modifications at inspection. The
submodule revision was checked directly, not inferred from an older note.

- [Default dispatch](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/crates/hermon-runtime/src/dispatch.rs):
  unset mode selects Batched; explicit paged mode remains preview-gated.
- [Paged helper and call sites](https://github.com/hermonai/hermon/blob/2a3fd5214e17ca7283656108847d02dcd6cdf7c5/crates/hermon-runtime/src/paged.rs):
  helper near 578, model validation near 929, Q/K rotation near 1220.
- Pinned submodule `src/models/llama.cpp`, Q/K `ggml_rope_ext` calls near
  126-159; `ggml/src/ggml-cpu/ops.cpp`, phase preparation and NORMAL/NEOX
  coordinate modes near 5835-5930. These are library paths under the recorded
  submodule revision, not descriptions of the teaching API.
- Hermon `crates/hermon-runtime/tests/backend_equivalence.rs`, especially
  `a_scrambled_block_table_matches_the_oracle` and incremental decode tests:
  inspected source, not executed here. Their scrambled placement deliberately
  separates logical order from physical block IDs. `paged.rs` repeat-prefix
  tests cover warm/cold synthetic behavior but do not by themselves prove our
  RoPE component equivalent to production.

## Technical review - 2026-09-12

This is one author's structured review, not an external peer-review claim.

| Reader lens | Question checked | Evidence / limitation |
|---|---|---|
| Beginner | Why these signs, and why radians? | Basis-vector construction, [1,2] hand calculation and coordinate-plane plate. |
| Software engineer | Can a late error corrupt earlier heads? | Full-owner bit snapshot after late output overflow; source preflight precedes every write. |
| ML engineer | Does checkpoint pairing match activation order? | Adjacent/split-half coordinate oracle and explicit permutation-conjugacy test; V unchanged in full composition. |
| Systems engineer | What is allocated and what is a position? | Payload pointer retention, O(R) coefficient scratch, strided/broadcast reference inputs, logical versus physical position plate. |
| Graduate reader | What exactly is proved? | Key-minus-query sign, group law, same-position score, common shift and phase-error chord bound; fixed-frequency premises stated. |
| Researcher | What is not certified by the tests? | F64 conversion range is not phase accuracy or context quality; base and position scaling differ; no real-model, optimized-kernel or benchmark claim. |

Validation: 25 new Rust tests, 218 unit/integration tests in the full workspace
plus two compile-fail doctests; formatting and warning-denying Clippy passed.
All five chapter parity gates passed. Maximum absolute errors against the
independent complex/real-arithmetic oracle:

| Checked output | Maximum absolute error |
|---|---:|
| Adjacent coordinates | 2.111e-7 |
| Split-half coordinates | 2.056e-7 |
| Rotated-vector dot | 9.314e-7 |
| Relative-rotation dot | 4.069e-7 |
| Composed positioned Q | 3.044e-7 |
| Composed positioned K | 1.688e-7 |
| Composed unchanged V | 1.177e-7 |

Acceptance is 1e-5 + 1e-5 times absolute oracle value, not a universal bound.
Both truncated and modified candidates are rejected. The phase-overflow test
was corrected during development: width 8 with the smallest positive base
does not overflow F64 at our position limit; width 128 does. This was a test
assumption defect, not evidence that the validation branch was broken.

## Editorial and visual review - 2026-09-12

Separate pass after numerical validation: read all 19 standalone pages and all
ten plates in color and grayscale. Pairing brackets, frequency-clock angles,
partial-tail flow, logical-position table and scaling slopes remain readable
without color. Browser checks passed at 1024/768/390 pixels with embedded SVG
and offline MathML, plus SVG canvas and panel-border bounds. Existing six
interactive sequences retain keyboard and reduced-motion behavior.

The first native build exposed a long inline four-coordinate vector extending
into the margin. It was changed to display math. The final native build has
187 pages with 65 semantic and 28 legacy TikZ sources, no raster/verbatim
graphs, no overfull boxes and no missing glyph warnings. Reviewed the entire
new native chapter, including high-resolution geometry/derivation and corrected
vector pages, plus the updated cover and Chapter 8 transition. The standalone
chapter remains 19 pages; conventional full PDF is 182 pages and atlas 65.
All extracted words in all eight publication PDFs remain inside page bounds.
Source comparisons remain explicitly CURRENT,
PREVIEW and LIBRARY; future attention and cache mechanisms are not marked
implemented. Labs 49-58 have inputs, expected artifacts, deliberate breaks
and cleanup. Future lab references were reconciled with the canonical index.
