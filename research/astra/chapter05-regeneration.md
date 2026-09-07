# Chapter 5 regeneration: one allocation, several interpretations

Status: COMPLETE. Started 2026-09-06 at book commit `cc18b1b`; completed 2026-09-07.
Scope: the Chapter 5 pilot in the regeneration plan. Existing APIs, Chapter 3
fixture, stress cases, 13 legacy diagrams and Labs 16–21 remain available.

## Before / after design

The original chapter has strong checked-layout and ownership explanations,
but switches between letter values, 3-D examples and a twelve-value slice.
The regeneration carries `A = [[1,2,3],[4,5,6]]` through eight sequential plates:
orientation/physical storage, transpose, materialization, reshape comparison,
column slice/extent, actual ownership UML, borrow lifetime, and production
representation boundary. Existing higher-rank and twelve-value examples remain
as generalization and stress exercises, not the reader's first visual objects.

Each plate has one source specification and exact fixture values. A Rust
trace executes the real tensor API; an independent Python enumerator checks
the emitted metadata, logical values and physical offsets against the shared
visual fixture. The figure builder validates that same fixture before drawing.
This establishes an equation/code/oracle/diagram chain rather than a second
illustration-only implementation.

## Fresh source review

Hermon HEAD and fetched origin/main on 2026-09-06:
`5e908d75e4d5ee102f97f6a880ed4d0968d949df`.
Pinned llama.cpp/GGML remains `389ff61d77b5c71cec0cf92fe4e5d01ace80b797`.
Inspected `hermon-llamacpp/src/linked.rs`: `tensor_info`, `tensor_row_f32`,
`TensorSession` and its Send/exclusive mutable pointer contract;
`hermon-runtime/src/dispatch.rs` default batched selection and paged gate;
`hermon-kernels/src/lib.rs` actual unsafe FFI calls; pinned `ggml.h` fields.

The chapter's “single allowed unsafe crate” claim is stale: the native kernel
crate also contains an FFI boundary. Replace it with the specific wrapper
ownership contract. `TensorSession` is Send, deliberately not Sync; its access
methods require exclusive mutation. API availability is LIBRARY evidence;
use on a gated paged route is PREVIEW, not a default hot-path claim.

Two educational source details also need precision. Derived `OwnedTensor::clone`
copies its Vec payload; `TensorView::clone` only clones metadata and a borrow.
`from_vec` validates count, vector length and canonical strides; it does not
call `checked_byte_count`. Byte accounting is a separate checked helper.

## Storyboard / parity gates

| Plate | Question | Executable evidence |
| --- | --- | --- |
| Storage | Which byte represents A[1,2]? | element 5, byte 20 |
| Transpose | Which values change logical coordinates? | shape [3,2], strides [1,3], offsets [0,3,1,4,2,5] |
| Materialize | Where does a new owner appear? | copied storage [1,4,2,5,3,6], source remains [1,2,3,4,5,6] |
| Reshape | Why is equal shape insufficient? | reshape [3,2]/[2,1] differs from transpose at [0,1] |
| Slice | Why do four logical values need six backing slots? | base 1, shape [2,2], strides [3,1], offsets [1,2,4,5] |
| Ownership UML | Which fields own, which borrow? | actual OwnedTensor, TensorView and TensorViewMut declarations |
| Lifetime | When may mutation begin? | shared borrow's last use precedes exclusive view; cloned owner is independent |
| Production | Which representation crosses the ABI? | GGML byte strides and packed type; wrapper-owned row conversion |

Completion requires reviewed prose and captions, exact numerical parity, fresh
source status, linked labs, deterministic builds, full regressions and a rendered
chapter PDF with the figures embedded. Review results will be appended below.

## Evidence links and implementation boundary

- [Hermon linked wrapper](https://github.com/hermonai/hermon/blob/5e908d75e4d5ee102f97f6a880ed4d0968d949df/crates/hermon-llamacpp/src/linked.rs): metadata, owned row conversion and exclusive session access.
- [Dispatch](https://github.com/hermonai/hermon/blob/5e908d75e4d5ee102f97f6a880ed4d0968d949df/crates/hermon-runtime/src/dispatch.rs): default batched route versus gated paged route.
- [Native kernels](https://github.com/hermonai/hermon/blob/5e908d75e4d5ee102f97f6a880ed4d0968d949df/crates/hermon-kernels/src/lib.rs): another actual unsafe FFI boundary.
- [Pinned GGML header](https://github.com/ggml-org/llama.cpp/blob/389ff61d77b5c71cec0cf92fe4e5d01ace80b797/ggml/include/ggml.h): typed storage, dimensions, byte strides and source/view metadata.

No production code or mini-engine implementation was changed. The Rust source
changes are documentation plus tests and an executable example. No API, model
fixture, operator, numerical precision or dependency was changed.

## Five parity reviews

| Gate | Result and evidence |
| --- | --- |
| Prose / mathematics | PASS: one numeric matrix, coordinate identity, offset/extent units and 48-byte analytical copy accounting. Existing zero-size and higher-rank cases retained. |
| Mathematics / code | PASS: the trace invokes the real checked API; from_vec byte-helper claim corrected; clone payload and coordinate-allocation costs explicit. |
| Code / oracle | PASS: independent Python row/column enumeration matches every emitted shape, stride, base, offset and value in five states. Four ownership/extent/mutation tests and two compile-fail doctests add non-numerical proofs. |
| Figures / semantics | PASS: eight canonical SVG/TXT pairs, a shared fixture for seven new plates, explicit hero-fixture parity, actual Rust fields, and non-color ownership labels. No diagram asserts an unimplemented tensor operation. |
| Production / source | PASS: dated pin above; LIBRARY API availability is distinguished from CURRENT default execution and PREVIEW paged integration. No new hardware or real-model result is claimed. |

## Six reader-perspective reviews

These are editorial self-reviews against six reading needs, not independent
human reviews or user-study results.

- Beginner: the same six numbers recur; coordinate, offset, byte displacement
  and address are distinguished. The original higher-rank exercise comes later.
- Software engineer: UML uses the real structs and fields; no inherited
  ownership or arbitrary mutable-stride constructor is invented. Clone costs
  and last-use borrowing are explicit and tested.
- ML engineer: transpose and reshape have the same output shape but different
  values. The alias/copy distinction is separate from the mathematical result.
- Systems engineer: logical count differs from backing extent; copy accounting
  excludes cache-line/allocator effects. Historical traversal measurements keep
  their original provenance, rather than becoming fresh benchmark claims.
- Graduate reader: the coordinate equation is connected to a finite trace,
  type/extent rejection and a pointer-identity test. The general zero-size,
  overflow and strided cases are not reduced to the six-value illustration.
- Researcher: pinned primary implementation evidence and scope limits are
  visible. GGML byte/block-aware strides are not equated with F32 strides, and
  an FFI API is not presented as evidence of default end-to-end execution.

## Visual and publication review

The manuscript grew from 6,083 to 7,468 whitespace-delimited words. It retains
all thirteen legacy diagrams and Labs 16-21, which now link the exact vector
and text checkpoints. Eight canonical plates are embedded at their teaching
points; the atlas now contains 17 total plates, including nine remaining
prototypes. All 38 generated SVG/TXT/animation artifacts are deterministic.

The PDF-skill render/inspect loop caught and corrected delayed floating figures
and a split synthesis table. Chapter headings now start new print pages.
The narrow-browser gate caught page-level overflow from equations/tables;
wide contracts now scroll within their own region in offline HTML. All eight
figures are embedded SVGs in HTML and remain vector objects in print.
MathML works without a remote renderer. Browser checks pass at 1024, 768 and
390 pixels; existing animation keyboard/reduced-motion checks still pass.

Remaining limits: PDF is not tagged PDF/UA; HTML and TXT are the accessible
reading alternatives, not a claim of certified accessibility. Small diagrams
benefit from zoom on phones. No animation was necessary for this static tensor
contract. Publication dependencies are pinned; byte-identical PDF output across
TeX versions is not promised. Later chapters have not passed this pilot's
regeneration gates, and this pass does not complete Chapter 8.

## Final validation, 2026-09-07

- PASS: Rust format, check all targets, Clippy with warnings denied, 167
  unit/integration tests and two compile-fail doctests.
- PASS: all five historical Python oracles and the new five-state visual parity
  checker; historical CLI still generates Rust then EOS.
- PASS: repository structure, relative links/images, 78 legacy diagram
  inventory/width checks, 115 display equations and 31 real-valued shape
  declarations across seven chapters, deterministic 17-scene/38-artifact build.
- PASS: full ebook 123 pages, Chapter 5 23 pages, vector atlas 17 pages.
  All chapter pages were rendered and reviewed in contact sheets; figures and
  ownership labels were additionally inspected at larger size and in grayscale.
  The chapter contains all eight figure IDs exactly once, no rasterized figures
  and no PDF replacement character. The build reported no missing glyphs.
- PASS: offline HTML embeds eight SVGs and native MathML; browser checks cover
  1024/768/390px and the existing keyboard/playback/reduced-motion sequences.
- PASS: credential guard and whitespace diff check. Pre-existing Chapter 1,
  runtime diagram and Chapter 8 research/status changes remain outside this commit.

The bundled dependency refresh removed the optional SVG converter during the
continuation; rebuilding from the pinned requirements in an isolated ignored
venv restored the build. No machine-specific dependency paths enter repository
sources. Next: Chapter 6 visual regeneration, not another tensor API expansion.
