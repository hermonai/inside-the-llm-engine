# First visual milestone: validation and review

Validation completed 2026-09-06; source inspection began 2026-09-05.
Baseline `8588ff839174c31754385403765c6eb2a127ab43`, branch
`astra-visual-rewrite`. This is a bounded prototype/publication milestone.

## Executable gates

| Check | Observed result |
| --- | --- |
| Rust fmt / check / tests / Clippy | PASS; 163 tests, no failures; no engine implementation changes |
| Python oracles | PASS; all five Chapters 3–7 oracles |
| Runnable examples | PASS; all four release examples execute; their timings are smoke evidence, not new published performance claims |
| Chapter 3 regression | PASS; hidden `[1,-0.5,2]`, logits `[-0.7,0.1,0.40000004,2.2]`, `Rust`, then EOS |
| Repository structure | PASS; 15 parts, 94 specifications, seven chapter artifact sets |
| Legacy diagrams | PASS; 78 inventoried; maximum 95 display columns |
| Mathematical structure | PASS; all seven chapters, 112 display blocks, 29 explicit real-valued shape declarations |
| Figure build and stale check | PASS; ten scenes, 24 deterministic SVG/TXT/HTML/player artifacts |
| Scene numerics | PASS; tensor offsets, GEMV, head count, rotation norm, masked attention probabilities/output, cache byte growth, final batch membership |
| Browser controls | PASS; three pages, frame visibility, Next/Previous, range keyboard input, timed play/end stop, reduced motion, no script errors |
| Viewports | PASS; no horizontal document overflow at 1024, 768 and 390 pixels |
| PDF | PASS; 113-page current manuscript and ten-page vector atlas; no missing-character warnings after fallback corrections |
| HTML | PASS; offline book with native MathML; atlas with embedded SVG and locally packaged players |
| Git whitespace / credential guard | PASS; no embedded remote credentials or whitespace errors |

Relative links, including image links, are checked by the expanded link gate.
CI now checks deterministic figures as well as the existing Rust/book gates.
PDF requires the documented TeX/Python dependencies; it is validated locally,
not newly asserted as a hosted site or an automatically published GitHub asset.

## Visual review and corrections

Reviewed all ten atlas pages as a rendered contact sheet, a grayscale RoPE
plate, the RMS/mean-square equations on manuscript pages 103–104, and the
Unicode/emoji lesson on page 22. Reviewed animation screenshots in a browser.
The first rendering revealed SVG font fallback to Helvetica, which lost several
mathematical subscripts/superscripts. PDF conversion now explicitly assigns
DejaVu Sans to every label. A tensor arrow now lands on element 5 rather than
an adjacent cell. Typography, borders and labels remain distinguishable in
grayscale. Static plates contain all states needed to understand the animations.

The full PDF build found a Chapter 7 form-feed in `\frac` and six missing
`\mathbf` backslashes. Those seven mathematical presentation defects are
corrected. Math discovery now covers every written chapter and rejects control
characters. Chinese uses Fandol; composite emoji use credited Twemoji vectors;
the source and HTML retain original Unicode. These publication transformations
are explicit and do not alter numerical semantics or tokenizer fixtures.

## Limits and next acceptance gates

The PDF is a working proof, not a release-certified typesetting pass across
every paragraph. It has selectable text but is not tagged PDF/UA. Browser SVGs
have title/description and captions; captions and controls reflow on phones,
but dense multi-panel labels benefit from zoom. Testing overflow is not a
claim that a ten-panel textbook works optimally at phone width.

UML prototypes show a scoped successful runtime iteration and a real borrow
relationship; the full ownership/composition and failure `alt` plates are
Chapter 5/1 pilot work. Future RoPE/attention/cache figures are educational
specifications, not mini-engine implementations or claims of production layout.
No new accelerator or real-model equivalence tests were run. The audit's local
equation flags are inventory assistance, not automatic mathematical proofs.

Next: Chapter 5 visual regeneration, with a single matrix carried through
logical layout, offsets, transpose, materialization and ownership UML, then
all five parity tests. Preserve historical and in-progress Chapter 8 work.
