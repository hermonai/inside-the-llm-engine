# Visual language v1

The active textbook is now authored in `tex/`; see `TEXTBOOK_STANDARD.md` and
`tex/README.md`. Native TikZ figures in that directory are directly editable.
The scene generators below continue to own the historical atlas and its
regression fixtures, not the active manuscript or its new figures.

Publication requirement, 2026-09-11: the book ships a native `.tex` edition.
New and regenerated chapter illustrations must have professional TikZ sources,
with geometric edges and typeset labels, not verbatim character-box graphs.
`publication/build-tex.py` preserves semantic SVG geometry as native TikZ;
`scripts/check-native-figures.py` enforces parity. Legacy Unicode sources remain
internal references and migration inputs, not the preferred print representation.

The system adds publication vectors and optional motion to the existing
[Unicode grammar](../docs/DIAGRAM_STYLE.md). That grammar remains authoritative
for legacy diagrams. Do not redefine an old arrow in a graphical counterpart.

## Objects and edges

| Entity | Redundant encoding | Color |
| --- | --- | --- |
| Parameter / persistent storage | rectangle, explicit owner and immutable label | pale ochre `#f4ead2` |
| Activation / result | indexed cells, shape and dtype | pale green `#dcefe7` |
| Operator / component | named rectangle with typed inputs and outputs | pale blue `#dcecf7` |
| Control / optional path | dashed, labeled event or status | ink `#152b3c` |
| Tensor movement | solid labeled arrow into an operator or storage | ink |
| Borrow | dashed dependency labeled “borrows”; never implies copy | ink |
| Failure | explicit error or terminal label | ink; optional red accent later |

For UML, standard solid calls/dashed returns supersede the local flow convention
and are explained in the caption. Arrows must name a purpose. A and B enter
multiply independently; never draw A becoming B merely to save space.

## Geometry and typography

Use a 1000×720 viewBox, 40-unit margins, 30-unit titles, 19-unit normal labels,
15-unit contract notes. DejaVu Sans supplies deterministic Unicode glyphs in
PDF; browsers may use a sans-serif fallback. Six panels maximum per plate.
At 180 mm print width the 19-unit labels are approximately 9.7 pt. Keep essential
information in normal labels and captions; contract notes supplement them.
No rasterized labels, decorative gradients or generated-image text.

The persistent zoom hierarchy is product → engine → model → block → operator →
kernel → memory. Chapter storyboards select a zoom explicitly. A complete
architecture plate synthesizes already-taught boundaries; it is not the first
figure shown to a beginner.

## Canonical source and three tiers

Existing `diagrams/*.txt` sources remain canonical for their historical IDs.
New prototypes use `figures/src/*.json` plus the deterministic geometry in
`figures/build.py` as their semantic and visual source. JSON fixtures, contracts,
evidence and frame lists produce TXT, SVG and optional HTML. Generated outputs
are checked byte-for-byte. Do not edit them manually. Each important future
chapter equation receives a stable figure ID and a direct implementation link.

`EDUCATIONAL MODEL` is printed on conceptual prototypes; `CURRENT`, `PREVIEW`,
and `LIBRARY` qualify production paths. An illustration of an unwritten chapter
cannot be used as evidence that mini-engine implements that chapter.

## Accessibility and print

Color is redundant with labels, cells and line styles. Dark ink on all pale
fills maintains high contrast. SVG includes title/description; HTML includes
captions, keyboard controls, reduced-motion behavior and a complete static link.
Animations never autoplay. Test color and grayscale PDF pages and a tablet-width
HTML view. A dense plate can zoom, but prose and controls must reflow. See
[the visual review](../research/astra/validation.md) for tested limits.
