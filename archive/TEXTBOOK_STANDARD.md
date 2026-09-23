# Textbook authoring standard — shared series revision

This policy implements the author's September 2026 request for the three named
projects: Inside the LLM Engine, DNA Computing, and Evolutor. It does not alter
the separately maintained training book or imply that any outline is complete.

## Source of truth

Author new and revised book prose in chapter-level LaTeX. A build may extract
tested code, compute tables, compile TikZ, or convert editable SVG to vector PDF;
it must not regenerate the manuscript from Markdown. Research notes, project
status, web summaries and accessibility descriptions may remain Markdown/plain
text. Historical accepted editions and their source hashes remain intact.

Use an authored book document: Pagella body and mathematics, Heros headings,
11-point main text, numbered chapters/sections/equations/figures, useful cross
references, an index and a selective glossary. Keep editorial status outside
the explanation except where a scientific boundary matters.

## Exposition

Begin with a concrete problem and a small, fully specified object. Ask the
reader to predict its behavior. Derive the operation with intermediate steps,
then implement and test it. Introduce the next abstraction only when the
current example exposes a limitation. Finish with an engineering or scientific
consequence and a question the next chapter must answer.

Every important equation names its symbols, dimensions/units and assumptions.
Every implementation example identifies actual companion code. Commands and
expected outputs are checked on the current revision. An illustrative number
is not a measurement. A working component is not an integrated system.

Adopt the general build-from-scratch learning approach exemplified by Sebastian
Raschka's book, while writing original prose, examples and illustrations.
Do not imitate an author's distinctive wording or claim equivalent validation.

## Mechanism illustrations

Draw the objects that change: graph edges, molecular orientations, tensor cells,
coordinate planes, physical addresses, computation dependencies or state
transitions. Arrows have named semantics. Avoid replacing a paragraph with
decorative boxes. Use native TikZ for new mathematical/system plates; preserve
sound editable SVG scientific artwork as vector PDF where appropriate.
Do not print character-box diagrams, ASCII workflow graphs or screenshots of
equations. Captions explain the inference the figure supports and its limits.
Color is reinforced by labels, patterns, geometry and line styles.

Use short numbered traces or tables for small exact mappings; do not force every
relationship into a diagram. Render at final print size and inspect grayscale.
No figure-count target substitutes for explanatory value.

## Exercises and acceptance

Include prediction, derivation, implementation, counterexample and experiment
problems where meaningful. Supply worked answers for closed problems and
explicit scoring rubrics for open investigations. Keep tested code and manual
explanation aligned, including negative and degenerate cases.

Separate author technical review, editorial review, render review and independent
specialist review. Do not describe one agent's multiple passes as independent
peer review. Freeze source hashes for a reviewed release; publish a changed
candidate under a new name until its acceptance gates are satisfied.

A migration to LaTeX is not a complete editorial rewrite. Record those statuses
separately. Preserve earlier scientific/engineering evidence and failed results.
