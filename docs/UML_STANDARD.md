# UML standard

Use recognized UML where software relationships are primary. v1 selects
structured JSON scene metadata plus version-controlled SVG geometry, not a
new PlantUML dependency. Runtime-sequence source is `figures/src/sequence.json`
and the `sequence` renderer in `figures/build.py`; both are required source.

Sequence diagrams use named participants, dashed lifelines, solid message
arrows and dashed returns. The current prototype traces submit, admission,
decode, logits/status, and stream. Its caption scopes it to a successful
iteration and records failure/cancellation behavior. Detailed `alt` branches
belong in the Chapter 1/78 regeneration; do not imply this is an exhaustive
protocol trace.

For class diagrams, `«struct»` names a real Rust struct. A stored owned field
is composition (filled diamond, cardinality); a borrowed slice is a labeled
dependency with a lifetime, not ownership or inheritance. Trait realization
requires an actual `impl`. No inheritance relationship is invented for Rust.
The tensor prototype shows a borrow dependency; the chapter pilot will add
the full composition plate for Vec storage and mutable/exclusive views.
