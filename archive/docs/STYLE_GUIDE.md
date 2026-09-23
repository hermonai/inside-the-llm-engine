# Style Guide

Write with technical seriousness, directness, curiosity, and respect for the
reader. Prefer concrete data, shapes, ownership, failure modes, and evidence to
metaphor. The engine is understandable machinery, never magic.

## Preferred movement

Open with a question or failure the reader can feel. Establish a small mental
model, then derive the math and implementation. Show what changes at scale and
what an optimization does not change. Use sentences such as:

- Here is the invariant.
- Here is why it exists.
- Here is the memory layout.
- Here is the failure mode.
- Here is how we test it.
- Here is what the optimization changes—and what it does not.

Avoid “simply,” “obviously,” “just,” “magic,” “basically,” and “everyone
knows” when they conceal work or prerequisites. Define jargon before relying on
it. Do not compress a central subject into its familiar slogan.

## Structure and voice

Use headings to expose reasoning, not to fragment prose. Prefer active voice
when ownership or responsibility matters. State uncertainty and status
explicitly. Historical failures deserve the same precision as successful
designs. Analogies to operating systems, databases, compilers, networking,
storage, and distributed systems must state both where the analogy works and
where it breaks.

## Examples and exercises

Name shapes, positions, dtypes, and expected results. Use CHECK, BUILD, BREAK,
and EXTEND to progress from explanation to engineering judgment. Do not force
identical chapter lengths or pad thin material.

## Terminology and typography

Follow `TERMINOLOGY.md`. Use “prefill” and “decode” for workload phases,
“KV cache” for stored key/value state, and “paged KV” for a block-mapped
layout—not as a synonym for any cache. Use code formatting for identifiers,
paths, commands, and literal configuration. Keep reusable diagrams monospaced.
Apply `DIAGRAM_STYLE.md` to visual artifacts and `MATH_STYLE.md` to equations;
style never overrides technical truth, dimensions, units, or source status.
