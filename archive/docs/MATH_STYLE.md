# Mathematical Style

Mathematics in this book is rigorous, dimensional, and connected to memory and
code. An equation is a contract, derivation, or cost model—not decoration.

## Every important equation answers

- What does each symbol mean and who owns the represented data?
- What are the dimensions, dtype, layout, and units?
- What allocation and memory traffic does it imply?
- What arithmetic and asymptotic work does it imply?
- What changes between prefill and decode?
- How does the equation appear in the reference and optimized code?
- What numerical failure modes and tolerances apply?

Never present only an isolated formula such as an attention expression. First
declare shapes, ownership, visibility, and the storage/accumulation rules.

## Notation

- Define symbols at first use, include a local symbol table for a sustained
  derivation, and register cross-chapter equations in `MATH_INDEX.md`.
- Use italic lowercase for scalars (`x`, `t`, `i`), bold lowercase for vectors
  (`\mathbf{x}`), and bold uppercase for matrices/tensors (`\mathbf{W}`). Use
  `V_{\mathrm{vocab}}` for vocabulary size so later `\mathbf{V}` can denote
  attention values without collision.
- Use uppercase dimension symbols (`B`, `T`, `M`, `N`, `K`, `D`, `H_q`,
  `H_{kv}`, `D_h`) and state whether intervals are half-open.
- Declare shapes with membership, for example
  `\mathbf{W}\in\mathbb{R}^{M\times K}`, not only with prose or bracket
  shorthand. Bracket shapes remain useful in diagrams and code contracts.
- Put units on byte, bandwidth, time, and throughput formulas.
- Distinguish logical tokens from physical batch entries and logical blocks
  from physical block identifiers.
- State accumulation precision separately from storage dtype.
- Mark approximations (`\approx`), proportionality (`\propto`), and bounds
  (`\le`, `\ge`) honestly; never use equality for an estimate. A FLOP formula
  may use equality only after declaring the exact counting convention.

## Display, inline, and code

- Use inline math for a compact symbol or relation. Use Markdown display math
  (`$$ ... $$`) for central derivations, contracts, and reusable results.
- Align related steps with `\begin{aligned}` and add words with `\text{...}`.
- Keep programming syntax, array literals, and API signatures in code fences.
  Do not use a `text` fence as a substitute for typeset mathematics.
- Introduce an equation in prose and interpret it afterward; equations do not
  float without a claim.
- Number equations semantically in `MATH_INDEX.md` rather than adding fragile
  manuscript-wide numeric labels.

## Shapes, layouts, and ownership

Every operator contract states input/output shapes, the dimension being
reduced or transformed, and layout restrictions. When a logical tensor is a
view, separate its shape/stride metadata from the storage owner. State storage
dtype and accumulation dtype separately when they differ.

For example:

$$
\mathbf{C}=\mathbf{A}\mathbf{B},\qquad
\mathbf{A}\in\mathbb{R}^{M\times K},\quad
\mathbf{B}\in\mathbb{R}^{K\times N},\quad
\mathbf{C}\in\mathbb{R}^{M\times N}.
$$

Here `K` is the contracted dimension. The equation does not imply a physical
layout; the surrounding contract must state strides or contiguity.

## Numerical examples

Begin with shapes small enough to compute by hand. Provide expected values and
tolerances. Explain overflow, underflow, cancellation, rounding, reduction
order, and nondeterminism when relevant. An optimized result is compared with
an independently implemented oracle, not with itself under another flag.

Every hand calculation that serves as evidence has an executable mirror under
`code/reference/` or a focused Rust test. Record the tolerance and explain
why exact equality is or is not appropriate.

## Cost models

Separate FLOPs, bytes read/written, transfers, launch/synchronization overhead,
queue delay, and allocation. State when an asymptotic result hides a constant
that dominates real decode. Tie cost models to measurements without presenting
predictions as measured facts.

## Review checklist

- Symbols are defined once and used consistently.
- Scalars, vectors, matrices, and dimensions follow the convention above.
- Every shape composes and every unit cancels correctly.
- Equality, approximation, and bounds say what the evidence supports.
- Numerical examples match their oracle and code fixture.
- Storage bytes, accumulation precision, ownership, and layout are explicit
  where they affect the result.
- Cost models are labeled analytical; benchmark tables are labeled measured.
