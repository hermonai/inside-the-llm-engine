# The book source

This directory is the manuscript, authored directly in LaTeX. Nothing here is
generated. The rules are in [`../AUTHORING.md`](../AUTHORING.md); the chapter
plan is [`../docs/STRUCTURE.md`](../docs/STRUCTURE.md).

| Path | Contents |
| --- | --- |
| `inside-the-llm-engine.tex` | Main file: title page, nine parts, appendices, back matter |
| `preamble.tex` | Fonts, colours, anatomy macros, ledger table, draft markers |
| `frontmatter/preface.tex` | How to read this book |
| `chapters/chNN-<slug>.tex` | The 42 chapters; first line `% status: …` |
| `appendices/` | `a0` Appendix A opener; `a1`–`a8` the first edition's Chapters 2–9 (Appendix A.1–A.8); `b`–`g` the reference appendices |
| `figures/<name>.tex` | Native TikZ (and pgfplots) figures, placed with `\EngineFigure` |
| `worked/<stem>.tex` | Worked problems for the chapter or appendix of the same stem |
| `references.bib` | Primary sources; every entry has a row in `research/FRONTIER.md` |
| `backmatter/glossary.tex` | Working vocabulary and evidence notes |

## Chapter anatomy macros

Each chapter uses these once, in order (`AUTHORING.md` §3); plain `\section`s
between the napkin math and the ledger are the mechanism.

```latex
\begin{ChapterIntent} … \end{ChapterIntent}
\OpeningSection{…}   \NapkinSection{…}   \section{…} …
\LedgerSection       \begin{ConstraintLedger} … \end{ConstraintLedger}
\LabSection{…}       \HappenedSection{…}   \FrontierSection{YYYY-MM-DD}
\ExercisesSection    % or \input{worked/<stem>.tex}
```

Draft markers — `\todo{…}`, `\FigureIntent{…}`, `\LabIntent{…}` — are allowed
in SKELETON and ZERO chapters and rejected in FULL ones. Cross-reference with
`\Chref{<slug>}`, `\Appref{<name>}` and `Part~\ref{part:<name>}`; never type a
chapter number.

## Build

```sh
make textbook-check   # structure, anatomy, status table, figures, CLI trace
make status           # regenerate the chapter table in docs/STATUS.md
make textbook         # PDF in output/pdf/ (XeLaTeX, latexmk, biber)
python3 scripts/review-textbook-pdf.py output/pdf/inside-the-llm-engine-textbook.pdf build/textbook/review
```

The build uses TeX Live 2025: XeLaTeX via latexmk, biber for the
bibliography, TeX Gyre Pagella/Heros with Pagella Math, DejaVu Sans Mono,
TikZ and pgfplots, fvextra, xeCJK/Fandol and twemojis. It fails on overfull
boxes, missing glyphs and unresolved references. A clean log does not prove a
good layout: look at the pages. Generated PDFs and logs stay in the ignored
`output/pdf/` and `build/`.

## Appendix A and Chapter 14

Appendix A.1–A.8 and Chapter 14 carry the first edition's reviewed text,
moved with `git mv` so their history follows them. Their commands are checked
against the real `mini-engine` executable by `scripts/check-textbook.py`.
Edit them lightly; the first edition's per-chapter research notes are in
`research/part-01/` and `research/part-02/`.
