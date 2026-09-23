# Figure and publication build

## Active authored textbook

Run `make textbook-check` and `make textbook` from the repository root.
The entry point is [the authored LaTeX manuscript](../tex/inside-the-llm-engine.tex);
see [its build and authoring guide](../tex/README.md). It does not invoke
Pandoc or read Markdown prose. New figures are native TikZ with typeset
mathematics. The commands below reproduce historical web/atlas artifacts;
they do not overwrite the active textbook.

## Historical atlas and web edition

Run from the repository root:

```sh
python3 figures/build.py
python3 figures/build.py --check
python3 scripts/check-tensor-visual-parity.py
python3 scripts/check-linear-visual-parity.py
python3 scripts/check-normalization-visual-parity.py
python3 scripts/check-qkv-visual-parity.py
python3 scripts/check-links.py
python3 scripts/check-diagram-style.py
python3 scripts/check-diagram-width.py
python3 scripts/check-math-style.py
```

SVG/TXT/HTML generation requires only Python 3.10+ standard library. Commit the
small generated vector/text/animation artifacts so GitHub readers need no build.
`--check` detects stale bytes, missing source/output, duplicate IDs, metadata
drift and invalid numerical fixtures. It does not claim to prove layout quality.

For publication, install Pandoc 3.9, XeLaTeX (TeX Live 2025), DejaVu fonts and
the Python packages in `publication/requirements.txt` into an isolated venv.
The chosen TeX distribution must include `fvextra`, `float`, `needspace`, `seqsplit`, `fontspec`, `unicode-math`,
`xeCJK`, `twemojis`, Fandol fonts, Noto Emoji and Latin Modern Math.
Font paths resolve with `kpsewhich`; `BOOK_FONT_DIR` can override discovery.
This override controls vector-PDF font discovery; the named TeX fonts must
also remain discoverable by the TeX distribution.

```sh
python3 -m venv .venv
.venv/bin/pip install -r publication/requirements.txt
.venv/bin/python publication/build.py
pdftoppm -scale-to 1200 -png output/pdf/visual-atlas.pdf build/atlas-page
```

Outputs: `output/pdf/inside-the-llm-engine.pdf` (all nine written chapters),
`output/pdf/chapter05-tensors-without-magic.pdf` and
`output/pdf/chapter06-matrix-multiplication.pdf` (regenerated chapters),
`output/pdf/chapter07-embeddings-and-normalization.pdf`,
`output/pdf/chapter08-queries-keys-and-values.pdf`,
`output/pdf/chapter09-position-rope.pdf`,
`output/pdf/visual-atlas.pdf` (65 vector plates), and `build/publication/book.html`
plus `chapter05.html`, `chapter06.html`, `chapter07.html`, `chapter08.html`, `chapter09.html`, `atlas.html` and six animation pages. HTML mathematics uses native MathML,
not a remote renderer. The build never advertises unwritten chapters as complete.
The atlas is a separately readable companion. Chapter 5 embeds eight canonical
figures, Chapter 6 fourteen and Chapters 7, 8 and 9 ten each: SVG in offline HTML and vector PDF in print. Other chapters
retain their legacy diagrams pending their bounded regeneration passes.
Binary/build outputs are ignored in Git. The audit inventory script records
baseline audits; do not regenerate historical inventories as a routine build step.

Inspect color and grayscale renders, text extraction, glyph warnings, page
edges and narrow HTML. Re-run fixture and artifact checks after scene edits.

For the Chapter 5 publication gate, render the entire current chapter before
making review contact sheets and checking vector/MathML completeness:

```sh
pdftoppm -scale-to 1000 -png output/pdf/chapter05-tensors-without-magic.pdf build/ch05-final
.venv/bin/python publication/check-chapter05.py
pdftoppm -scale-to 1400 -png output/pdf/chapter06-matrix-multiplication.pdf build/ch06-final
.venv/bin/python publication/check-chapter06.py
.venv/bin/python publication/check-chapter07.py
.venv/bin/python publication/check-chapter08.py
.venv/bin/python publication/check-chapter09.py
.venv/bin/python publication/check-page-bounds.py
```

The QA helper requires Pillow in addition to the publication requirements.
Contact sheets and rendered pages are review intermediates, not published assets.
CI checks deterministic figures and retains the complete Rust/Markdown gates.
Publication dependencies are explicit; the full PDF is also built locally for
this milestone. PDF byte identity is not required across TeX versions.

Optional browser QA requires Playwright and a Chromium installation:
Build the publication first; this check also verifies the offline Chapters 5/6/7/8/9
figures, native math and narrow-screen layout.

```sh
node scripts/check-figure-browser.cjs
```

## Native LaTeX publication

The preferred print-source edition now ships in `publication/latex/`, with
native TikZ plates and geometric legacy diagrams instead of verbatim graphs.
After the publication preparation above, run:

```sh
python3 publication/build-tex.py
python3 publication/build-tex.py --check
make -C publication/latex
```

This emits `output/pdf/inside-the-llm-engine-tex.pdf`. No machine paths or binary
figure dependencies are embedded in the portable `.tex` tree. Commit `.tex`
sources; keep compiled products ignored. See [the edition guide](../publication/latex/README.md).

Set `CHROME_PATH` to an installed browser executable when not using Playwright's
downloaded Chromium. Composite emoji graphics in print come from Twemoji
(Twitter and contributors), CC BY 4.0, via TeX Live; the PDF colophon records
the attribution. All book diagrams remain original deterministic vectors.
