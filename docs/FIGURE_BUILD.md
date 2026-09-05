# Figure and publication build

Run from the repository root:

```sh
python3 figures/build.py
python3 figures/build.py --check
python3 scripts/astra-audit-inventory.py
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
The chosen TeX distribution must include `fvextra`, `fontspec`, `unicode-math`,
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

Outputs: `output/pdf/inside-the-llm-engine.pdf` (all seven written chapters),
`output/pdf/visual-atlas.pdf` (ten vector plates), and `build/publication/book.html`
plus `atlas.html` and three animation pages. HTML mathematics uses native MathML,
not a remote renderer. The build never advertises unwritten chapters as complete.
The atlas is a separately readable companion; chapter insertion happens during
the bounded regeneration passes. Binary/build outputs are ignored in Git.

Inspect color and grayscale renders, text extraction, glyph warnings, page
edges and narrow HTML. Re-run fixture and artifact checks after scene edits.
CI checks deterministic figures and retains the complete Rust/Markdown gates.
Publication dependencies are explicit; the full PDF is also built locally for
this milestone. PDF byte identity is not required across TeX versions.

Optional browser QA requires Playwright and a Chromium installation:

```sh
node scripts/check-figure-browser.cjs
```

Set `CHROME_PATH` to an installed browser executable when not using Playwright's
downloaded Chromium. Composite emoji graphics in print come from Twemoji
(Twitter and contributors), CC BY 4.0, via TeX Live; the PDF colophon records
the attribution. All book diagrams remain original deterministic vectors.
