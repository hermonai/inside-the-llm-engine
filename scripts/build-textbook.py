"""Build the authored LaTeX edition. No Markdown or Pandoc input is used."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / "build/textbook"
    output = ROOT / "output/pdf/inside-the-llm-engine-textbook.pdf"
    build.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH="/Library/TeX/texbin:" + os.environ["PATH"])
    args = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
            "-outdir=../build/textbook", "inside-the-llm-engine.tex"]
    result = subprocess.run(args, cwd=ROOT/"tex", env=env, capture_output=True, text=True)
    (build/"console.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit((result.stdout + result.stderr)[-6000:])
    log = (build/"inside-the-llm-engine.log").read_text()
    failures = [line for line in log.splitlines() if any(s in line for s in
                ("Overfull", "Missing character:", "undefined references",
                 "multiply defined", "LaTeX Warning: Reference"))]
    if failures:
        raise SystemExit("\n".join(failures))
    shutil.copyfile(build/"inside-the-llm-engine.pdf", output)
    sources = sorted((ROOT/"tex").rglob("*.tex"))
    sources += [ROOT/"Makefile", ROOT/"TEXTBOOK_STANDARD.md",
                ROOT/"scripts/build-textbook.py", ROOT/"scripts/check-textbook.py"]
    record = {"edition": "latex-first textbook revision; partial editorial rewrite",
              "pdf": str(output.relative_to(ROOT)),
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources}}
    (build/"build-record.json").write_text(json.dumps(record, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
