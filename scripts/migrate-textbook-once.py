"""One-time import of reviewed prose, never part of the textbook build.

Refuses to overwrite authored chapters. Imported material is not automatically
an editorial rewrite: the edition manifest records the distinction.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "tex/chapters"
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("Refusing to overwrite the authored LaTeX manuscript")
    destination.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "publication/latex/main.tex").read_text()
    chapters = list(re.finditer(r"\\section\{Chapter (\d+) -+ (.*?)\}\\label\{([^}]+)\}", source, re.S))
    assert len(chapters) == 9
    figures = ROOT / "tex/figures"
    figures.mkdir(parents=True, exist_ok=True)
    imported = set()
    for i, match in enumerate(chapters):
        number = int(match[1])
        end = chapters[i+1].start() if i+1 < len(chapters) else source.index(r"\subsection{Publication colophon}")
        body = source[match.end():end].strip()
        body = re.sub(r"\\sub(sub)?section\{", lambda m: "\\" + ("subsection" if m[1] else "section") + "{", body)
        def plate(m):
            name, caption = m[1], m[2]
            imported.add(name)
            return "\\EngineFigure{"+name+"}{"+caption+"}{fig:"+name+"}"
        body = re.sub(r"\\begin\{figure\}\[H\]\\centering\\resizebox\{\\linewidth\}\{!\}\{\\input\{figures/([^}]+)\.tex\}\}\\caption\{(.*?)\}\\end\{figure\}", plate, body)
        # Legacy character-grid geometry is deliberately excluded from this edition.
        # Explicit semantic replacements are authored below; this is not an SVG rasterization.
        body = re.sub(r"\\begin\{center\}\\begin\{adjustbox\}\{max width=\\linewidth\}\\input\{figures/legacy-(\d+)\.tex\}\\end\{adjustbox\}\\end\{center\}", lambda m: "\\LegacyReplacement{"+m[1]+"}", body)
        body = body.replace("\\clearpage", "")
        # Every display now has a chapter-scoped number for discussions and errata.
        body = body.replace("\\[", "\\begin{equation}").replace("\\]", "\\end{equation}")
        title = " ".join(match[2].split())
        (destination/f"ch{number:02}.tex").write_text("% Authoritative LaTeX; edit this file, not the historical Markdown.\n\\chapter{"+title+"}\\label{"+match[3]+"}\n"+body+"\n\\input{worked/ch"+f"{number:02}"+".tex}\n")
    for name in sorted(imported):
        original = (ROOT/"publication/latex/figures"/(name+".tex")).read_text().splitlines()
        lines = ["% Editable textbook TikZ, initially imported from the verified semantic plate."]
        for line in original[1:]:
            if "use as bounding box" in line:
                lines.append(r"\path[use as bounding box] (30,170) rectangle (975,700);")
            elif "fill=white,draw=none" in line and "1000.0,720.0" in line:
                continue
            elif re.search(r"at \(40,(38|84|140)\)", line) or "(40,109) -- (960,109)" in line:
                continue
            else:
                lines.append(line)
        (figures/(name+".tex")).write_text("\n".join(lines)+"\n")
    print(f"Imported {len(chapters)} chapters and {len(imported)} semantic TikZ plates; editorial review still required")


if __name__ == "__main__":
    main()
