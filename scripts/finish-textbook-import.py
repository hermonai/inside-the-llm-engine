"""One-time mechanical cleanup of imported typesetting, never a build step."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def escape(text):
    return "".join({"\\":r"\textbackslash{}", "{":r"\{", "}":r"\}",
                    "_":r"\_", "&":r"\&", "%":r"\%", "#":r"\#",
                    "$":r"\$", "^":r"\textasciicircum{}", "~":r"\textasciitilde{}"}.get(c,c)
                   for c in text)


def main():
    count = 0
    for path in sorted((ROOT/"tex/chapters").glob("ch*.tex")):
        if path.stem == "ch01":
            continue
        text = path.read_text()
        def block(m):
            nonlocal count
            raw = m[1]
            if not ("->" in raw or "-->" in raw):
                return m[0]
            # Preserve actual code and API signatures; transform only diagram-like prose.
            if any(s in raw for s in ("fn ", "Result<", "let ", "pub ", "for ", "Model::", "Selector::", "SamplerState::")):
                return m[0]
            count += 1
            lines = [s.strip() for s in raw.strip().splitlines() if s.strip()]
            items = []
            for line in lines:
                line = re.sub(r"[-+|]+>", " → ", line).strip()
                line = line.lstrip("-+| ")
                items.append(r"\item " + escape(line).replace("→", r"\(\longrightarrow\)"))
            return "\\begin{quote}\n\\begin{enumerate}\n"+"\n".join(items)+"\n\\end{enumerate}\n\\end{quote}"
        text = re.sub(r"\\begin\{verbatim\}\n(.*?)\\end\{verbatim\}", block, text, flags=re.S)
        # Do not remove bare \item: Pandoc puts ordinary bullet content on the next line.
        text = re.sub(r"^\\item[ \t]+[|=+v^ -]+\n", "", text, flags=re.M)
        text = re.sub(r"(?:\\item\s*){2,}", lambda _: "\\item\n", text)
        text = text.replace(r"\texttt{{[}1,2,3{]}\ →\ 23}", r"\([1,2,3]\mapsto23\)")
        path.write_text(text)
    for path in sorted((ROOT/"tex/figures").glob("*.tex")):
        text = path.read_text()
        superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
        text = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+",
                      lambda m: r"\({}^{" + m[0].translate(superscripts) + r"}\)", text)
        text = text.replace("ᵀ", r"\({}^{\mathsf{T}}\)").replace("ᵢ", r"\({}_i\)")
        text = text.replace("↔", r"\(\leftrightarrow\)")
        path.write_text(text)
    print(f"Converted {count} character-flow snippets into typeset ordered traces; code signatures preserved")


if __name__ == "__main__":
    main()
