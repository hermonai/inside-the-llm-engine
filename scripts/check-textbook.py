"""Source-first publication and independent worked-example regression gates."""
import codecs
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    chapters = sorted((ROOT/"tex/chapters").glob("ch*.tex"))
    assert len(chapters) == 9
    assert "\\documentclass[11pt,oneside,openany]{book}" in (ROOT/"tex/inside-the-llm-engine.tex").read_text()
    used = []
    for chapter in chapters:
        text = chapter.read_text()
        assert text.count("\\chapter{") == 1, chapter
        assert "LegacyReplacement" not in text
        assert not re.search(r"\\item\s*\\item", text), chapter
        assert not re.search(r"\\input\{[^}]+\.(md|txt|svg)\}", text)
        assert not re.search("[\u2500-\u257f]", text), chapter
        worked = ROOT/"tex/worked"/chapter.name
        assert worked.is_file() and worked.read_text().count("\\textbf{Problem.") == 3
        assert worked.read_text().count("\\textbf{Solution") >= 2 or chapter.stem == "ch01"
        for figure in re.findall(r"\\EngineFigure\{([^}]+)\}", text):
            path = ROOT/"tex/figures"/(figure+".tex")
            assert path.is_file(), path
            source = path.read_text()
            assert "\\begin{tikzpicture}" in source and "\\includegraphics" not in source
            assert "Legacy topology" not in source
            used.append(figure)
    assert len(set(used)) == 65, len(set(used))
    builder = (ROOT/"scripts/build-textbook.py").read_text()
    assert '"pandoc"' not in builder and "book-print.md" not in builder

    # Actual CLI, not copied prose output. Keep stdout and stderr separate.
    engine = ROOT/"code/mini-engine"
    subprocess.run(["cargo","build","-q","-p","engine0"],cwd=engine,check=True)
    binary = engine/"target/debug/engine0"
    cases = [
        (["--trace","I like"],0,"completed:end_of_sequence",1,1),
        (["--max-tokens","1","I like"],0,"completed:max_tokens",1,1),
        (["--cancel-at","1","I like"],0,"cancelled",1,1),
        (["--fail-at","1","I like"],1,"failed:model error:",1,1),
        (["I"],0,"completed:end_of_sequence",0,0),
    ]
    results = []
    for args, code, terminal, tokens, texts in cases:
        run = subprocess.run([str(binary),*args],capture_output=True,text=True)
        assert run.returncode == code, (args,run.stderr)
        events = [s for s in run.stdout.splitlines() if s.startswith("stream ")]
        assert sum(s.startswith("stream terminal ") for s in events) == 1
        assert events[-1].startswith("stream terminal "+terminal), events
        assert sum(s.startswith("stream token[") for s in events) == tokens
        assert sum(s.startswith("stream text[") for s in events) == texts
        if texts:
            assert 'stream text[through=0] " Rust"' in events
        if "--trace" in args:
            assert run.stderr.count("ModelInvoked") == 2
            assert "values: [-0.7, 0.1, 0.40000004, 2.2]" in run.stderr
        results.append({"args":args,"exit":code,"events":events})
    # Leading spaces are part of vocabulary pieces, not optional formatting.
    invalid = subprocess.run([str(binary),"like"],capture_output=True,text=True)
    assert invalid.returncode == 2 and "no token" in invalid.stderr

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    assert decoder.decode(bytes.fromhex("e4b8")) == ""
    assert decoder.decode(bytes.fromhex("96"), final=True) == "\u4e16"
    try:
        bytes.fromhex("e4b8").decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("Incomplete scalar unexpectedly decoded")
    hidden = [1.,-.5,2.]
    weight = [1.,-.4,.25]
    original = sum(a*b for a,b in zip(hidden,weight))+.5
    changed = sum(a*b for a,b in zip(hidden,[1.,-.4,.35]))+.5
    assert math.isclose(original,2.2) and math.isclose(changed,2.4)
    assert (12+12+4)*4 == 112 and (3+4)*4 == 28
    assert [.55/.8,.25/.8] == [.6875,.3125]
    assert [1,2,3,4,5,6][2*1+1*3] == 6
    assert 1+(3-1)*4+1 == 10
    assert (min(8,5)-4,min(8,7)-4,min(4,3)-2) == (1,3,1)
    assert 2*5*7*3 == 210
    norm = math.sqrt((3**2+4**2)/2+.5)
    assert norm == math.sqrt(13)
    assert 8*8+4*8+4*8 == 128
    assert [h//2 for h in range(4)] == [0,0,1,1]
    assert 2*2*3*2*2*4 == 192
    rotate = lambda p,z: z*complex(math.cos(p),math.sin(p))
    q,k,m,n = complex(1,2),complex(-3,.5),.7,1.2
    lhs = (rotate(m,q).conjugate()*rotate(n,k)).real
    rhs = (q.conjugate()*rotate(n-m,k)).real
    assert math.isclose(lhs,rhs,abs_tol=1e-12)
    record = {"chapters":9,"native_figures":len(set(used)),
              "worked_problems":27,"cli_cases":results,
              "status":"source, CLI and independent worked-calculation checks passed"}
    output = ROOT/"build/textbook/checks.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k!="cli_cases"}))


if __name__ == "__main__":
    main()
