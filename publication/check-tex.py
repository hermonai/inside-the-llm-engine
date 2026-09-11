"""Check the portable TeX artifact and render every page for visual review."""
import json
from pathlib import Path
import re
import subprocess
from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
source = root/'publication/latex'
manifest = json.loads((source/'manifest.json').read_text())
assert manifest['canonical_and_reference_plates'] == 45
assert manifest['legacy_vector_diagrams'] == 28
main = (source/'main.tex').read_text()
assert '\\includegraphics' not in main, 'native edition must not embed plates as images'
assert '/Users/' not in main and '/home/' not in main
for target in re.findall(r'\\input\{([^}]+)\}', main):
    assert (source/target).is_file(), target
for block in re.findall(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', main, re.S):
    assert not re.search('[┌┐└┘─│]',block), 'character diagram survived migration'
pdf = root/'output/pdf/inside-the-llm-engine-tex.pdf'
reader = PdfReader(pdf)
assert not any(p.images for p in reader.pages), 'raster image in native edition'
text = '\n'.join(p.extract_text() or '' for p in reader.pages)
assert '\ufffd' not in text
assert 'From model weights and KV memory' in text
out = root/'build/tex-qa'
out.mkdir(parents=True,exist_ok=True)
subprocess.run(['pdftoppm','-scale-to','750','-png',str(pdf),str(out/'page')],check=True)
paths = [out/f'page-{i:03}.png' for i in range(1,len(reader.pages)+1)]
assert all(p.is_file() for p in paths)
for start in range(0,len(paths),20):
    sheet = Image.new('RGB',(1600,1840),'#dfe5ea')
    draw = ImageDraw.Draw(sheet)
    for offset,path in enumerate(paths[start:start+20]):
        tile = ImageOps.contain(Image.open(path).convert('RGB'),(300,420))
        x,y = offset%5*320+10,offset//5*460+25
        sheet.paste(tile,(x,y)); draw.text((x,y-18),path.stem,fill='black')
    sheet.save(out/f'overview-{start//20+1:02}.png')
print('Native TeX QA:',len(reader.pages),'pages, 45 semantic + 28 legacy TikZ sources, no raster/verbatim graphs')
