#!/usr/bin/env python3
"""Check vector/offline publication and render all new material for review."""
import json
import math
from pathlib import Path
import subprocess
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
QA=ROOT/'build/chapter07-qa'
QA.mkdir(parents=True,exist_ok=True)
chapter=PdfReader(OUT/'chapter07-embeddings-and-normalization.pdf')
entries=json.loads((ROOT/'figures/manifest.json').read_text())['figures']
text='\n'.join(p.extract_text() or '' for p in chapter.pages)
assert '\ufffd' not in text and 'qquad' not in text
assert not any(p.images for p in chapter.pages)
assert len(PdfReader(OUT/'visual-atlas.pdf').pages)==len(entries)
for e in entries:
    if e['id'].startswith('FIG-CH07-'): assert text.count(e['id'])==1,e['id']
content=(ROOT/'build/publication/chapter07.html').read_text()
assert content.count('data:image/svg+xml')==10 and '<math ' in content
subprocess.run(['pdftoppm','-scale-to','1500','-png',str(OUT/'chapter07-embeddings-and-normalization.pdf'),str(QA/'page')],check=True)

def sheets(paths,prefix,columns=3,cell=(500,720),gray=False):
    for start in range(0,len(paths),6):
        group=paths[start:start+6]
        sheet=Image.new('RGB',(columns*cell[0],math.ceil(len(group)/columns)*cell[1]),'#e1e6e9')
        for i,path in enumerate(group):
            tile=Image.open(path).convert('RGB')
            if gray: tile=ImageOps.grayscale(tile).convert('RGB')
            tile=ImageOps.contain(tile,(cell[0]-20,cell[1]-35))
            x,y=i%columns*cell[0]+10,i//columns*cell[1]+25
            sheet.paste(tile,(x,y)); ImageDraw.Draw(sheet).text((x,y-18),path.stem,fill='black')
        sheet.save(QA/f'{prefix}-{start//6+1}.png')

paths=[QA/f'page-{i:02}.png' for i in range(1,len(chapter.pages)+1)]
assert all(p.is_file() for p in paths)
sheets(paths,'pages')
plates=[]
for i,e in enumerate(entries,1):
    if not e['id'].startswith('FIG-CH07-'): continue
    prefix=QA/e['id']
    subprocess.run(['pdftoppm','-f',str(i),'-l',str(i),'-singlefile','-scale-to','1500','-png',str(OUT/'visual-atlas.pdf'),str(prefix)],check=True)
    plates.append(prefix.with_suffix('.png'))
sheets(plates,'plates',columns=2,cell=(750,660))
sheets(plates,'grayscale',columns=2,cell=(750,660),gray=True)
print('Chapter 7 publication PASS:',len(chapter.pages),'pages, 10 vectors, offline SVG/MathML; review sheets in',QA)
