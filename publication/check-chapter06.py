#!/usr/bin/env python3
"""Check Chapter 6 artifacts and prepare every page for human-visible QA."""
from pathlib import Path
import json
import math
import re
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
chapter=PdfReader(ROOT/'output/pdf/chapter06-matrix-multiplication.pdf')
book=PdfReader(ROOT/'output/pdf/inside-the-llm-engine.pdf')
atlas=PdfReader(ROOT/'output/pdf/visual-atlas.pdf')
entries=json.loads((ROOT/'figures/manifest.json').read_text())['figures']
ids=[e['id'] for e in entries if e['id'].startswith('FIG-CH06-')]
assert len(ids)==14 and len(atlas.pages)==len(entries)
text='\n'.join(p.extract_text() or '' for p in chapter.pages)
for identifier in ids:
    assert text.count(identifier)==1, identifier
assert '\ufffd' not in text
assert not any(p.images for p in chapter.pages), 'chapter figures must remain vectors'
embedded=sum(len(re.findall(r'!\[[^\]]*\]\([^)]*figures/generated/[^)]*\.svg\)',path.read_text())) for path in (ROOT/'manuscript').glob('part-*/chapter-*.md'))
for filename,count in [('chapter06.html',14),('book.html',embedded)]:
    content=(ROOT/'build/publication'/filename).read_text()
    assert content.count('data:image/svg+xml')==count
    assert '<math ' in content
paths=sorted((ROOT/'build').glob('ch06-final-*.png'))
assert len(paths)==len(chapter.pages), 'render every chapter page first'
for start in range(0,len(paths),9):
    group=paths[start:start+9]
    sheet=Image.new('RGB',(1500,math.ceil(len(group)/3)*710),'#d5dce0')
    for i,path in enumerate(group):
        tile=ImageOps.contain(Image.open(path).convert('RGB'),(480,675))
        x,y=(i%3)*500+10,(i//3)*710+25
        sheet.paste(tile,(x,y)); ImageDraw.Draw(sheet).text((x,y-18),'page '+str(start+i+1),fill='black')
    sheet.save(ROOT/'build'/('chapter06-contact-'+str(start//9+1)+'.png'))
print('Chapter 6 publication PASS:',len(chapter.pages),'chapter pages;',len(book.pages),
      'book pages;',len(atlas.pages),'atlas pages; 14 vectors; offline MathML')
