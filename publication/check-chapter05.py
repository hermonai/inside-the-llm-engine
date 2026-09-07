#!/usr/bin/env python3
"""Inspect publication structure and make contact sheets for visual review."""
from pathlib import Path
import json
import math
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
chapter=PdfReader(ROOT/'output/pdf/chapter05-tensors-without-magic.pdf')
book=PdfReader(ROOT/'output/pdf/inside-the-llm-engine.pdf')
atlas=PdfReader(ROOT/'output/pdf/visual-atlas.pdf')
assert len(atlas.pages)==len(json.loads((ROOT/'figures/manifest.json').read_text())['figures'])
ids=[entry['id'] for entry in json.loads((ROOT/'figures/manifest.json').read_text())['figures'] if entry['chapter']==5]
text='\n'.join(page.extract_text() or '' for page in chapter.pages)
for identifier in ids:
    assert text.count(identifier)==1, identifier+' must appear exactly once in chapter'
assert '\ufffd' not in text, 'replacement glyph'
assert not any(page.images for page in chapter.pages), 'unexpected rasterized chapter figure'
for filename in ('chapter05.html','book.html'):
    content=(ROOT/'build/publication'/filename).read_text()
    assert content.count('data:image/svg+xml')==8, filename+' missing embedded SVG'
    assert '<math ' in content, filename+' missing native MathML'
paths=sorted((ROOT/'build').glob('ch05-final-*.png'))
assert len(paths)==len(chapter.pages), 'render all chapter pages first'
for start in range(0,len(paths),9):
    group=paths[start:start+9]
    sheet=Image.new('RGB',(1500,math.ceil(len(group)/3)*710),'#d5dce0')
    for i,path in enumerate(group):
        tile=ImageOps.contain(Image.open(path).convert('RGB'),(480,675))
        x,y=(i%3)*500+10,(i//3)*710+25
        sheet.paste(tile,(x,y))
        ImageDraw.Draw(sheet).text((x,y-18),'page '+str(start+i+1),fill='black')
    sheet.save(ROOT/'build'/('chapter05-contact-'+str(start//9+1)+'.png'))
print('Publication structure PASS:',len(book.pages),'book pages;',len(chapter.pages),
      'chapter pages;',len(atlas.pages),'atlas plates; eight vector figures; offline MathML')
