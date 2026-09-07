#!/usr/bin/env python3
"""Optional visual QA helper: make a labeled atlas contact sheet from Poppler PNGs."""
from pathlib import Path
import json
import math
from PIL import Image, ImageOps, ImageDraw

paths=sorted(Path('build').glob('atlas-page-*.png'))
expected=len(json.loads(Path('figures/manifest.json').read_text())['figures'])
assert len(paths)==expected, 'render the complete current atlas first'
sheet=Image.new('RGB',(1500,math.ceil(len(paths)/2)*520),'#d5dce0')
for i,path in enumerate(paths):
    tile=ImageOps.contain(Image.open(path).convert('RGB'),(730,490))
    x,y=(i%2)*750+10,(i//2)*520+24
    sheet.paste(tile,(x,y)); ImageDraw.Draw(sheet).text((x,y-18),str(i+1),fill='black')
sheet.save('build/atlas-contact.png')
