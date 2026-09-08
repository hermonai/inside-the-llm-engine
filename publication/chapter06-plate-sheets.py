#!/usr/bin/env python3
"""Prepare color/grayscale review sheets from the current rendered atlas."""
from pathlib import Path
import math
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
for prefix in ('atlas-page','ch06-gray'):
    paths=[ROOT/'build'/f'{prefix}-{number:02}.png' for number in range(18,32)]
    assert all(p.is_file() for p in paths), 'render all Chapter 6 atlas plates first'
    for start in range(0,14,6):
        group=paths[start:start+6]
        sheet=Image.new('RGB',(1500,math.ceil(len(group)/2)*670),'#d5dce0')
        for i,path in enumerate(group):
            tile=ImageOps.contain(Image.open(path).convert('RGB'),(730,640))
            x,y=(i%2)*750+10,(i//2)*670+24
            sheet.paste(tile,(x,y))
            ImageDraw.Draw(sheet).text((x,y-18),'atlas plate '+str(start+i+18),fill='black')
        sheet.save(ROOT/'build'/f'ch06-{prefix}-sheet-{start//6+1}.png')
print('Prepared color and grayscale contact sheets for all 14 Chapter 6 plates')
