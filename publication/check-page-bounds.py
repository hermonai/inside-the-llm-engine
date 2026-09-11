#!/usr/bin/env python3
"""Use Poppler word geometry to reject off-page PDF text; not a visual-review substitute."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
for name in ('inside-the-llm-engine-tex.pdf','inside-the-llm-engine.pdf','chapter05-tensors-without-magic.pdf',
             'chapter06-matrix-multiplication.pdf','chapter07-embeddings-and-normalization.pdf',
             'chapter08-queries-keys-and-values.pdf','visual-atlas.pdf'):
    output=subprocess.run(['pdftotext','-bbox',str(ROOT/'output/pdf'/name),'-'],
                          check=True,capture_output=True).stdout
    tree=ET.fromstring(output)
    pages=tree.findall('.//{*}page')
    failures=[]
    for number,page in enumerate(pages,1):
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a=word.attrib
            if float(a['xMin'])<0 or float(a['yMin'])<0 or float(a['xMax'])>width or float(a['yMax'])>height:
                failures.append((number,word.text,a))
    assert not failures,(name,failures)
    print(name,':',len(pages),'pages, all extracted words inside page bounds')
