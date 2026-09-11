#!/usr/bin/env python3
"""Check all native semantic plate sources without Pandoc or a TeX install."""
from pathlib import Path
import importlib.util
import json
import tempfile
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_tex', root/'publication/build-tex.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
entries = json.loads((root/'figures/manifest.json').read_text())['figures']
for entry in entries:
    svg = root/entry['generated'][0]
    target = root/'publication/latex/figures'/f'{svg.stem}.tex'
    expected = '% Generated from '+entry['generated'][0]+'; do not hand edit.\n'+module.svg_tikz(svg)
    assert target.read_text() == expected, target
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'invalid.svg'
    tree = ET.parse(root/entries[0]['generated'][0])
    rectangle = tree.getroot().find('{*}rect')
    rectangle.set('transform', 'rotate(30)')
    tree.write(path)
    try:
        module.svg_tikz(path)
    except ValueError:
        pass
    else:
        raise AssertionError('unsupported transformation was silently discarded')
print('Native figure parity passed:',len(entries),'TikZ plates; unsupported transform rejected')
