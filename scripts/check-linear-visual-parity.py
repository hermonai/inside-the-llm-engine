#!/usr/bin/env python3
"""Require real kernels to agree with the independent visual specification."""
from pathlib import Path
import importlib.util
import json
import subprocess
import re
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('oracle',ROOT/'code/reference/python/chapter06_visual_oracle.py')
oracle=importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
expected=oracle.derive()
run=subprocess.run(['cargo','run','--quiet','--example','chapter06_visual_trace'],cwd=ROOT/'code/mini-engine',capture_output=True,text=True)
if run.returncode: raise RuntimeError(run.stderr)
actual=json.loads(run.stdout)
for key in ('shape','y','c','products','sums','w_offsets','w_strides','b_strides','c_strides'):
    assert actual[key]==expected[key],key
assert actual['candidate']==expected['c'] and actual['dot']==expected['y'][0]
# Every loop order visits exactly the same triples and valid physical slots.
assert sorted(expected['ijk'])==sorted(expected['ikj'])
assert len(expected['ijk'])==3*4*2
assert len(expected['tails'])==8 and expected['tails'][-1]==[4,5,4,7,2,3]
# Publication annotations deliberately describe this bounded fixture. A new
# fixture needs a reviewed storyboard, not silently changed arithmetic labels.
assert expected['w']==[1,-2,3,0,0,1,-1,2,2,0,1,-1]
assert expected['x']==[2,1,-1,0.5] and expected['b']==[2,1,1,-1,-1,2,0.5,0]
assert expected['y']==[-3,3,2.5] and expected['c']==[-3,9,3,-3,2.5,4]
assert expected['products']==[2,-2,-3,0] and expected['sums']==[2,0,-3,-3]
assert expected['w_bytes'][6]==24 and expected['w'][6]==-1
assert expected['tail_shape']==[5,7,3] and expected['tail_block']==[4,4,2]
manuscript=(ROOT/'manuscript/part-02/chapter-06-matrix-multiplication-the-engine-room.md').read_text()
source=''.join((ROOT/'code/mini-engine/crates/engine0/src/linear.rs').read_text().split())
snippets=re.findall(r'```rust\n(.*?)\n```',manuscript,re.S)
for marker in ('left.get(&[index])','matrix.get2(row, k)','right.get2(k, j)','ii.saturating_add'):
    matching=[body for body in snippets if marker in body]
    assert len(matching)==1 and ''.join(matching[0].split()) in source, marker+' source excerpt drift'
entries=[e for e in json.loads((ROOT/'figures/manifest.json').read_text())['figures'] if e['id'].startswith('FIG-CH06-')]
assert len(entries)==14
for entry in entries:
    scene=json.loads((ROOT/entry['source']).read_text())
    assert scene['fixture']['source']=='code/reference/fixtures/chapter06-visual.json'
    assert manuscript.count('](../../'+entry['generated'][0]+')')==1
    assert scene['publication_status']=='canonical'
print('Chapter 6 visual parity PASS: real kernels, fixture annotations, addresses, loop visits, tails, four source excerpts, 14 embedded scenes')
