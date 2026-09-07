#!/usr/bin/env python3
"""Compare the real Rust API trace with independent row/column enumeration."""
from pathlib import Path
import json
import subprocess
from itertools import product

ROOT=Path(__file__).resolve().parents[1]
fixture=json.loads((ROOT/'code/reference/fixtures/chapter05-visual.json').read_text())
source=[[1,2,3],[4,5,6]]
flat=[x for row in source for x in row]
expected={
    'source':flat,
    'transpose':[source[i][j] for j in range(3) for i in range(2)],
    'reshape':flat,
    'slice':[source[i][j] for i in range(2) for j in range(1,3)]}
expected['copy']=expected['transpose']
hero=json.loads((ROOT/'figures/src/tensor.json').read_text())['fixture']
assert hero['values']==flat and hero['shape']==fixture['source']['shape']
assert hero['strides']==fixture['source']['strides']
assert hero['element_offset']==5 and hero['byte_offset']==20
trace=subprocess.run(['cargo','run','--quiet','-p','engine0','--example','chapter05_visual_trace'],cwd=ROOT/'code/mini-engine',check=True,capture_output=True,text=True)
actual=json.loads(trace.stdout)
assert set(actual)==set(expected)
for name,values in expected.items():
    row=fixture[name]
    assert row['values']==values, name+' independent values'
    offsets=[row['base']+sum(i*s for i,s in zip(index,row['strides'])) for index in product(*(range(d) for d in row['shape']))]
    assert row['offsets']==offsets, name+' offsets'
    assert actual[name]==row, name+' Rust metadata/value parity'
assert fixture['slice_required_elements']==1+max(fixture['slice']['offsets'])
assert fixture['source_payload_bytes']==4*len(flat)
assert fixture['copy_payload_traffic_bytes']==2*4*len(flat)
print('Chapter 5 visual parity PASS: five views/owners, exact values, strides, offsets and bytes')
