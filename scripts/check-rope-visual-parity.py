#!/usr/bin/env python3
"""Compare actual Rust rotations/composition with an independent complex oracle."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/reference/python'))
from chapter09_rope_oracle import check, rotate, dot

expected = check()
actual = json.loads(subprocess.check_output(
    ['cargo','run','--quiet','--example','chapter09_rope_trace'],
    cwd=ROOT/'code/mini-engine', text=True))


def close(a,b):
    assert len(a)==len(b), 'length mismatch'
    for x,y in zip(a,b):
        assert math.isfinite(x) and abs(x-y)<=1e-5+1e-5*abs(y), (x,y)


for name in ['adjacent','split_half','relative_lhs','relative_rhs']:
    a,b=actual[name],expected[name]
    if not isinstance(a,list): a,b=[a],[b]
    close(a,b)
    print(name, 'max absolute error:', max(abs(x-y) for x,y in zip(a,b)))
for name in ['query','key','value']:
    close(actual['composed'][name], expected['composed'][name])
    print('composed', name, 'max absolute error:', max(abs(x-y) for x,y in zip(actual['composed'][name],expected['composed'][name])))
for bad in [actual['adjacent'][:-1], [99]+actual['adjacent'][1:]]:
    try: close(bad,actual['adjacent'])
    except AssertionError: pass
    else: raise AssertionError('comparator accepted changed output')

# Analytical claims used by the manuscript and plates.
assert 32768*128*4 == 16*2**20
assert rotate([1,2],7,2,100)==rotate([1,2],7,2,10000)
q,k=expected['input'][:4],expected['key']
wrong=dot(q,rotate(k,-3,4,100))
assert abs(wrong-expected['relative_lhs'])>1
assert math.cos(6)>math.cos(3)  # not pointwise monotonic in separation
assert [100**(-2*i/4) for i in range(2)] == [1,0.1]
assert [10000**(-2*i/4) for i in range(2)] == [1,0.01]
for angle,error in [(0.2,1e-3),(7,0.5),(100,1.0)]:
    x=[1,2]
    a=complex(*x)*complex(math.cos(angle),math.sin(angle))
    b=complex(*x)*complex(math.cos(angle+error),math.sin(angle+error))
    chord=2*math.sqrt(5)*abs(math.sin(error/2))
    assert abs(abs(a-b)-chord)<1e-12
    assert chord<=math.sqrt(5)*abs(error)

entries=[e for e in json.loads((ROOT/'figures/manifest.json').read_text())['figures'] if e['id'].startswith('FIG-CH09-')]
assert len(entries)==10
chapter=(ROOT/'manuscript/part-02/chapter-09-position-rope.md').read_text()
for e in entries:
    s=json.loads((ROOT/e['source']).read_text())
    assert s['fixture']['source']=='code/reference/fixtures/chapter09-rope.json'
    assert s['publication_status']=='canonical'
    assert chapter.count('](../../'+e['generated'][0]+')')==1
print('RoPE parity PASS: both layouts, relative sign, complete Q/K/V composition, 10 scenes, analytical geometry')
