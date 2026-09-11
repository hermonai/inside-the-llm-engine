#!/usr/bin/env python3
"""Cross-check actual Rust operators, independent math, scene links and stages."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/reference/python'))
from chapter07_visual_oracle import fixture

expected=fixture()
actual=json.loads(subprocess.check_output(
    ['cargo','run','--quiet','--example','chapter07_visual_trace'],
    cwd=ROOT/'code/mini-engine',text=True))
for key in ['x','sequence','offsets','squares','sums']:
    assert actual[key]==expected[key], key
for a,b,c in zip(actual['output'],expected['output'],expected['ideal']):
    assert math.isfinite(a) and abs(a-b)<=2e-6 and abs(a-c)<=2e-6
assert expected['sums']==[1,5,14,30]
assert expected['bytes']==[16,20,24,28]
assert expected['stress']==['finite']*4+['square overflow']
assert expected['deltas'][0]>2 and expected['deltas'][2]==0
assert 1.34 < math.sqrt(sum(n*n for n in expected['output'])/4) < 1.35
for alpha in expected['alphas']:
    from chapter07_embedding_rmsnorm_oracle import rmsnorm
    scale=alpha*math.sqrt(7.5+1e-5)/math.sqrt(alpha*alpha*7.5+1e-5)
    values=rmsnorm([alpha*n for n in expected['x']],expected['weight'],1e-5)
    assert all(math.isclose(a,scale*b,rel_tol=1e-12) for a,b in zip(values,expected['ideal']))
manifest=json.loads((ROOT/'figures/manifest.json').read_text())
entries=[e for e in manifest['figures'] if e['id'].startswith('FIG-CH07-')]
assert len(entries)==10
chapter=(ROOT/'manuscript/part-02/chapter-07-embeddings-and-normalization.md').read_text()
for entry in entries:
    scene=json.loads((ROOT/entry['source']).read_text())
    assert scene['fixture']['source']=='code/reference/fixtures/chapter07-visual.json'
    assert scene['publication_status']=='canonical'
    assert chapter.count('](../../'+entry['generated'][0]+')')==1
assert ',qquad' not in chapter
source=(ROOT/'code/mini-engine/crates/engine0/src/normalization.rs').read_text()
for statement in ['let square = value * value;', 'sum_squares += square;',
                  'let mean_square = sum_squares / dimension as f32;',
                  'let inverse_rms = 1.0_f32 / (mean_square + epsilon).sqrt();']:
    assert statement in source and statement in chapter
print('Chapter 7 parity PASS: real operators, full vector, addresses, two-pass stages, 10 scenes, scale/range evidence')
