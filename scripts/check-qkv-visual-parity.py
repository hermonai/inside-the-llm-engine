#!/usr/bin/env python3
"""Actual Rust composition vs independent real-arithmetic oracle and figure contract."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/reference/python'))
from chapter08_qkv_oracle import check

expected = check()
actual = json.loads(subprocess.check_output(
    ['cargo','run','--quiet','--example','chapter08_qkv_trace'],
    cwd=ROOT/'code/mini-engine', text=True))


def close(a, b):
    assert len(a) == len(b), 'length mismatch'
    for x, y in zip(a,b):
        assert math.isfinite(x) and abs(x-y) <= 1e-5+1e-5*abs(y), (x,y)


assert actual['input'] == expected['input']
assert actual['offsets'] == expected['offsets']
close(actual['normalized'], expected['normalized'])
for mode in ['raw','composed']:
    for name in ['query','key','value']:
        flat = [n for head in expected[mode][name] for n in head]
        close(actual[mode][name], flat)
        if mode == 'raw': assert actual[mode][name] == flat
        print(mode, name, 'max absolute error:', max(abs(a-b) for a,b in zip(actual[mode][name],flat)))
close(actual['head1'], expected['composed']['query'][1])
# Prove the comparator rejects a truncated or changed output, not only success.
for bad in [actual['composed']['query'][:-1], [99]+actual['composed']['query'][1:]]:
    try: close(bad, actual['composed']['query'])
    except AssertionError: pass
    else: raise AssertionError('comparison accepted bad candidate')

entries = [e for e in json.loads((ROOT/'figures/manifest.json').read_text())['figures'] if e['chapter']==8 and e['id'].startswith('FIG-CH08-')]
assert len(entries)==10
chapter=(ROOT/'manuscript/part-02/chapter-08-queries-keys-and-values.md').read_text()
for e in entries:
    s=json.loads((ROOT/e['source']).read_text())
    assert s['fixture']['source']=='code/reference/fixtures/chapter08-qkv.json'
    assert s['publication_status']=='canonical'
    assert chapter.count('](../../'+e['generated'][0]+')')==1
# Disambiguate reshape from permutation in the manuscript's sequence example.
assert 0*2*2+1*2+0 == 1*2+0*4+0 == 2
assert 1*2*2+0*2+0 == 4
assert [c['parameters'] for c in expected['costs']] == [50331648,25165824,17825792]
print('QKV parity PASS: full raw/composed vectors, head view, offsets, analytical costs, 10 scenes')
