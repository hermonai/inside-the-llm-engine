"""Independent complex-number RoPE oracle, using real-arithmetic tolerances."""
import cmath
import json
import math
from pathlib import Path
from chapter08_qkv_oracle import fixture as qkv_fixture

ROOT = Path(__file__).resolve().parents[3]


def rotate(values, position, rotary_dim, base, pairing='adjacent'):
    result = list(values)
    for i in range(rotary_dim // 2):
        a, b = (2*i, 2*i+1) if pairing == 'adjacent' else (i, i+rotary_dim//2)
        z = complex(values[a], values[b]) * cmath.exp(1j * position * base**(-2*i/rotary_dim))
        result[a], result[b] = z.real, z.imag
    return result


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def fixture():
    f = json.loads((ROOT/'code/reference/fixtures/chapter09-rope.json').read_text())
    rows = [f['input'][i:i+4] for i in range(0, len(f['input']), 4)]
    for pairing in ['adjacent', 'split_half']:
        f[pairing] = [v for row in rows for v in rotate(row, f['position'], 4, f['base'], pairing)]
    q, k = rows[0], f['key']
    p, n = f['query_position'], f['key_position']
    f['relative_lhs'] = dot(rotate(q,p,4,100), rotate(k,n,4,100))
    f['relative_rhs'] = dot(q, rotate(k,n-p,4,100))
    previous = qkv_fixture()['composed']
    f['composed'] = {name: [v for row in previous[name] for v in
        (rotate(row, f['composition_position'],2,10000) if name != 'value' else row)]
        for name in ['query','key','value']}
    f['frequencies'] = [1.0,0.1]
    f['periods'] = [2*math.pi/w for w in f['frequencies']]
    return f


def check():
    f = fixture()
    assert abs(f['relative_lhs']-f['relative_rhs']) < 1e-12
    assert abs(f['adjacent'][0] - (math.cos(1)-2*math.sin(1))) < 1e-12
    for pairing in ['adjacent','split_half']:
        for pos in [0,1,7,2048,32768,-7]:
            x = [1,2,3,4,5]
            y = rotate(x,pos,4,100,pairing)
            assert abs(dot(x,x)-dot(y,y)) < 1e-12
            assert y[-1] == x[-1]
            assert max(abs(a-b) for a,b in zip(x,rotate(y,-pos,4,100,pairing))) < 1e-12
    # A wrong pairing can pass every norm test. A coordinate oracle is essential.
    assert max(abs(a-b) for a,b in zip(f['adjacent'], f['split_half'])) > 0.5
    return f


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
