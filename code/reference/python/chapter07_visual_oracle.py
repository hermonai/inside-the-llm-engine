#!/usr/bin/env python3
"""One input-only fixture; independently evaluate values, addresses and range.

The mathematical output uses the existing wider oracle. The separate F32
trace rounds every operation in the teaching reduction, including epsilon.
No generated expected-output file is an authority.
"""
import json
import math
from pathlib import Path
from chapter07_embedding_rmsnorm_oracle import f32, rmsnorm, naive_f32_mean_square

ROOT = Path(__file__).resolve().parents[3]


def fixture():
    data = json.loads((ROOT / 'code/reference/fixtures/chapter07-visual.json').read_text())
    v, d = data['shape']
    assert (v, d) == (3, 4) and len(data['table']) == v * d
    assert data['tokens'] == [1, 0, 1] and len(data['weight']) == d
    # Geometry contains reviewed teaching labels; changed inputs require a
    # deliberate storyboard revision, not silently different numbers.
    assert data['table'] == [0,0,0,0,1,-2,3,-4,2,1,0,-1]
    assert data['weight'] == [1,0.5,2,-1] and data['epsilon'] == [1e-5]
    assert data['alphas'] == [1e-8,0.1,1,10,100]
    assert data['magnitudes'] == [1e-20,1e-10,1,1e10,1e20]
    x = data['table'][d:2*d]
    eps = f32(data['epsilon'][0])
    squares = [f32(n*n) for n in x]
    sums = []
    total = 0.0
    for n in squares:
        total = f32(total+n)
        sums.append(total)
    mean = f32(total/d)
    inverse = f32(1/f32(math.sqrt(f32(mean+eps))))
    output = [f32(f32(n*inverse)*w) for n, w in zip(x, data['weight'])]
    ideal = rmsnorm(x, data['weight'], data['epsilon'][0])
    sequence = [n for t in data['tokens'] for n in data['table'][t*d:(t+1)*d]]
    deltas = [max(abs(a-b) for a,b in zip(rmsnorm([alpha*n for n in x], data['weight'], data['epsilon'][0]), ideal)) for alpha in data['alphas']]
    stress = [naive_f32_mean_square([m, -m])[1] for m in data['magnitudes']]
    return {**data, 'x': x, 'squares': squares, 'sums': sums, 'mean': mean,
            'inverse': inverse, 'output': output, 'ideal': ideal,
            'sequence': sequence, 'offsets': list(range(d,2*d)),
            'bytes': list(range(4*d,8*d,4)), 'deltas': deltas, 'stress': stress}


if __name__ == '__main__':
    print(json.dumps(fixture(), allow_nan=False))
