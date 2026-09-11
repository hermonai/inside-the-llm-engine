"""Independent real-arithmetic oracle; no Rust or tensor-runtime dependency."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def geometry(d, hq, hkv, dh):
    if any(type(n) is not int or n <= 0 for n in (d, hq, hkv, dh)):
        raise ValueError('positive integer geometry required')
    if d != hq * dh or hq % hkv:
        raise ValueError('incompatible head geometry')
    return hq * dh, hkv * dh


def project(x, weights, heads, dh):
    if len(weights) != heads * dh or any(len(row) != len(x) for row in weights):
        raise ValueError('weight shape')
    flat = [math.fsum(a * b for a, b in zip(row, x)) for row in weights]
    return [flat[h * dh:(h + 1) * dh] for h in range(heads)]


def fixture():
    f = json.loads((ROOT / 'code/reference/fixtures/chapter08-qkv.json').read_text())
    d, hq, hkv, dh = f['geometry']
    geometry(d, hq, hkv, dh)
    x = f['input']
    r = 1 / math.sqrt(math.fsum(a * a for a in x) / d + f['epsilon'][0])
    normalized = [a * r * w for a, w in zip(x, f['gain'])]
    result = {'input': x, 'normalized': normalized, 'raw': {}, 'composed': {}}
    for name, heads in [('query', hq), ('key', hkv), ('value', hkv)]:
        rows = [f[name][i:i+d] for i in range(0, len(f[name]), d)]
        result['raw'][name] = project(x, rows, heads, dh)
        result['composed'][name] = project(normalized, rows, heads, dh)
    result['offsets'] = [h * dh + j for h in range(hq) for j in range(dh)]
    result['bytes'] = [4 * n for n in result['offsets']]
    # Analytical examples, not measured model runs. D=4096, Hq=32, Dh=128.
    result['costs'] = []
    for hkv in [32, 8, 1]:
        pq, pkv = geometry(4096, 32, hkv, 128)
        parameters = 4096 * (pq + 2 * pkv)
        result['costs'].append({'hkv': hkv, 'parameters': parameters,
            'flops': 2 * parameters, 'kv_bytes_per_token_layer_f16': 4 * pkv,
            'kv_bytes_8192_tokens_32_layers_f16': 4 * pkv * 8192 * 32})
    return result


def check():
    f = fixture()
    assert f['raw'] == {'query': [[7, 2], [-2, -3]],
                        'key': [[1, -3], [3, 3.5]],
                        'value': [[-3, -1], [0.5, -2]]}
    for args in [(1,1,1,1), (6,3,1,2), (8,4,2,2), (8,4,4,2), (3,3,1,1)]:
        pq, pkv = geometry(*args)
        d, hq, hkv, dh = args
        for h in [hq, hkv]:
            weights = [[(i-j)/4 for j in range(d)] for i in range(h*dh)]
            assert len(project([1]*d, weights, h, dh)) == h
    for args in [(0,1,1,1), (4,0,1,2), (4,2,0,2), (4,2,2,0), (5,2,1,2), (6,3,2,2)]:
        try: geometry(*args)
        except ValueError: pass
        else: raise AssertionError(args)
    assert [c['kv_bytes_8192_tokens_32_layers_f16'] for c in f['costs']] == [2**32, 2**30, 2**27]
    return f


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
