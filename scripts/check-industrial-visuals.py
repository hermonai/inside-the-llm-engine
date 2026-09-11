#!/usr/bin/env python3
"""Ensure every displayed numeric fixture field is protected from silent drift."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('industrial',ROOT/'figures/industrial.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rejected = 0
for kind in ('latency','memory'):
    scene = json.loads((ROOT/f'figures/src/industrial-{kind}.json').read_text())
    module.validate(scene)
    for field in scene['fixture']:
        changed = deepcopy(scene)
        value = changed['fixture'][field]
        if isinstance(value,list):
            value[0] += 1
        else:
            changed['fixture'][field] += 1
        try:
            module.validate(changed)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError(f'unprotected fixture field: {kind}.{field}')
assert rejected == 15
print(f'Industrial numerical QA passed: 2 fixtures, {rejected} rejected mutations')
