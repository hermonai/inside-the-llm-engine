#!/usr/bin/env python3
"""Independent exact indexing oracle for Chapter 5.

This file imports no mini-engine code. It keeps the arithmetic deliberately
small enough to inspect by hand.
"""

from itertools import product
import json


def canonical_strides(shape: list[int]) -> list[int]:
    strides = [0] * len(shape)
    suffix = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = suffix
        suffix *= shape[axis]
    return strides


def offset(indices: list[int], strides: list[int], base: int = 0) -> int:
    return base + sum(index * stride for index, stride in zip(indices, strides))


shape_3d = [2, 3, 4]
strides_3d = canonical_strides(shape_3d)
assert strides_3d == [12, 4, 1]
assert offset([1, 2, 3], strides_3d) == 23

matrix_shape = [2, 3]
matrix_strides = canonical_strides(matrix_shape)
storage = ["A", "B", "C", "D", "E", "F"]
matrix_indices = list(product(range(2), range(3)))
matrix_offsets = [offset(list(index), matrix_strides) for index in matrix_indices]
assert matrix_offsets == [0, 1, 2, 3, 4, 5]

transpose_shape = [3, 2]
transpose_strides = [matrix_strides[1], matrix_strides[0]]
transpose_indices = list(product(range(3), range(2)))
transpose_offsets = [
    offset(list(index), transpose_strides) for index in transpose_indices
]
transpose_logical = [storage[index] for index in transpose_offsets]
assert transpose_strides == [1, 3]
assert transpose_offsets == [0, 3, 1, 4, 2, 5]
assert transpose_logical == ["A", "D", "B", "E", "C", "F"]

reshape_shape = [3, 2]
reshape_strides = canonical_strides(reshape_shape)
reshape_offsets = [
    offset(list(index), reshape_strides)
    for index in product(range(3), range(2))
]
assert reshape_offsets == [0, 1, 2, 3, 4, 5]

result = {
    "shape_3d": shape_3d,
    "strides_3d": strides_3d,
    "offset_1_2_3": 23,
    "matrix_offsets": matrix_offsets,
    "transpose_shape": transpose_shape,
    "transpose_strides": transpose_strides,
    "transpose_offsets": transpose_offsets,
    "transpose_logical": transpose_logical,
    "reshape_shape": reshape_shape,
    "reshape_strides": reshape_strides,
    "reshape_offsets": reshape_offsets,
}

print(json.dumps(result, indent=2))
print("chapter05 tensor oracle: PASS")
