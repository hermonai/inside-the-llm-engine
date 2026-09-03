#!/usr/bin/env python3
"""Independent Chapter 6 dot/GEMV/GEMM oracle using only Python's stdlib.

The `f32` helper rounds after every multiply and add, matching ENGINE-2's
declared scalar accumulation policy instead of Python's native binary64 loop.
"""

import json
import struct


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def dot(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right)
    total = f32(0.0)
    for a, b in zip(left, right):
        total = f32(total + f32(f32(a) * f32(b)))
    return total


def gemv(matrix: list[list[float]], vector: list[float]) -> list[float]:
    assert all(len(row) == len(vector) for row in matrix)
    return [dot(row, vector) for row in matrix]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    inner = len(right)
    columns = len(right[0]) if right else 0
    assert all(len(row) == inner for row in left)
    assert all(len(row) == columns for row in right)
    return [
        [dot(left_row, [right[k][j] for k in range(inner)]) for j in range(columns)]
        for left_row in left
    ]


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1.0e-5 + 1.0e-5 * abs(expected)


left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
right = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
product = matmul(left, right)
assert product == [[58.0, 64.0], [139.0, 154.0]]

vector = [2.0, -1.0, 0.5]
matrix_vector = gemv(left, vector)
assert matrix_vector == [1.5, 6.0]
assert dot([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]) == 12.0
assert dot([], []) == 0.0

fractional = matmul(
    [[0.25, -1.5, 2.0]],
    [[4.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
)
assert all(close(actual, expected) for actual, expected in zip(fractional[0], [-1.75, -4.5]))

transposed = [list(column) for column in zip(*left)]
transpose_product = matmul(transposed, [[1.0, 2.0], [3.0, 4.0]])
assert transpose_product == [[13.0, 18.0], [17.0, 24.0], [21.0, 30.0]]

result = {
    "dot": 12.0,
    "gemv": matrix_vector,
    "gemm": product,
    "fractional_gemm": fractional,
    "transpose_view_logical_gemm": transpose_product,
    "empty_dot": 0.0,
    "arithmetic": "round-to-f32 after every multiply and add",
    "tolerance": {"absolute": 1e-5, "relative": 1e-5},
}

print(json.dumps(result, indent=2))
print("chapter06 matrix multiplication oracle: PASS")
