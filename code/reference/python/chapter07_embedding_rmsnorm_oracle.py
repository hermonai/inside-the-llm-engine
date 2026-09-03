#!/usr/bin/env python3
"""Independent Chapter 7 embedding and RMSNorm mathematical oracle.

Uses only the Python standard library. The primary oracle evaluates the
equations with Python floats and math.fsum; a separate helper rounds every
operation to IEEE binary32 to expose the teaching kernel's finite range.
"""

from __future__ import annotations

import math
import struct


def close(actual: float, expected: float, tolerance: float = 2.0e-6) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance * (
        1.0 + abs(expected)
    )


def embedding(table: list[list[float]], token: int) -> list[float]:
    if not table:
        raise ValueError("empty vocabulary")
    width = len(table[0])
    if width == 0:
        raise ValueError("empty model dimension")
    if any(len(row) != width for row in table):
        raise ValueError("ragged table")
    if token < 0 or token >= len(table):
        raise IndexError("token outside vocabulary")
    return list(table[token])


def embed_sequence(table: list[list[float]], tokens: list[int]) -> list[list[float]]:
    return [embedding(table, token) for token in tokens]


def rms(values: list[float], epsilon: float) -> float:
    if not values:
        raise ValueError("empty normalization dimension")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    mean_square = math.fsum(value * value for value in values) / len(values)
    return math.sqrt(mean_square + epsilon)


def rmsnorm(values: list[float], weight: list[float], epsilon: float) -> list[float]:
    if len(values) != len(weight):
        raise ValueError("shape mismatch")
    denominator = rms(values, epsilon)
    return [value / denominator * gain for value, gain in zip(values, weight)]


def f32(value: float) -> float:
    """Round one Python float to IEEE binary32, preserving overflow as inf."""

    try:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def naive_f32_mean_square(values: list[float]) -> tuple[float, str]:
    """Model the Rust reference reduction without duplicating its API."""

    total = f32(0.0)
    for value in values:
        rounded = f32(value)
        square = f32(rounded * rounded)
        if not math.isfinite(square):
            return square, "square overflow"
        total = f32(total + square)
        if not math.isfinite(total):
            return total, "reduction overflow"
    return f32(total / f32(float(len(values)))), "finite"


def expect_raises(kind: type[BaseException], function, *args) -> None:
    try:
        function(*args)
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def main() -> None:
    table = [
        [0.0, 1.0, 2.0],
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
    ]
    selected = embedding(table, 2)
    assert selected == [20.0, 21.0, 22.0]
    selected[0] = -999.0
    assert table[2][0] == 20.0, "oracle lookup must return owned activation data"
    assert embed_sequence(table, [3, 0, 3]) == [
        [30.0, 31.0, 32.0],
        [0.0, 1.0, 2.0],
        [30.0, 31.0, 32.0],
    ]

    values = [1.0, -2.0, 3.0, -4.0]
    weight = [1.0, 0.5, 2.0, -1.0]
    epsilon = 1.0e-5
    denominator = rms(values, epsilon)
    output = rmsnorm(values, weight, epsilon)
    expected = [
        0.3651481282,
        -0.3651481282,
        2.1908887693,
        1.4605925129,
    ]
    assert close(denominator, math.sqrt(7.50001))
    assert all(close(actual, target) for actual, target in zip(output, expected))

    assert rmsnorm([0.0] * 4, [1.0, 2.0, 3.0, 4.0], epsilon) == [0.0] * 4
    assert all(close(value, 1.0) for value in rmsnorm([4.0] * 4, [1.0] * 4, 1e-12))
    assert all(
        close(actual, target)
        for actual, target in zip(
            rmsnorm([1.0, 1.0, 1.0], [0.5, 1.0, 2.0], 1e-12),
            [0.5, 1.0, 2.0],
        )
    )

    baseline = rmsnorm(values, weight, epsilon)
    print("scale experiment (max absolute delta from alpha=1):")
    for alpha in [1.0e-8, 0.1, 1.0, 10.0, 100.0]:
        scaled = rmsnorm([alpha * value for value in values], weight, epsilon)
        delta = max(abs(a - b) for a, b in zip(scaled, baseline))
        print(f"  alpha={alpha:>8g}  max_abs_delta={delta:.9g}")

    print("binary32 square/reduction stress:")
    for magnitude in [1.0e-20, 1.0e-10, 1.0, 1.0e10, 1.0e20]:
        mean_square, status = naive_f32_mean_square([magnitude, -magnitude])
        print(f"  magnitude={magnitude:>8g}  mean_square={mean_square:<14g}  {status}")

    expect_raises(ValueError, rms, [], epsilon)
    for bad_epsilon in [0.0, -1.0, math.nan, math.inf, -math.inf]:
        expect_raises(ValueError, rms, [1.0], bad_epsilon)
    expect_raises(ValueError, rmsnorm, [1.0], [1.0, 2.0], epsilon)
    expect_raises(ValueError, embedding, [], 0)
    expect_raises(ValueError, embedding, [[]], 0)
    expect_raises(IndexError, embedding, table, len(table))

    print("chapter07 embedding/RMSNorm oracle: PASS")


if __name__ == "__main__":
    main()
