#!/usr/bin/env python3
"""Pedagogical scalar projection scaling probe; not an LLM benchmark."""

from statistics import median
from time import perf_counter_ns


def forward(vocab_size: int, hidden_dim: int, hidden: list[float], weights: list[float]) -> float:
    checksum = 0.0
    for output in range(vocab_size):
        accumulator = 0.0
        row = output * hidden_dim
        for dimension in range(hidden_dim):
            accumulator += weights[row + dimension] * hidden[dimension]
        checksum += accumulator
    return checksum


def run(vocab_size: int, hidden_dim: int, repetitions: int = 21) -> None:
    hidden = [((index % 7) - 3) * 0.125 for index in range(hidden_dim)]
    weights = [((index % 11) - 5) * 0.0625 for index in range(vocab_size * hidden_dim)]
    forward(vocab_size, hidden_dim, hidden, weights)

    samples = []
    checksum = 0.0
    for _ in range(repetitions):
        started = perf_counter_ns()
        checksum = forward(vocab_size, hidden_dim, hidden, weights)
        samples.append(perf_counter_ns() - started)

    parameters = 2 * vocab_size * hidden_dim + vocab_size
    print(
        f"V={vocab_size:5d} D={hidden_dim:4d} "
        f"parameters={parameters:9d} parameter_bytes={parameters * 4:10d} "
        f"median_forward_ns={int(median(samples)):10d} checksum={checksum:.6f}"
    )


def main() -> None:
    print("warning=pedagogical Python scalar probe; do not extrapolate to LLM throughput")
    for vocab_size, hidden_dim in [(4, 3), (100, 16), (1_000, 64), (2_000, 128)]:
        run(vocab_size, hidden_dim)


if __name__ == "__main__":
    main()
