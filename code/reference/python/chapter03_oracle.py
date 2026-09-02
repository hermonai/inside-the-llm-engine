#!/usr/bin/env python3
"""Independent plain-Python oracle for Chapter 3's ENGINE-1 fixture."""

VOCABULARY = ["<eos>", "I", "like", "Rust"]
EMBEDDING = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, -0.5, 2.0],
    [-1.0, 0.0, 0.0],
]
OUTPUT_WEIGHT = [
    [-0.5, 0.4, 0.1],
    [0.2, 0.2, 0.0],
    [0.3, 0.2, 0.1],
    [1.0, -0.4, 0.25],
]
OUTPUT_BIAS = [-0.2, 0.0, 0.0, 0.5]
EXPECTED = [-0.7, 0.1, 0.4, 2.2]


def forward(token_id: int) -> tuple[list[float], list[float]]:
    """Evaluate h=E[x], z=W h+b with loops independent from Rust."""
    hidden = EMBEDDING[token_id].copy()
    logits = []
    for row, bias in zip(OUTPUT_WEIGHT, OUTPUT_BIAS):
        accumulator = bias
        for weight, value in zip(row, hidden):
            accumulator += weight * value
        logits.append(accumulator)
    return hidden, logits


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-9 + 1e-9 * abs(expected)


def main() -> None:
    token_id = VOCABULARY.index("like")
    hidden, logits = forward(token_id)
    assert hidden == [1.0, -0.5, 2.0]
    assert all(close(actual, expected) for actual, expected in zip(logits, EXPECTED))

    print(f"input={token_id}:{VOCABULARY[token_id]}")
    print(f"hidden={hidden}")
    print(f"logits={logits}")
    print("expected=[-0.7, 0.1, 0.4, 2.2]")
    print("oracle=PASS")


if __name__ == "__main__":
    main()
