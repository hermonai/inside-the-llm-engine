#!/usr/bin/env python3
"""Independent scalar oracle for Chapter 4 sampling stages.

This file intentionally does not import ENGINE-1. It uses Python's standard
library and accepts an artificial draw so categorical selection can be checked
without coupling the proof to either implementation's PRNG.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    probabilities: list[float]
    retained: list[int]
    selected: int


def stable_softmax(scores: list[float]) -> list[float]:
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError("scores must be finite and non-empty")
    maximum = max(scores)
    numerators = [math.exp(value - maximum) for value in scores]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def normalize(probabilities: list[float]) -> list[float]:
    total = sum(probabilities)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("candidate mass must be finite and positive")
    return [value / total for value in probabilities]


def process(
    logits: list[float],
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("logits must be finite and non-empty")

    scaled = [value / temperature for value in logits]
    active = set(range(len(scaled)))
    if top_k is not None:
        order = sorted(active, key=lambda token_id: (-scaled[token_id], token_id))
        active = set(order[: min(top_k, len(order))])

    active_ids = [token_id for token_id in range(len(scaled)) if token_id in active]
    active_probabilities = stable_softmax([scaled[token_id] for token_id in active_ids])
    probabilities = [0.0] * len(scaled)
    for token_id, probability in zip(active_ids, active_probabilities):
        probabilities[token_id] = probability

    if top_p is not None and top_p < 1:
        order = sorted(active, key=lambda token_id: (-probabilities[token_id], token_id))
        retained: set[int] = set()
        cumulative = 0.0
        for token_id in order:
            retained.add(token_id)
            cumulative += probabilities[token_id]
            if cumulative >= top_p:
                break
        probabilities = [
            probability if token_id in retained else 0.0
            for token_id, probability in enumerate(probabilities)
        ]
        probabilities = normalize(probabilities)

    return probabilities


def categorical_select(probabilities: list[float], draw: float) -> int:
    if not 0 <= draw < 1:
        raise ValueError("draw must be in [0, 1)")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")
    cumulative = 0.0
    final_positive = None
    for token_id, probability in enumerate(probabilities):
        if probability > 0:
            final_positive = token_id
        cumulative += probability
        if draw < cumulative:
            return token_id
    if final_positive is None:
        raise ValueError("all candidates were filtered")
    return final_positive


def oracle(
    logits: list[float],
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    draw: float,
) -> Result:
    probabilities = process(logits, temperature, top_k, top_p)
    return Result(
        probabilities=probabilities,
        retained=[index for index, value in enumerate(probabilities) if value > 0],
        selected=categorical_select(probabilities, draw),
    )


def self_check() -> None:
    basic = stable_softmax([1.0, 2.0, 3.0])
    expected = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
    assert all(math.isclose(a, e, rel_tol=1e-15) for a, e in zip(basic, expected))
    assert stable_softmax([1000.0, 999.0, 998.0]) == list(reversed(basic))

    nucleus = oracle(
        [math.log(0.40), math.log(0.30), math.log(0.15), math.log(0.10), math.log(0.05)],
        temperature=1.0,
        top_k=None,
        top_p=0.80,
        draw=0.63,
    )
    assert nucleus.retained == [0, 1, 2]
    assert nucleus.selected == 1
    assert math.isclose(sum(nucleus.probabilities), 1.0, abs_tol=1e-15)

    combined = oracle([3.0, 2.0, 1.0, 0.0], 1.0, 2, 0.60, 0.99)
    assert combined.retained == [0]
    assert combined.selected == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--draw", type=float, default=0.63)
    args = parser.parse_args()
    self_check()
    result = oracle(args.logits, args.temperature, args.top_k, args.top_p, args.draw)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    print("chapter04 sampling oracle: PASS")


if __name__ == "__main__":
    main()
