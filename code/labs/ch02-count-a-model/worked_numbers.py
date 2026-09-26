#!/usr/bin/env python3
"""Chapter 2's worked examples, computed the way the capstone engine computes them.

Every number the chapter prints for its small examples comes from this file:
RMSNorm, one attention head over a four-token cache, rotary embedding in the
rotate-half layout, SwiGLU's gate, a mixture-of-experts router and sampling
temperature. The kernels mirror code/mini-engine/crates/capstone/src/kernels.rs
(in float64 here, so the printed digits are the exact values).

    python3 worked_numbers.py
"""
import math


def rms_norm(x, weight, eps=1e-5):
    scale = 1 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * scale * w for v, w in zip(x, weight)]


def softmax(v, temperature=1.0):
    m = max(v)
    e = [math.exp((x - m) / temperature) for x in v]
    z = sum(e)
    return [x / z for x in e]


def attend(q, keys, values):
    """One head, one query: scaled dot products, softmax, weighted sum of values."""
    scale = 1 / math.sqrt(len(q))
    scores = [sum(a * b for a, b in zip(q, k)) * scale for k in keys]
    p = softmax(scores)
    out = [sum(p[t] * values[t][j] for t in range(len(values))) for j in range(len(values[0]))]
    return scores, p, out


def rope(head, pos, base):
    """Rotate element i with element i + d/2 by pos * base^(-2i/d)."""
    d, h = len(head), list(head)
    for i in range(d // 2):
        angle = pos * base ** (-2 * i / d)
        c, s = math.cos(angle), math.sin(angle)
        a, b = h[i], h[i + d // 2]
        h[i], h[i + d // 2] = a * c - b * s, a * s + b * c
    return h


def silu(x):
    return x / (1 + math.exp(-x))


def route(logits, k):
    """Top-k experts, weighted by a softmax over the chosen logits."""
    top = sorted(range(len(logits)), key=lambda i: -logits[i])[:k]
    return top, softmax([logits[i] for i in top])


EXAMPLES = {
    "rms_norm": rms_norm([3, -1, 2, 0], [1, 2, 0.5, 1]),
    "attention": attend([1, 0, 1, 0],
                        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
                        [[2, 0, 1, 0], [0, 2, 0, 1], [1, 1, 1, 1], [0, 0, 2, 2]]),
    "rope_pos1": rope([1, 0, 0, 1], 1, 10000.0),
    "rope_relative": [sum(a * b for a, b in zip(rope([1, 0, 1, 0], m, 10000.0),
                                                rope([0.5, 1, 0, 0.5], m - 2, 10000.0)))
                      for m in (5, 12, 102)],
    "swiglu": [silu(g) * u for g, u in zip([2, -1, 0, 3], [0.5, 2, 4, -1])],
    "router": route([1.2, -0.3, 2.1, 0.4, -1.0, 1.9, 0.0, 0.7], 2),
    "temperature": {t: softmax([2, 1, 0.5, -1], t) for t in (1.0, 0.5, 2.0)},
}


def main():
    fmt = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    print("RMSNorm of [3, -1, 2, 0] with gains [1, 2, 0.5, 1]:", fmt(EXAMPLES["rms_norm"]))
    scores, p, out = EXAMPLES["attention"]
    print("attention scores:", fmt(scores), " weights:", fmt(p), " output:", fmt(out))
    print("rotary embedding of [1, 0, 0, 1] at position 1:", fmt(EXAMPLES["rope_pos1"]))
    print("q.k for positions (5, 3), (12, 10), (102, 100):", fmt(EXAMPLES["rope_relative"]))
    print("SwiGLU gate * up:", fmt(EXAMPLES["swiglu"]))
    top, w = EXAMPLES["router"]
    print(f"router: experts {top} with weights {fmt(w)}")
    for t, probs in EXAMPLES["temperature"].items():
        print(f"temperature {t}: {fmt(probs)}")


if __name__ == "__main__":
    main()
