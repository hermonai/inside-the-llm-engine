"""Small, readable decoder for Part I. Standard library only; not trained.

Matrices store output rows: linear(x, W)[j] = sum_i W[j][i] * x[i].
The full-prefix and cached paths deliberately have different control flow.
They share scalar arithmetic; hand calculations test that arithmetic separately.
"""
from dataclasses import dataclass
import math


def linear(x, weights):
    if not x or any(len(row) != len(x) for row in weights):
        raise ValueError("linear shape mismatch")
    return [math.fsum(a * b for a, b in zip(row, x)) for row in weights]


def softmax(scores):
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("softmax needs a nonempty finite, unmasked score row")
    maximum = max(scores)
    exps = [math.exp(s - maximum) for s in scores]
    total = math.fsum(exps)
    return [e / total for e in exps]


def attention_row(query, keys, values):
    if not keys or len(keys) != len(values):
        raise ValueError("nonempty, matching key/value history required")
    width = len(query)
    if not width or any(len(k) != width for k in keys):
        raise ValueError("query/key width mismatch")
    value_width = len(values[0])
    if not value_width or any(len(v) != value_width for v in values):
        raise ValueError("value width mismatch")
    scores = [math.fsum(q * k for q, k in zip(query, key)) / math.sqrt(width)
              for key in keys]
    probabilities = softmax(scores)
    output = [math.fsum(p * v[j] for p, v in zip(probabilities, values))
              for j in range(value_width)]
    return probabilities, output


def causal_attention(queries, keys, values):
    if not queries or len(queries) != len(keys) or len(keys) != len(values):
        raise ValueError("self-attention needs equal positive sequence lengths")
    # Slicing is the causal mask: future positions cannot enter softmax.
    return [attention_row(q, keys[:i + 1], values[:i + 1])[1]
            for i, q in enumerate(queries)]


def silu(x):
    # Both branches avoid exponentiating a large positive number.
    return x / (1 + math.exp(-x)) if x >= 0 else x * math.exp(x) / (1 + math.exp(x))


def swiglu(x, gate_weights, up_weights, down_weights):
    gate, up = linear(x, gate_weights), linear(x, up_weights)
    if len(gate) != len(up):
        raise ValueError("gate/up widths differ")
    return linear([silu(g) * u for g, u in zip(gate, up)], down_weights)


def rms_norm(x, epsilon=1e-5):
    if not x or epsilon <= 0:
        raise ValueError("positive width and epsilon required")
    scale = 1 / math.sqrt(math.fsum(v * v for v in x) / len(x) + epsilon)
    return [v * scale for v in x]  # unit gain in this teaching model


def rope(x, position, base=10_000.0):
    if not x or len(x) % 2 or position < 0:
        raise ValueError("RoPE needs an even width and nonnegative position")
    half = len(x) // 2
    out = list(x)
    for j in range(half):
        angle = position * base ** (-2 * j / len(x))
        c, s = math.cos(angle), math.sin(angle)
        a, b = x[j], x[j + half]
        out[j], out[j + half] = a * c - b * s, a * s + b * c
    return out


@dataclass(frozen=True)
class Config:
    vocab: int = 5
    width: int = 4
    heads: int = 2
    kv_heads: int = 1
    hidden: int = 6
    layers: int = 2

    def __post_init__(self):
        if min(self.vocab, self.width, self.heads, self.kv_heads,
               self.hidden, self.layers) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("head dimensions must divide")
        if self.head_width % 2:
            raise ValueError("rotary head width must be even")

    @property
    def head_width(self):
        return self.width // self.heads


class Decoder:
    def __init__(self, config=Config()):
        self.cfg = config
        serial = 0

        def matrix(rows, cols):
            nonlocal serial
            result = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    serial += 1
                    row.append(math.sin(serial * 1.7) * 0.2)
                result.append(row)
            return result

        c = config
        self.embedding = matrix(c.vocab, c.width)
        kv_width = c.kv_heads * c.head_width
        self.weights = [
            {"q": matrix(c.width, c.width), "k": matrix(kv_width, c.width),
             "v": matrix(kv_width, c.width), "o": matrix(c.width, c.width),
             "gate": matrix(c.hidden, c.width), "up": matrix(c.hidden, c.width),
             "down": matrix(c.width, c.hidden)} for _ in range(c.layers)]
        self.output = matrix(c.vocab, c.width)

    def _token(self, token):
        if not isinstance(token, int) or not 0 <= token < self.cfg.vocab:
            raise ValueError("token outside vocabulary")
        return list(self.embedding[token])

    def _project(self, x, weights, position):
        c = self.cfg
        h = rms_norm(x)
        d = c.head_width
        q, k, v = [linear(h, weights[name]) for name in ("q", "k", "v")]
        return ([rope(q[i:i+d], position) for i in range(0, len(q), d)],
                [rope(k[i:i+d], position) for i in range(0, len(k), d)],
                [v[i:i+d] for i in range(0, len(v), d)])

    def _finish(self, x, attended, weights):
        updated = [a+b for a, b in zip(x, linear(attended, weights["o"]))]
        ffn = swiglu(rms_norm(updated), weights["gate"], weights["up"], weights["down"])
        return [a+b for a, b in zip(updated, ffn)]

    def full(self, tokens):
        """Recompute every prefix position; return a logit row per input token."""
        if not tokens:
            raise ValueError("empty prompt has no last row")
        x = [self._token(t) for t in tokens]
        group = self.cfg.heads // self.cfg.kv_heads
        for weights in self.weights:
            projections = [self._project(row, weights, i) for i, row in enumerate(x)]
            next_rows = []
            for i, row in enumerate(x):
                attended = []
                for head in range(self.cfg.heads):
                    kv_head = head // group
                    keys = [p[1][kv_head] for p in projections[:i+1]]
                    values = [p[2][kv_head] for p in projections[:i+1]]
                    attended += attention_row(projections[i][0][head], keys, values)[1]
                next_rows.append(self._finish(row, attended, weights))
            x = next_rows
        return [linear(rms_norm(row), self.output) for row in x]

    def cache(self):
        """One request's cache; bound to this decoder instance."""
        return Cache(self)


class Cache:
    def __init__(self, model):
        self.model = model
        self.keys = [[] for _ in model.weights]
        self.values = [[] for _ in model.weights]
        self.position = 0

    def append(self, token):
        """Evaluate one token, append one K/V per layer, return its logits."""
        model, c = self.model, self.model.cfg
        x = model._token(token)  # validate before mutating the cache
        group = c.heads // c.kv_heads
        for layer, weights in enumerate(model.weights):
            q, k, v = model._project(x, weights, self.position)
            self.keys[layer].append(k)
            self.values[layer].append(v)
            attended = []
            for head in range(c.heads):
                kv_head = head // group
                keys = [row[kv_head] for row in self.keys[layer]]
                values = [row[kv_head] for row in self.values[layer]]
                attended += attention_row(q[head], keys, values)[1]
            x = model._finish(x, attended, weights)
        self.position += 1
        return linear(rms_norm(x), model.output)
