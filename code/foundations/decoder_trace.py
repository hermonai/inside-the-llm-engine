"""Inspect Chapters 9--11 without timing or downloading a trained model."""
import json
import math

from model import Decoder, attention_row, linear, rms_norm, silu


def chunk_attention(queries, keys, values, prefix):
    """All keys include prefix plus this chunk; masks use logical positions."""
    if (not isinstance(prefix, int) or prefix < 0 or not queries
            or len(keys) != prefix + len(queries) or len(values) != len(keys)):
        raise ValueError("history must contain prefix plus the complete chunk")
    rows = []
    for local, query in enumerate(queries):
        end = prefix + local + 1
        probabilities, output = attention_row(query, keys[:end], values[:end])
        rows.append({"position": end - 1, "permitted": list(range(end)),
                     "probabilities": probabilities, "output": output})
    return rows


def residual_trace():
    """Chosen two-coordinate pre-norm block; attention output is supplied."""
    x, attended = [3., 4.], [1., -2.]
    attention_input = rms_norm(x, epsilon=.5)
    # Identity output projection: this isolates the two residual additions.
    r = [a + b for a, b in zip(x, attended)]
    h = rms_norm(r, epsilon=.5)
    gate = linear(h, [[1., 0.], [0., 1.]])
    up = linear(h, [[1., 1.], [1., -1.]])
    product = [silu(g) * u for g, u in zip(gate, up)]
    update = linear(product, [[1., -1.], [.5, 1.]])
    return {"x": x, "attention_input": attention_input,
            "attention_update": attended, "r": r, "ffn_input": h,
            "gate": gate, "up": up, "product": product, "ffn_update": update,
            "next": [a + b for a, b in zip(r, update)]}


def tiled_swiglu(x, gate, up, down, tile):
    """Stream hidden coordinates in tiles; return update and scratch count."""
    if (not isinstance(tile, int) or tile <= 0 or not x or not gate
            or len(gate) != len(up) or not down
            or any(len(row) != len(x) for row in gate + up)
            or any(len(row) != len(gate) for row in down)):
        raise ValueError("incompatible projections or nonpositive tile")
    output = [0.] * len(down)
    peak = 0
    for start in range(0, len(gate), tile):
        end = min(start + tile, len(gate))
        products = [silu(math.fsum(a*b for a,b in zip(x, gate[j])))
                    * math.fsum(a*b for a,b in zip(x, up[j]))
                    for j in range(start, end)]
        peak = max(peak, len(products))
        for out, row in enumerate(down):
            output[out] += math.fsum(row[j] * products[j-start]
                                     for j in range(start, end))
    return output, peak


class InspectDecoder(Decoder):
    """Observe the existing model, not an independent numerical oracle."""
    def __init__(self):
        super().__init__()
        self.events = []

    def _project(self, x, weights, position):
        result = super()._project(x, weights, position)
        layer = next(i for i, w in enumerate(self.weights) if w is weights)
        self.events.append({"layer": layer, "position": position,
                            "residual": list(x), "q": result[0],
                            "k": result[1], "v": result[2]})
        return result


def generation(prompt=(0, 1, 2), budget=4, end=None):
    """Selected IDs are included in output but evaluated only for feedback."""
    if not isinstance(budget, int) or budget <= 0 or not prompt:
        raise ValueError("positive budget and nonempty prompt required")
    model = InspectDecoder()
    if end is not None and (not isinstance(end, int) or not 0 <= end < model.cfg.vocab):
        raise ValueError("end ID outside vocabulary")
    cache, history, events = model.cache(), list(prompt), []
    for token in prompt:
        logits = cache.append(token)
    for selection in range(budget):
        expected = model.full(history)[-1]
        error = max(abs(a-b) for a,b in zip(logits, expected))
        if error > 1e-12:
            raise AssertionError("cached/full logit disagreement")
        token = max(range(len(logits)), key=logits.__getitem__)
        events.append({"evaluated": cache.position, "selected": token,
                       "max_logit_error": error, "logits": logits})
        if token == end or selection + 1 == budget:
            break
        history.append(token)
        logits = cache.append(token)
    return {"events": events, "cache_position": cache.position,
            "cache_layer_lengths": [len(k) for k in cache.keys]}


def demonstrate():
    # Scores are all equal, so every answer is an independently known mean.
    rows = chunk_attention([[0.,0.]]*2, [[1.,2.]]*5,
                           [[float(i)] for i in range(5)], prefix=3)
    model = InspectDecoder()
    logits = model.full([0,1,2])[-1]
    c = model.cfg
    matrices = len(model.embedding)*c.width + len(model.output)*c.width
    layer_parameters = sum(len(row) for w in model.weights[0].values() for row in w)
    return {"chunk": rows, "residual": residual_trace(),
            "layer_zero_position_two": model.events[2],
            "matrix_parameters": matrices + c.layers*layer_parameters,
            "layer_matrix_parameters": layer_parameters,
            "last_logits": logits, "generation": generation()}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
