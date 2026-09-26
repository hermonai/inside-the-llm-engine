#!/usr/bin/env python3
"""Chapter 2 lab: count a real model from its file.

Reads a GGUF file's header (no weights are loaded), groups its tensors by the
operator that uses them, and prices one decode step: parameters, FLOPs per
token, and bytes read at batch one and at a larger batch. For a mixture of
experts it separates resident parameters from active ones.

The oracle is the model's own metadata: the program recomputes every layer's
parameter count from the shape keys (width, heads, FFN width, experts) with
Chapter 2's equations and stops if a tensor in the file disagrees.

    python3 count_model.py model.gguf [--context 4096] [--batch 16]

Standard library only.
"""
import argparse
import math
import re
import struct
from collections import defaultdict

# type id: (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))


def _read_str(f):
    (n,) = _read(f, "Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    if vtype in SCALARS:
        return _read(f, SCALARS[vtype])[0]
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (etype,) = _read(f, "I")
        (n,) = _read(f, "Q")
        if etype == 8:  # string arrays (the vocabulary) are skipped, not kept
            for _ in range(n):
                _read_str(f)
            return {"array_len": n}
        return [_read_value(f, etype) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(path, with_offsets=False):
    """Return (metadata dict, list of tensors) without reading any weights."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        _read(f, "I")  # format version
        (n_tensors,) = _read(f, "Q")
        (n_kv,) = _read(f, "Q")
        meta = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = _read(f, "I")
            meta[key] = _read_value(f, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (ndim,) = _read(f, "I")
            dims = _read(f, "Q" * ndim)
            (ttype,) = _read(f, "I")
            (offset,) = _read(f, "Q")  # from the start of the aligned data section
            tname, block, bpb = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 0))
            n = math.prod(dims)
            tensors.append({"name": name, "dims": dims, "type": tname,
                            "params": n, "bytes": n // block * bpb, "offset": offset})
        if with_offsets:
            align = meta.get("general.alignment", 32)
            meta["data_start"] = (f.tell() + align - 1) // align * align
    return meta, tensors


def read_f32(path, name):
    """The values of one small F32 tensor, such as rope_freqs.weight."""
    meta, tensors = read_header(path, with_offsets=True)
    t = next(t for t in tensors if t["name"] == name)
    if t["type"] != "F32":
        raise SystemExit(f"{name} is {t['type']}, not F32")
    with open(path, "rb") as f:
        f.seek(meta["data_start"] + t["offset"])
        return struct.unpack(f"<{t['params']}f", f.read(4 * t["params"]))


# The operator each tensor feeds, by llama.cpp's tensor names; first match wins.
GROUPS = [
    ("input embedding", r"^token_embd\.weight$"),
    ("output projection", r"^output\.weight$"),
    ("attention projections", r"\.attn_(q|k|v|qkv|output)\.weight$"),
    ("router", r"\.ffn_gate_inp\.weight$"),
    ("routed experts", r"\.ffn_(gate|up|down)_exps\.weight$"),
    ("shared expert", r"\.ffn_(gate|up|down)_shexp\.weight$"),
    ("feed-forward network", r"\.ffn_(gate|up|down)\.weight$"),
    ("other tensors", r"."),  # norms, biases, rotary tables
]
MATMUL = {"output projection", "attention projections", "router", "routed experts",
          "shared expert", "feed-forward network"}


def group_of(name):
    return next(label for label, pattern in GROUPS if re.search(pattern, name))


def shape(meta):
    """The shape keys Chapter 2's equations need, from the metadata alone."""
    arch = meta.get("general.architecture", "llama")
    key = lambda k, default=None: meta.get(f"{arch}.{k}", default)
    d = key("embedding_length")
    n_head = key("attention.head_count")
    s = {
        "layers": key("block_count"), "d": d, "n_head": n_head,
        "n_kv": key("attention.head_count_kv", n_head),
        "d_h": key("attention.key_length") or d // n_head,
        "d_ff": key("feed_forward_length"),
        "n_exp": key("expert_count", 0) or 0, "k_exp": key("expert_used_count", 0) or 0,
    }
    s["d_ff_exp"] = key("expert_feed_forward_length", s["d_ff"])
    return s


def check_layers(s, tensors):
    """The oracle: each layer's matrices must hold what the equations predict."""
    attn = s["d"] * s["n_head"] * s["d_h"] + 2 * s["d"] * s["n_kv"] * s["d_h"] \
        + s["n_head"] * s["d_h"] * s["d"]                                   # Eq. (params-attn)
    dense = 3 * s["d"] * s["d_ff"]                                          # Eq. (params-ffn)
    moe = s["n_exp"] * 3 * s["d"] * s["d_ff_exp"] + s["d"] * s["n_exp"]    # experts and router
    layers = defaultdict(lambda: {"attn": 0, "ffn": 0, "moe": False})
    for t in tensors:
        m = re.match(r"^blk\.(\d+)\.", t["name"])
        if not m:
            continue
        g, layer = group_of(t["name"]), layers[int(m.group(1))]
        if g == "attention projections":
            layer["attn"] += t["params"]
        elif g in ("feed-forward network", "routed experts", "router"):
            layer["ffn"] += t["params"]
            layer["moe"] |= g != "feed-forward network"
    for i, got in sorted(layers.items()):
        want = (attn, moe if got["moe"] else dense)
        if (got["attn"], got["ffn"]) != want:
            raise SystemExit(
                f"oracle failed at layer {i}: the file holds {got['attn']:,} attention and "
                f"{got['ffn']:,} FFN parameters; the metadata's shape keys imply {want[0]:,} "
                f"and {want[1]:,}.\nThe equations assume every layer has one shape; look for "
                f"per-layer keys in the metadata (for example *_swa).")
    return len(layers)


def price(meta, tensors, context, batch):
    s = shape(meta)
    n_checked = check_layers(s, tensors)
    groups = defaultdict(lambda: {"params": 0, "bytes": 0})
    for t in tensors:
        g = groups[group_of(t["name"])]
        g["params"] += t["params"]
        g["bytes"] += t["bytes"]
    embd = groups["input embedding"]
    vocab = next(t["dims"][1] for t in tensors if t["name"] == "token_embd.weight")
    tied = "output projection" not in groups
    frac = s["k_exp"] / s["n_exp"] if s["n_exp"] else 1.0

    rows = [("input embedding", embd["params"], 0, embd["bytes"] / vocab)]  # one row gathered
    if tied:  # the same table, read in full, scores the vocabulary
        rows.append(("output projection (tied)", 0, 2 * embd["params"], embd["bytes"]))
    for label, _ in GROUPS[1:]:
        if label not in groups:
            continue
        g = groups[label]
        share = frac if label == "routed experts" else 1.0
        flops = 2 * g["params"] * share if label in MATMUL else 0
        rows.append((label, g["params"], flops, g["bytes"] * share))
    kv_per_token = 2 * s["layers"] * s["n_kv"] * s["d_h"] * 2  # keys and values, 16-bit
    rows.append(("attention over the cache", 0, 4 * context * s["n_head"] * s["d_h"] * s["layers"],
                 context * kv_per_token))

    weights_read = sum(r[3] for r in rows[:-1])
    expert_bytes = groups["routed experts"]["bytes"] if s["n_exp"] else 0
    union = 1 - (1 - frac) ** batch if s["n_exp"] else 1.0  # experts a batch touches, uniform routing
    return {
        "shape": s, "rows": rows, "tied": tied, "layers_checked": n_checked,
        "params": sum(t["params"] for t in tensors), "file_bytes": sum(t["bytes"] for t in tensors),
        "flops": sum(r[2] for r in rows), "bytes": sum(r[3] for r in rows),
        "active": sum(r[2] for r in rows[:-1]) / 2, "weights_read": weights_read,
        "kv_per_token": kv_per_token, "union": union,
        "batch_bytes": weights_read + expert_bytes * (union - frac) + batch * context * kv_per_token,
    }


def human(x, unit=""):
    for scale, suffix in ((1e12, " T"), (1e9, " G"), (1e6, " M"), (1e3, " K")):
        if abs(x) >= scale:
            return f"{x / scale:.2f}{suffix}{unit}"
    return f"{x:.0f} {unit}".rstrip()


def report(meta, p, context, batch):
    s = p["shape"]
    lines = [f"{meta.get('general.name', 'model')}: {p['params']:,} parameters in "
             f"{p['file_bytes']:,} bytes of tensors{' (tied embeddings)' if p['tied'] else ''}"]
    if s["n_exp"]:
        lines.append(f"mixture of experts: {s['k_exp']} of {s['n_exp']} experts per token, "
                     f"{p['active'] / 1e9:.2f} B parameters active per token")
    lines.append(f"oracle: all {p['layers_checked']} layers match the metadata's shapes")
    lines.append(f"\none decode step at batch 1, {context} tokens of 16-bit cache:")
    lines.append(f"{'operator':<28}{'parameters':>12}{'FLOPs':>12}{'bytes read':>13}")
    for label, params, flops, nbytes in p["rows"]:
        lines.append(f"{label:<28}{human(params):>12}{human(flops):>12}{human(nbytes, 'B'):>13}")
    lines.append(f"{'one step':<28}{human(p['params']):>12}{human(p['flops']):>12}"
                 f"{human(p['bytes'], 'B'):>13}")
    lines.append(f"\nweights read per token: {human(p['weights_read'], 'B')}, "
                 f"{100 * p['weights_read'] / p['file_bytes']:.1f}% of the tensor bytes")
    lines.append(f"arithmetic intensity: {p['flops'] / p['bytes']:.2f} FLOP/byte")
    lines.append(f"KV cache: {p['kv_per_token'] / 1024:.0f} KiB per token")
    touched = (f"uniform routing touches {100 * p['union']:.0f}% of each layer's experts; "
               if s["n_exp"] else "")
    lines.append(f"batch of {batch}: {touched}a step reads {human(p['batch_bytes'], 'B')} "
                 f"for {batch} tokens")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=4096, help="tokens already in the cache")
    ap.add_argument("--batch", type=int, default=16, help="sequences decoded together")
    ap.add_argument("--dump", metavar="TENSOR", help="print the values of a small F32 tensor")
    a = ap.parse_args()
    if a.dump:
        print(" ".join(f"{v:.4g}" for v in read_f32(a.model, a.dump)))
        return
    meta, tensors = read_header(a.model)
    print(report(meta, price(meta, tensors, a.context, a.batch), a.context, a.batch))


if __name__ == "__main__":
    main()
