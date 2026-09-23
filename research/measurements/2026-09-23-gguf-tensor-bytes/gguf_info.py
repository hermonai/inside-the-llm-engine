"""Minimal GGUF v2/v3 reader: metadata and tensor byte sizes (stdlib only)."""
import struct, sys, json
from collections import defaultdict

GGML_TYPES = {  # type id: (name, block size in elements, bytes per block)
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

def read(f, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, f.read(size))

def read_str(f):
    (n,) = read(f, "Q")
    return f.read(n).decode("utf-8", errors="replace")

def read_value(f, vtype):
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if vtype in scalar:
        return read(f, scalar[vtype])[0]
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        (etype,) = read(f, "I"); (n,) = read(f, "Q")
        items = [read_value(f, etype) for _ in range(n)]
        return {"array_len": n, "head": items[:3]} if n > 8 else items
    raise ValueError(f"unknown GGUF value type {vtype}")

def main(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", magic
        (version,) = read(f, "I"); (n_tensors,) = read(f, "Q"); (n_kv,) = read(f, "Q")
        meta = {}
        for _ in range(n_kv):
            key = read_str(f); (vtype,) = read(f, "I")
            meta[key] = read_value(f, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f); (ndim,) = read(f, "I")
            dims = read(f, "Q" * ndim); (ttype,) = read(f, "I"); (offset,) = read(f, "Q")
            tname, block, bpb = GGML_TYPES.get(ttype, (f"type{ttype}", None, None))
            n = 1
            for d in dims: n *= d
            nbytes = n // block * bpb if block else None
            tensors.append({"name": name, "dims": dims, "type": tname, "elements": n, "bytes": nbytes})
    total = sum(t["bytes"] for t in tensors)
    by_type = defaultdict(int)
    for t in tensors: by_type[t["type"]] += t["bytes"]
    keep = {k: v for k, v in meta.items() if not k.startswith("tokenizer.ggml.") or k.endswith(("model", "pre"))}
    embd = [t for t in tensors if t["name"] == "token_embd.weight"]
    out = [t for t in tensors if t["name"] == "output.weight"]
    print(json.dumps({"version": version, "n_tensors": n_tensors, "metadata": keep,
        "total_tensor_bytes": total, "bytes_by_type": by_type,
        "token_embd": embd, "output": out,
        "params": sum(t["elements"] for t in tensors)}, indent=1, default=str))

main(sys.argv[1])
