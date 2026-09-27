"""Chapter 10 oracle: ggml's own decoders and Q4_K quantizer, called through ctypes.

1. Reads rows 0-3 of blk.0.attn_q.weight (Q4_K) and blk.0.attn_v.weight (Q6_K)
   from the GGUF file, decodes them with ggml's dequantize_row_q4_K / _q6_K from
   the Homebrew libggml-base (build 8660, commit d006858), and compares the f32
   bits with the lab's decoders (the lab's --dump output).
2. Quantizes all of blk.0.attn_v.weight to Q4_K with ggml's reference quantizer
   (quantize_row_q4_K_ref, the path llama-quantize takes without an importance
   matrix), and measures the error in the layer's output on the lab's evaluation
   tokens, computed the same way as the lab (RMSNorm of the embedding times the
   layer's attn_norm weights).

    python3 ch10_ggml_oracle.py <model.gguf> <dump-dir> <tokens.txt>
"""
import ctypes
import struct
import sys

import numpy as np

LIB = ctypes.CDLL("/opt/homebrew/lib/libggml-base.dylib")
BLOCK = {0: (1, 4), 12: (256, 144), 14: (256, 210)}


def header(path):
    def rd(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

    def rs(f):
        (n,) = rd(f, "Q")
        return f.read(n).decode("utf-8", "replace")

    def rv(f, t):
        sc = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in sc:
            return rd(f, sc[t])[0]
        if t == 8:
            return rs(f)
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        return [rv(f, et) for _ in range(n)]

    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        rd(f, "I")
        (nt,) = rd(f, "Q")
        (nkv,) = rd(f, "Q")
        meta = {}
        for _ in range(nkv):
            k = rs(f)
            (t,) = rd(f, "I")
            v = rv(f, t)
            if not k.startswith("tokenizer.ggml."):
                meta[k] = v
        tensors = {}
        for _ in range(nt):
            name = rs(f)
            (nd,) = rd(f, "I")
            dims = rd(f, "Q" * nd)
            (ty,) = rd(f, "I")
            (off,) = rd(f, "Q")
            tensors[name] = (dims, ty, off)
        align = meta.get("general.alignment", 32)
        start = (f.tell() + align - 1) // align * align
    return meta, tensors, start


def rows(path, start, t, r0, n):
    dims, ty, off = t
    be, bb = BLOCK[ty]
    rb = dims[0] // be * bb
    with open(path, "rb") as f:
        f.seek(start + off + r0 * rb)
        return f.read(n * rb), ty, dims[0]


def dequant(raw, ty, k):
    out = (ctypes.c_float * k)()
    fn = {12: LIB.dequantize_row_q4_K, 14: LIB.dequantize_row_q6_K}[ty]
    fn(ctypes.c_char_p(raw), out, ctypes.c_int64(k))
    return np.frombuffer(out, dtype=np.float32).copy()


def main(path, dump, tokens):
    meta, tensors, start = header(path)
    print("1. ggml's decoders against the lab's, rows 0-3, bit for bit")
    for name in ["blk.0.attn_q.weight", "blk.0.attn_v.weight"]:
        raw, ty, cols = rows(path, start, tensors[name], 0, 4)
        ref = dequant(raw, ty, 4 * cols)
        lab = np.array([int(x, 16) for x in open(f"{dump}/{name}.rows0-3.txt")], dtype=np.uint32).view(np.float32)
        same = int((ref.view(np.uint32) == lab.view(np.uint32)).sum())
        print(f"  {name}: {same} of {ref.size} values bit-identical; largest difference {np.abs(ref - lab).max():.3e}")

    print("2. ggml's Q4_K quantizer (no importance matrix) on blk.0.attn_v.weight")
    raw, ty, cols = rows(path, start, tensors["blk.0.attn_v.weight"], 0, 1024)
    w = dequant(raw, ty, 1024 * cols).reshape(1024, cols)
    n_blocks = w.size // 256
    buf = ctypes.create_string_buffer(n_blocks * 144)
    LIB.quantize_row_q4_K_ref(w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf, ctypes.c_int64(w.size))
    wq = dequant(buf.raw, 12, w.size).reshape(1024, cols)
    ids = [int(v) for line in open(tokens) if not line.startswith("#") for v in line.split()]
    evals = ids[1024:]
    uniq, counts = np.unique(np.array(evals), return_counts=True)
    embd = tensors["token_embd.weight"]
    gamma = np.frombuffer(rows(path, start, tensors["blk.0.attn_norm.weight"], 0, 1)[0], dtype=np.float32)
    eps = meta["llama.attention.layer_norm_rms_epsilon"]
    xs = []
    for t in uniq:
        e_raw, e_ty, e_cols = rows(path, start, embd, int(t), 1)
        e = dequant(e_raw, e_ty, e_cols).astype(np.float64)
        x = (e / np.sqrt((e * e).mean() + eps)).astype(np.float32) * gamma
        xs.append(x)
    x = np.array(xs, dtype=np.float64)
    y, yq = x @ w.T.astype(np.float64), x @ wq.T.astype(np.float64)
    err = np.sqrt((counts[:, None] * (y - yq) ** 2).sum() / (counts[:, None] * y * y).sum())
    werr = np.linalg.norm(w - wq) / np.linalg.norm(w)
    print(f"  4.5 bits per weight: weight error {werr:.4f}, output error on the evaluation tokens {err:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
