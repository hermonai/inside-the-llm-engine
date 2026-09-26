"""Checks for the Chapter 2 lab, on tiny GGUF files written here (no model needed).

    python3 -m unittest discover -s code/labs/ch02-count-a-model
"""
import os
import struct
import tempfile
import unittest

import count_model
import worked_numbers

F32, Q8_0 = 0, 8


def write_gguf(path, meta, tensors, values=None):
    """A minimal GGUF v3 file: metadata, tensor infos, then data (zeros unless
    `values` gives an F32 tensor's contents)."""
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(meta))]
    for key, value in meta.items():
        if isinstance(value, str):
            out += [s(key), struct.pack("<I", 8), s(value)]
        else:
            out += [s(key), struct.pack("<II", 4, value)]
    offset = 0
    data = b""
    for name, dims, ttype in tensors:
        _, block, bpb = count_model.GGML_TYPES[ttype]
        n = 1
        for dim in dims:
            n *= dim
        size = n // block * bpb
        out += [s(name), struct.pack("<I", len(dims)), struct.pack(f"<{len(dims)}Q", *dims),
                struct.pack("<IQ", ttype, offset)]
        raw = struct.pack(f"<{n}f", *values[name]) if values and name in values else b"\0" * size
        data += raw + b"\0" * ((-size) % 32)
        offset += (size + 31) // 32 * 32
    header = b"".join(out)
    with open(path, "wb") as f:
        f.write(header + b"\0" * ((-len(header)) % 32) + data)


def llama_like(d=64, n_head=4, n_kv=2, d_ff=128, layers=2, vocab=100, tied=True, experts=None,
               bad_ffn=False):
    d_h = d // n_head
    meta = {"general.architecture": "llama", "general.name": "tiny", "llama.block_count": layers,
            "llama.embedding_length": d, "llama.attention.head_count": n_head,
            "llama.attention.head_count_kv": n_kv, "llama.feed_forward_length": d_ff}
    tensors = [("token_embd.weight", (d, vocab), F32), ("output_norm.weight", (d,), F32)]
    if not tied:
        tensors.append(("output.weight", (d, vocab), Q8_0))
    if experts:
        meta["llama.expert_count"], meta["llama.expert_used_count"] = experts
    for i in range(layers):
        b = f"blk.{i}."
        tensors += [(b + "attn_norm.weight", (d,), F32), (b + "ffn_norm.weight", (d,), F32),
                    (b + "attn_q.weight", (d, n_head * d_h), F32),
                    (b + "attn_k.weight", (d, n_kv * d_h), F32),
                    (b + "attn_v.weight", (d, n_kv * d_h), F32),
                    (b + "attn_output.weight", (n_head * d_h, d), F32)]
        ff = d_ff + (1 if bad_ffn and i == 1 else 0)
        if experts:
            e = experts[0]
            tensors += [(b + "ffn_gate_inp.weight", (d, e), F32),
                        (b + "ffn_gate_exps.weight", (d, ff, e), F32),
                        (b + "ffn_up_exps.weight", (d, ff, e), F32),
                        (b + "ffn_down_exps.weight", (ff, d, e), F32)]
        else:
            tensors += [(b + "ffn_gate.weight", (d, ff), F32), (b + "ffn_up.weight", (d, ff), F32),
                        (b + "ffn_down.weight", (ff, d), F32)]
    return meta, tensors


class CountModel(unittest.TestCase):
    def price(self, context=0, batch=4, **kw):
        meta, tensors = llama_like(**kw)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.gguf")
            write_gguf(path, meta, tensors)
            meta_r, tensors_r = count_model.read_header(path)
        return count_model.price(meta_r, tensors_r, context, batch)

    def test_a_dense_tied_model_counts_by_the_equations(self):
        p = self.price()
        attn = 64 * 64 + 2 * 64 * 32 + 64 * 64          # Eq. (params-attn), d_h = 16
        ffn = 3 * 64 * 128                               # Eq. (params-ffn)
        self.assertEqual(p["params"], 2 * (attn + ffn) + 100 * 64 + 64 + 2 * 2 * 64)
        self.assertTrue(p["tied"])
        # every matrix is used once per token, and the tied table scores the vocabulary
        self.assertEqual(p["flops"], 2 * (2 * (attn + ffn) + 100 * 64))
        # the tied table is read in full plus one row: every tensor byte, and one row more
        self.assertAlmostEqual(p["weights_read"], p["file_bytes"] + 64 * 4)

    def test_an_untied_model_reads_one_row_of_its_input_table(self):
        p = self.price(tied=False)
        self.assertFalse(p["tied"])
        self.assertAlmostEqual(p["weights_read"], p["file_bytes"] - 100 * 64 * 4 + 64 * 4)
        q8 = dict((r[0], r) for r in p["rows"])["output projection"]
        self.assertEqual(q8[3], 64 * 100 // 32 * 34)     # Q8_0: 34 bytes per 32 weights

    def test_attention_over_the_cache_grows_with_context(self):
        short, long = self.price(context=0), self.price(context=1000)
        self.assertEqual(long["flops"] - short["flops"], 4 * 1000 * 4 * 16 * 2)
        self.assertEqual(long["bytes"] - short["bytes"], 1000 * 2 * 2 * 2 * 16 * 2)

    def test_a_mixture_of_experts_is_priced_by_its_active_experts(self):
        p = self.price(experts=(8, 2), batch=4)
        experts = 2 * 8 * 3 * 64 * 128
        router = 2 * 64 * 8
        attn = 2 * (64 * 64 + 2 * 64 * 32 + 64 * 64)
        self.assertEqual(p["active"], attn + router + experts / 4 + 100 * 64)
        self.assertAlmostEqual(p["union"], 1 - 0.75 ** 4)

    def test_a_small_f32_tensor_can_be_read_back(self):
        meta, tensors = llama_like()
        tensors.append(("rope_freqs.weight", (4,), F32))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.gguf")
            write_gguf(path, meta, tensors, {"rope_freqs.weight": [1.0, 1.5, 4.0, 32.0]})
            self.assertEqual(count_model.read_f32(path, "rope_freqs.weight"), (1.0, 1.5, 4.0, 32.0))

    def test_the_oracle_stops_on_a_layer_the_metadata_does_not_describe(self):
        with self.assertRaises(SystemExit) as e:
            self.price(bad_ffn=True)
        self.assertIn("layer 1", str(e.exception.code))


class WorkedNumbers(unittest.TestCase):
    def test_the_chapter_prints_these_values(self):
        ex = worked_numbers.EXAMPLES
        self.assertEqual([round(v, 4) for v in ex["rms_norm"]], [1.6036, -1.0690, 0.5345, 0.0])
        scores, p, out = ex["attention"]
        self.assertEqual(scores, [0.5, 0.0, 1.0, 0.5])
        self.assertEqual([round(v, 3) for v in p], [0.235, 0.143, 0.387, 0.235])
        self.assertEqual([round(v, 3) for v in out], [0.857, 0.673, 1.092, 1.0])
        self.assertEqual(ex["router"][0], [2, 5])
        self.assertEqual([round(v, 3) for v in ex["router"][1]], [0.550, 0.450])
        self.assertEqual([round(v, 3) for v in ex["temperature"][0.5]], [0.842, 0.114, 0.042, 0.002])
        # rotary embedding: the score depends only on the distance between positions
        r = ex["rope_relative"]
        self.assertAlmostEqual(r[0], r[1], places=12)
        self.assertAlmostEqual(r[0], r[2], places=12)


if __name__ == "__main__":
    unittest.main()
