"""Ch 21: synthesize LoRA adapters for Llama 3.2 3B in llama.cpp's GGUF adapter format.
A is random, B is zero, so the adapter's contribution is exactly zero: outputs must
match the base model token for token, while every adapter matmul still runs.
Usage: python3 make_lora.py <out.gguf> <rank> <seed>"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else "gguf-py")
import gguf

out, rank, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d, d_kv, d_ff, n_layer = 3072, 1024, 8192, 28
shapes = {  # name: (in_features, out_features)
    "attn_q": (d, d), "attn_k": (d, d_kv), "attn_v": (d, d_kv), "attn_output": (d, d),
    "ffn_gate": (d, d_ff), "ffn_up": (d, d_ff), "ffn_down": (d_ff, d),
}
rng = np.random.default_rng(seed)
w = gguf.GGUFWriter(out, "llama")
w.add_type("adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", float(rank))
n_params = 0
for i in range(n_layer):
    for name, (fin, fout) in shapes.items():
        a = (rng.standard_normal((rank, fin)) * 0.02).astype(np.float16)   # ggml ne = [in, rank]
        b = np.zeros((fout, rank), dtype=np.float16)                        # ggml ne = [rank, out]
        w.add_tensor(f"blk.{i}.{name}.weight.lora_a", a)
        w.add_tensor(f"blk.{i}.{name}.weight.lora_b", b)
        n_params += a.size + b.size
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: rank {rank}, {n_params:,} parameters, {2 * n_params:,} bytes of F16")
