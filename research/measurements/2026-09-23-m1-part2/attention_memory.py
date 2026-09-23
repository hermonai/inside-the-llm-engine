"""Ch 8: naive attention vs fused scaled_dot_product_attention on MPS.
Causal, batch 1, 8 heads, head width 128, float16. Oracle: outputs must agree."""
import json, statistics, time, torch
import torch.nn.functional as F
dev = torch.device("mps")
H, D = 8, 128
def naive(q, k, v):
    s = (q @ k.transpose(-1, -2)) / D ** 0.5
    S = q.shape[-2]
    mask = torch.ones(S, S, device=dev, dtype=torch.bool).triu(1)
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v
def timeit(fn, reps=5, warm=2):
    for _ in range(warm): fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
out = {"torch": torch.__version__, "runs": []}
for S in (1024, 2048, 4096, 8192):
    q, k, v = (torch.randn(1, H, S, D, device=dev, dtype=torch.float16) for _ in range(3))
    torch.mps.empty_cache(); base = torch.mps.driver_allocated_memory()
    o1 = naive(q, k, v); torch.mps.synchronize(); peak_naive = torch.mps.driver_allocated_memory() - base
    o2 = F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.mps.synchronize()
    err = float((o1.float() - o2.float()).abs().max())
    del o1, o2; torch.mps.empty_cache(); base2 = torch.mps.driver_allocated_memory()
    o2 = F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.mps.synchronize()
    peak_sdpa = torch.mps.driver_allocated_memory() - base2
    del o2
    tn = timeit(lambda: naive(q, k, v)); ts = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    out["runs"].append({"S": S, "naive_s": tn, "sdpa_s": ts, "naive_extra_bytes": peak_naive,
        "sdpa_extra_bytes": peak_sdpa, "score_matrix_bytes_fp16": H * S * S * 2, "max_abs_diff": err})
    del q, k, v; torch.mps.empty_cache()
print(json.dumps(out, indent=1))
