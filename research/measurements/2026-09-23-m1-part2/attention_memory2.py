"""Ch 8: causal attention three ways on MPS under a fixed allocator cap.
naive - materializes the scores (fp16 scores, fp32 softmax), as eager PyTorch does;
sdpa  - torch.nn.functional.scaled_dot_product_attention;
tiled - blockwise online softmax (the chapter's equation), fp32 running state,
        written in eager PyTorch, so each block's scores still round-trip memory.
Oracle: float64 CPU attention for query rows 0, S/2 and S-1 (the last row sees every key).
One (variant, S) per process: python3 attention_memory2.py <variant> <S>."""
import json, statistics, sys, time, torch
import torch.nn.functional as F

CAP_GIB = 6
dev = torch.device("mps")
torch.mps.set_per_process_memory_fraction(CAP_GIB * 2**30 / torch.mps.recommended_max_memory())
H, D = 8, 128
variant, S = sys.argv[1], int(sys.argv[2])

def naive(q, k, v):
    s = (q @ k.transpose(-1, -2)) / D ** 0.5
    n = q.shape[-2]
    mask = torch.ones(n, n, device=dev, dtype=torch.bool).triu(1)
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def tiled(q, k, v, blk=1024):
    n = q.shape[-2]
    out = torch.empty_like(q)
    scale = D ** -0.5
    for i in range(0, n, blk):
        qi = q[:, :, i:i + blk].float() * scale
        rows = qi.shape[-2]
        m = torch.full((1, H, rows, 1), float("-inf"), device=dev)
        l = torch.zeros((1, H, rows, 1), device=dev)
        o = torch.zeros((1, H, rows, D), device=dev)
        for j in range(0, i + rows, blk):          # causal: blocks up to the diagonal
            kj = k[:, :, j:j + blk].float()
            vj = v[:, :, j:j + blk].float()
            s = qi @ kj.transpose(-1, -2)
            if j + kj.shape[-2] > i:                # the diagonal block: mask the future
                qpos = torch.arange(i, i + rows, device=dev)[:, None]
                kpos = torch.arange(j, j + kj.shape[-2], device=dev)[None, :]
                s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            c = torch.exp(m - m_new)
            l = l * c + p.sum(-1, keepdim=True)
            o = o * c + p @ vj
            m = m_new
        out[:, :, i:i + rows] = (o / l).to(q.dtype)
    return out

def oracle(q, k, v, rows):
    qc, kc, vc = (t[0].cpu().double() for t in (q, k, v))
    res = []
    for r in rows:
        s = (kc[:, :r + 1] @ qc[:, r].unsqueeze(-1)).squeeze(-1) / D ** 0.5
        res.append((torch.softmax(s, dim=-1).unsqueeze(1) @ vc[:, :r + 1]).squeeze(1))
    return torch.stack(res, 1)

torch.manual_seed(0)
q, k, v = (torch.randn(1, H, S, D, device=dev, dtype=torch.float16) for _ in range(3))
fn = {"naive": naive, "sdpa": sdpa, "tiled": tiled}[variant]
if variant == "sdpa" and S >= 32768:
    # Added mid-run: at 16,384 tokens the fused path allocated 16.5 GiB outside
    # the allocator cap (its internal buffers are not governed by it); 32,768
    # would ask for about four times that on a 16 GB machine.
    print(json.dumps({"variant": variant, "S": S, "status": "skipped",
                      "reason": "internal allocations at 16384 exceeded the cap (16.5 GiB); not attempted"}))
    sys.exit(0)
torch.mps.synchronize()
base = torch.mps.driver_allocated_memory()
rec = {"variant": variant, "S": S, "heads": H, "d_head": D, "cap_gib": CAP_GIB,
       "torch": torch.__version__, "qkvo_bytes": 4 * H * S * D * 2, "score_bytes_fp16": H * S * S * 2}
try:
    t0 = time.perf_counter(); out = fn(q, k, v); torch.mps.synchronize(); first = time.perf_counter() - t0
except RuntimeError as e:
    rec.update(status="failed", error=str(e).splitlines()[0][:240])
    print(json.dumps(rec)); sys.exit(0)
rec["driver_bytes_after_first_call"] = torch.mps.driver_allocated_memory() - base
rec["current_bytes_after_first_call"] = torch.mps.current_allocated_memory()
rows = sorted({0, S // 2, S - 1})
rec["max_abs_err_vs_fp64"] = float((out[0, :, rows].cpu().double() - oracle(q, k, v, rows)).abs().max())
del out
ts = []
for _ in range(3 if S <= 8192 else 2):
    torch.mps.synchronize(); t0 = time.perf_counter(); fn(q, k, v); torch.mps.synchronize()
    ts.append(time.perf_counter() - t0)
rec.update(status="ok", first_call_s=first, s_median=statistics.median(ts), s_all=ts)
print(json.dumps(rec))
