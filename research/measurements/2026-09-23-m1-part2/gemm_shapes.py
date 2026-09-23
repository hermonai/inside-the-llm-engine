"""Ch 7: matmul throughput vs shape on Apple M1 (MPS GPU and Accelerate CPU).
Every timed product is first checked against a float64 NumPy oracle on a sample."""
import json, statistics, time
import numpy as np, torch

def t_mps(fn, reps=10, warm=3):
    for _ in range(warm): fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def t_cpu(fn, reps=5, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

import io, contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    np.show_config()
res = {"torch": torch.__version__, "numpy": np.__version__, "numpy_blas": [l.strip() for l in _buf.getvalue().splitlines() if "name" in l.lower()][:6], "mps": [], "cpu": [], "same_flops": {}}
M = K = 4096
dev = torch.device("mps")
for dtype in (torch.float16, torch.float32):
    A = torch.randn(M, K, device=dev, dtype=dtype) / 64
    for N in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096):
        B = torch.randn(K, N, device=dev, dtype=dtype) / 64
        C = A @ B
        ref = A[:8].cpu().double().numpy() @ B.cpu().double().numpy()
        err = float(np.max(np.abs(C[:8].cpu().double().numpy() - ref)))
        t = t_mps(lambda: A @ B)
        flops = 2 * M * K * N
        byts = (M * K + K * N + M * N) * A.element_size()
        res["mps"].append({"dtype": str(dtype), "N": N, "s": t, "gflops": flops / t / 1e9,
                           "gbps_compulsory": byts / t / 1e9, "max_abs_err_8rows": err})
    del A, B, C
    torch.mps.empty_cache()
An = np.random.randn(M, K).astype(np.float32) / 64
for N in (1, 8, 64, 512, 4096):
    Bn = np.random.randn(K, N).astype(np.float32) / 64
    Cn = An @ Bn
    err = float(np.max(np.abs(Cn[:8].astype(np.float64) - An[:8].astype(np.float64) @ Bn.astype(np.float64))))
    t = t_cpu(lambda: An @ Bn)
    res["cpu"].append({"dtype": "float32", "N": N, "s": t, "gflops": 2 * M * K * N / t / 1e9,
                       "max_abs_err_8rows": err})
# Same FLOPs: one 4096^3 GEMM vs 4096 separate GEMVs (timed on 512 and scaled).
A = torch.randn(M, K, device=dev, dtype=torch.float16) / 64
B = torch.randn(K, 4096, device=dev, dtype=torch.float16) / 64
cols = [B[:, i:i+1].contiguous() for i in range(512)]
def gemvs():
    for c in cols: A @ c
Cg = A @ B
chk = float((Cg[:, :1].float() - (A @ cols[0]).float()).abs().max().cpu())
t_gemm = t_mps(lambda: A @ B)
t_512 = t_mps(gemvs, reps=5, warm=2)
res["same_flops"] = {"gemm_4096_s": t_gemm, "gemv_x512_s": t_512, "gemv_x4096_s_scaled": t_512 * 8,
                     "ratio": t_512 * 8 / t_gemm, "gemm_vs_gemv_col0_max_abs_diff": chk}
print(json.dumps(res, indent=1))
