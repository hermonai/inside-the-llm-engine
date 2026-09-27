"""Chapter 7: time of A (4096 x 4096) @ B (4096 x N) on the Apple M1 GPU through
PyTorch's MPS backend, for every N from 1 to 32 and some larger ones, in float16
and float32. Every product is checked against a float64 NumPy oracle on 8 rows
before it is timed; each time is the median of 20 after 5 warm-ups."""
import json, statistics, time, sys
import numpy as np, torch

def timed(fn, reps=20, warm=5):
    for _ in range(warm):
        fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

M = K = 4096
dev = torch.device("mps")
out = {"torch": torch.__version__, "rows": []}
for dtype in (torch.float16, torch.float32):
    torch.manual_seed(0)
    A = torch.randn(M, K, device=dev, dtype=dtype) / 64
    for N in list(range(1, 33)) + [40, 48, 64, 96, 128, 256, 512]:
        B = torch.randn(K, N, device=dev, dtype=dtype) / 64
        C = A @ B
        ref = A[:8].cpu().double().numpy() @ B.cpu().double().numpy()
        err = float(np.max(np.abs(C[:8].cpu().double().numpy() - ref)))
        tol = 5e-3 if dtype == torch.float16 else 1e-4
        if err > tol:
            sys.exit(f"oracle failed: {dtype} N={N} max abs error {err}")
        t = timed(lambda: A @ B)
        out["rows"].append({"dtype": str(dtype).replace("torch.", ""), "N": N, "ms": t * 1e3,
                            "gflops": 2 * M * K * N / t / 1e9,
                            "weight_gbps": M * K * A.element_size() / t / 1e9, "max_abs_err": err})
    del A, B, C
    torch.mps.empty_cache()
print(json.dumps(out, indent=1))
