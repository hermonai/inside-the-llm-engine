"""Achievable GPU memory bandwidth on Apple silicon via PyTorch MPS.

read:  y = x.sum()        -> reads N bytes (one scalar written)
copy:  y.copy_(x)         -> reads N bytes, writes N bytes
Median of 20 timed repetitions after 5 warm-ups; synchronizes around each.
"""
import json, statistics, sys, time, platform
import torch

def bench(fn, nbytes, reps=20, warm=5):
    for _ in range(warm):
        fn(); torch.mps.synchronize()
    times = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"median_s": med, "gbps": nbytes / med / 1e9,
            "best_gbps": nbytes / min(times) / 1e9, "reps": reps}

dev = torch.device("mps")
results = {"torch": torch.__version__, "machine": platform.machine(), "tests": {}}
for mib in (256, 512):
    n = mib * 1024 * 1024 // 4
    x = torch.rand(n, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    nb = n * 4
    results["tests"][f"read_sum_{mib}MiB"] = bench(lambda: x.sum(), nb)
    results["tests"][f"copy_{mib}MiB"] = bench(lambda: y.copy_(x), 2 * nb)
    del x, y
    torch.mps.empty_cache()
print(json.dumps(results, indent=1))
