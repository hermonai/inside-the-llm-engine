#!/usr/bin/env python3
"""Summarize test-backend-ops SQL output for MUL_MAT_ID at Qwen3-30B-A3B's expert
shape (128 experts, 8 used, 2,048 -> 768): time per call against batch, with the
expected expert union under uniform routing and the implied weight bandwidth."""
import glob, re, statistics as st, sys
from collections import defaultdict

out = sys.argv[1]
E, K, M, KDIM = 128, 8, 768, 2048
BYTES = {"q4_K": M * KDIM * 144 // 256, "f16": M * KDIM * 2}   # bytes per expert matrix
rows = defaultdict(list)
for f in sorted(glob.glob(f"{out}/moe-*-[0-9].sql")):
    for line in open(f):
        if not line.startswith("INSERT"):
            continue
        vals = re.findall(r"'([^']*)'", line.split("VALUES", 1)[1])
        params, time_us, n_runs = vals[4], float(vals[9]), int(vals[13])
        t = re.search(r"type_a=(\w+)", params).group(1)
        n = int(re.search(r",n=(\d+),", params).group(1))
        rows[(t, n)].append((time_us, n_runs))
print(f"{'type':5} {'batch':>5} {'union':>6} {'time us (3 runs)':>30} {'mean':>9} {'us/token':>9} {'GB/s':>6}")
for (t, n) in sorted(rows, key=lambda x: (x[0], x[1])):
    ts = [r[0] for r in rows[(t, n)]]
    u = E * (1 - (1 - K / E) ** n)
    mean = st.mean(ts)
    gbs = u * BYTES[t] / (mean * 1e-6) / 1e9
    print(f"{t:5} {n:>5} {u:>6.1f} {', '.join(f'{x:.1f}' for x in ts):>30} {mean:>9.1f} {mean / n:>9.2f} {gbs:>6.1f}")
print("bytes per expert matrix:", BYTES)
