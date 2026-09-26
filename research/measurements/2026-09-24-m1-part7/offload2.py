#!/usr/bin/env python3
"""Offload re-run: Qwen3-8B decode (tg64 x3) at -ngl 0/9/18/27/36/99, one llama-bench
invocation per configuration, three rounds in different orders, each started only when
the one-minute load average is below 4. Plus one verbose run per split to record the
graph splits and buffer placement."""
import json, os, subprocess, sys, time
QWEN8 = os.path.expanduser("~/.ollama/models/blobs/sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def wait(limit=4.0, max_wait=1800):
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(10)
    return round(time.time() - t0), os.getloadavg()[0]

res = {"verbose": {}, "rounds": []}
for ngl in (9, 0, 99):
    waited, l1 = wait()
    r = subprocess.run(["llama-bench", "-m", QWEN8, "-fa", "1", "-ngl", str(ngl), "-p", "0", "-n", "8", "-r", "1", "-v"],
                       capture_output=True, text=True)
    path = os.path.join(out, f"offload2-verbose-ngl{ngl}.txt")
    with open(path, "w") as f:
        f.write(r.stderr.replace(QWEN8, "<qwen3-8b.gguf>") + "\n" + r.stdout.replace(QWEN8, "<qwen3-8b.gguf>"))
    res["verbose"][ngl] = [l for l in r.stderr.splitlines() if "graph splits" in l or "buffer size" in l or "offloaded" in l]
orders = [(0, 9, 18, 27, 36, 99), (99, 36, 27, 18, 9, 0), (18, 99, 0, 36, 9, 27)]
for ri, order in enumerate(orders):
    for ngl in order:
        waited, l1 = wait()
        t0 = time.perf_counter()
        r = subprocess.run(["llama-bench", "-m", QWEN8, "-fa", "1", "-ngl", str(ngl), "-p", "0", "-n", "64", "-r", "3", "-o", "json"],
                           capture_output=True, text=True)
        row = json.loads(r.stdout)[0] if r.returncode == 0 else {}
        res["rounds"].append({"round": ri, "ngl": ngl, "waited_s": waited, "load1_start": round(l1, 2),
                              "load1_end": round(os.getloadavg()[0], 2), "seconds": round(time.perf_counter() - t0, 1),
                              "avg_ts": row.get("avg_ts"), "samples_ts": row.get("samples_ts")})
        print(res["rounds"][-1], flush=True)
res["_meta"] = {"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(res, open(os.path.join(out, "offload2.json"), "w"), indent=1)
print("done")
