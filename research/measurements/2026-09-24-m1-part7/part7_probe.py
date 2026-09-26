#!/usr/bin/env python3
"""Part VII probes for 'Inside the LLM Engine' on an Apple M1 (unified memory).

  python3 part7_probe.py <experiment> <outdir>

hog         (helper) copy a 256 MiB array into another in a loop until a stop file
            appears, then write the bytes copied and seconds to a JSON file.
contention  Llama 3.2 3B decode (llama-bench, Homebrew build 8660, -p 0 -n 128 -r 3)
            alone and while 1, 2 or 4 CPU processes copy memory, in the order
            0, 1, 0, 2, 0, 4, 0 hog processes; plus the hogs' own bandwidth alone.
mmapload    llama-server with the Qwen3-8B-architecture model, memory-mapped (the
            default) against --no-mmap, alternating: seconds until /health answers,
            the first short completion, the model-buffer lines of the log, and the
            server's resident memory.
"""
import json
import os
import subprocess
import sys
import time

import serve_probe as sp

BLOBS = os.path.expanduser("~/.ollama/models/blobs")
LLAMA = os.path.join(BLOBS, "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
QWEN8 = os.path.join(BLOBS, "sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d")
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 18301


def hog(out_json, stop_file):
    import numpy as np
    n = 256 << 20
    src = np.ones(n, dtype=np.uint8)
    dst = np.empty_like(src)
    copied = 0
    t0 = time.perf_counter()
    while not os.path.exists(stop_file):
        np.copyto(dst, src)
        copied += 2 * n                      # read n bytes and write n bytes
    t = time.perf_counter() - t0
    with open(out_json, "w") as f:
        json.dump({"bytes_moved": copied, "seconds": t, "GBps": copied / t / 1e9}, f)


def start_hogs(k, out, tag):
    stop = os.path.join(out, f"stop-{tag}")
    if os.path.exists(stop):
        os.remove(stop)
    procs, files = [], []
    for i in range(k):
        fj = os.path.join(out, f"hog-{tag}-{i}.json")
        files.append(fj)
        procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, "part7_probe.py"), "hog", fj, stop]))
    time.sleep(2.0 if k else 0)
    return stop, procs, files


def stop_hogs(stop, procs, files):
    open(stop, "w").close()
    for p in procs:
        p.wait()
    res = []
    for fj in files:
        with open(fj) as f:
            res.append(json.load(f))
    os.remove(stop)
    return res


def bench_decode():
    r = subprocess.run(["llama-bench", "-m", LLAMA, "-fa", "1", "-p", "0", "-n", "128", "-r", "3", "-o", "json"],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)[0]
    return {"avg_ts": d["avg_ts"], "samples_ts": d["samples_ts"]}


def exp_contention(out):
    res = {"alone": [], "runs": []}
    for k in (1, 2, 4):                                  # hog bandwidth without llama-bench
        stop, procs, files = start_hogs(k, out, f"alone{k}")
        time.sleep(10)
        res["alone"].append({"hogs": k, "each": stop_hogs(stop, procs, files)})
    for i, k in enumerate((0, 1, 0, 2, 0, 4, 0)):
        stop, procs, files = start_hogs(k, out, f"run{i}")
        t0 = time.perf_counter()
        dec = bench_decode()
        t = time.perf_counter() - t0
        hogs = stop_hogs(stop, procs, files) if k else []
        res["runs"].append({"order": i, "hogs": k, "decode": dec, "bench_s": round(t, 2),
                            "hog_GBps_each": [round(h["GBps"], 2) for h in hogs]})
    return res


def rss_kib(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def exp_mmapload(out):
    res = {"runs": []}
    for i, mode in enumerate(("mmap", "nommap", "mmap", "nommap")):
        extra = ["--no-mmap"] if mode == "nommap" else []
        t0 = time.perf_counter()
        srv = sp.Server(out, f"mmapload-{i}-{mode}", ["-np", "1", "-c", "4096", *extra], port=PORT, model=QWEN8)
        t_ready = time.perf_counter() - t0
        try:
            st, d, dt = sp.post("/completion", {"prompt": "The capital of France is", "n_predict": 8,
                                                "temperature": 0}, port=PORT)
            rss = rss_kib(srv.p.pid)
        finally:
            srv.stop()
        with open(srv.log_path) as f:
            lines = [l.rstrip() for l in f if "buffer size" in l or "mmap" in l.lower()]
        res["runs"].append({"order": i, "mode": mode, "seconds_to_ready": round(t_ready, 2),
                            "first_completion_s": round(dt, 3), "content": d.get("content"),
                            "rss_kib_after": rss, "log": lines})
    return res


def exp_gpucopy(out):
    """GPU copy bandwidth (PyTorch MPS, 256 MiB tensors) alone and while 1, 2 or 4 CPU
    processes copy memory: the shared-bandwidth question without an inference engine."""
    import torch
    n = 256 << 20
    a = torch.ones(n, dtype=torch.uint8, device="mps")
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.mps.synchronize()

    def gbps(seconds=6.0):
        reps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for _ in range(10):
                b.copy_(a)
            torch.mps.synchronize()
            reps += 10
        return 2 * n * reps / (time.perf_counter() - t0) / 1e9

    res = {"runs": []}
    for i, k in enumerate((0, 1, 0, 2, 0, 4, 0)):
        stop, procs, files = start_hogs(k, out, f"gpu{i}")
        g = gbps()
        hogs = stop_hogs(stop, procs, files) if k else []
        res["runs"].append({"order": i, "hogs": k, "gpu_GBps": round(g, 2),
                            "hog_GBps_each": [round(h["GBps"], 2) for h in hogs],
                            "hog_GBps_total": round(sum(h["GBps"] for h in hogs), 2)})
    return res


def bench(args, tag, out):
    """One llama-bench invocation on the Qwen3-8B-architecture model; JSON rows kept whole."""
    cmd = ["llama-bench", "-m", QWEN8, "-fa", "1", "-o", "json", *args]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(out, f"bench-{tag}.stderr.txt"), "w") as f:
        f.write(r.stderr)
    rows = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else []
    keep = ("n_gpu_layers", "n_threads", "no_kv_offload", "no_op_offload", "tensor_buft_overrides",
            "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts", "samples_ts", "model_size", "model_n_params")
    return {"tag": tag, "cmd": " ".join(cmd[1:]).replace(QWEN8, "<qwen3-8b.gguf>"), "rc": r.returncode,
            "seconds": round(time.perf_counter() - t0, 1), "load1_start": os.getloadavg()[0],
            "rows": [{k: row.get(k) for k in keep} for row in rows]}


def exp_offload(out):
    """Qwen3-8B (36 layers) with a fraction of its layers on the GPU, FFN tensors on the CPU,
    the KV cache on the CPU, and CPU-only; two passes, the second in reverse order."""
    configs = [
        ("layers", ["-ngl", "99,27,18,9,0", "-p", "512", "-n", "64", "-r", "2"]),
        ("cpu-noopoffload", ["-ngl", "0", "-nopo", "1", "-p", "512", "-n", "0", "-r", "2"]),
        ("ffn-on-cpu", ["-ngl", "99", "-ot", "ffn_.*=CPU", "-nopo", "0,1", "-p", "512", "-n", "64", "-r", "2"]),
        ("kv-on-cpu", ["-ngl", "99", "-nkvo", "0,1", "-d", "0,4096", "-p", "512", "-n", "64", "-r", "2"]),
        ("cpu-8threads", ["-ngl", "0", "-t", "8", "-p", "0", "-n", "64", "-r", "2"]),
    ]
    res = {"passes": []}
    for pas, order in enumerate((configs, configs[::-1])):
        runs = []
        for tag, args in order:
            wait_load()
            runs.append(bench(args, f"p{pas}-{tag}", out))
        res["passes"].append(runs)
    return res


def wait_load(limit=6.0, max_wait=1200):
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(15)


if __name__ == "__main__":
    if sys.argv[1] == "hog":
        hog(sys.argv[2], sys.argv[3])
        sys.exit(0)
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    result = {"contention": exp_contention, "mmapload": exp_mmapload, "offload": exp_offload,
              "gpucopy": exp_gpucopy}[name](out)
    result["_meta"] = {"experiment": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                       "seconds": round(time.time() - t0, 1), "loadavg_end": os.getloadavg()}
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(name, "ok", result["_meta"])
