#!/usr/bin/env python3
"""What a CPU/GPU split costs in memory on unified memory: llama-batched-bench (build 8660)
on the Qwen3-8B-architecture model with 0 or 9 layers on the GPU, CPU weights repacked
(the default) or not (--no-repack). Records decode and prefill rates, the model-buffer
lines of the log, and the process's peak resident size from /usr/bin/time -l. One round (time-boxed on a
thrashing machine), each run started when the one-minute load is below 6 or after 5 minutes."""
import json, os, re, subprocess, sys, time
QWEN8 = os.path.expanduser("~/.ollama/models/blobs/sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def vmstat():
    r = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(k + r":\s+(\d+)", r).group(1))
    return {"pageouts": get("Pageouts"), "swapins": get("Swapins"), "swapouts": get("Swapouts"),
            "pageins": get("Pageins"), "compressions": get("Compressions"),
            "decompressions": get("Decompressions")}


def wait(limit=6.0, max_wait=300):
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(10)
    return round(time.time() - t0), os.getloadavg()[0]

configs = [(9, False), (9, True), ("ffn", False), ("ffn", True), (0, True), (0, False)]
res = {"runs": []}
for rnd, order in enumerate((configs,), start=1):
    for ngl, repack in order:
        waited, l1 = wait()
        extra = ["-ngl", "99", "-ot", "ffn_.*=CPU"] if ngl == "ffn" else ["-ngl", str(ngl)]
        cmd = ["/usr/bin/time", "-l", "llama-batched-bench", "-m", QWEN8, *extra, "-fa", "on",
               "-t", "4", "-c", "2048", "-npp", "128", "-ntg", "32", "-npl", "1"] + ([] if repack else ["--no-repack"])
        v0 = vmstat()
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        v1 = vmstat()
        tag = f"r{rnd}-ngl{ngl}-{'repack' if repack else 'norepack'}"
        text = (r.stdout + "\n" + r.stderr).replace(QWEN8, "<qwen3-8b.gguf>")
        with open(os.path.join(out, f"repack-{tag}.txt"), "w") as f:
            f.write(text)
        m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
        pf = re.search(r"(\d+)\s+peak memory footprint", r.stderr)
        rows = [l for l in r.stdout.splitlines() if l.strip().startswith("|") and re.search(r"\|\s*\d", l)]
        res["runs"].append({"round": rnd, "ngl": ngl, "repack": repack, "rc": r.returncode, "waited_s": waited,
                            "load1_start": round(l1, 2), "seconds": round(time.perf_counter() - t0, 1),
                            "max_rss_bytes": int(m.group(1)) if m else None,
                            "peak_footprint_bytes": int(pf.group(1)) if pf else None,
                            "table": rows,
                            "vm_delta_pages": {k: v1[k] - v0[k] for k in v0},
                            "splits": [l for l in text.splitlines() if "graph splits" in l],
                            "buffers": [l for l in text.splitlines() if "model buffer size" in l]})
        print(json.dumps(res["runs"][-1]), flush=True)
json.dump(res, open(os.path.join(out, "repack-round1.json"), "w"), indent=1)
print("done")
