#!/usr/bin/env python3
"""Chapter 41's opening: one prompt, one model file, three engines on the M1.

llama.cpp's server (Homebrew build 8660), Ollama (its own runner over the same GGUF
blob) and Hermon (its native path over its vendored llama.cpp), each greedy for 128
tokens on the same raw prompt, three timed runs after a warm-up. Records each
engine's own timing report and the generated text.

    python3 engines3.py <outdir> <hermon-binary>
"""
import http.client
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_probe as sp  # noqa: E402

BLOB = os.path.expanduser("~/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
N = 128
text = re.sub(r"\s+", " ", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc.txt"),
                               encoding="utf-8").read())
i = text.find(". ", 30000) + 2
PROMPT = text[i:i + 600].rsplit(" ", 1)[0]


def post(port, path, body, timeout=600):
    c = http.client.HTTPConnection("localhost", port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    return d, time.perf_counter() - t0


def run_llamacpp(out):
    res = []
    srv = sp.Server(out, "engines3-llamacpp", ["-np", "1", "-c", "4096"], port=18501, model=BLOB)
    try:
        body = {"prompt": PROMPT, "n_predict": N, "temperature": 0, "cache_prompt": False, "ignore_eos": True}
        post(18501, "/completion", body)
        for _ in range(3):
            d, wall = post(18501, "/completion", body)
            t = d["timings"]
            res.append({"wall_s": round(wall, 3), "prompt_tokens": t["prompt_n"], "prompt_ms": t["prompt_ms"],
                        "decode_tok_s": t["predicted_per_second"], "tokens": t["predicted_n"], "text": d["content"]})
    finally:
        srv.stop()
    return res


def run_ollama():
    res = []
    body = {"model": "llama3.2:3b", "prompt": PROMPT, "raw": True, "stream": False,
            "options": {"temperature": 0, "num_predict": N, "seed": 1, "num_ctx": 4096}}
    post(11434, "/api/generate", body)
    for _ in range(3):
        d, wall = post(11434, "/api/generate", body)
        res.append({"wall_s": round(wall, 3), "prompt_tokens": d.get("prompt_eval_count"),
                    "prompt_ms": d.get("prompt_eval_duration", 0) / 1e6,
                    "decode_tok_s": d["eval_count"] / (d["eval_duration"] / 1e9), "tokens": d["eval_count"],
                    "load_ms": d.get("load_duration", 0) / 1e6, "text": d["response"]})
    post(11434, "/api/generate", {"model": "llama3.2:3b", "prompt": "", "keep_alive": 0, "stream": False})
    return res


def run_hermon(binary):
    res = []
    for _ in range(4):
        t0 = time.perf_counter()
        r = subprocess.run([binary, "native", BLOB, PROMPT, "--max-tokens", str(N), "--temperature", "0"],
                           capture_output=True, text=True, timeout=900)
        wall = time.perf_counter() - t0
        m = re.search(r"\[([\d.]+)s wall \| (\d+) tokens in / (\d+) out @ ([\d.]+) tok/s", r.stderr)
        res.append({"wall_s": round(wall, 3), "rc": r.returncode, "reported": m.group(0) if m else None,
                    "prompt_tokens": int(m.group(2)) if m else None, "tokens": int(m.group(3)) if m else None,
                    "decode_tok_s": float(m.group(4)) if m else None, "text": r.stdout.strip(),
                    "stderr_tail": r.stderr[-1500:]})
    return res[1:]                      # the first run warms the page cache and Metal pipelines


if __name__ == "__main__":
    out, hermon = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    result = {"prompt": PROMPT, "n_predict": N, "llamacpp": run_llamacpp(out), "ollama": run_ollama(),
              "hermon": run_hermon(hermon)}
    v = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    result["_meta"] = {"ollama_version": v, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(result, open(os.path.join(out, "engines3.json"), "w"), indent=1)
    for k in ("llamacpp", "ollama", "hermon"):
        for r in result[k]:
            print(k, r.get("decode_tok_s"), r.get("prompt_ms"), repr(r.get("text", "")[:60]))
