#!/usr/bin/env python3
"""Part V follow-up probe for 'Inside the LLM Engine' on an Apple M1.

  python3 part5b_probe.py hybrid_prefix <outdir>

hybrid_prefix  Prefix reuse with a pure-attention model (Llama 3.2 3B) and a hybrid
               model (Qwen3.5-9B architecture: 24 Gated DeltaNet + 8 attention
               layers) in llama-server (build 8660, llama.cpp d006858). The same
               five prompts go to each server in the same order; the server's own
               timings report how many prompt tokens it had to process, and its
               log records the recurrent-state checkpoints it created and restored.
               The hybrid runs twice: with the default checkpoint policy, and with
               a checkpoint every 512 prompt tokens.
"""
import json
import os
import re
import sys
import time

import serve_probe as sp

BLOBS = os.path.expanduser("~/.ollama/models/blobs")
LLAMA = os.path.join(BLOBS, "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
HYBRID = os.path.join(BLOBS, "sha256-2bfb097d01f8633cd5fcf6cc3aeeeded7c7211b41e9d241e871d7d595a42fc49")
PORT = 18196


def prompts():
    with open(sp.DOC, encoding="utf-8") as f:
        words = f.read().split()
    doc = " ".join(words[:2400])
    third = len(doc) // 3
    cut = doc.rfind(" ", 0, third)
    edited = doc[:cut] + " (an edited sentence appears here)" + doc[cut:]
    q1 = "\n\nQuestion: what is the main subject of the text above? Answer in one word:"
    q2 = "\n\nQuestion: name one technical term used in the text above. Answer in one word:"
    return [
        ("P1 doc+q1 (cold)", doc + q1),
        ("P2 doc+q2 (new question)", doc + q2),
        ("P3 doc+q1 (earlier prompt again)", doc + q1),
        ("P4 edited doc+q1 (change at one third)", edited + q1),
        ("P5 doc+q1 (after the edit)", doc + q1),
    ]


def ntok(text, port):
    st, d, _ = sp.post("/tokenize", {"content": text, "add_special": True}, port=port)
    return len(d["tokens"])


def run(out, tag, model, extra):
    srv = sp.Server(out, f"hybrid_prefix-{tag}", ["-np", "1", "-c", "16384", "-fa", "on", *extra],
                    port=PORT, model=model)
    rows = []
    try:
        for label, p in prompts():
            n = ntok(p, PORT)
            st, d, dt = sp.post("/completion", {"prompt": p, "n_predict": 8, "temperature": 0,
                                                "cache_prompt": True}, port=PORT)
            assert st == 200, (st, d)
            tm = d.get("timings", {})
            rows.append({"prompt": label, "prompt_tokens": n, "processed": tm.get("prompt_n"),
                         "cached": d.get("tokens_cached"), "prompt_ms": tm.get("prompt_ms"),
                         "elapsed_s": round(dt, 2), "content": d.get("content")})
            time.sleep(0.5)
    finally:
        srv.stop()
    with open(srv.log_path) as f:
        events = [l.rstrip() for l in f if re.search(
            r"checkpoint|forcing full prompt|RS buffer|KV buffer size", l)]
    return {"rows": rows, "log_events": events}


def exp_hybrid_prefix(out):
    return {
        "llama3.2-3b": run(out, "llama", LLAMA, []),
        "hybrid default": run(out, "hybrid-default", HYBRID, []),
        "hybrid every 512": run(out, "hybrid-every512", HYBRID, ["--checkpoint-every-n-tokens", "512"]),
    }


def exp_hybrid_prefix_b512(out):
    """Follow-up: with the default logical batch of 2,048 tokens, a checkpoint can be
    made only once per batch, so --checkpoint-every-n-tokens 512 produced checkpoints
    at 2,048 and 4,096 and the edit at about 1,570 tokens still forced a full
    re-processing. A 512-token batch lets the checkpoints fall every 512 tokens."""
    return {"hybrid every 512, batch 512": run(out, "hybrid-every512-b512", HYBRID,
                                               ["--checkpoint-every-n-tokens", "512", "-b", "512"])}


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    result = {"hybrid_prefix": exp_hybrid_prefix, "hybrid_prefix_b512": exp_hybrid_prefix_b512}[name](out)
    result["_meta"] = {"experiment": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                       "seconds": round(time.time() - t0, 1), "loadavg_end": os.getloadavg()}
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(name, "ok", result["_meta"])
