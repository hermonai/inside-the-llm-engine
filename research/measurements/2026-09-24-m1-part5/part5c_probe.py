#!/usr/bin/env python3
"""Part V rerun for 'Inside the LLM Engine' on an Apple M1.

  python3 part5c_probe.py ollama_ctx2 <outdir>

ollama_ctx2  Gemma 4 E4B (35 of 42 layers with a 512-token sliding window) and
             Llama 3.2 3B (all layers global) in Ollama 0.34.3 with a 16,384-token
             context: decode and prefill speed after prompts of about 0, 3,000 and
             12,000 tokens, and the memory Ollama reports for each loaded model.
             Replaces the first attempt (ollama_ctx), whose raw-mode prompts made
             Gemma stop after one token: prompts now go through each model's chat
             template, Gemma's thinking is off, and every repetition starts with a
             different first line so that Ollama cannot reuse a cached prompt.
"""
import http.client
import json
import os
import sys
import time

import serve_probe as sp


def ollama(path, body, timeout=3600):
    c = http.client.HTTPConnection("localhost", 11434, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    d["_wall_s"] = time.perf_counter() - t0
    d["_status"] = r.status
    return d


def ollama_ps():
    c = http.client.HTTPConnection("localhost", 11434, timeout=30)
    c.request("GET", "/api/ps")
    d = json.loads(c.getresponse().read())
    c.close()
    return [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in d.get("models", [])]


def summarize(d):
    pe = (d.get("prompt_eval_duration") or 0) / 1e9
    ev = (d.get("eval_duration") or 0) / 1e9
    return {"status": d.get("_status"), "error": d.get("error"),
            "prompt_eval_count": d.get("prompt_eval_count"), "prompt_eval_s": round(pe, 3),
            "eval_count": d.get("eval_count"), "eval_s": round(ev, 3),
            "prefill_tok_s": round((d.get("prompt_eval_count") or 0) / pe, 2) if pe else None,
            "decode_tok_s": round((d.get("eval_count") or 0) / ev, 2) if ev else None,
            "wall_s": round(d["_wall_s"], 2), "done_reason": d.get("done_reason"),
            "response_head": (d.get("response") or "")[:120]}


def exp_ollama_ctx2(out):
    with open(sp.DOC, encoding="utf-8") as f:
        words = f.read().split()
    task = "\n\nSummarize the text above in one sentence."
    prompts = {"short": "Say hello in five words.",
               "~3k": " ".join(words[:1400]) + task,
               "~12k": " ".join(words[:5600]) + task}
    res = {"runs": []}
    for model in ("gemma4", "llama3.2:3b"):
        extra = {"think": False} if model == "gemma4" else {}
        opts = {"temperature": 0, "num_predict": 64, "num_ctx": 16384}
        warm = ollama("/api/generate", {"model": model, "prompt": "Hi.", "stream": False, "keep_alive": "10m",
                                        "options": {"num_predict": 4, "num_ctx": 16384}, **extra})
        res["runs"].append({"model": model, "warmup": summarize(warm), "ps_after_load": ollama_ps()})
        for label, p in prompts.items():
            for rep in range(2):
                body = {"model": model, "prompt": f"Request {rep + 1} of 2.\n" + p, "stream": False,
                        "keep_alive": "10m", "options": opts, **extra}
                d = ollama("/api/generate", body)
                res["runs"].append({"model": model, "prompt": label, "rep": rep, **summarize(d)})
        res["runs"].append({"model": model, "ps_before_unload": ollama_ps()})
        ollama("/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    res["loaded_after"] = ollama_ps()
    return res


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    result = {"ollama_ctx2": exp_ollama_ctx2}[name](out)
    result["_meta"] = {"experiment": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                       "seconds": round(time.time() - t0, 1), "loadavg_end": os.getloadavg()}
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(name, "ok", result["_meta"])
