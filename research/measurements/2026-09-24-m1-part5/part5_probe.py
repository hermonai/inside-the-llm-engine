#!/usr/bin/env python3
"""Part V probes for 'Inside the LLM Engine' on an Apple M1.

  python3 part5_probe.py <experiment> <outdir>

kvsizes    Start llama-server (build 8660, llama.cpp d006858) with a 16,384-token
           context for four local models and record the KV-cache and recurrent-
           state allocations it logs.
reasoning  DeepSeek-R1-0528-Qwen3-8B on eight original arithmetic problems with
           an unlimited thinking budget, a budget of 256 tokens, and none; answers
           are checked against the known results.
ollama_ctx Gemma 4 (sliding-window layers) and Llama 3.2 3B served by Ollama at
           prompts of about 0, 2,000 and 6,000 tokens: decode and prefill speeds.
vision     Gemma 4 in Ollama with no image and with images of three sizes, plus a
           repeated image: prompt tokens and prompt-processing time.
"""
import base64
import http.client
import json
import os
import re
import subprocess
import sys
import time

import serve_probe as sp

BLOBS = os.path.expanduser("~/.ollama/models/blobs")
MODELS = {
    "llama3.2-3b": os.path.join(BLOBS, "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"),
    "qwen2.5-coder-3b": os.path.join(BLOBS, "sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba"),
    "qwen3-8b (deepseek-r1)": os.path.join(BLOBS, "sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d"),
    "qwen3.5-9b (hybrid)": os.path.join(BLOBS, "sha256-2bfb097d01f8633cd5fcf6cc3aeeeded7c7211b41e9d241e871d7d595a42fc49"),
}
PORT = 18195

PROBLEMS = [
    ("A train leaves at 3:15 pm and arrives at 5:40 pm the same day. How many minutes does the trip take?", 145),
    ("What is 17 multiplied by 23?", 391),
    ("If 3 pencils cost 45 cents, how many cents do 7 pencils cost?", 105),
    ("A rectangle has a perimeter of 30 and a length of 9. What is its area?", 54),
    ("How many positive divisors does 36 have?", 9),
    ("Sam has twice as many apples as Lee. Together they have 27 apples. How many apples does Sam have?", 18),
    ("What is the remainder when 2 to the power 10 is divided by 7?", 2),
    ("An 80-dollar item is discounted by 25 percent, then 10 percent tax is added to the discounted price. "
     "What is the final price in dollars?", 66),
]
SUFFIX = " End your reply with a line of the form 'Answer: N', where N is the number."


def exp_kvsizes(out):
    res = {}
    pat = re.compile(r"(llama_kv_cache.*size = .*|llama_memory_recurrent.*|.*KV buffer size.*|.*RS buffer size.*|"
                     r"llama_kv_cache_iswa.*|.*n_swa.*|.*n_ctx_seq.*|.*SWA.*cache.*)")
    for name, path in MODELS.items():
        tag = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
        srv = sp.Server(out, f"kvsizes-{tag}", ["-np", "1", "-c", "16384", "-fa", "on"], port=PORT, model=path)
        try:
            pass
        finally:
            srv.stop()
        with open(srv.log_path) as f:
            lines = [l.rstrip() for l in f if pat.search(l)]
        res[name] = lines
    return res


def chat(content, budget, max_tokens=2048):
    body = {"messages": [{"role": "user", "content": content + SUFFIX}], "temperature": 0, "seed": 1,
            "max_tokens": max_tokens, "cache_prompt": False}
    if budget is not None:
        body["thinking_budget_tokens"] = budget
    st, d, dt = sp.post("/v1/chat/completions", body, port=PORT)
    assert st == 200, (st, d)
    msg = d["choices"][0]["message"]
    tm = d.get("timings", {})
    return msg.get("content") or "", msg.get("reasoning_content") or "", d.get("usage", {}), tm, dt


def ntok(text):
    if not text:
        return 0
    st, d, _ = sp.post("/tokenize", {"content": text, "add_special": False}, port=PORT)
    return len(d["tokens"])


def exp_reasoning(out):
    res = {"runs": []}
    srv = sp.Server(out, "reasoning", ["-np", "1", "-c", "8192", "--reasoning-format", "deepseek"], port=PORT,
                    model=MODELS["qwen3-8b (deepseek-r1)"])
    try:
        chat("Say hi.", 0, max_tokens=16)
        for budget in (-1, 256, 0):
            for i, (q, ans) in enumerate(PROBLEMS):
                content, reasoning, usage, tm, dt = chat(q, budget)
                m = re.findall(r"Answer:\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)", content)
                got = float(m[-1].replace(",", "")) if m else None
                res["runs"].append({
                    "budget": budget, "problem": i, "expected": ans, "got": got,
                    "correct": got is not None and abs(got - ans) < 1e-9,
                    "reasoning_tokens": ntok(reasoning), "answer_tokens": ntok(content),
                    "completion_tokens": usage.get("completion_tokens"), "predicted_ms": tm.get("predicted_ms"),
                    "prompt_ms": tm.get("prompt_ms"), "elapsed_s": round(dt, 2),
                    "finish_hit_limit": (usage.get("completion_tokens") or 0) >= 2048,
                    "content_tail": content[-160:], "reasoning_head": reasoning[:200]})
    finally:
        srv.stop()
    return res


def ollama(path, body, timeout=1800):
    c = http.client.HTTPConnection("localhost", 11434, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    d["_wall_s"] = time.perf_counter() - t0
    return d


def ollama_ps():
    c = http.client.HTTPConnection("localhost", 11434, timeout=30)
    c.request("GET", "/api/ps")
    d = json.loads(c.getresponse().read())
    c.close()
    return [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in d.get("models", [])]


def summarize_gen(d):
    return {"prompt_eval_count": d.get("prompt_eval_count"),
            "prompt_eval_s": (d.get("prompt_eval_duration") or 0) / 1e9,
            "eval_count": d.get("eval_count"), "eval_s": (d.get("eval_duration") or 0) / 1e9,
            "load_s": (d.get("load_duration") or 0) / 1e9, "wall_s": round(d["_wall_s"], 2),
            "decode_tok_s": (d.get("eval_count") or 0) / max(1e-9, (d.get("eval_duration") or 0) / 1e9),
            "prefill_tok_s": (d.get("prompt_eval_count") or 0) / max(1e-9, (d.get("prompt_eval_duration") or 0) / 1e9),
            "response_head": (d.get("response") or "")[:120]}


def exp_ollama_ctx(out):
    res = {"loaded_before": ollama_ps(), "runs": []}
    with open(sp.DOC, encoding="utf-8") as f:
        words = f.read().split()
    prompts = {"short": "Say hello in five words.",
               "~2k": " ".join(words[:1400]) + "\n\nSummarize the text above in one sentence.",
               "~6k": " ".join(words[:4200]) + "\n\nSummarize the text above in one sentence."}
    for model in ("gemma4", "llama3.2:3b"):
        opts = {"temperature": 0, "num_predict": 64, "num_ctx": 8192}
        ollama("/api/generate", {"model": model, "prompt": "Hi.", "stream": False, "keep_alive": "10m",
                                 "options": {"num_predict": 4, "num_ctx": 8192}})
        res["runs"].append({"model": model, "ps_after_load": ollama_ps()})
        for label, p in prompts.items():
            for rep in range(2):
                d = ollama("/api/generate", {"model": model, "prompt": p, "stream": False, "keep_alive": "10m",
                                             "raw": True, "options": opts})
                res["runs"].append({"model": model, "prompt": label, "rep": rep, **summarize_gen(d)})
        ollama("/api/generate", {"model": model, "prompt": "", "keep_alive": 0})   # unload
    res["loaded_after"] = ollama_ps()
    return res


def make_images(out):
    """Render page 1 of the book's PDF (the author's own content) at three sizes."""
    pdf = os.path.expanduser("~/ClaudeProjects/inside-the-llm-engine/output/pdf/inside-the-llm-engine-textbook.pdf")
    paths = {}
    for px in (256, 768, 1536):
        base = os.path.join(out, f"page-{px}")
        subprocess.run(["pdftoppm", "-f", "20", "-l", "20", "-png", "-singlefile", "-scale-to", str(px), pdf, base],
                       check=True)
        paths[px] = base + ".png"
    return paths


def exp_vision(out):
    res = {"loaded_before": ollama_ps(), "runs": []}
    imgs = make_images(out)
    prompt = "Describe this image in one sentence."
    opts = {"temperature": 0, "num_predict": 32, "num_ctx": 8192}
    ollama("/api/generate", {"model": "gemma4", "prompt": "Hi.", "stream": False, "keep_alive": "10m",
                             "options": {"num_predict": 4, "num_ctx": 8192}})
    d = ollama("/api/generate", {"model": "gemma4", "prompt": prompt, "stream": False, "keep_alive": "10m",
                                 "options": opts})
    res["runs"].append({"image": "none", **summarize_gen(d)})
    for px, path in imgs.items():
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        for rep in range(2):
            d = ollama("/api/generate", {"model": "gemma4", "prompt": prompt, "images": [b64], "stream": False,
                                         "keep_alive": "10m", "options": opts})
            res["runs"].append({"image": f"{px}px", "bytes": os.path.getsize(path), "rep": rep, **summarize_gen(d)})
    ollama("/api/generate", {"model": "gemma4", "prompt": "", "keep_alive": 0})
    res["loaded_after"] = ollama_ps()
    return res


EXPERIMENTS = {"kvsizes": exp_kvsizes, "reasoning": exp_reasoning, "ollama_ctx": exp_ollama_ctx,
               "vision": exp_vision}

if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    result = EXPERIMENTS[name](out)
    result["_meta"] = {"experiment": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                       "seconds": round(time.time() - t0, 1), "loadavg_end": os.getloadavg()}
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(name, "ok", result["_meta"])
