#!/usr/bin/env python3
"""Part IV probes for 'Inside the LLM Engine': speculative decoding in llama-server
(llama.cpp build 8660, commit d006858) on an Apple M1.

  python3 spec_probe.py <experiment> <outdir>

ngram      Llama 3.2 3B, a baseline server and a prompt-lookup server (--spec-type
           ngram-simple) side by side; requests alternate between them so that
           load drift affects both. Four workloads, greedy and at temperature 0.8.
           Greedy outputs are compared token for token.
exact      Ten greedy prompts on the same pair of servers; outputs compared.
concurrent Four simultaneous quote requests (-np 4), baseline against prompt lookup.
draftmodel An 8B target (DeepSeek-R1-0528-Qwen3-8B) with no speculation, prompt
           lookup, or a 3B draft model (Qwen2.5-Coder-3B, vocabularies translated).

Every speculative result carries the server's own draft counts
(timings.draft_n, timings.draft_n_accepted)."""
import json
import os
import sys
import threading
import time

import serve_probe as sp

BLOBS = os.path.expanduser("~/.ollama/models/blobs")
M3B = os.path.join(BLOBS, "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
M8B = os.path.join(BLOBS, "sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d")
MDRAFT = os.path.join(BLOBS, "sha256-4a188102020e9c9530b687fd6400f775c45e90a0d7baafe65bd0a36963fbb7ba")
P_BASE, P_SPEC = 18190, 18191

PASSAGE = (
    "A serving engine spends most of its life waiting on memory. Each decode step reads every "
    "weight of the model once, performs a handful of arithmetic operations per weight, and "
    "produces a single token for each sequence in the batch. The arithmetic units finish their "
    "share long before the next bytes arrive, so the step's duration is set by bandwidth rather "
    "than by compute. This imbalance is the reason continuous batching pays: every additional "
    "sequence in the batch reuses the same weights that were already read, adding arithmetic "
    "that the machine had to spare. It is also the reason speculative decoding works. A step that "
    "verifies five guessed tokens reads the same weights as a step that produces one, and on "
    "hardware with idle arithmetic it costs little more. The catch is that the guesses must be "
    "right often enough, and cheap enough to make, for the saved steps to outweigh the extra "
    "work. When the batch is already large, the arithmetic is no longer idle, and the same trade "
    "that halved latency can make every request slower. Engines therefore treat speculation as a "
    "policy rather than a switch: they measure how often drafts are accepted, how much a draft "
    "costs, and how busy the machine is, and they adjust the number of drafted tokens at every step.")

WORKLOADS = {
    "quote": "Repeat the following passage exactly, word for word, and output nothing else.\n\n" + PASSAGE,
    "edit": ("Rewrite the following passage, replacing the word \"batch\" with \"group\" everywhere "
             "and changing nothing else. Output only the rewritten passage.\n\n" + PASSAGE),
    "summarize": "Summarize the following passage in three sentences.\n\n" + PASSAGE,
    "free": "Write a short story of about 200 words about a lighthouse keeper who collects lost letters.",
}

EXTRA = [
    "Explain how TCP slow start works, step by step.",
    "Write a Python function that merges two sorted lists, with a docstring and three doctests.",
    "List the planets of the solar system in order, with one fact about each.",
    "Translate into French: The weather was cold, so we stayed inside and read books all afternoon.",
    "Continue this passage for a paragraph: " + PASSAGE[:600],
    "Write a JSON array of five objects, each with fields id, name and email, for fictional users.",
]


def chat(port, content, temperature=0.0, seed=1, max_tokens=400):
    body = {"messages": [{"role": "user", "content": content}], "temperature": temperature,
            "seed": seed, "max_tokens": max_tokens, "cache_prompt": False}
    st, d, dt = sp.post("/v1/chat/completions", body, port=port)
    assert st == 200, (st, d)
    tm = d.get("timings", {})
    return {"text": d["choices"][0]["message"]["content"], "elapsed_s": round(dt, 3),
            "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
            "tok_s": tm.get("predicted_per_second"), "prompt_n": tm.get("prompt_n"),
            "prompt_ms": tm.get("prompt_ms"), "draft_n": tm.get("draft_n", 0),
            "draft_n_accepted": tm.get("draft_n_accepted", 0)}


def pair(out, tag, spec_args, np_=1, ctx=8192, model=None):
    base = sp.Server(out, f"{tag}-base", ["-np", str(np_), "-c", str(ctx)], port=P_BASE, model=model)
    try:
        spec = sp.Server(out, f"{tag}-spec", ["-np", str(np_), "-c", str(ctx)] + spec_args, port=P_SPEC,
                         model=model)
    except Exception:
        base.stop()
        raise
    return base, spec


def exp_ngram(out):
    res = {"runs": []}
    base, spec = pair(out, "ngram", ["--spec-type", "ngram-simple"])
    try:
        chat(P_BASE, "Say hi.", max_tokens=8); chat(P_SPEC, "Say hi.", max_tokens=8)   # warm both
        for temp in (0.0, 0.8):
            for name, content in WORKLOADS.items():
                for rep in range(3):
                    order = [("base", P_BASE), ("spec", P_SPEC)]
                    if rep % 2:
                        order.reverse()
                    got = {}
                    for label, port in order:
                        got[label] = chat(port, content, temperature=temp, seed=100 + rep)
                    rec = {"workload": name, "temperature": temp, "rep": rep, "base": got["base"],
                           "spec": got["spec"]}
                    if temp == 0.0:
                        rec["identical"] = got["base"]["text"] == got["spec"]["text"]
                    res["runs"].append(rec)
    finally:
        base.stop(); spec.stop()
    return res


def exp_exact(out):
    res = {"runs": []}
    base, spec = pair(out, "exact", ["--spec-type", "ngram-simple"])
    try:
        for i, content in enumerate(list(WORKLOADS.values()) + EXTRA):
            b = chat(P_BASE, content, max_tokens=300)
            s = chat(P_SPEC, content, max_tokens=300)
            first_diff = next((k for k, (x, y) in enumerate(zip(b["text"], s["text"])) if x != y), None)
            if first_diff is None and len(b["text"]) != len(s["text"]):
                first_diff = min(len(b["text"]), len(s["text"]))
            res["runs"].append({"prompt": i, "identical": b["text"] == s["text"], "first_diff_char": first_diff,
                                "draft_n": s["draft_n"], "draft_n_accepted": s["draft_n_accepted"],
                                "base_chars": len(b["text"]), "spec_chars": len(s["text"]),
                                "base_text": b["text"], "spec_text": s["text"]})
    finally:
        base.stop(); spec.stop()
    return res


def exp_concurrent(out):
    res = {"runs": []}
    base, spec = pair(out, "concurrent", ["--spec-type", "ngram-simple"], np_=4, ctx=16384)
    try:
        chat(P_BASE, "Say hi.", max_tokens=8); chat(P_SPEC, "Say hi.", max_tokens=8)
        for rep in range(3):
            for label, port in (("base", P_BASE), ("spec", P_SPEC)) if rep % 2 == 0 else (("spec", P_SPEC), ("base", P_BASE)):
                box = {}
                t0 = time.perf_counter()

                def run(i):
                    box[i] = chat(port, WORKLOADS["quote"], seed=200 + i)
                ths = [threading.Thread(target=run, args=(i,)) for i in range(4)]
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
                wall = time.perf_counter() - t0
                toks = sum(box[i]["predicted_n"] for i in range(4))
                res["runs"].append({"rep": rep, "config": label, "wall_s": round(wall, 2),
                                    "aggregate_tok_s": round(toks / wall, 2),
                                    "per_request_tok_s": [box[i]["tok_s"] for i in range(4)],
                                    "draft_n": sum(box[i]["draft_n"] for i in range(4)),
                                    "draft_n_accepted": sum(box[i]["draft_n_accepted"] for i in range(4)),
                                    "tokens": toks})
        # the single-request reference on the same servers
        for label, port in (("base", P_BASE), ("spec", P_SPEC)):
            r = chat(port, WORKLOADS["quote"], seed=300)
            res["runs"].append({"rep": "single", "config": label, "tok_s": r["tok_s"],
                                "draft_n": r["draft_n"], "draft_n_accepted": r["draft_n_accepted"]})
    finally:
        base.stop(); spec.stop()
    return res


def exp_draftmodel(out):
    res = {"runs": []}
    configs = {"none": [], "ngram": ["--spec-type", "ngram-simple"],
               "draft": ["-md", MDRAFT, "-ngld", "99"]}
    order = ["none", "draft", "ngram", "ngram", "draft", "none"]
    for k, cfg in enumerate(order):
        srv = sp.Server(out, f"draftmodel-{k}-{cfg}", ["-np", "1", "-c", "4096"] + configs[cfg],
                        port=P_BASE, model=M8B)
        try:
            chat(P_BASE, "Say hi.", max_tokens=8)
            for name in ("quote", "free"):
                r = chat(P_BASE, WORKLOADS[name], max_tokens=256)
                res["runs"].append({"block": k, "config": cfg, "workload": name, **{x: r[x] for x in r if x != "text"},
                                    "text_head": r["text"][:300]})
        finally:
            srv.stop()
    return res


EXPERIMENTS = {"ngram": exp_ngram, "exact": exp_exact, "concurrent": exp_concurrent, "draftmodel": exp_draftmodel}

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
