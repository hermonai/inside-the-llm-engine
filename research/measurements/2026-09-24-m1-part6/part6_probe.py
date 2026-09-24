#!/usr/bin/env python3
"""Part VI probes for 'Inside the LLM Engine' on an Apple M1.

  python3 part6_probe.py <experiment> <outdir>

ssd       Sequential read bandwidth of the internal SSD with caching off: a 2 GiB
          file written and read back with F_NOCACHE, and the same file read
          again through the page cache.
kvmove    llama-server (build 8660, llama.cpp d006858), one slot, a 20,480-token
          context and --slot-save-path. For Llama 3.2 3B and the Qwen3-8B
          architecture and prompts of about 1,024, 4,096 and 16,384 tokens: the
          cold prefill; saving the slot's cache to a file; restoring it from the
          host-memory prompt cache (--cache-ram) after an unrelated prompt;
          restoring it from the saved file (just written, so likely from the
          page cache); and restoring it from an uncached copy of the file (read
          from the SSD). After each restore the same prompt is sent again and its
          one-token answer must equal the cold run's.
disagg    Prefill on one llama-server, decode on another: server P prefills a
          prompt of about 4,000 tokens and saves the slot; server D restores the
          file and generates 64 tokens. The greedy text must equal a single
          server's text for the same prompt; time to first token is split into
          its parts.
"""
import fcntl
import json
import os
import shutil
import sys
import time

import serve_probe as sp

BLOBS = os.path.expanduser("~/.ollama/models/blobs")
MODELS = {
    "llama3.2-3b": os.path.join(BLOBS, "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"),
    "qwen3-8b": os.path.join(BLOBS, "sha256-e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d"),
}
F_NOCACHE = 48          # <sys/fcntl.h> on macOS
PORT_A, PORT_B = 18201, 18202
CHUNK = 16 << 20


def nocache_fd(path, flags, mode=0o644):
    fd = os.open(path, flags, mode)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def copy_uncached(src, dst):
    """Copy src to dst with F_NOCACHE on the destination, so dst is not left in
    the page cache and a later read of it comes from the SSD."""
    fi = os.open(src, os.O_RDONLY)
    fo = nocache_fd(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        while True:
            b = os.read(fi, CHUNK)
            if not b:
                break
            os.write(fo, b)
        os.fsync(fo)
    finally:
        os.close(fi)
        os.close(fo)


def exp_ssd(out):
    path = os.path.join(out, "ssd-test.bin")
    size = 2 << 30
    block = os.urandom(CHUNK)
    fd = nocache_fd(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    t0 = time.perf_counter()
    for _ in range(size // CHUNK):
        os.write(fd, block)
    os.fsync(fd)
    t_write = time.perf_counter() - t0
    os.close(fd)
    res = {"bytes": size, "write_nocache_s": round(t_write, 3), "write_nocache_GBps": round(size / t_write / 1e9, 3),
           "reads": []}
    for label, nocache in (("nocache-1", True), ("nocache-2", True), ("cached-1", False), ("cached-2", False)):
        fd = nocache_fd(path, os.O_RDONLY) if nocache else os.open(path, os.O_RDONLY)
        t0 = time.perf_counter()
        n = 0
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            n += len(b)
        t = time.perf_counter() - t0
        os.close(fd)
        res["reads"].append({"read": label, "bytes": n, "s": round(t, 3), "GBps": round(n / t / 1e9, 3)})
    os.remove(path)
    return res


def prompt_of(n_tokens, port, salt):
    """Words of doc.txt, preceded by a salt line, with about n_tokens tokens."""
    with open(sp.DOC, encoding="utf-8") as f:
        words = f.read().split()
    lo, hi = 1, len(words)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text = f"Document {salt}.\n" + " ".join(words[:mid])
        st, d, _ = sp.post("/tokenize", {"content": text, "add_special": True}, port=port)
        n = len(d["tokens"])
        if n <= n_tokens:
            best = (text, n)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def completion(prompt, port, n_predict=1):
    st, d, dt = sp.post("/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                                        "cache_prompt": True}, port=port)
    assert st == 200, (st, d)
    tm = d.get("timings", {})
    return {"content": d.get("content"), "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
            "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"), "wall_s": round(dt, 3)}


def slot(action, port, filename=None):
    body = {"filename": filename} if filename else {}
    st, d, dt = sp.post(f"/slots/0?action={action}", body, port=port)
    assert st == 200, (action, st, d)
    d["wall_s"] = round(dt, 3)
    return d


def exp_kvmove(out):
    slots_dir = os.path.join(out, "slots")
    os.makedirs(slots_dir, exist_ok=True)
    res = {}
    for tag, model in MODELS.items():
        srv = sp.Server(out, f"kvmove-{tag}", ["-np", "1", "-c", "20480", "-fa", "on",
                                              "--slot-save-path", slots_dir], port=PORT_A, model=model)
        rows = []
        try:
            completion("Hello.", PORT_A)                        # warm-up
            for target in (1024, 4096, 16384):
                text, n_tok = prompt_of(target, PORT_A, f"{tag}-{target}")
                slot("erase", PORT_A)
                cold = completion(text, PORT_A)
                fname = f"{tag}-{target}.bin"
                saved = slot("save", PORT_A, fname)
                # host-memory tier: an unrelated prompt displaces the slot's cache into
                # the host prompt cache; the original prompt then restores it
                completion(f"Unrelated request {target}: say one word.", PORT_A)
                ram = completion(text, PORT_A)
                # file tier, just written (page cache likely warm)
                slot("erase", PORT_A)
                rest_pc = slot("restore", PORT_A, fname)
                after_pc = completion(text, PORT_A)
                # file tier, uncached copy (SSD)
                ssd_name = f"{tag}-{target}-nocache.bin"
                t0 = time.perf_counter()
                copy_uncached(os.path.join(slots_dir, fname), os.path.join(slots_dir, ssd_name))
                t_copy = time.perf_counter() - t0
                slot("erase", PORT_A)
                rest_ssd = slot("restore", PORT_A, ssd_name)
                after_ssd = completion(text, PORT_A)
                rows.append({
                    "target_tokens": target, "prompt_tokens": n_tok,
                    "cold": cold, "save": saved, "ram_restore": ram,
                    "file_restore_pagecache": rest_pc, "after_pagecache_restore": after_pc,
                    "uncached_copy_s": round(t_copy, 3),
                    "file_restore_ssd": rest_ssd, "after_ssd_restore": after_ssd,
                    "file_bytes": os.path.getsize(os.path.join(slots_dir, fname)),
                    "same_answer": (cold["content"] == ram["content"] == after_pc["content"] == after_ssd["content"]),
                })
                for f in (fname, ssd_name):
                    os.remove(os.path.join(slots_dir, f))
        finally:
            srv.stop()
        res[tag] = rows
    shutil.rmtree(slots_dir, ignore_errors=True)
    return res


def exp_disagg(out):
    slots_dir = os.path.join(out, "slots")
    os.makedirs(slots_dir, exist_ok=True)
    model = MODELS["llama3.2-3b"]
    args = ["-np", "1", "-c", "8192", "-fa", "on", "--slot-save-path", slots_dir]
    p = sp.Server(out, "disagg-prefill", args, port=PORT_A, model=model)
    d = None
    res = {"runs": []}
    try:
        d = sp.Server(out, "disagg-decode", args, port=PORT_B, model=model)
        completion("Hello.", PORT_A)
        completion("Hello.", PORT_B)
        for rep in range(3):
            text, n_tok = prompt_of(4000, PORT_A, f"disagg-{rep}")
            text += "\n\nQuestion: summarize the text above in two sentences.\nAnswer:"
            # single-server baseline on D, from a cold slot
            slot("erase", PORT_B)
            base = completion(text, PORT_B, n_predict=64)
            # disaggregated: prefill on P, move the cache through a file, decode on D
            slot("erase", PORT_A)
            pre = completion(text, PORT_A, n_predict=1)
            fname = f"disagg-{rep}.bin"
            saved = slot("save", PORT_A, fname)
            slot("erase", PORT_B)
            restored = slot("restore", PORT_B, fname)
            dec = completion(text, PORT_B, n_predict=64)
            res["runs"].append({"rep": rep, "prompt_tokens": n_tok, "baseline": base, "prefill_on_P": pre,
                                "save_on_P": saved, "restore_on_D": restored, "decode_on_D": dec,
                                "file_bytes": os.path.getsize(os.path.join(slots_dir, fname)),
                                "identical_text": base["content"] == dec["content"]})
            os.remove(os.path.join(slots_dir, fname))
    finally:
        p.stop()
        if d is not None:
            d.stop()
    shutil.rmtree(slots_dir, ignore_errors=True)
    return res


EXPERIMENTS = {"ssd": exp_ssd, "kvmove": exp_kvmove, "disagg": exp_disagg}

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
