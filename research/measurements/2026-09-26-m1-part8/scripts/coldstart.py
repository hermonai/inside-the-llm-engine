#!/usr/bin/env python3
"""Chapter 40: how long a scale-up takes on the M1, from process start to first token.

llama-server (build 8660): time from spawn to /health 200, then the first and second
requests' time to first token. Ollama 0.34.4: the model is unloaded (keep_alive 0),
then one request's reported load_duration and total time. Three repetitions each.
The model file is in the page cache (it was just used), so storage is not measured;
the record says so.

    python3 coldstart.py <outdir>
"""
import http.client, json, os, subprocess, sys, time

BLOB = os.path.expanduser("~/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
PORT = 18402

def req(port, path, body=None, timeout=300, method="POST"):
    c = http.client.HTTPConnection("localhost", port, timeout=timeout)
    c.request(method, path, body=json.dumps(body) if body is not None else None,
              headers={"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read(); c.close()
    return r.status, d

def first_token(port):
    c = http.client.HTTPConnection("localhost", port, timeout=300)
    t0 = time.perf_counter()
    c.request("POST", "/completion", body=json.dumps({"prompt": "The capital of France is", "n_predict": 8,
              "temperature": 0, "stream": True, "cache_prompt": False}), headers={"Content-Type": "application/json"})
    r = c.getresponse(); t_first = None
    while True:
        line = r.readline()
        if not line: break
        if line.startswith(b"data: "):
            d = json.loads(line[6:])
            if d.get("content") and t_first is None: t_first = time.perf_counter() - t0
            if d.get("stop"): break
    c.close()
    return t_first

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
res = {"llama_server": [], "ollama": []}
for rep in range(3):
    log = open(os.path.join(out, f"coldstart-server-{rep}.log"), "w")
    t0 = time.perf_counter()
    p = subprocess.Popen(["llama-server", "-m", BLOB, "--port", str(PORT), "-ngl", "99", "-np", "4", "-c", "8192"],
                         stdout=log, stderr=subprocess.STDOUT)
    ready = None
    while time.perf_counter() - t0 < 300:
        try:
            st, _ = req(PORT, "/health", method="GET", timeout=2)
            if st == 200: ready = time.perf_counter() - t0; break
        except OSError: pass
        time.sleep(0.05)
    f1 = first_token(PORT); f2 = first_token(PORT)
    p.terminate(); p.wait(timeout=30); log.close()
    res["llama_server"].append({"ready_s": round(ready, 3), "first_ttft_s": round(f1, 3), "second_ttft_s": round(f2, 3)})
    print("llama-server", res["llama_server"][-1], flush=True)
    time.sleep(3)
for rep in range(3):
    req(11434, "/api/generate", {"model": "llama3.2:3b", "prompt": "", "keep_alive": 0, "stream": False})
    time.sleep(2)
    t0 = time.perf_counter()
    st, d = req(11434, "/api/generate", {"model": "llama3.2:3b", "prompt": "The capital of France is", "raw": True,
                "stream": False, "options": {"temperature": 0, "num_predict": 8}})
    wall = time.perf_counter() - t0
    d = json.loads(d)
    res["ollama"].append({"wall_s": round(wall, 3), "load_s": round(d.get("load_duration", 0) / 1e9, 3),
                          "prompt_s": round(d.get("prompt_eval_duration", 0) / 1e9, 3)})
    print("ollama", res["ollama"][-1], flush=True)
req(11434, "/api/generate", {"model": "llama3.2:3b", "prompt": "", "keep_alive": 0, "stream": False})
res["_meta"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load": os.getloadavg(),
                "ollama": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()}
json.dump(res, open(os.path.join(out, "coldstart.json"), "w"), indent=1)
print(res["_meta"])
