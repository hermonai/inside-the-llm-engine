"""Ch 21: what switching models costs in Ollama on this machine. Each request asks
for 8 tokens; keep_alive=0 unloads the model after the request, so the next request
for it must load it again. Durations are Ollama's own (nanoseconds in its API).
Leaves no model loaded when it finishes."""
import http.client, json, sys, time
def gen(model, keep_alive):
    c = http.client.HTTPConnection("127.0.0.1", 11434, timeout=600)
    body = {"model": model, "prompt": "Say hello in five words.", "stream": False,
            "keep_alive": keep_alive, "options": {"temperature": 0, "num_predict": 8}}
    t0 = time.perf_counter()
    c.request("POST", "/api/generate", body=json.dumps(body), headers={"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read()); c.close()
    wall = time.perf_counter() - t0
    return {"model": model, "keep_alive": keep_alive, "wall_s": round(wall, 3),
            "load_s": d.get("load_duration", 0) / 1e9, "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
            "eval_s": d.get("eval_duration", 0) / 1e9, "eval_count": d.get("eval_count"),
            "total_s": d.get("total_duration", 0) / 1e9, "response": d.get("response")}
def ps():
    c = http.client.HTTPConnection("127.0.0.1", 11434, timeout=30); c.request("GET", "/api/ps")
    d = json.loads(c.getresponse().read()); c.close(); return [m["name"] for m in d.get("models", [])]
res = {"loaded_before": ps(), "runs": []}
A, B = "llama3.2:3b", "qwen2.5-coder:3b"
for i in range(3):                      # cold switches: each request loads its model
    res["runs"].append({"phase": "switch", **gen(A, 0)})
    res["runs"].append({"phase": "switch", **gen(B, 0)})
res["runs"].append({"phase": "warm-first", **gen(A, "5m")})   # load once, keep resident
for i in range(3):
    res["runs"].append({"phase": "warm", **gen(A, "5m")})
res["runs"].append({"phase": "unload", **gen(A, 0)})
res["loaded_after"] = ps()
json.dump(res, open(sys.argv[1], "w"), indent=1)
for r in res["runs"]:
    print(r["phase"], r["model"], "wall %.2f load %.2f prompt %.2f eval %.2f" % (r["wall_s"], r["load_s"], r["prompt_eval_s"], r["eval_s"]))
print("loaded before", res["loaded_before"], "after", res["loaded_after"])
