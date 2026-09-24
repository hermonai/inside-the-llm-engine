"""Ch 14: bytes on the wire per streamed token event (llama-server /completion and
/v1/chat/completions, stream=true). Counts the raw SSE bytes of each event."""
import http.client, json, sys, statistics
import serve_probe as sp
out = sys.argv[1]
srv = sp.Server(out, "sse_bytes", ["-np", "1", "-c", "2048"])
res = {}
try:
    p = sp.doc_tokens(50)
    for path, body in (("/completion", {"prompt": p, "n_predict": 64, "temperature": 0, "ignore_eos": True, "stream": True}),
                       ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write two sentences about rivers."}],
                                                 "max_tokens": 64, "temperature": 0, "stream": True})):
        c = http.client.HTTPConnection("127.0.0.1", sp.PORT, timeout=600)
        c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        sizes, last = [], None
        while True:
            line = r.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                sizes.append(len(line) + 1)      # event line plus the blank line that ends it
                last = line
        c.close()
        res[path] = {"events": len(sizes), "median_bytes": statistics.median(sizes[:-1]), "max_bytes": max(sizes[:-1]),
                     "final_event_bytes": sizes[-1], "example": sizes[:3]}
finally:
    srv.stop()
json.dump(res, open(f"{out}/sse_bytes.json", "w"), indent=1)
print(json.dumps(res, indent=1))
