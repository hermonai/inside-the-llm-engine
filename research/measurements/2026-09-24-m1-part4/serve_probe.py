#!/usr/bin/env python3
"""Part III probes for 'Inside the LLM Engine': drive llama-server (llama.cpp build
8660, commit d006858) through request-lifecycle, batching, KV-memory, prefix-cache,
chunked-prefill, overload, structured-output and adapter experiments.

Stdlib only. One experiment per invocation:
    python3 serve_probe.py <experiment> <outdir>
Each experiment starts its own server, writes <outdir>/<experiment>.json and the
server's log, and stops the server. Checks come before timings: greedy outputs are
compared token for token where the experiment claims equivalence, and failures are
recorded, not retried away."""
import http.client
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time

MODEL = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "doc.txt")          # source text for long prompts
PORT = 18187
BOS = 128000


class Server:
    def __init__(self, out, name, args, port=None, model=None):
        global PORT
        self.port = port or PORT
        self.log_path = os.path.join(out, f"{name}.server.log")
        self.log = open(self.log_path, "w")
        self.cmd = ["llama-server", "-m", model or MODEL, "--port", str(self.port), "-ngl", "99", *args]
        self.p = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                st, _ = get("/health", timeout=2, port=self.port)
                if st == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become healthy: " + " ".join(self.cmd))

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def get(path, timeout=30, port=None):
    c = http.client.HTTPConnection("localhost", port or PORT, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, (json.loads(body) if body else None)


def post(path, body, timeout=1800, port=None):
    c = http.client.HTTPConnection("localhost", port or PORT, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    t1 = time.perf_counter()
    c.close()
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        parsed = {"raw": data.decode(errors="replace")}
    return r.status, parsed, t1 - t0


def stream(path, body, close_after=None, t_ref=None):
    """POST with stream=true. Returns (t_send, events, how). Each event is
    (seconds since t_ref or t_send, parsed JSON). how is 'done', 'closed' or 'error'."""
    c = http.client.HTTPConnection("localhost", PORT, timeout=1800)
    t_send = time.perf_counter()
    base = t_ref if t_ref is not None else t_send
    c.request("POST", path, body=json.dumps({**body, "stream": True}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    events, n_content, how = [], 0, "done"
    if r.status != 200:
        data = r.read()
        c.close()
        return t_send, [(time.perf_counter() - base, {"http_status": r.status,
                                                      "body": data.decode(errors="replace")})], "error"
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        t = time.perf_counter() - base
        d = json.loads(payload)
        events.append((t, d))
        if "error" in d:
            how = "error"
            break
        if d.get("stop"):
            break
        if d.get("content") or d.get("tokens"):
            n_content += 1
        if close_after is not None and n_content >= close_after:
            c.sock.shutdown(socket.SHUT_RDWR)
            c.close()
            return t_send, events, "closed"
    c.close()
    return t_send, events, how


def tokenize(text):
    st, d, _ = post("/tokenize", {"content": text, "add_special": False})
    return d["tokens"]


def doc_tokens(n, offset=0):
    """n tokens of the book's own text, with BOS in front."""
    with open(DOC, encoding="utf-8") as f:
        toks = tokenize(f.read())
    assert len(toks) >= offset + n, (len(toks), offset, n)
    return [BOS] + toks[offset:offset + n]


def content_times(events):
    return [t for t, d in events if not d.get("stop") and (d.get("content") or d.get("tokens"))]


def summarize_itl(times):
    gaps = [b - a for a, b in zip(times, times[1:])]
    if not gaps:
        return {}
    gs = sorted(gaps)
    return {"n": len(gaps), "median_ms": 1e3 * statistics.median(gaps),
            "p90_ms": 1e3 * gs[int(0.9 * (len(gs) - 1))], "max_ms": 1e3 * gs[-1],
            "mean_ms": 1e3 * statistics.mean(gaps)}


def final_timings(events):
    for _, d in reversed(events):
        if "timings" in d:
            return d["timings"]
    return None


def poll_release(slot_id, t_from, limit=20.0):
    """Poll /slots until slot_id is idle; return (seconds after t_from, n_decoded)."""
    while time.perf_counter() - t_from < limit:
        st, slots = get("/slots", timeout=5)
        s = [x for x in slots if x["id"] == slot_id][0]
        if not s["is_processing"]:
            nd = s.get("next_token", [{}])[0].get("n_decoded")
            return time.perf_counter() - t_from, nd
        time.sleep(0.02)
    return None, None


# ---------------------------------------------------------------- experiments

def exp_lifecycle(out):
    """Ch 14: TTFT and inter-token latency of one streamed request; cancellation
    of a streamed and of a non-streamed request when the client disconnects."""
    res = {}
    srv = Server(out, "lifecycle", ["-np", "1", "-c", "4096"])
    try:
        prompt = doc_tokens(200)
        runs = []
        for rep in range(6):
            _, ev, how = stream("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": False})
            ts = content_times(ev)
            runs.append({"rep": rep, "how": how, "ttft_ms": 1e3 * ts[0], "events": len(ts),
                         "itl": summarize_itl(ts), "total_ms": 1e3 * ev[-1][0],
                         "server_timings": final_timings(ev)})
        res["stream_runs"] = runs[1:]          # first run warms the server
        res["warmup_run"] = runs[0]
        # cancellation of a streamed request after 20 content events
        _, ev, how = stream("/completion", {"prompt": prompt, "n_predict": 2000, "temperature": 0,
                                            "ignore_eos": True, "cache_prompt": False}, close_after=20)
        t_close = time.perf_counter()
        dt, nd = poll_release(0, t_close)
        res["cancel_stream"] = {"how": how, "events_before_close": len(content_times(ev)),
                                "release_after_close_ms": None if dt is None else 1e3 * dt,
                                "n_decoded_at_release": nd}
        # cancellation of a non-streamed request: close the socket after 3 s
        body = json.dumps({"prompt": prompt, "n_predict": 2000, "temperature": 0,
                           "ignore_eos": True, "cache_prompt": False}).encode()
        s = socket.create_connection(("localhost", PORT))
        s.sendall(b"POST /completion HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        time.sleep(3.0)
        st, slots = get("/slots")
        nd_close = slots[0].get("next_token", [{}])[0].get("n_decoded")
        s.close()
        t_close = time.perf_counter()
        dt, nd = poll_release(0, t_close)
        res["cancel_nonstream"] = {"n_decoded_at_close": nd_close,
                                   "release_after_close_ms": None if dt is None else 1e3 * dt,
                                   "n_decoded_at_release": nd}
    finally:
        srv.stop()
    return res


def exp_contbatch(out):
    """Ch 15: a 32-token request arrives 2 s into a 512-token request, with
    continuous batching on and off (-nocb: new prompts wait for an empty batch)."""
    res = {}
    for mode, flags in (("cb_on", []), ("cb_off", ["-nocb"])):
        srv = Server(out, f"contbatch-{mode}", ["-np", "4", "-c", "16384"] + flags)
        runs = []
        try:
            pa, pb = doc_tokens(100, 0), doc_tokens(100, 1000)
            for rep in range(3):
                t_ref = time.perf_counter()
                box = {}

                def run_a():
                    box["a"] = stream("/completion", {"prompt": pa, "n_predict": 512, "temperature": 0,
                                                      "ignore_eos": True, "cache_prompt": False}, t_ref=t_ref)
                th = threading.Thread(target=run_a)
                th.start()
                time.sleep(2.0)
                _, evb, howb = stream("/completion", {"prompt": pb, "n_predict": 32, "temperature": 0,
                                                      "ignore_eos": True, "cache_prompt": False}, t_ref=t_ref)
                th.join()
                _, eva, howa = box["a"]
                ta, tb = content_times(eva), content_times(evb)
                runs.append({"rep": rep, "a_how": howa, "b_how": howb,
                             "a_first_ms": 1e3 * ta[0], "a_done_ms": 1e3 * eva[-1][0],
                             "b_arrival_ms": 2000.0, "b_first_ms": 1e3 * tb[0], "b_done_ms": 1e3 * evb[-1][0],
                             "b_ttft_ms": 1e3 * tb[0] - 2000.0, "a_itl": summarize_itl(ta),
                             "b_itl": summarize_itl(tb), "a_tokens": len(ta), "b_tokens": len(tb)})
        finally:
            srv.stop()
        res[mode] = runs
    return res


def exp_kvlayout(out):
    """Ch 16: a 3,000-token prompt against 8,192 KV cells split into four
    per-slot regions (no unified KV) and against one unified pool."""
    res = {}
    prompt = None
    for mode, flags in (("per_slot", ["-no-kvu"]), ("unified", ["-kvu"])):
        srv = Server(out, f"kvlayout-{mode}", ["-np", "4", "-c", "8192"] + flags)
        try:
            prompt = prompt or doc_tokens(3000)
            st, slots = get("/slots")
            st2, d, dt = post("/completion", {"prompt": prompt, "n_predict": 16, "temperature": 0,
                                               "ignore_eos": True, "cache_prompt": False})
            res[mode] = {"slot_n_ctx": [s["n_ctx"] for s in slots], "prompt_tokens": len(prompt),
                         "http_status": st2, "elapsed_ms": 1e3 * dt,
                         "error": d.get("error") if isinstance(d, dict) else None,
                         "timings": d.get("timings") if isinstance(d, dict) else None}
        finally:
            srv.stop()
    return res


def exp_prefix(out):
    """Ch 17: time to first token for a ~3,000-token shared document, cold, warm,
    with one changed leading line, and with caching disabled; then eviction to the
    host-memory prompt cache and restoration."""
    res = {"runs": []}
    srv = Server(out, "prefix", ["-np", "1", "-c", "8192"])
    try:
        doc = doc_tokens(3000, 0)[1:]
        other = doc_tokens(3000, 4000)[1:]
        q = [tokenize(f"\n\nQuestion {i}: summarize the section above in one sentence.\nAnswer:") for i in range(8)]
        d1, d2 = tokenize("Date: 2026-09-24\n"), tokenize("Date: 2026-09-25\n")
        def ask(label, toks, cache=True):
            _, ev, how = stream("/completion", {"prompt": [BOS] + toks, "n_predict": 8, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": cache})
            ts = content_times(ev)
            tm = final_timings(ev) or {}
            res["runs"].append({"label": label, "prompt_tokens": len(toks) + 1, "how": how,
                                "ttft_ms": 1e3 * ts[0], "prompt_n": tm.get("prompt_n"),
                                "prompt_ms": tm.get("prompt_ms"), "cache_n": tm.get("cache_n")})
        ask("cold: document + question 0", doc + q[0])
        ask("warm: same document + question 1", doc + q[1])
        ask("warm again: same document + question 2", doc + q[2])
        ask("caching off: document + question 3", doc + q[3], cache=False)
        ask("date line 2026-09-24 + document + question 4", d1 + doc + q[4])
        ask("date line 2026-09-25 + document + question 4", d2 + doc + q[4])
        ask("different document (evicts the first to host memory)", other + q[5])
        ask("first document again (restored from host memory?)", doc + q[6])
    finally:
        srv.stop()
    return res


def exp_chunked(out):
    """Ch 18: a 4,000-token prompt arrives 3 s into a streamed 400-token decode;
    the decode's inter-token gaps for logical batch sizes 2048, 512 and 256."""
    res = {}
    for nb, nub in ((2048, 512), (512, 512), (256, 256)):
        srv = Server(out, f"chunked-b{nb}", ["-np", "2", "-c", "16384", "-b", str(nb), "-ub", str(nub)])
        try:
            pa, pb = doc_tokens(100, 0), doc_tokens(4000, 200)
            runs = []
            for rep in range(2):
                t_ref = time.perf_counter()
                box = {}

                def run_a():
                    box["a"] = stream("/completion", {"prompt": pa, "n_predict": 400, "temperature": 0,
                                                      "ignore_eos": True, "cache_prompt": False}, t_ref=t_ref)
                th = threading.Thread(target=run_a)
                th.start()
                time.sleep(3.0)
                _, evb, howb = stream("/completion", {"prompt": pb, "n_predict": 8, "temperature": 0,
                                                      "ignore_eos": True, "cache_prompt": False}, t_ref=t_ref)
                th.join()
                _, eva, howa = box["a"]
                ta, tb = content_times(eva), content_times(evb)
                gaps = [(round(1e3 * a, 1), round(1e3 * (b - a), 1)) for a, b in zip(ta, ta[1:])]
                runs.append({"rep": rep, "a_how": howa, "b_how": howb, "b_ttft_ms": 1e3 * tb[0] - 3000.0,
                             "a_itl": summarize_itl(ta), "a_gaps_over_200ms": [g for g in gaps if g[1] > 200],
                             "a_times_ms": [round(1e3 * t, 1) for t in ta]})
            res[f"b{nb}_ub{nub}"] = runs
        finally:
            srv.stop()
    return res


def exp_overload(out):
    """Ch 19: four streamed requests whose combined KV demand (4 x ~1,400 tokens)
    exceeds a 4,096-cell unified cache; then the same load against 8,192 cells."""
    res = {}
    for cells in (4096, 8192):
        srv = Server(out, f"overload-c{cells}", ["-np", "4", "-c", str(cells), "-kvu"])
        try:
            prompts = [doc_tokens(200, 500 * i) for i in range(4)]
            t_ref = time.perf_counter()
            box = {}

            def run(i):
                box[i] = stream("/completion", {"prompt": prompts[i], "n_predict": 1200, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": False}, t_ref=t_ref)
            ths = [threading.Thread(target=run, args=(i,)) for i in range(4)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            reqs = []
            for i in range(4):
                _, ev, how = box[i]
                ts = content_times(ev)
                err = [d["error"] for _, d in ev if "error" in d]
                reqs.append({"how": how, "tokens": len(ts), "last_ms": 1e3 * ev[-1][0] if ev else None,
                             "error": err[0] if err else None})
            res[f"cells_{cells}"] = reqs
        finally:
            srv.stop()
    return res


BOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "author": {"type": "string"},
        "year": {"type": "integer"}, "genres": {"type": "array", "items": {"type": "string"}},
        "rating": {"type": "number"}},
    "required": ["title", "author", "year", "genres", "rating"],
    "additionalProperties": False,
}


def valid_book(text):
    try:
        o = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(o, dict) or set(o) != set(BOOK_SCHEMA["properties"]):
        return False
    return (isinstance(o["title"], str) and isinstance(o["author"], str) and type(o["year"]) is int
            and isinstance(o["genres"], list) and all(isinstance(g, str) for g in o["genres"])
            and isinstance(o["rating"], (int, float)) and not isinstance(o["rating"], bool))


def exp_structured(out):
    """Ch 20: 20 sampled answers per condition to a request for a JSON book record:
    unconstrained, JSON-schema constrained, and constrained by a schema whose genre
    field is an enum of 2,000 strings. Validity is checked against the schema."""
    res = {}
    srv = Server(out, "structured", ["-np", "1", "-c", "4096"])
    try:
        msg = ("Return a JSON object describing a fictional book, with exactly these fields: "
               "title (string), author (string), year (integer), genres (array of strings), "
               "rating (number from 0 to 5). Output only the JSON object.")
        enum_schema = json.loads(json.dumps(BOOK_SCHEMA))
        enum_schema["properties"]["genres"]["items"] = {"enum": [f"genre-{i:04d}" for i in range(2000)]}
        conditions = {"unconstrained": None, "schema": BOOK_SCHEMA, "schema_enum2000": enum_schema}
        for name, schema in conditions.items():
            runs = []
            for seed in range(1, 21):
                body = {"messages": [{"role": "user", "content": msg}], "temperature": 0.8, "seed": seed,
                        "max_tokens": 200}
                if schema is not None:
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "book", "schema": schema}}
                st, d, dt = post("/v1/chat/completions", body)
                text = d["choices"][0]["message"]["content"] if st == 200 else None
                tm = d.get("timings", {}) if isinstance(d, dict) else {}
                runs.append({"seed": seed, "status": st, "valid": valid_book(text), "text": text,
                             "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                             "prompt_ms": tm.get("prompt_ms"), "elapsed_ms": 1e3 * dt})
            res[name] = runs
    finally:
        srv.stop()
    return res


def exp_lora(out):
    """Ch 21: decode speed with no adapter, a rank-16 and a rank-64 adapter (B = 0,
    so outputs must equal the base model's), and two concurrent streams that use the
    same or different adapters (llama-server batches only equal adapter sets)."""
    res = {}
    loras = [os.path.join(HERE, f) for f in ("lora-r16-a.gguf", "lora-r16-b.gguf", "lora-r64.gguf")]
    args = ["-np", "2", "-c", "8192", "--lora-init-without-apply", "--lora", ",".join(loras)]
    srv = Server(out, "lora", args)
    try:
        prompt = doc_tokens(100)
        def one(lora, tag):
            _, ev, how = stream("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": False, "lora": lora})
            ts = content_times(ev)
            text = "".join(d.get("content", "") for _, d in ev)
            return {"tag": tag, "how": how, "itl": summarize_itl(ts), "text": text,
                    "timings": final_timings(ev)}
        off = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}]
        r16a = [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}]
        r16b = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}]
        r64 = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 1.0}]
        one(off, "warmup")
        single = [one(off, "none"), one(r16a, "rank16"), one(r64, "rank64"),
                  one(off, "none-repeat"), one(r16a, "rank16-repeat"), one(r64, "rank64-repeat")]
        base_text = single[0]["text"]
        for s in single:
            s["same_text_as_base"] = s["text"] == base_text
        res["single"] = single
        def pair(la, lb, tag):
            box = {}
            t_ref = time.perf_counter()
            def run(k, lora):
                box[k] = stream("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0,
                                                "ignore_eos": True, "cache_prompt": False, "lora": lora}, t_ref=t_ref)
            ths = [threading.Thread(target=run, args=("x", la)), threading.Thread(target=run, args=("y", lb))]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            outp = {"tag": tag}
            for k in ("x", "y"):
                _, ev, how = box[k]
                ts = content_times(ev)
                outp[k] = {"how": how, "tokens": len(ts), "done_ms": 1e3 * ev[-1][0], "itl": summarize_itl(ts),
                           "same_text_as_base": "".join(d.get("content", "") for _, d in ev) == base_text}
            outp["aggregate_tok_s"] = (outp["x"]["tokens"] + outp["y"]["tokens"]) / (
                max(outp["x"]["done_ms"], outp["y"]["done_ms"]) / 1e3)
            return outp
        res["pairs"] = [pair(r16a, r16a, "same adapter"), pair(r16a, r16b, "different adapters"),
                        pair(off, off, "no adapter"), pair(r16a, r16a, "same adapter (repeat)"),
                        pair(r16a, r16b, "different adapters (repeat)")]
    finally:
        srv.stop()
    return res


def raw_post_then_close(body, close_after_s):
    """Send a non-streamed POST /completion on a raw socket and close it after
    close_after_s seconds without reading the response."""
    data = json.dumps(body).encode()
    s = socket.create_connection(("localhost", PORT))
    s.sendall(b"POST /completion HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(close_after_s)
    s.close()
    return time.perf_counter()


def poll_slot_slow(slot_id, t_from, period=1.5, limit=150.0):
    """Poll /slots every period seconds (longer than the server's 1 s disconnect
    polling, so the observer does not keep resetting it). Returns the samples."""
    samples = []
    while time.perf_counter() - t_from < limit:
        st, slots = get("/slots", timeout=5)
        sl = [x for x in slots if x["id"] == slot_id][0]
        nd = sl.get("next_token", [{}])[0].get("n_decoded")
        samples.append((round(time.perf_counter() - t_from, 2), sl["is_processing"], nd))
        if not sl["is_processing"]:
            break
        time.sleep(period)
    return samples


def exp_cancel2(out):
    """Ch 14 follow-up: a non-streamed request whose client disconnects after 3 s,
    on a quiet server and while another client streams on the other slot. The
    server checks for a closed connection only when its 1 s wait for results times
    out, and any result for any request restarts that wait."""
    res = {}
    srv = Server(out, "cancel2", ["-np", "2", "-c", "8192", "--log-timestamps"])
    try:
        px, py = doc_tokens(200, 0), doc_tokens(200, 3000)
        bx = {"prompt": px, "n_predict": 1500, "temperature": 0, "ignore_eos": True,
              "cache_prompt": False, "id_slot": 0}
        t_close = raw_post_then_close(bx, 3.0)
        res["quiet"] = {"samples_after_close": poll_slot_slow(0, t_close)}
        time.sleep(2.0)
        box = {}
        t_ref = time.perf_counter()

        def run_y():
            box["y"] = stream("/completion", {"prompt": py, "n_predict": 500, "temperature": 0,
                                              "ignore_eos": True, "cache_prompt": False, "id_slot": 1}, t_ref=t_ref)
        th = threading.Thread(target=run_y)
        th.start()
        time.sleep(1.0)
        t_close = raw_post_then_close(bx, 3.0)
        samples = poll_slot_slow(0, t_close)
        th.join()
        _, evy, howy = box["y"]
        res["busy"] = {"samples_after_close": samples, "y_how": howy,
                       "y_done_after_close_s": round(evy[-1][0] - (t_close - t_ref), 2),
                       "y_tokens": len(content_times(evy))}
    finally:
        srv.stop()
    return res


EXPERIMENTS = {"lifecycle": exp_lifecycle, "contbatch": exp_contbatch, "kvlayout": exp_kvlayout,
               "prefix": exp_prefix, "chunked": exp_chunked, "overload": exp_overload,
               "structured": exp_structured, "lora": exp_lora,
               "cancel2": exp_cancel2}

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
