#!/usr/bin/env python3
"""Load generators for Chapters 39 and 40 of 'Inside the LLM Engine'.

Stdlib only. Drives llama-server (build 8660) with Llama 3.2 3B through
serve_probe.Server, streaming every completion so the first token's arrival is
seen by the client.

  python3 loadgen.py stall <outdir>      closed loop vs open loop, same server, a
                                         5-second SIGSTOP of the server mid-run
  python3 loadgen.py sweep <outdir>      open loop at rising Poisson rates, with
                                         /metrics scraped every 2 s

Closed loop: C users, each sends its next request when the previous one ends;
latency is timed from the actual send. Open loop: arrivals are drawn up front
from a Poisson process; each request is timed from its scheduled arrival, so
time spent waiting to be sent counts, as it would for a user.
"""
import http.client
import json
import os
import random
import re
import signal
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_probe as sp  # noqa: E402

PORT = 18401
N_PREDICT = 48


def prompts():
    text = re.sub(r"\s+", " ", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc.txt"),
                                   encoding="utf-8").read())
    out, i = [], 0
    while len(out) < 400 and i < len(text) - 1200:
        j = text.find(". ", i + 700)
        out.append(text[i:j + 1])
        i = j + 2
    return out


PROMPTS = prompts()


def one_request(prompt, t_sched, rec):
    """Stream one completion; fill rec with send, first-token and end times."""
    rec["t_sched"] = t_sched
    try:
        c = http.client.HTTPConnection("localhost", PORT, timeout=600)
        rec["t_send"] = time.perf_counter()
        c.request("POST", "/completion", body=json.dumps({
            "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "stream": True,
            "cache_prompt": False, "ignore_eos": True}), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        n = 0
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            d = json.loads(line[6:])
            if d.get("content") and "t_first" not in rec:
                rec["t_first"] = time.perf_counter()
            if d.get("content"):
                n += 1
            if d.get("stop"):
                break
        rec["t_end"] = time.perf_counter()
        rec["tokens"] = n
        c.close()
    except (OSError, http.client.HTTPException) as e:
        rec["error"] = repr(e)
        rec["t_end"] = time.perf_counter()


def closed_loop(users, seconds, t0, recs, stop_evt):
    def user(u):
        k = u
        while time.perf_counter() - t0 < seconds and not stop_evt.is_set():
            rec = {"mode": "closed", "user": u}
            recs.append(rec)
            one_request(PROMPTS[k % len(PROMPTS)], time.perf_counter(), rec)
            k += users
    ths = [threading.Thread(target=user, args=(u,)) for u in range(users)]
    for t in ths:
        t.start()
    return ths


def open_loop(rate, seconds, t0, recs, seed):
    rng = random.Random(seed)
    t, arrivals = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= seconds:
            break
        arrivals.append(t)
    ths = []

    def launcher():
        for i, a in enumerate(arrivals):
            now = time.perf_counter() - t0
            if a > now:
                time.sleep(a - now)
            rec = {"mode": "open", "i": i}
            recs.append(rec)
            th = threading.Thread(target=one_request, args=(PROMPTS[i % len(PROMPTS)], t0 + a, rec))
            th.start()
            ths.append(th)
    lt = threading.Thread(target=launcher)
    lt.start()
    return lt, ths, len(arrivals)


def scrape(stop_evt, t0, out):
    while not stop_evt.is_set():
        try:
            c = http.client.HTTPConnection("localhost", PORT, timeout=5)
            c.request("GET", "/metrics")
            body = c.getresponse().read().decode()
            c.close()
            row = {"t": round(time.perf_counter() - t0, 2)}
            for line in body.splitlines():
                if line.startswith("llamacpp:"):
                    k, v = line.split()[:2]
                    row[k[9:]] = float(v)
            out.append(row)
        except (OSError, http.client.HTTPException, ValueError):
            out.append({"t": round(time.perf_counter() - t0, 2), "error": True})
        stop_evt.wait(2.0)


def summarize(recs, t0):
    ok = [r for r in recs if "t_end" in r and "error" not in r and "t_first" in r]
    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None
    ttft = [r["t_first"] - r["t_sched"] for r in ok]
    ttft_send = [r["t_first"] - r["t_send"] for r in ok]
    e2e = [r["t_end"] - r["t_sched"] for r in ok]
    return {"n_ok": len(ok), "n_err": sum("error" in r for r in recs),
            "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99), "ttft_max": pct(ttft, 100),
            "ttft_from_send_p99": pct(ttft_send, 99),
            "e2e_p50": pct(e2e, 50), "e2e_p99": pct(e2e, 99),
            "tokens": sum(r.get("tokens", 0) for r in ok)}


def exp_stall(out):
    res = {}
    srv = sp.Server(out, "loadgen-stall", ["-np", "4", "-c", "8192", "--metrics"], port=PORT)
    try:
        # warm-up
        one_request(PROMPTS[0], time.perf_counter(), {})
        for mode in ("closed", "open"):
            recs, seconds, t0 = [], 90.0, time.perf_counter()
            stop_evt = threading.Event()
            if mode == "closed":
                ths = closed_loop(4, seconds, t0, recs, stop_evt)
            else:
                lt, ths, n_arr = open_loop(0.45, seconds, t0, recs, seed=7)
            time.sleep(40.0)
            os.kill(srv.p.pid, signal.SIGSTOP)            # the server freezes for 5 s
            t_stall = time.perf_counter() - t0
            time.sleep(5.0)
            os.kill(srv.p.pid, signal.SIGCONT)
            if mode == "closed":
                for t in ths:
                    t.join()
            else:
                lt.join()
                for t in ths:
                    t.join()
            res[mode] = {"summary": summarize(recs, t0), "stall_at_s": round(t_stall, 2),
                         "requests": [{k: (round(v - t0, 4) if k.startswith("t_") else v) for k, v in r.items()}
                                      for r in recs]}
            print(mode, res[mode]["summary"], flush=True)
            time.sleep(10.0)
    finally:
        srv.stop()
    return res


def exp_sweep(out):
    res = {"rates": []}
    srv = sp.Server(out, "loadgen-sweep", ["-np", "4", "-c", "8192", "--metrics"], port=PORT)
    try:
        one_request(PROMPTS[0], time.perf_counter(), {})
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            recs, seconds, t0 = [], 60.0, time.perf_counter()
            stop_evt, metrics = threading.Event(), []
            st = threading.Thread(target=scrape, args=(stop_evt, t0, metrics))
            st.start()
            lt, ths, n_arr = open_loop(rate, seconds, t0, recs, seed=int(rate * 1000))
            lt.join()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            stop_evt.set()
            st.join()
            s = summarize(recs, t0)
            s.update({"rate": rate, "arrivals": n_arr, "wall_s": round(wall, 2),
                      "throughput_tok_s": round(s["tokens"] / wall, 2),
                      "completed_req_s": round(s["n_ok"] / wall, 3)})
            res["rates"].append({"summary": s, "metrics": metrics})
            print(rate, s, flush=True)
            time.sleep(5.0)
    finally:
        srv.stop()
    return res


def exp_sweep2(out):
    """The sweep again on a quieter machine: 120-s windows, load recorded per rate."""
    res = {"rates": []}
    srv = sp.Server(out, "loadgen-sweep2", ["-np", "4", "-c", "8192", "--metrics"], port=PORT)
    try:
        one_request(PROMPTS[0], time.perf_counter(), {})
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            recs, seconds, t0 = [], 120.0, time.perf_counter()
            load_start = os.getloadavg()
            stop_evt, metrics = threading.Event(), []
            st = threading.Thread(target=scrape, args=(stop_evt, t0, metrics))
            st.start()
            lt, ths, n_arr = open_loop(rate, seconds, t0, recs, seed=int(rate * 1000) + 1)
            lt.join()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            stop_evt.set()
            st.join()
            s = summarize(recs, t0)
            s.update({"rate": rate, "arrivals": n_arr, "window_s": seconds, "realized_rate": round(n_arr / seconds, 3),
                      "wall_s": round(wall, 2), "throughput_tok_s": round(s["tokens"] / wall, 2),
                      "completed_req_s": round(s["n_ok"] / wall, 3), "load_start": load_start,
                      "load_end": os.getloadavg()})
            res["rates"].append({"summary": s, "metrics": metrics,
                                 "requests": [{k: (round(v - t0, 4) if k.startswith("t_") else v) for k, v in r.items()}
                                              for r in recs]})
            print(rate, s, flush=True)
            time.sleep(5.0)
    finally:
        srv.stop()
    return res


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t = time.time()
    r = {"stall": exp_stall, "sweep": exp_sweep, "sweep2": exp_sweep2}[name](out)
    r["_meta"] = {"experiment": name, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
                  "seconds": round(time.time() - t, 1), "n_predict": N_PREDICT, "load_end": os.getloadavg()}
    json.dump(r, open(os.path.join(out, f"loadgen-{name}.json"), "w"), indent=1)
    print("done", r["_meta"])
