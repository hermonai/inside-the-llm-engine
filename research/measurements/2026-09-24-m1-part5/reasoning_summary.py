#!/usr/bin/env python3
"""Summarize reasoning.json: strict scoring (the requested 'Answer: N' line) and
lenient scoring (also accept the last \\boxed{N}); tokens and times per budget."""
import json, re, statistics as st, sys

d = json.load(open(sys.argv[1]))
BOX = re.compile(r"\\boxed\{\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\}")
print(f"{'budget':>7} {'strict':>6} {'lenient':>7} {'hit limit':>9} {'reasoning mean/median':>22} {'answer mean':>11} {'completion mean':>15} {'total tokens':>12} {'seconds mean':>12}")
for b in (-1, 256, 0):
    rs = [r for r in d["runs"] if r["budget"] == b]
    strict = sum(r["correct"] for r in rs)
    lenient = 0
    for r in rs:
        ok = r["correct"]
        if not ok and r["got"] is None:
            m = BOX.findall(r["content_tail"])
            ok = bool(m) and abs(float(m[-1].replace(",", "")) - r["expected"]) < 1e-9
        lenient += ok
    hit = sum(r["finish_hit_limit"] for r in rs)
    reas = [r["reasoning_tokens"] for r in rs]
    print(f"{b:>7} {strict:>4}/8 {lenient:>5}/8 {hit:>9} {st.mean(reas):>12.0f} / {st.median(reas):<7.0f} "
          f"{st.mean(r['answer_tokens'] for r in rs):>11.0f} {st.mean(r['completion_tokens'] for r in rs):>15.0f} "
          f"{sum(r['completion_tokens'] for r in rs):>12} {st.mean(r['elapsed_s'] for r in rs):>12.1f}")
rates = [r["completion_tokens"] / (r["predicted_ms"] / 1000) for r in d["runs"]]
print(f"decode rate across all 24 runs: {min(rates):.2f}-{max(rates):.2f} tokens/s (server timings)")
