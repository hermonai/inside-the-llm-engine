#!/usr/bin/env python3
"""Analyse a routing trace recorded by probe8 (granite3.1-moe:3b, 32 layers x 40 experts,
8 per token): popularity skew, token-to-token reuse, cache-policy replay and batch union,
each beside what uniform routing would give.

    python3 analyze_routing.py route-granite.json out.json
"""
import heapq
import json
import math
import sys
from collections import Counter, OrderedDict

d = json.load(open(sys.argv[1]))
prompts = d["prompts"]
L = len(prompts[0]["decode"][0])
K = prompts[0]["k"]
E = 40
res = {"layers": L, "experts": E, "k": K,
       "decode_tokens": sum(len(p["decode"]) for p in prompts),
       "prompt_tokens": sum(p["n_prompt"] for p in prompts)}

# 1. popularity: share of selections that go to each layer's 8 most-used experts
shares, entropies = [], []
for layer in range(L):
    c = Counter(e for p in prompts for step in p["decode"] for e in step[layer])
    total = sum(c.values())
    shares.append(sum(n for _, n in c.most_common(K)) / total)
    entropies.append(-sum(n / total * math.log2(n / total) for n in c.values()))
res["top8_share"] = {"mean": sum(shares) / L, "min": min(shares), "max": max(shares), "uniform": K / E}
res["entropy_bits"] = {"mean": sum(entropies) / L, "uniform": math.log2(E)}

# 2. reuse: overlap between consecutive tokens' expert sets in the same layer
overlaps = []
for p in prompts:
    dec = p["decode"]
    for t in range(1, len(dec)):
        for layer in range(L):
            overlaps.append(len(set(dec[t][layer]) & set(dec[t - 1][layer])) / K)
res["consecutive_overlap"] = {"mean": sum(overlaps) / len(overlaps), "uniform": K / E}

# 3. cache replay over the concatenated session (one user, prompts in order)
trace = [(layer, step[layer]) for p in prompts for step in p["decode"] for layer in range(L)]
half = (len(trace) // L // 2) * L


def lru(n_blocks, warm_from):
    cache, h = OrderedDict(), [0, 0]
    for c, (layer, ids) in enumerate(trace):
        pinned, pending = set(), []
        for e in ids:
            key = (layer, e)
            if key in cache:
                h[0] += c >= warm_from
                pinned.add(key)
                cache.move_to_end(key)
                continue
            if len(cache) + len(pending) >= n_blocks:
                victim = next(kk for kk in cache if kk not in pinned)
                del cache[victim]
            pending.append(key)
            h[1] += c >= warm_from
        for key in pending:
            cache[key] = True
    return 100.0 * h[0] / (h[0] + h[1])


def learned(n_blocks, warm_from):
    freq = Counter((layer, e) for layer, ids in trace[:warm_from] for e in ids)
    keep = {k for k, _ in freq.most_common(n_blocks)}
    tot = sum(len(ids) for _, ids in trace[warm_from:])
    return 100.0 * sum((layer, e) in keep for layer, ids in trace[warm_from:] for e in ids) / tot


def belady(n_blocks, warm_from):
    seq = [(layer, e) for layer, ids in trace for e in ids]
    nxt, last = [0] * len(seq), {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], math.inf)
        last[seq[i]] = i
    resident, heap, i, hw, tw = {}, [], 0, 0, 0
    for c, (layer, ids) in enumerate(trace):
        held = set()
        for e in ids:
            key = (layer, e)
            hit = key in resident
            if not hit and len(resident) >= n_blocks:
                stash = []
                while True:
                    nu, kk = heapq.heappop(heap)
                    if resident.get(kk) != -nu:
                        continue
                    if kk in held:
                        stash.append((nu, kk))
                        continue
                    del resident[kk]
                    break
                for s in stash:
                    heapq.heappush(heap, s)
            if c >= warm_from:
                tw += 1
                hw += hit
            resident[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            held.add(key)
            i += 1
    return 100.0 * hw / tw


total = L * E
res["cycle_bound_records"] = (L - 1) * K
res["replay"] = []
for f in (0.05, 0.10, 0.20, 0.25, 0.50):
    nb = int(f * total)
    res["replay"].append({"cache_frac": f, "blocks": nb, "lru": lru(nb, half),
                          "learned": learned(nb, half), "belady": belady(nb, half)})

# 4. union across B concurrent sequences: experts touched per layer when B different
#    prompts decode their step s together (steps shared by every prompt considered)
steps = min(len(p["decode"]) for p in prompts)
res["union"] = []
for B in (1, 2, 4, 8):
    fr = []
    for s in range(steps):
        for layer in range(L):
            u = set()
            for p in prompts[:B]:
                u |= set(p["decode"][s][layer])
            fr.append(len(u) / E)
    res["union"].append({"B": B, "measured": sum(fr) / len(fr), "uniform": 1 - (1 - K / E) ** B})

json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
