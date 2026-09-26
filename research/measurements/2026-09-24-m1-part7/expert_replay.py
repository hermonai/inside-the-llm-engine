#!/usr/bin/env python3
"""Replay Hermon's expert-routing trace against an expert cache, in Python.

Chapter 36 lab for 'Inside the LLM Engine'. The routing trace is regenerated
exactly as Hermon's crates/hermon-kernels/csrc/bench_experts.c (commit 2a3fd52)
generates it: the same 64-bit LCG and seed, the same per-layer permutation, the
same uniform and power-law draws and the same de-duplication. The cache follows
csrc/experts.c: one LRU over every resident (layer, expert) record, the blocks a
layer has already acquired are pinned until the layer is done, and a layer's
misses are installed at the most-recent end after its hits are touched.

Nothing here reads a disk. What the replay can check is the hit rate, which is a
property of the trace and the policy alone; bytes and tokens per second follow
from the record sizes and a device bandwidth.

    python3 expert_replay.py <out.json>
"""
import heapq
import json
import math
import sys
import time
from collections import OrderedDict, Counter

MASK = (1 << 64) - 1
SEED = 0x9E3779B97F4A7C15
L, E, K = 48, 128, 8                   # Qwen3-30B-A3B: layers, experts per layer, routed per token
FRACS = (0.02, 0.05, 0.10, 0.25, 0.50)
# Record bytes per layer: gate and up are Q4_K; down is Q6_K in 24 layers (llama.cpp's
# Q4_K_M rule: first and last eighth, and every third layer between) and Q4_K elsewhere.
Q4K = 768 * 2048 // 256 * 144          # 884,736 bytes
Q6K = 768 * 2048 // 256 * 210          # 1,290,240 bytes


def down_is_q6k(i, n=L):
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


REC = [2 * Q4K + (Q6K if down_is_q6k(i) else Q4K) for i in range(L)]
MAXREC = max(REC)


class LCG:
    def __init__(self):
        self.g = SEED

    def rnd(self, n):
        self.g = (self.g * 6364136223846793005 + 1442695040888963407) & MASK
        return (self.g >> 33) % n


def zipf(rng, n):
    u = rng.rnd(1 << 30) / float(1 << 30)
    idx = int(math.pow(float(n), 1.0 - u) - 1.0)
    return n - 1 if idx >= n else idx


def permutations(rng):
    perm = []
    for _ in range(L):
        p = list(range(E))
        for i in range(E, 1, -1):
            j = rng.rnd(i)
            p[i - 1], p[j] = p[j], p[i - 1]
        perm.append(p)
    return perm


def trace(mode, tokens, era="2a3fd52"):
    """[(layer, [expert ids])] in the order the benchmark acquires them. era "be0288a"
    reproduces the K6.1 benchmark's de-duplication loop, which restarts its scan at
    j = 1 after a bump (the loop's own increment follows the reset to 0), so a bumped
    id can still equal ids[0]; era "2a3fd52" is the corrected loop."""
    rng = LCG()
    perm = permutations(rng)
    rng.g = SEED                          # bench_experts.c resets the seed per configuration
    out = []
    for _ in range(tokens):
        for l in range(L):
            ids = []
            for i in range(K):
                pick = rng.rnd(E) if mode == "uniform" else zipf(rng, E)
                e = perm[l][pick]
                if era == "be0288a":
                    ids.append(e)
                    j = 0
                    while j < i:
                        if ids[j] == ids[i]:
                            ids[i] = (ids[i] + 1) % E
                            j = 0
                        j += 1
                    continue
                clash = True
                while clash:
                    clash = False
                    for j in range(i):
                        if ids[j] == e:
                            e = (e + 1) % E
                            clash = True
                            break
                ids.append(e)
            out.append((l, ids))
    return out


def lru(tr, n_blocks, warm_from=0):
    """Hermon's pinned LRU. Returns hits, misses, evictions, bytes read (all lookups and
    those at or after call index warm_from)."""
    cache = OrderedDict()                  # first = least recently used
    h = m = ev = 0
    hw = mw = 0
    byts = 0
    for c, (l, ids) in enumerate(tr):
        pinned, pending = set(), []
        for e in ids:
            key = (l, e)
            if key in cache:
                h += 1
                hw += c >= warm_from
                pinned.add(key)
                cache.move_to_end(key)
                continue
            if len(cache) + len(pending) >= n_blocks:
                victim = next((kk for kk in cache if kk not in pinned), None)
                if victim is None:
                    raise RuntimeError("every cached expert is pinned")
                del cache[victim]
                ev += 1
            pending.append(key)
        for key in pending:
            cache[key] = True
        m += len(pending)
        mw += len(pending) * (c >= warm_from)
        byts += len(pending) * REC[l]
    return {"hits": h, "misses": m, "evictions": ev, "hit_pct": 100.0 * h / (h + m),
            "warm_hit_pct": 100.0 * hw / max(1, hw + mw), "bytes_read": byts,
            "bytes_if_max_record": m * MAXREC}


def lru_single_pass(tr, n_blocks):
    """The K6.1-era acquire (be0288a): a miss claims a block, is read and is installed at
    the most-recent end at once, so a repeated id later in the same call is a hit."""
    cache = OrderedDict()
    h = m = ev = byts = 0
    dups = 0
    for l, ids in tr:
        pinned = set()
        dups += len(ids) - len(set(ids))
        for e in ids:
            key = (l, e)
            if key in cache:
                h += 1
                pinned.add(key)
                cache.move_to_end(key)
                continue
            if len(cache) >= n_blocks:
                victim = next((kk for kk in cache if kk not in pinned), None)
                if victim is None:
                    raise RuntimeError("every cached expert is pinned")
                del cache[victim]
                ev += 1
            cache[key] = True
            pinned.add(key)
            m += 1
            byts += REC[l]
    return {"hits": h, "misses": m, "evictions": ev, "hit_pct": 100.0 * h / (h + m),
            "bytes_read": byts, "bytes_if_max_record": m * MAXREC, "duplicate_ids": dups}


def belady(tr, n_blocks, warm_from=0):
    """Offline optimum for the same trace: on a miss with a full cache, evict the resident
    record whose next use is farthest away (never one the current layer holds)."""
    seq = [(l, e) for l, ids in tr for e in ids]
    call = [c for c, (_, ids) in enumerate(tr) for _ in ids]
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], math.inf)
        last[seq[i]] = i
    resident = {}                          # key -> next use
    heap = []                              # (-next use, key), lazily invalidated
    h = m = hw = mw = 0
    i = 0
    for c, (l, ids) in enumerate(tr):
        held = set()
        for e in ids:
            key = (l, e)
            if key in resident:
                h += 1
                hw += c >= warm_from
            else:
                if len(resident) >= n_blocks:
                    stash = []
                    while True:
                        nu, kk = heapq.heappop(heap)
                        if kk in resident and resident[kk] == -nu and kk not in held:
                            del resident[kk]
                            break
                        if kk in resident and resident[kk] == -nu:
                            stash.append((nu, kk))
                    for s in stash:
                        heapq.heappush(heap, s)
                m += 1
                mw += c >= warm_from
            resident[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            held.add(key)
            i += 1
    return {"hit_pct": 100.0 * h / (h + m), "warm_hit_pct": 100.0 * hw / max(1, hw + mw)}


def static_top(tr, n_blocks, warm_from=0):
    """Pin the n_blocks records the trace uses most, chosen with hindsight: the best any
    frequency-based policy could do on a stationary router."""
    freq = Counter((l, e) for l, ids in tr for e in ids)
    keep = {k for k, _ in freq.most_common(n_blocks)}
    tot = hit = tw = hw = 0
    for c, (l, ids) in enumerate(tr):
        for e in ids:
            tot += 1
            hit += (l, e) in keep
            if c >= warm_from:
                tw += 1
                hw += (l, e) in keep
    return {"hit_pct": 100.0 * hit / tot, "warm_hit_pct": 100.0 * hw / max(1, tw)}


def static_learned(tr, n_blocks, warm_from):
    """Pin the n_blocks records used most in the first part of the trace (calls before
    warm_from) and score the rest: a frequency policy learned online, no hindsight."""
    freq = Counter((l, e) for l, ids in tr[:warm_from] for e in ids)
    keep = {k for k, _ in freq.most_common(n_blocks)}
    tw = hw = 0
    for l, ids in tr[warm_from:]:
        for e in ids:
            tw += 1
            hw += (l, e) in keep
    return {"warm_hit_pct": 100.0 * hw / max(1, tw)}


def main(path):
    t0 = time.time()
    total = L * E
    res = {"shape": {"layers": L, "experts": E, "routed": K, "record_bytes_q6k_down": REC[0],
                     "record_bytes_q4k_down": REC[6], "q6k_down_layers": sum(map(down_is_q6k, range(L))),
                     "bytes_per_token_all_miss": K * sum(REC), "container_bytes": E * sum(REC)},
           "replication_64_tokens": [], "replication_k61_be0288a": [], "long_2048_tokens": []}
    for mode in ("uniform", "skewed"):
        tr61 = trace(mode, 64, era="be0288a")
        for f in FRACS:
            nb = max(int(f * total), K + 1)
            r = lru_single_pass(tr61, nb)
            r.update({"routing": mode, "cache_frac": f, "blocks": nb, "cache_GiB": nb * MAXREC / 2**30,
                      "read_GiB": r["bytes_read"] / 2**30,
                      "read_GiB_if_max_record": r["bytes_if_max_record"] / 2**30})
            res["replication_k61_be0288a"].append(r)
        tr = trace(mode, 64)
        for f in FRACS:
            nb = max(int(f * total), K + 1)
            r = lru(tr, nb)
            r.update({"routing": mode, "cache_frac": f, "blocks": nb,
                      "cache_GiB": nb * MAXREC / 2**30, "read_GiB": r["bytes_read"] / 2**30,
                      "read_GiB_if_max_record": r["bytes_if_max_record"] / 2**30})
            res["replication_64_tokens"].append(r)
        tr = trace(mode, 2048)
        warm = 1024 * L                    # score only the second half of the trace
        for f in FRACS:
            nb = max(int(f * total), K + 1)
            row = {"routing": mode, "cache_frac": f, "blocks": nb,
                   "lru": lru(tr, nb, warm), "belady": belady(tr, nb, warm),
                   "static_top": static_top(tr, nb, warm),
                   "static_learned": static_learned(tr, nb, warm)}
            res["long_2048_tokens"].append(row)
    res["seconds"] = round(time.time() - t0, 1)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("K6.1 replication (be0288a trace and single-pass cache, 64 tokens, cold):")
    for r in res["replication_k61_be0288a"]:
        print(f"  {r['routing']:8s} {r['cache_GiB']:5.2f} GiB {r['blocks']:5d} blocks  hit {r['hit_pct']:5.1f}%"
              f"  read {r['read_GiB']:5.1f} GiB (x max record: {r['read_GiB_if_max_record']:5.1f})"
              f"  duplicate ids {r['duplicate_ids']}")
    print("replication (64 tokens, cold):")
    for r in res["replication_64_tokens"]:
        print(f"  {r['routing']:8s} {r['cache_GiB']:5.2f} GiB {r['blocks']:5d} blocks  hit {r['hit_pct']:5.1f}%"
              f"  read {r['read_GiB']:5.1f} GiB (x max record: {r['read_GiB_if_max_record']:5.1f})")
    print("steady state (tokens 1024-2047 of 2048):")
    for r in res["long_2048_tokens"]:
        print(f"  {r['routing']:8s} {r['cache_frac']:4.2f}  LRU {r['lru']['warm_hit_pct']:5.1f}%"
              f"  static-top {r['static_top']['warm_hit_pct']:5.1f}%  learned {r['static_learned']['warm_hit_pct']:5.1f}%"
              f"  Belady {r['belady']['warm_hit_pct']:5.1f}%")
    print("seconds", res["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
