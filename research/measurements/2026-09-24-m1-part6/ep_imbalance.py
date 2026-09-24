#!/usr/bin/env python3
"""Derived, not measured: expected straggler ratio (busiest device's token-expert
pairs over the mean) under expert parallelism with uniformly random routing.
Each token picks k distinct experts of E uniformly; experts are placed
contiguously, E/G per device; 300 trials per case, seed 1. Compares with the
normal approximation 1 + sqrt(2 ln G / mu), mu = B k / G."""
import math
import random
import statistics as st

random.seed(1)


def sim(E, k, G, B, trials=300):
    per = E // G
    ratios = []
    for _ in range(trials):
        load = [0] * G
        for _ in range(B):
            for e in random.sample(range(E), k):
                load[e // per] += 1
        ratios.append(max(load) / (B * k / G))
    return st.mean(ratios)


print("E k G tokens_per_step mean_pairs_per_device simulated approximation")
for (E, k, G, B) in [(256, 8, 32, 4096), (256, 8, 32, 512), (256, 8, 32, 128), (256, 8, 128, 4096), (256, 8, 128, 512)]:
    mu = B * k / G
    print(E, k, G, B, f"{mu:.0f}", f"{sim(E, k, G, B):.3f}", f"{1 + math.sqrt(2 * math.log(G) / mu):.3f}")
