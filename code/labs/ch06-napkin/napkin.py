#!/usr/bin/env python3
"""Chapter 6 lab: a napkin-math calculator for decode.

Prices one decode step from a model's shape and a machine's datasheet with the
chapter's step-time model,

    t_step(B, S) = max((W + B*S*m_kv) / (eta_b*beta), B*F(S) / (eta_p*pi)) + t0,

and derives everything a deployment is judged by: time per output token,
tokens per second per user and in aggregate, the largest batch that fits, the
context above which decode stays bandwidth-bound at any batch, the batch a
latency objective allows (goodput) and dollars per million output tokens.

    python3 napkin.py --model qwen3-8b --machine h100 --context 4096
    python3 napkin.py --model qwen3-8b --machine h100 --bytes 1 --context 4096 --tpot 15
    python3 napkin.py --config path/to/config.json --machine b200 --bytes 1
    python3 napkin.py --model llama-3.2-3b --machine m1-gpu --bytes 0.626 --kv-bytes 2 \
        --context 2064 --eta-b 0.72 --eta-p 0.7 --batches 1,2,4,8

Efficiencies default to eta_b = 0.8 and eta_p = 0.6 (illustrative); measure
yours with the labs of Chapters 3 and 4 and pass --eta-b and --eta-p.
Standard library only.
"""
import argparse
import json
import math

# Published configurations (research/FRONTIER.md), in Hugging Face's key names.
MODELS = {
    "qwen3-8b": {"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
                 "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
                 "vocab_size": 151936, "tie_word_embeddings": False},
    "llama-3.2-3b": {"hidden_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 24,
                     "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 8192,
                     "vocab_size": 128256, "tie_word_embeddings": True},
    "qwen3-30b-a3b": {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
                      "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 6144,
                      "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
                      "vocab_size": 151936, "tie_word_embeddings": False},
}

# Datasheet numbers (research/FRONTIER.md): bandwidth B/s, dense peaks FLOP/s by
# bytes per operand, memory bytes, price per hour (illustrative, volatile).
MACHINES = {
    "m1-gpu": {"beta": 68e9, "pi": {2: 2.6e12, 1: 2.6e12}, "memory": 12123 * 2**20, "price": 0.0},
    "h100": {"beta": 3.35e12, "pi": {2: 989e12, 1: 1979e12}, "memory": 80e9, "price": 3.99},
    "h200": {"beta": 4.8e12, "pi": {2: 989e12, 1: 1979e12}, "memory": 141e9, "price": 0.0},
    "b200": {"beta": 8e12, "pi": {2: 2250e12, 1: 4500e12}, "memory": 180e9, "price": 6.69},
}


class Model:
    """Bytes and FLOPs per decode token from a config, at b bytes per weight and cache element."""

    def __init__(self, cfg, b_w, b_kv):
        d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
        hq, hkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        dh = cfg.get("head_dim") or d // hq
        V = cfg["vocab_size"]
        attn = d * hq * dh + 2 * d * hkv * dh + hq * dh * d
        self.experts = cfg.get("num_experts", 0)
        self.k = cfg.get("num_experts_per_tok", 0)
        if self.experts:
            expert = 3 * d * cfg["moe_intermediate_size"]
            self.expert_params = L * self.experts * expert
            dense_ffn = L * d * self.experts  # the router
        else:
            self.expert_params = 0
            dense_ffn = L * 3 * d * cfg["intermediate_size"]
        table = V * d
        self.params = L * attn + dense_ffn + self.expert_params + table * (1 if cfg.get("tie_word_embeddings") else 2)
        # weights a step multiplies: everything but the input table (the output
        # projection is the table itself when tied)
        self.shared = L * attn + dense_ffn + table
        self.b_w, self.layers, self.hq, self.dh = b_w, L, hq, dh
        self.m_kv = 2 * L * hkv * dh * b_kv
        self.resident = self.params * b_w

    def weight_bytes(self, batch):
        """Weight bytes one step reads; a mixture of experts reads the union of the batch's experts."""
        union = 1 - (1 - self.k / self.experts) ** batch if self.experts else 0.0
        return (self.shared + self.expert_params * union) * self.b_w

    def flops(self, context):
        """FLOPs per token: two per multiplied weight, plus attention over the context."""
        active = self.shared + (self.expert_params * self.k / self.experts if self.experts else 0)
        return 2 * active + 4 * self.layers * self.hq * self.dh * context


def step(model, machine, batch, context, eta_b, eta_p, t0):
    """Seconds per decode step and which roof binds."""
    b = model.b_w if model.b_w in machine["pi"] else 2
    t_mem = (model.weight_bytes(batch) + batch * context * model.m_kv) / (eta_b * machine["beta"])
    t_math = batch * model.flops(context) / (eta_p * machine["pi"][b])
    return max(t_mem, t_math) + t0, ("bandwidth" if t_mem >= t_math else "compute")


def b_max(model, machine, context, reserve):
    return int((machine["memory"] - model.resident - reserve) // (context * model.m_kv))


def knee_context(model, machine, eta_b, eta_p):
    """Mean context above which adding sequences adds more memory time than compute time."""
    b = model.b_w if model.b_w in machine["pi"] else 2
    # F(S) grows with S too; solve S * m_kv / (eta_b beta) = F(S) / (eta_p pi) by bisection
    lo, hi = 0.0, 1e7
    f = lambda s: s * model.m_kv / (eta_b * machine["beta"]) - model.flops(s) / (eta_p * machine["pi"][b])
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lo


def dollars_per_million(machine, batch, t):
    return machine["price"] / 3600 / (batch / t) * 1e6 if machine["price"] else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3-8b", choices=sorted(MODELS))
    ap.add_argument("--config", help="a Hugging Face config.json instead of a preset")
    ap.add_argument("--machine", default="h100", choices=sorted(MACHINES))
    ap.add_argument("--bytes", type=float, default=2, help="bytes per weight (2 = BF16, 1 = FP8, 0.63 = Q4_K_M)")
    ap.add_argument("--kv-bytes", type=float, help="bytes per cache element (default: same as --bytes)")
    ap.add_argument("--batches", help="comma-separated batch sizes to print (default: 1,8,32,64 and the largest)")
    ap.add_argument("--context", type=int, default=4096, help="mean tokens in each sequence's cache")
    ap.add_argument("--eta-b", type=float, default=0.8)
    ap.add_argument("--eta-p", type=float, default=0.6)
    ap.add_argument("--t0", type=float, default=0.0, help="fixed per-step overhead in ms")
    ap.add_argument("--reserve", type=float, default=4.0, help="GB kept for activations and workspace")
    ap.add_argument("--tpot", type=float, help="time-per-output-token objective in ms")
    a = ap.parse_args()
    cfg = json.load(open(a.config)) if a.config else MODELS[a.model]
    model, machine = Model(cfg, a.bytes, a.kv_bytes or a.bytes), MACHINES[a.machine]
    t0 = a.t0 / 1e3
    top = b_max(model, machine, a.context, a.reserve * 1e9)
    print(f"{a.config or a.model} on {a.machine}: {model.params / 1e9:.2f} B parameters, "
          f"{model.resident / 1e9:.1f} GB resident, {model.weight_bytes(1) / 1e9:.1f} GB read per step at B = 1, "
          f"{model.m_kv / 1024:.0f} KiB of cache per token, {model.flops(a.context) / 1e9:.1f} GFLOP per token")
    print(f"batch that fits at {a.context} tokens of context: {top}")
    print(f"{'B':>5} {'step ms':>9} {'tok/s/user':>11} {'tok/s':>9} {'bound':>10} {'$/M tokens':>11}")
    batches = [int(x) for x in a.batches.split(",")] if a.batches else sorted({1, 8, 32, 64, top} - {0})
    for bsz in batches:
        if bsz > top:
            continue
        t, bound = step(model, machine, bsz, a.context, a.eta_b, a.eta_p, t0)
        print(f"{bsz:>5} {t * 1e3:>9.1f} {1 / t:>11.0f} {bsz / t:>9.0f} {bound:>10} "
              f"{dollars_per_million(machine, bsz, t):>11.2f}")
    print(f"decode stays bandwidth-bound at any batch above {knee_context(model, machine, a.eta_b, a.eta_p):.0f} "
          f"tokens of mean context")
    if a.tpot:
        best = max((bsz for bsz in range(1, top + 1)
                    if step(model, machine, bsz, a.context, a.eta_b, a.eta_p, t0)[0] * 1e3 <= a.tpot), default=0)
        if best:
            t, _ = step(model, machine, best, a.context, a.eta_b, a.eta_p, t0)
            print(f"TPOT <= {a.tpot:g} ms allows B <= {best}: {best / t:.0f} tok/s, "
                  f"${dollars_per_million(machine, best, t):.2f} per million")
        else:
            print(f"TPOT <= {a.tpot:g} ms is not reachable at any batch")


if __name__ == "__main__":
    main()
