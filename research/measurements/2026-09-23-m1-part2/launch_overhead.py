"""Ch 13: per-operation dispatch overhead on MPS.
N dependent tiny ops (x = x + 1 on 1,024 floats) vs one op doing the same arithmetic."""
import json, statistics, time, torch
dev = torch.device("mps")
def timeit(fn, reps=7, warm=2):
    for _ in range(warm): fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
res = {"torch": torch.__version__, "runs": []}
x0 = torch.zeros(1024, device=dev)
for n in (1, 10, 100, 1000):
    def many():
        x = x0
        for _ in range(n): x = x + 1
        return x
    def one():
        return x0 + float(n)
    assert torch.equal(many(), one())
    t_many, t_one = timeit(many), timeit(one)
    res["runs"].append({"ops": n, "many_s": t_many, "one_s": t_one, "per_op_us": (t_many - t_one) / max(n - 1, 1) * 1e6})
print(json.dumps(res, indent=1))
