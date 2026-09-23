"""Ch 13: per-operation dispatch cost on MPS, and what compilation removes.
many     - n dependent tiny ops (x = x + 1 on 1,024 floats), one dispatch each;
one      - a single op producing the same result;
compiled - `many` under torch.compile, which can fuse the chain into one kernel.
Every variant must equal `one` exactly before it is timed."""
import json, statistics, time, torch
dev = torch.device("mps")
def timeit(fn, reps=15, warm=3):
    for _ in range(warm): fn(); torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
res = {"torch": torch.__version__, "runs": []}
x0 = torch.zeros(1024, device=dev)
for n in (1, 10, 100, 1000):
    def many(x, n=n):
        for _ in range(n): x = x + 1
        return x
    def one():
        return x0 + float(n)
    assert torch.equal(many(x0), one())
    rec = {"ops": n, "many_s": timeit(lambda: many(x0)), "one_s": timeit(one)}
    rec["per_op_us"] = (rec["many_s"] - rec["one_s"]) / max(n - 1, 1) * 1e6
    try:
        torch._dynamo.reset()
        cm = torch.compile(many)
        t0 = time.perf_counter(); r = cm(x0); torch.mps.synchronize(); rec["compile_first_call_s"] = time.perf_counter() - t0
        assert torch.equal(r, one())
        rec["compiled_s"] = timeit(lambda: cm(x0))
    except Exception as e:  # record, do not hide, a compiler failure
        rec["compiled_error"] = (type(e).__name__ + ": " + str(e)).splitlines()[0][:240]
    res["runs"].append(rec)
print(json.dumps(res, indent=1))
