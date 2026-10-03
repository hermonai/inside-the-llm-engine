"""Request accounting plus an optional synchronous Python experiment.

Default timestamps are illustrative. --measure times toy integer matrix work,
not a model, accelerator, or inference service. Only the standard library.
"""
import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import get_clock_info, perf_counter_ns


@dataclass(frozen=True)
class Request:
    name: str
    planned_ns: int
    send_ns: int
    tokens_ns: tuple
    terminal_ns: int
    status: str = "ok"
    correct: bool = True
    clock: str = "client-monotonic"

    def __post_init__(self):
        times = (self.planned_ns, self.send_ns, *self.tokens_ns, self.terminal_ns)
        if any(type(t) is not int or t < 0 for t in times):
            raise ValueError("timestamps must be nonnegative integer nanoseconds")
        if tuple(sorted(times)) != times:
            raise ValueError("events must be ordered in one clock domain")
        if self.status not in {"ok", "error", "timeout", "cancelled"}:
            raise ValueError("unknown terminal status")
        if self.status == "ok" and not self.tokens_ns:
            raise ValueError("this token-workload contract needs a delivered token")
        if not self.name or not self.clock or type(self.correct) is not bool:
            raise ValueError("missing identity, clock, or correctness decision")

    def metrics(self):
        gaps = tuple(b-a for a,b in zip(self.tokens_ns,self.tokens_ns[1:]))
        first = self.tokens_ns[0] if self.tokens_ns else None
        return {"client_delay_ns": self.send_ns-self.planned_ns,
                "ttft_from_send_ns": None if first is None else first-self.send_ns,
                "ttft_from_plan_ns": None if first is None else first-self.planned_ns,
                "gaps_ns": gaps,
                "mean_gap_ns": Fraction(sum(gaps),len(gaps)) if gaps else None,
                "terminal_from_send_ns": self.terminal_ns-self.send_ns}


def percentile(values, probability, method="nearest-rank"):
    """Descriptive sample quantile, not a confidence interval."""
    data = sorted(Fraction(v) for v in values)
    p = Fraction(probability)
    if not data or not 0 <= p <= 1:
        raise ValueError("need nonempty data and a probability in [0,1]")
    if method == "nearest-rank":
        return data[max(1,math.ceil(p*len(data)))-1]
    if method == "inclusive-linear":
        position = p*(len(data)-1)
        lower = position.numerator//position.denominator
        upper = min(lower+1,len(data)-1)
        return data[lower]+(position-lower)*(data[upper]-data[lower])
    raise ValueError("unknown percentile method")


def cohort(requests, start_ns, stop_ns, ttft_limit_ns, gap_limit_ns):
    """A fully drained planned-arrival cohort, not a streaming-window estimator.

    Good tokens require correct successful completion, planned-arrival TTFT
    within the limit and request mean gap within the limit, when defined.
    """
    requests = tuple(requests)
    if any(type(t) is not int or t < 0 for t in
           (start_ns,stop_ns,ttft_limit_ns,gap_limit_ns)) or stop_ns <= start_ns:
        raise ValueError("invalid observation window or target")
    if len({r.name for r in requests}) != len(requests):
        raise ValueError("request identities must be unique")
    if len({r.clock for r in requests}) > 1:
        raise ValueError("cannot merge unsynchronized clock domains")
    if any(r.planned_ns < start_ns or r.planned_ns >= stop_ns or
           r.terminal_ns > stop_ns for r in requests):
        raise ValueError("cohort must arrive in window and drain by its end")
    wire = valid = good = errors = wrong = 0
    for r in requests:
        # Count this drained cohort's complete output, including an event
        # at the drain endpoint; do not apply a second streaming-window cut.
        wire += len(r.tokens_ns)
        if r.status != "ok":
            errors += 1
            continue
        if not r.correct:
            wrong += 1
            continue
        valid += len(r.tokens_ns)
        m = r.metrics()
        if (m["ttft_from_plan_ns"] <= ttft_limit_ns and
            (m["mean_gap_ns"] is None or m["mean_gap_ns"] <= gap_limit_ns)):
            good += len(r.tokens_ns)
    seconds = Fraction(stop_ns-start_ns,10**9)
    return {"requests":len(requests),"failed_requests":errors,
            "incorrect_requests":wrong,"wire_tokens":wire,
            "valid_completed_tokens":valid,"good_tokens":good,
            "wire_tokens_per_s":wire/seconds,
            "valid_completed_tokens_per_s":valid/seconds,
            "good_tokens_per_s":good/seconds}


def fixture():
    ms = 1_000_000
    def r(name,planned,send,tokens,terminal,status="ok",correct=True):
        return Request(name,planned*ms,send*ms,tuple(t*ms for t in tokens),
                       terminal*ms,status,correct)
    return (r("A",0,0,[5,9,13],14), r("B",2,8,[12,30],32),
            r("C",4,4,[15],40,"timeout"), r("D",6,6,[8,10],11,correct=False))


def fcfs(arrivals, service):
    """Derived single-worker deterministic queue; no inference scheduling."""
    if len(arrivals) != len(service) or list(arrivals) != sorted(arrivals):
        raise ValueError("ordered arrivals and one service demand per arrival required")
    if any(type(v) is not int or v < 0 for v in (*arrivals,*service)):
        raise ValueError("nonnegative integer times required")
    free = 0
    rows = []
    for arrival,demand in zip(arrivals,service):
        start = max(arrival,free)
        free = start+demand
        rows.append((arrival,start,free))
    return rows


def matrices(n):
    return ([[((3*i+k)%11)-5 for k in range(n)] for i in range(n)],
            [[((2*k+j)%7)-3 for j in range(n)] for k in range(n)])


def oracle(a,b):
    """Independent k/i/j loop, exact integer arithmetic."""
    n = len(a)
    out = [[0]*n for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                out[i][j] += a[i][k]*b[k][j]
    return out


def baseline(a,b):
    n = len(a)
    out = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[i][j] += a[i][k]*b[k][j]
    return out


def pack(b):
    return list(zip(*b))


def packed_dot(a,bt):
    return [[sum(x*y for x,y in zip(row,col)) for col in bt] for row in a]


def checked(actual, expected):
    if actual != expected:
        raise ValueError("oracle failed: no valid performance result")


def measure(n=24, repetitions=9, warmups=2):
    if (type(n) is not int or not 1 <= n <= 64 or
        type(repetitions) is not int or not 1 <= repetitions <= 31 or
        type(warmups) is not int or not 0 <= warmups <= 10):
        raise ValueError("bounded experiment: size 1..64, runs 1..31, warmups 0..10")
    a,b = matrices(n)
    expected = oracle(a,b)
    checked(baseline(a,b),expected)
    checked(packed_dot(a,pack(b)),expected)
    for _ in range(warmups):
        checked(baseline(a,b),expected)
        checked(packed_dot(a,pack(b)),expected)
    rows = []
    for run in range(repetitions):
        order = ("baseline","packed") if run % 2 == 0 else ("packed","baseline")
        for variant in order:
            begin = perf_counter_ns()
            if variant == "packed":
                bt = pack(b)
                ready = perf_counter_ns()
                actual = packed_dot(a,bt)
            else:
                ready = begin
                actual = baseline(a,b)
            end = perf_counter_ns()  # synchronous return: output exists
            checked(actual,expected)  # verification excluded, never skipped
            rows.append({"pair":run,"variant":variant,"setup_ns":ready-begin,
                         "compute_ns":end-ready,"total_ns":end-begin})
    clock = get_clock_info("perf_counter")
    canonical = json.dumps([a,b],separators=(",",":"))
    return {"evidence_kind":"measured synchronous Python toy workload",
            "size":n,"repetitions":repetitions,"warmups_per_variant":warmups,
            "oracle":"exact k/i/j integer matrix product, every output",
            "scope":"pack plus compute; output allocation included; oracle excluded",
            "order":"alternating AB/BA pairs; no randomized-order claim",
            "python":platform.python_version(),"machine":platform.machine(),
            "system":platform.system(),"release":platform.release(),
            "clock":{"implementation":clock.implementation,
                     "monotonic":clock.monotonic,"resolution_s":clock.resolution},
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs_sha256":hashlib.sha256(canonical.encode()).hexdigest(),
            "raw":rows,
            "median_total_ns":{v:median(r["total_ns"] for r in rows if r["variant"]==v)
                               for v in ("baseline","packed")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--size",type=int,default=24)
    parser.add_argument("--repetitions",type=int,default=9)
    parser.add_argument("--warmups",type=int,default=2)
    args = parser.parse_args()
    if args.measure:
        result = measure(args.size,args.repetitions,args.warmups)
    else:
        result = {"evidence_kind":"derived from illustrative timestamps",
                  "cohort":cohort(fixture(),0,40_000_000,10_000_000,10_000_000),
                  "sample_p95_nearest":percentile([9,10,10,11,60],Fraction(95,100)),
                  "sample_p95_inclusive":percentile([9,10,10,11,60],Fraction(95,100),"inclusive-linear"),
                  "open_loop_queue_ms":fcfs([0,4,8,12,16],[4,4,20,4,4])}
    print(json.dumps(result,default=str,indent=2))


if __name__ == "__main__":
    main()
