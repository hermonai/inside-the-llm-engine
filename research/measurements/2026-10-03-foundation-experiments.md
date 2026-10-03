# Accounting and experiment lessons — 2026-10-03

Printed Chapters 16–17. Most quantities are **derived from illustrative
inputs**. The optional Python matrix experiment below is **measured**:
it is not a native GEMM, an accelerator, a trained model or a service.

## Environment and executed source

Apple M1; macOS 26.6.2 (25G83), Darwin 25.6.0, arm64; Python 3.9.6.
The process ran in the ordinary local environment; no cache purge, thermal
control, process isolation or machine-state stabilization is claimed.
No model/quantization applies.

Measurements were collected on top of source revision 6c87313 with the
new teaching files uncommitted. Exact executed source identity is SHA-256
5899f7a1bb6d46454758473c9a87b3afbcfb2903147d8b4c04e1d071b8b4d774
for code/foundations/experiment_trace.py.
Canonical JSON inputs [a,b], separators (',',':'), have SHA-256
0413e2c3d8b41099eb25648efb142701c2b275799956e5464ec21c5d93778a5a.
The full PDF build record hashes the lesson and test sources.

## Commands and scope

```bash
python3 code/foundations/performance_contract.py
python3 code/foundations/experiment_trace.py
python3 -m unittest discover -s code/foundations -p 'test_*.py'
python3 code/foundations/experiment_trace.py \
  --measure --size 24 --repetitions 9 --warmups 2
```

The last command was invoked twice. All commands exited zero. Default
trace/accounting modes contain no timed work.

Input matrices use a[i][k]=((3*i+k)%11)-5 and
b[k][j]=((2*k+j)%7)-3, for each index in 0..23.
Each output contains 24 exact integer products. No floating-point
FLOP-rate inference is made.

An independent k/i/j loop constructs every expected integer output.
The baseline is an explicit i/j/k loop; the candidate packs right-hand
columns, then computes generator-based dot products.
Both outputs are verified before warm-up and after every timed call.
Both variants have two warm-ups. Nine pairs alternate AB/BA order.
This is alternating-order teaching code, not randomized order or an
independent-run statistical guarantee.

Elapsed clock: perf_counter_ns, implementation mach_absolute_time(),
monotonic true, reported resolution 4.166666666666667e-08 seconds.
Nanosecond units are not an accuracy claim. Calls are synchronous.

Total begins before packing (candidate) and ends after output construction.
Output allocation is included. Input creation, reference computation,
warm-up and result checking are excluded. Candidate setup and compute are
also retained separately. Baseline setup is zero by this convention.
Each row's total equals setup plus compute. Output verification is not timed
but is never skipped.

## Derived outputs and independent checks

Projection: 128 weights; eight illustrative sixteen-weight groups with
eight code bytes and two scale bytes each occupy 80 bytes.
Three work rows need 96 input bytes and 48 output bytes, giving 224
one-boundary payload bytes, 768 conventional FLOPs and intensity 24/7.
This includes metadata, not padding, conversion or observed transactions.

Private cache with lengths [1,5,9], 64 bytes/token and four-token blocks:
960 logical bytes, 1536 reserved bytes, 576 unused bytes, six entries
occupying 24 bytes in the chosen four-byte table-entry format.
Conditional resource times: memory 1/50 s, compute 1/125 s;
ideal-overlap bound 1/50 s, serial resource model 7/250 s.

Four-request cohort, fully drained at 40 ms:
eight wire tokens, five correct-completion tokens, three accepted tokens.
Rates: 200, 125 and 75 tokens/s. One timeout and one incorrect response
are retained. Acceptance uses intended-arrival TTFT at most 10 ms and
request mean gap at most 10 ms, when defined.

Sample [9,10,10,11,60] ms: nearest-rank p95 is 60 ms; inclusive linear
p95 is 251/5 = 50.2 ms. The latter agrees with Python statistics.
Neither estimates an extreme population tail reliably.

The suite adds eighteen tests: seven unit/count contracts and eleven
experiment/trace tests. Work is also enumerated independently of its
formula; private padding is enumerated independently of ceiling division.
Trace failures include out-of-order timestamps, mixed clock domains,
duplicate identities, undrained cohorts and incorrect candidate output.
Integer-product correctness fixtures cover sizes 1 through 11 before the
24-wide measured workload. A record test also recomputes both medians
from the raw CSV rows and checks each duration decomposition.

## Raw run 1

Rows preserve the execution order. Units for every duration: nanoseconds.

```csv
pair,variant,setup_ns,compute_ns,total_ns
0,baseline,0,1487750,1487750
0,packed,2083,812209,814292
1,packed,2833,804584,807417
1,baseline,0,1490458,1490458
2,baseline,0,1490500,1490500
2,packed,2750,798125,800875
3,packed,2708,796833,799541
3,baseline,0,1605833,1605833
4,baseline,0,1503666,1503666
4,packed,4416,809709,814125
5,packed,2792,799958,802750
5,baseline,0,1483458,1483458
6,baseline,0,1491000,1491000
6,packed,2625,800375,803000
7,packed,2625,803209,805834
7,baseline,0,1482334,1482334
8,baseline,0,1506625,1506625
8,packed,2791,816917,819708
```

Median total: baseline 1,490,500 ns; candidate 805,834 ns.
Total ranges: baseline 1,482,334–1,605,833 ns; candidate 799,541–819,708 ns.

## Raw run 2

```csv
pair,variant,setup_ns,compute_ns,total_ns
0,baseline,0,1501083,1501083
0,packed,4209,812125,816334
1,packed,2709,806625,809334
1,baseline,0,1510041,1510041
2,baseline,0,1488417,1488417
2,packed,3167,803833,807000
3,packed,2750,799875,802625
3,baseline,0,1488250,1488250
4,baseline,0,1539125,1539125
4,packed,2708,894834,897542
5,packed,4209,821000,825209
5,baseline,0,1534292,1534292
6,baseline,0,1508000,1508000
6,packed,3458,814000,817458
7,packed,2875,806333,809208
7,baseline,0,1489333,1489333
8,baseline,0,1488041,1488041
8,packed,2667,802750,805417
```

Median total: baseline 1,501,083 ns; candidate 809,334 ns.
Total ranges: baseline 1,488,041–1,539,125 ns; candidate 802,625–897,542 ns.

## Falsification and limits

The test corrupts a candidate coordinate and independently requires the
output check to raise. A patched incorrect candidate on the measured route
fails before any performance report is accepted. This is a deliberately
broken program, not a naturally observed optimization regression.

Candidate medians were lower in both invocations, including packing.
These are tiny synchronous Python measurements under one uncontrolled
machine state. Indexing, interpreter-loop structure, allocation and memory
traversal all changed, so the data does not isolate cache behavior.
There is no inference throughput, GPU speedup, confidence interval or
production SLO result. The source and input hashes identify the experiment,
not a promise to reproduce identical nanosecond values.
