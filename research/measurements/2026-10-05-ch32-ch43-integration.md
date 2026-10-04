# Chapters 32–43 integration — 2026-10-05

This is a deterministic mechanism-integration record, not a GPU benchmark.
The public source snapshots checked for this pass are in [FRONTIER.md](../FRONTIER.md).
Earlier M1 measurements remain unchanged. No independent-review VERIFIED status is asserted.

## Package provenance

The supplied archives were inspected as data before integration. All entry names
were relative, remained within their extraction root, and contained no symlinks.
The already-dirty checkout was backed up outside the repository. The integration
script was not rerun over that checkout; the remaining steps were completed in place.

| Input archive | SHA-256 |
| --- | --- |
| ch32-continuous-batching-package.zip | 70b408cf06eaceb97ae75e0c644e8775c016cc7570bd3295037bd2e554336a7a |
| ch33-paged-kv-package.zip | 5035fd9983e34a9cc5d5c4a2c6b853d2fa461fb249db695bc02867e6560a4815 |
| ch34-prefix-caching-package.zip | 92ef222505fca29ae5ba42a6f39356988a754748c30fb53d2ada81fe177400fb |
| ch35-chunked-prefill-package.zip | 55ebbd544ac2252133cf955742e4df69e7841aecddf9bb561775598067e31e5c |
| ch36-scheduling-pressure-package.zip | 4017598c43ca23b06ae80c776dd72c946e0f6af6e254b69fb4ff505e5c509354 |
| ch37-structured-output-package.zip | fa7949725e82b0797c9b9fed23fb225d80676f44b31f5e9a2e0e3976396a2c39 |
| ch38-multi-model-serving-package.zip | ad2ffead009731abe0ed1f383f1bb70b85b0ebfb16ec23a731fe9fe78fa442d7 |
| ch39-speculative-decoding-package.zip | b62b55bdd82bfd2126db96780cc692e8175d362c22ecb2378cf5c5ca1c7a5a8a |
| ch40-modern-speculation-package.zip | 6cf6ac1bb574fb36121463102f88e31969efb5936fd15b6c0a0cf6e8c2bcfb12 |
| ch41-when-speculation-loses-package.zip | 3ca58876200f01aab2f0184a0e7a047ad57b49ec84ab456354848f480e7e8ca7 |
| ch42-shrinking-kv-package.zip | 37a5f4466fe01b261b87946cc86df2b8a26b6357ca39b30aa9a5738e282dc19c |
| ch43-mixture-of-experts-package.zip | 469f9fff8dc379040f478e9f4508aea5b15c715abf4c4e9838a508798e18c574 |

## Repairs and scope

- Repaired invalid Rust float literals without corrupting ranges, tuple fields or format precision. The initial partial integration had damaged older lab syntax; those unrelated older sources now match their starting Git versions.
- Added all twelve crates to the workspace; formatted, tested and linted the whole workspace.
- Connected twelve native worked-answer files; replaced an undefined environment and corrected TeX-relative paths. Restored sixteen existing measured/mechanism figures rather than abandoning their evidence.
- Added 33 native TikZ figures, fixing an invalid arrow and crowded routes/labels. Figure roles and captions distinguish illustrative examples from executable traces.
- Added substantive Chapter 32 trace, identity/ticket, binary-interface and batch-building explanations. Only Chapter 32 crosses the systems FULL gate; Chapters 33–43 remain enriched ZERO drafts.
- Fixed batching duplicate replies, repeated-budget planning and input validation; separated waiting cancellation from active-ticket retirement.
- Checked all candidates in a colliding prefix-hash bucket; added a real independent byte-prefix grammar oracle.
- Corrected speculative KV lag after a correction, monotonic publication, ancestor-closed tree budgets and cycle rejection.
- Corrected hysteresis on negative penalized scores and exercised EWMA updates in the control CLI.
- Corrected XGrammar-2 compilation versus generation claims and bibliography aliases/title; retained the MoE compute-knee equation used by the later expert-parallelism chapter.

## Portable lab evidence

Each program below was run twice from the final source. Both invocations returned
zero and produced identical stdout. Times and throughputs appearing in these
outputs are derived or illustrative inputs, not hardware measurements.

### continuous-batching-lab

Command: `cargo run -q -p continuous-batching-lab`; 10 tests passed.

```text
t=00 Arrive("A")
t=00 Arrive("B")
t=00 Admit("A", 0, 1)
t=00 Admit("B", 1, 2)
t=00 Schedule("A", 3)
t=01 Arrive("C")
t=01 Schedule("A", 1)
t=01 Schedule("B", 1)
t=02 Schedule("A", 1)
t=02 Schedule("B", 1)
t=03 Arrive("D")
t=03 Schedule("A", 1)
t=03 Schedule("B", 1)
t=03 Terminal("B", Completed)
t=04 Retire("B", 1, 2)
t=04 Admit("C", 1, 3)
t=04 Schedule("A", 1)
t=04 Schedule("C", 2)
t=04 Terminal("A", Completed)
t=05 Terminal("D", Cancelled)
t=05 Retire("A", 0, 1)
t=05 Schedule("C", 1)
t=06 Schedule("C", 1)
t=07 Schedule("C", 1)
t=07 Terminal("C", Completed)
t=08 Retire("C", 1, 3)
```

### paged-kv-lab

Command: `cargo run -q -p paged-kv-lab`; 5 tests passed.

```text
A table: [Handle { block: 0, generation: 1 }, Handle { block: 1, generation: 1 }]
A logical history: [10, 11, 12, 13, 14, 15]
fork B table: [Handle { block: 0, generation: 1 }, Handle { block: 1, generation: 1 }]
after B append:
  A table/history: [Handle { block: 0, generation: 1 }, Handle { block: 1, generation: 1 }] [10, 11, 12, 13, 14, 15]
  B table/history: [Handle { block: 0, generation: 1 }, Handle { block: 2, generation: 1 }] [10, 11, 12, 13, 14, 15, 99]
free blocks: 3
```

### ch17-prefix-caching

Command: `cargo run -q -p ch17-prefix-caching`; 6 tests passed.

```text
same tenant hit tokens = 4
other tenant hit tokens = 0
resident entries = 2, blocks = 3
```

### ch18-chunked-prefill

Command: `cargo run -q -p ch18-chunked-prefill`; 5 tests passed.

```text
step 0: decode=64 prefill=960 committed 6144 -> 7104
step 1: decode=80 prefill=944 committed 7104 -> 8048
step 2: decode=96 prefill=928 committed 8048 -> 8976
step 3: decode=112 prefill=912 committed 8976 -> 9888
step 4: decode=128 prefill=352 committed 9888 -> 10240
```

### ch19-scheduling

Command: `cargo run -q -p ch19-scheduling`; 5 tests passed.

```text
tick 0: used=6 service=[1, 0] completed=[]
tick 1: used=6 service=[1, 1] completed=[]
tick 2: used=6 service=[2, 1] completed=[]
tick 3: used=6 service=[2, 2] completed=[]
tick 4: used=6 service=[3, 2] completed=[]
tick 5: used=6 service=[3, 3] completed=[]
tick 6: used=6 service=[4, 3] completed=[]
tick 7: used=3 service=[4, 4] completed=[2]
```

### ch20-structured-output

Command: `cargo run -q -p ch20-structured-output`; 4 tests passed.

```text
state State(0): allowed ["{\"ok\":"]
state State(6): allowed ["true", "false", "tr", "fa"]
state State(14): allowed ["ue}"]
accepting=true output={"ok":true}
```

### ch21-multi-lora

Command: `cargo run -q -p ch21-multi-lora`; 4 tests passed.

```text
dynamic=[9.0, 16.0, 9.0]
merged =[9.0, 16.0, 9.0]
lease=Handle { slot: 0, generation: 1, adapter_id: 42 } pin=true
```

### ch22-speculative

Command: `cargo run -q -p ch22-speculative`; 5 tests passed.

```text
target=[0.5, 0.3, 0.2]
exact =[0.5, 0.29999999999999993, 0.2]
wrong =[0.6, 0.15999999999999998, 0.24]
tentative=Frontiers { committed: 100, drafted: 104, evaluated: 104, published: 100 }
reconciled=Frontiers { committed: 103, drafted: 103, evaluated: 102, published: 100 }
```

### ch23-draft-tree

Command: `cargo run -q -p ch23-draft-tree`; 7 tests passed.

```text
row 0: depth=1 path=[10] visible=[10]
row 1: depth=2 path=[10, 20] visible=[10, 20]
row 2: depth=2 path=[10, 30] visible=[10, 30]
row 3: depth=3 path=[10, 20, 40] visible=[10, 20, 40]
row 4: depth=3 path=[10, 30, 50] visible=[10, 30, 50]
budget 3: [0, 2, 1]
```

### ch24-break-even

Command: `cargo run -q -p ch24-break-even`; 8 tests passed.

```text
budget=4 speedup=2.667 profitable=true
EWMA=2.667 selected_budget=4
budget=4 speedup=0.889 profitable=false
EWMA=1.778 selected_budget=4
budget=4 speedup=1.667 profitable=false
EWMA=0.722 selected_budget=0
```

### ch25-shrinking-kv

Command: `cargo run -q -p ch25-shrinking-kv`; 6 tests passed.

```text
Llama-style: 112 KiB/token
Qwen-style: 36 KiB/token
MLA example: 68.6 KiB/token
8Q:2KV mapping: [0, 0, 0, 0, 1, 1, 1, 1]
```

### ch26-mixture-of-experts

Command: `cargo run -q -p ch26-mixture-of-experts`; 5 tests passed.

```text
B=  1: expected union 8.00/128
B=  8: expected union 51.62/128
B= 32: expected union 111.77/128
B= 64: expected union 125.94/128
B=128: expected union 127.97/128
occupancy = [1, 1, 3, 1]
combined outputs = [2.6399999999999997, 5.04, -2.8800000000000003]
```

## Validation boundary

Book source/anatomy/diagram contracts, all foundation tests, existing visual
parity checks, Markdown links, preservation/secrets guards, both Rust workspaces
(fmt/tests/clippy), and the two Python chapter labs passed locally. The portable
C/NEON fixture passed all 516 length/offset cases without instrumentation.

The AddressSanitizer/UndefinedBehaviorSanitizer binary did not finish its startup
probe on this Mac; that instrumented result is **unverified**, not a pass. The
unchanged Linux sanitizer step remains in CI. No CUDA experiment was needed to
validate these CPU mechanism fixtures; no GPU performance or model-quality result
was invented.

## Book build and visual QA

- Lab integration commit: `999ee3f` (twelve crates, 71 passing tests).
- PDF: `output/pdf/inside-the-llm-engine-textbook.pdf`, 793 A4 pages.
- Final PDF SHA-256: `aa943612e658a3351be5f7304931a5895a6f6b4faf7c2cb318936ef166bd2a11`.
- Whole-book gates: 59 chapters, 32 FULL, 27 ZERO, zero VERIFIED;
  215 native figures, 175 worked problems, five existing measurement TODOs.
- `make textbook` succeeded under XeLaTeX/biber. The strict log check found
  no overfull boxes, missing glyphs, unresolved references/citations or
  multiply defined labels. The build record separately fingerprints manuscript,
  bibliography, listing sources and build/check scripts.
- Initial visual review inspected 46 figure pages across the integrated
  chapters. Release review inspected 37 pages: PDF pages 498, 500, 501, 513,
  514, 552 and 594; the worked-answer entry pages 509, 521, 533, 546, 559,
  570, 584, 598, 611, 623, 636 and 653 and each following page; contents
  pages 4–5 and bibliography pages 780–783. Pages 500, 514, 552 and 594
  were also checked in grayscale. The final two label changes on pages 514
  and 552 were rerendered and reinspected after the last build.
- Review repaired crowded ragged offsets, lifecycle labels, address/pool
  routes and memory-pressure branches, plus detached worked headings and
  first problem/solution splits. The checked drawings remain native TikZ,
  not screenshots or ASCII diagrams.

This scope is author QA of the integration, not independent technical review,
an assertion that every unchanged page was inspected anew, or a claim that
the remaining ZERO drafts meet the FULL chapter contract. Generated review
images and the pre-integration backup are kept outside the public source tree.
