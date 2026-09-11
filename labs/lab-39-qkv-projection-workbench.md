# Labs 39–48 — QKV Projection Workbench

Prerequisites: Chapters 5–8 and Labs 16–38. Scope: bias-free one-token F32
projection, checked head geometry, independent outputs and immutable views.
Use a scratch branch or temporary notes for deliberate breaks; do not commit
broken reference code. No model download, credential, GPU or network is needed.

## Common fixture and commands

Read [the input fixture](../code/reference/fixtures/chapter08-qkv.json) before
running anything. Geometry is D=4, Hq=2, Hkv=2, Dh=2. The direct diagnostic
input is `[1,-2,3,-4]`; the composed trace first looks it up and normalizes it.
Keep these two experiments distinct in your notebook.

From `code/mini-engine`, run `cargo test --test qkv` and
`cargo run --quiet --example chapter08_qkv_trace`. From the repository root,
run `python3 scripts/check-qkv-visual-parity.py`. The independent oracle is
[chapter08_qkv_oracle.py](../code/reference/python/chapter08_qkv_oracle.py).

## Lab 39 — One row, one coordinate

**CHECK:** Write the four products for every Wq row. Expected flat Q is
`[7,2,-2,-3]`; the first row contributes `[1,0,6,0]`.

**BUILD:** Produce a notebook table with output index, row, products and sum.
**BREAK:** Transpose the nonsymmetric square matrix. Shape still passes;
explain the changed values. **EXTEND:** Choose an off-block weight to disprove
the claim that head 0 only consumes the first half of the input.

## Lab 40 — Three learned maps

**CHECK:** Calculate K=`[1,-3,3,3.5]` and V=`[-3,-1,0.5,-2]` independently.
**BUILD:** Call `project_qkv_reference` and compare all twelve values.
**BREAK:** Swap K and V weights. A shape checker cannot identify this semantic
name swap; record which value assertions detect it. **EXTEND:** Explain which
loader metadata must protect projection identity.

## Lab 41 — Regroup without recomputing

**CHECK:** Partition Q into `[7,2]` and `[-2,-3]`. **BUILD:** Use both
`query_head` views and verify their shapes and base offsets. **BREAK:** Treat
the output as `[Dh,Hq]` when Dh and Hq differ; select a non-square geometry.
**EXTEND:** Distinguish moving a Vec from copying its scalar payload.

## Lab 42 — Follow the address

**CHECK:** List `(h,j)`, element offset and F32 byte displacement for all query
coordinates. Expected offsets are 0,1,2,3; displacements 0,4,8,12.
**BUILD:** Compare borrowed cell addresses with the corresponding owner cells
as the safe Rust test does. **BREAK:** Use bytes as element offsets; record
the bounds failure or wrong cell. **EXTEND:** Recalculate for a hypothetical
F16 storage representation without claiming the teaching tensor supports it.

## Lab 43 — Fail before the first product

**CHECK:** List all four dimension-zero errors and both checked-product
overflows. **BUILD:** Test wrong rank and wrong input/output width for Q, K
and V independently. **BREAK:** Give a two-KV-head configuration full-width
MHA weights. **EXTEND:** Specify a typed resource-admission policy, separate
from shape validation. Current Vec allocation exhaustion is not recoverable
through QkvError; do not provoke a machine-wide allocation failure.

## Lab 44 — Keep queries, shrink KV

**CHECK:** For D=8,Hq=4,Dh=2 derive all weight and output shapes with Hkv=4,2,1.
**BUILD:** Create constant-weight fixtures and check every output value and
shape. **BREAK:** Try Hq=3,Hkv=2; require rejection, not integer truncation.
**EXTEND:** Draw the future contiguous grouping convention for four query
heads. Label links as associations, not copies. No attention implementation
is part of this lab.

## Lab 45 — Compose the component pipeline

**CHECK:** Derive normalized input using gain `[1,0.5,2,-1]`, epsilon=1e-5.
**BUILD:** Follow the trace through table lookup, RMSNorm, QKV and query head
1, expected approximately `[3.651481,0.365148]`.
**BREAK:** Accidentally compare normalized outputs with raw fixture outputs.
**EXTEND:** Explain why a model layer may contain other work even though this
component pipeline is now executable.

## Lab 46 — Use an independent oracle

**CHECK:** Explain `math.fsum` versus ordered F32 accumulation.
**BUILD:** Run the parity gate and record maximum absolute error separately
for Q, K, V and the normalized intermediate. Use abs+rel tolerance 1e-5+1e-5
times the reference magnitude for the small composed fixture.
**BREAK:** Truncate a candidate output and change one coordinate; the checker
must reject both. **EXTEND:** Design wider cancellation-heavy tests with a
justified tolerance rather than silently increasing this fixture's tolerance.

## Lab 47 — Locate the missing position input

**CHECK:** Label equal input vectors with positions 0 and 7.
**BUILD:** Call the stateless operator twice; require equal values and separate
owners. **BREAK:** Infer from this that equal token IDs always have equal
later-layer activations. Explain why earlier context can invalidate that
assumption. **EXTEND:** Write the interface contract that the next chapter's
position operator must add, without implementing it here.

## Lab 48 — Cost an architecture, not a slogan

**CHECK:** Reproduce all three analytical rows in the chapter cost plate.
**BUILD:** Calculate projection parameters/FLOPs and future KV payload for
D=4096,Hq=32,Dh=128,Hkv in {32,8,1}, L=32,T=8192,s=2.
Expected KV payloads are 4 GiB, 1 GiB and 128 MiB.
**BREAK:** Claim fourfold fewer KV heads means fourfold fewer QKV parameters.
Show the unchanged query term. **EXTEND:** List the missing quantities needed
to turn payload into a safe concurrency admission budget.

## Evidence and cleanup

Submit ten short lab sections with calculations, actual command outputs,
deliberate-break diagnosis and the relevant figure ID. These are correctness
and analytical labs, not performance measurements. Remove only your own
scratch artifacts or undo only your own deliberate changes. Re-run the full
QKV test and parity commands; the checked-in fixture must remain unchanged.
