#!/usr/bin/env python3
"""Generate review-only curriculum/figure plans without mutating the book outline.

Existing chapter-specific questions, experiments and prerequisites remain the
source of truth. Industrial amendments below are explicit editorial proposals.
--check validates coverage, graph acyclicity and byte-for-byte reproducibility.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Each amendment has one accountable chapter home, not a new parallel manuscript.
AMENDMENTS = {
    1: 'Use the four industrial atlas plates as optional overview/zoom references. Separate model semantics, execution and serving; revisit rather than overload the opening chapter.',
    4: 'Add a forward reference to constraint-state ownership in Chapter 67; preserve the current tested sampler API and distinguish token choice from output validity.',
    14: 'Compare a pinned safetensors specification with GGUF: metadata, tensor naming, layout, integrity and loading obligations. Do not imply container conversion guarantees model equivalence.',
    16: 'Separate weight-only, activation and KV quantization. Show scale groups, storage bits versus effective bytes and calibration/error budgets; keep the first packed fixture small.',
    17: 'Trace packed load, scale application, accumulation dtype and output tolerance. Compare optimized kernels only against the scalar oracle on identical logical values.',
    20: 'Use explicit client/server timing boundaries and workload-dependent roofline reasoning. Add the latency plate; prefill/decode names do not determine the bottleneck.',
    22: 'Count valid, allocated and reserved bytes separately. Add dense/GQA/sliding-window applicability limits and a quantized-KV metadata budget; defer MLA to Chapter 80.',
    26: 'Make chunked prefill a concrete scheduling policy with per-iteration token budget and decode latency tradeoff. Trace one long prompt beside short decodes.',
    27: 'Test fairness, bounded queues, cancellation and admission under a synthetic arrival trace; do not replace tail-latency analysis with average throughput.',
    31: 'State every cache-key assumption: model revision, tokenization/template, adapter and relevant position/execution context. Prefix equality alone is not universal cache validity.',
    34: 'Explicitly distinguish paged KV addressing, prefix reuse and tiled attention. Add quantized-KV read/dequantization to the layout contract, without claiming every backend supports it.',
    40: 'Connect the stable online softmax recurrence to tiled attention with an exact small oracle. Read the full FlashAttention algorithm before asserting kernel-level equivalence.',
    41: 'Make partition merge order, partial normalization state and determinism visible. Explain when scheduling or reduction changes floating-point results.',
    45: 'Separate model graph, compiler IR, fused kernel and captured launch graph. Show a dependency before a performance optimization.',
    48: 'Own the first bounded graph lowering/capture case study: dynamic shapes, specialization, buffer addresses, warmup and replay invalidation. Measure transfer/launch overhead before explaining a CPU/GPU result.',
    51: 'Read the full exact speculative-sampling paper; derive acceptance/correction and show first rejection plus KV rollback. Distinguish greedy verification from stochastic distribution preservation.',
    52: 'Contrast prompt lookup/ngram proposals with draft models; add a source-verified comparison sidebar for EAGLE/MTP with their model/training assumptions. These remain optional proposal mechanisms.',
    53: 'Charge draft, verification, rollback and memory costs to useful accepted tokens; include a counterexample where speculation loses.',
    54: 'Trace top-k expert routing, dispatch, expert compute and combine; separate activation communication from expert-weight residency.',
    58: 'Use an allocation ownership ledger across weight/KV/expert pools. Do not equate a unified policy with a physically unified address space.',
    67: 'Own grammar/JSON-schema constrained decoding: compiler/automaton state, allowed-token mask, unsatisfiable continuation, stop behavior and cancellation. Syntactic validity is not semantic truth.',
    68: 'Own adapter identity, loading, residency and routing; introduce multimodal encoder/projector inputs and per-request state. Include reuse isolation across adapters and model revisions.',
    70: 'Name timestamps and units for TTFT, ITL, TPOT, E2E, queue delay and goodput. Use trace IDs and censoring/failure policy; never infer token timing solely from transport chunks.',
    72: 'Threat-model untrusted model files, templates, adapters, tenant cache reuse, prompt logging and resource exhaustion. Pin a parser boundary and test rejection.',
    73: 'Report SLO-qualified goodput, tail distributions, warm/cold phases, arrival process, concurrency, model, precision and hardware. No synthetic timing fixture becomes a speed claim.',
    74: 'Update the case-study ledger to the inspected Hermon pin without rewriting historical measurements. Keep CURRENT, PREVIEW, LIBRARY and TARGET distinct.',
    80: 'Separate externally deployed architectural variants from research hypotheses. Add typed multimodal input and MLA/sliding-window/hybrid state-accounting exceptions, sourced per actual model.',
    81: 'A recurrent STATE page stores a bounded recurrence state, not a dense KV history. Define rollback/checkpoint behavior before using speculation.',
    82: 'Compare physical tiers with explicit transfer latency/bandwidth and ownership. A cost model is an analytical prediction, not a measured migration policy.',
    83: 'Read after Chapter 84 in the proposed production order. Build a two-process KV-transfer reference with layout/version handshake, ownership acknowledgement, duplicate delivery and failure injection before any performance claim.',
    84: 'Read before Chapter 83 in the proposed production order. Separate TP/PP/DP/context/expert parallelism; add real two-worker correctness execution where hardware permits, with a CPU reference path and no GPU-scaling claim from simulation.',
    85: 'Extend the multi-worker contract to multi-node placement, collective failure, timeout and recovery. Require an actual distributed integration record; a simulator alone cannot establish production readiness.',
    87: 'Revisit IR and executable-plan contracts introduced in Chapters 45/48. Compare a minimal interchange boundary against concrete consumers; do not present a proposed protocol as an adopted standard.',
    89: 'Select supported feature combinations and explicitly exclude unimplemented ones. Graduation scope is an evidence-backed engine, not a checklist of every surveyed vendor feature.',
    94: 'Replace one narrowly bounded component only after semantic, ownership, ABI, failure and performance gates. Require a reversible integration and an honest status label.',
}

# A branching prerequisite spine; exact prose prerequisites are also preserved.
PART_PARENTS = {
    1: [], 2: [4], 3: [13], 4: [18], 5: [20,21], 6: [22,27],
    7: [34], 8: [41], 9: [27,34], 10: [18,34,48], 11: [13,34,41],
    12: [27,48,66], 13: [73], 14: [58,73,79], 15: [66,73,85,88],
}
PLATE_RULES = {
    1: ('Request and token flow', 'request / sampler / decoder', 'invalid input, stop or disconnect'),
    2: ('Typed tensor and algorithm trace', 'weight owner / activation buffer', 'shape, mask or numerical mismatch'),
    3: ('Byte layout and decoding trace', 'mapped artifact / decoded tensor', 'truncation, alignment or unsupported encoding'),
    4: ('Timeline and physical byte accounting', 'profiler clock / retained state', 'wrong boundary or double-counted memory'),
    5: ('Swimlane and sequence lifecycle', 'scheduler / sequence / stream', 'starvation, cancellation or queue pressure'),
    6: ('Logical-to-physical memory map', 'sequence handles / block manager', 'aliasing, stale handle or exhausted pool'),
    7: ('Kernel dataflow and lifetime trace', 'caller / native allocation / kernel', 'ABI mismatch, refcount or reduction failure'),
    8: ('Hardware execution and transfer diagram', 'host / device / completion event', 'dispatch mismatch or uncharged transfer'),
    9: ('Proposal, verification and rollback trace', 'sequence / draft / target cache', 'rejection, invalid reuse or negative speedup'),
    10: ('Routing and memory-tier swimlane', 'router / expert pool / transfer queue', 'miss, eviction race or capacity pressure'),
    11: ('Counterexample and proof-boundary diagram', 'oracle / optimized implementation / harness', 'plausible output that violates an invariant'),
    12: ('Protocol and operational lifecycle', 'tenant / gateway / runtime / observer', 'disconnect, overload, malformed artifact or SLO miss'),
    13: ('Pinned source-to-runtime architecture', 'API / dispatcher / worker / bridge', 'mistaking available library code for the default path'),
    14: ('Typed distributed state and communication map', 'worker / state shard / transport', 'layout mismatch, lost owner or failed collective'),
    15: ('Acceptance-gate and integration diagram', 'component / test harness / release decision', 'a failed gate incorrectly reported as completion'),
}


def records():
    text = (ROOT/'docs/OUTLINE.md').read_text()
    result = []
    part = 0
    for chunk in re.split(r'(?=^## Part |^### Chapter )', text, flags=re.M):
        if chunk.startswith('## Part '):
            part += 1
        match = re.match(r'### Chapter (\d+) — (.+)', chunk)
        if not match:
            continue
        chapter_body = chunk.split('\n## ', 1)[0]
        fields = dict(re.findall(r'^- \*\*(.+?):\*\* (.+)$', chapter_body, re.M))
        assert len(fields) == 8, (match.group(1), fields.keys())
        result.append({'chapter': int(match.group(1)), 'title': match.group(2),
                       'part': part, 'fields': fields})
    assert [r['chapter'] for r in result] == list(range(1,95))
    assert part == 15
    return result


def make():
    rows = records()
    edges = []
    order = []
    for part in range(1,16):
        chapters = [r['chapter'] for r in rows if r['part'] == part]
        if part == 14:
            chapters[chapters.index(83)], chapters[chapters.index(84)] = 84,83
        order += chapters
        edges += [(p, chapters[0]) for p in PART_PARENTS[part]]
        edges += list(zip(chapters, chapters[1:]))
    assert all(order.index(a) < order.index(b) for a,b in edges)
    assert len(set(edges)) == len(edges)
    dependencies = {n: sorted(a for a,b in edges if b == n) for n in order}
    manifest = json.loads((ROOT/'figures/manifest.json').read_text())
    architecture = ['# Proposed industrial book architecture', '',
        'REVIEW PROPOSAL · 2026-09-10 · 15 existing parts, 94 stable chapter IDs.', '',
        'Generated from the canonical outline plus explicit amendments in',
        '`scripts/build-industrial-plan.py`. Canonical chapter status, titles, paths,',
        'numerical APIs and ENGINE milestone IDs are unchanged. No future chapter is',
        'marked complete by this map. Source evidence: [ledger](../research/industrial-source-map.md).', '',
        'Each chapter retains its question, prerequisites, mathematics, systems,',
        'hardware, implementation, case study, diagram, experiment and acceptance',
        'contract. Added industrial material has one primary home, not a duplicate book.', '']
    migration = ['# Industrial architecture migration proposal', '',
        'REVIEW ONLY. No manuscript moves or renumbering have been applied.', '',
        '| Chapter | Action | Rationale / amendment |', '| --- | --- | --- |']
    illustrations = ['# Chapter-by-chapter illustration production plan', '',
        'This is a production plan, not a claim of 94 finished illustrated chapters.',
        'Four new reference-atlas SVG/TXT plates are implemented in this pass.',
        'Chapter 5/6/7 canonical plates are complete; Chapter 8 is the next bounded curriculum task.', '',
        'Every chapter needs a teaching question, a mechanism figure and a failure',
        'or comparison figure where these clarify its actual logic. Use editable SVG',
        'with Unicode TXT companion, semantic labels, a precise caption and evidence.',
        'Never turn prose into decorative boxes or label planned figures as published.', '',
        'Acceptance: at most six outer panels per plate; minimum 15 px body labels;',
        'named tensor shapes/units/owners; distinguish parameters, activations,',
        'operators and persistent state; check numeric fixtures, SVG bounds, grayscale',
        'and print legibility. More detail belongs in a linked zoom, not smaller type.', '']
    plan = []
    for r in rows:
        n, f, part = r['chapter'], r['fields'], r['part']
        amendment = AMENDMENTS.get(n, 'Keep the existing chapter contract; add no vendor-specific detour without a claim-level primary source and a reproducible acceptance test.')
        rule, owner, failure = PLATE_RULES[part]
        source = [e for e in manifest['figures'] if e['chapter'] == n]
        figure_ids = [e['id'] for e in source]
        architecture += [f'## Chapter {n} — {r["title"]}', '',
            f'Part {part}; dependency spine: '+(', '.join(map(str,dependencies[n])) or 'entry point')+'.', '']
        architecture += [f'- **{key}:** {value}' for key,value in f.items()]
        architecture += [f'- **Industrial amendment:** {amendment}',
            f'- **Illustration contract:** {rule}; follow {owner}; expose {failure}.', '']
        action = 'KEEP NUMBER'
        if n in (83,84): action += '; MOVE READING ORDER (proposal only)'
        migration.append(f'| {n}. {r["title"]} | {action} | {amendment} |')
        illustrations += [f'## Chapter {n} — {r["title"]}', '',
            f'- **Reader question:** {f["Purpose / key question"]}',
            f'- **Planned visual content / experiment:** {f["Diagrams / experiments"]}',
            f'- **Visual grammar:** {rule}. Owners: {owner}.',
            f'- **Required counterexample / boundary:** {failure}.',
            f'- **Test anchor:** {f["Correctness / benchmark"]}',
            f'- **Industrial refinement:** {amendment}',
            '- **Existing manifest assets (mixed status, not proof of chapter completion):** '+(', '.join(figure_ids) or 'none yet')+'.',
            f'- **Next scene IDs:** FIG-CH{n:02d}-MECHANISM-001 and FIG-CH{n:02d}-BOUNDARY-001 (reserved plan only; reuse a matching existing plate before creating another).', '']
        plan.append({'chapter':n,'title':r['title'],'part':part,'status':'PLAN',
                     'dependencies':dependencies[n],'question':f['Purpose / key question'],
                     'visual_content':f['Diagrams / experiments'],'grammar':rule,
                     'owners':owner,'failure':failure,'acceptance':f['Correctness / benchmark'],
                     'amendment':amendment,'existing_manifest_ids':figure_ids})
    migration += ['', 'Part XIV proposed retitle: **Distributed Inference and Emerging Architectures**.',
        'This separates externally documented systems from speculative research.',
        'No chapter is split, merged, removed or appended in this pass. The existing',
        '14 appendices remain unchanged; no additional chapter count is justified yet.', '']
    dot = ['digraph curriculum {','  rankdir=LR;','  node [shape=box];']
    dot += [f'  c{r["chapter"]} [label="{r["chapter"]}: {r["title"]}"];' for r in rows]
    dot += [f'  c{a} -> c{b};' for a,b in edges] + ['}', '']
    graphtext = ['Industrial curriculum dependency spine (proposal)', '',
        'Each arrow means “study prerequisite before dependent chapter”, not execution.',
        'Exact chapter prerequisites remain in the generated architecture document.', '']
    for part in range(1,16):
        chapters = [n for n in order if rows[n-1]['part']==part]
        graphtext += [f'Part {part}: '+ ' → '.join(map(str,chapters)),
            'Entry prerequisites: '+(', '.join(map(str,PART_PARENTS[part])) or 'none'), '']
    return {
        'docs/INDUSTRIAL_BOOK_ARCHITECTURE.md':'\n'.join(architecture),
        'docs/INDUSTRIAL_ARCHITECTURE_MIGRATION.md':'\n'.join(migration),
        'figures/CHAPTER_ILLUSTRATION_PLAN.md':'\n'.join(illustrations),
        'figures/industrial-chapter-plan.json':json.dumps(plan,indent=2,ensure_ascii=False)+'\n',
        'docs/industrial-dependencies.json':json.dumps({'status':'PROPOSAL','order':order,'edges':edges},indent=2)+'\n',
        'diagrams/runtime/industrial-curriculum.dot':'\n'.join(dot),
        'diagrams/runtime/industrial-curriculum.txt':'\n'.join(graphtext),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for name,data in make().items():
        path=ROOT/name
        if args.check:
            assert path.is_file() and path.read_text()==data, 'stale: '+name
        else:
            path.write_text(data)
    print('Industrial plan passed: 94 chapters, 15 parts, acyclic dependencies, 7 deterministic artifacts')


if __name__=='__main__':
    main()
