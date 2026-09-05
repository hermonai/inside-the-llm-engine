#!/usr/bin/env python3
"""Create review inventories. Presence heuristics are explicitly not proofs."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/astra'
OUT.mkdir(parents=True,exist_ok=True)
chapters=sorted((ROOT/'manuscript').glob('part-*/chapter-*.md'))
code={1:'lib.rs',2:'tokenizer.rs',3:'model.rs',4:'sampling.rs',5:'tensor.rs',6:'linear.rs',7:'normalization.rs'}
eq=['# Display-equation review inventory','',
    'Snapshot: 2026-09-05. Each row names a display block, including numeric',
    'examples and repeated definitions. Scope means the enclosing section.',
    'The flags detect nearby declarations and examples; they do not prove',
    'symbol completeness. Chapter-level code/oracle evidence is in the book',
    'audit. Every block still requires an explicit figure binding in its',
    'regeneration pass. Cost/rate equations need analytical labels; prose',
    'definitions have no numerical oracle when one would be meaningless.','',
    '| Chapter / line | Section / equation excerpt | Local evidence | Code family | Decision |',
    '| --- | --- | --- | --- | --- |']
snips=['# Code example audit','',
    'Source examples and oracles are executable; manuscript fences are',
    'classified separately as excerpts, command recipes or non-executable',
    'representations. Rust build/test/Clippy checks do not prove isolated',
    'fences compile. Similar oracle arithmetic is intentional independence.','',
    '| Source | Role | Decision / validation |','| --- | --- | --- |']
for pattern,role in [('code/mini-engine/crates/engine0/examples/*.rs','experiment'),('code/reference/python/*.py','independent oracle')]:
    for p in sorted(ROOT.glob(pattern)):
        snips.append(f'| [{p.name}](../../{p.relative_to(ROOT)}) | {role} | KEEP; run directly; no competing implementation layer |')
snips+=['','## Manuscript fence inventory','','| Chapter / line | Language | Classification |','| --- | --- | --- |']
for p in chapters:
    text=p.read_text(); n=int(p.name.split('-')[1]); lines=text.splitlines()
    for m in re.finditer(r'^\$\$\s*\n(.*?)^\$\$',text,re.M|re.S):
        line=text[:m.start()].count('\n')+1
        heads=re.findall(r'^#{1,4} (.+)$',text[:m.start()],re.M)
        section=heads[-1] if heads else 'opening'
        context=text[max(0,m.start()-1000):m.end()+800]
        flags=[]
        for label,pattern in [('shapes',r'\\mathbb|shape|dimension'),('symbols',r'where |Here |means|denotes'),('example',r'example|by hand|\[\s*-?\d'),('visual',r'diagram|figure|\.txt')]:
            flags.append(label+(' nearby' if re.search(pattern,context,re.I) else ' review'))
        excerpt=re.sub(r'\s+',' ',m.group(1)).replace('|','¦')[:95]
        eq.append(f'| {n}:{line} | {section.replace("|","/")} — `{excerpt}` | {"; ".join(flags)} | `{code[n]}` | KEEP mathematics; bind figure; review local notation |')
    for m in re.finditer(r'^```([^\n]*)\n(.*?)^```',text,re.M|re.S):
        line=text[:m.start()].count('\n')+1; lang=m.group(1) or 'unlabeled'
        role='reference excerpt; covered by workspace, not standalone' if lang=='rust' else 'command recipe; environment/workload dependent' if lang in ('sh','bash','shell') else 'diagram/data/pseudocode; not an executable program'
        snips.append(f'| {n}:{line} | {lang} | {role} |')
(OUT/'equation-audit.md').write_text('\n'.join(eq)+'\n')
(OUT/'code-audit.md').write_text('\n'.join(snips)+'\n')
audit=['# Per-diagram disposition','',
       'All 78 historical artifacts are preserved. These decisions govern their',
       'next visual regeneration; KEEP does not assert that a new vector exists.','',
       '| ID | Source | Disposition | Reason / next check |','| --- | --- | --- | --- |']
for row in (ROOT/'diagrams/INDEX.md').read_text().splitlines():
    if not re.match(r'\| D\d',row): continue
    cols=[x.strip() for x in row.split('|')]
    ident,kind=cols[1],cols[4]; path=re.search(r'\]\(([^)]+)\)',row).group(1)
    if ident in ('D049','D050','D054'):
        decision='REDRAW'; reason='Incorrect operand-edge implication or mismatched reduction connectors; see visual audit.'
    elif kind=='ownership':
        decision='CONVERT TO UML'; reason='Actual owner composition and borrow dependencies; retain text contract.'
    elif kind in ('numerical flow','state machine','control flow'):
        decision='EXPAND INTO SEQUENCE'; reason='Separate before/transition/after; preserve terminal or numerical boundary.'
    elif kind=='architecture':
        decision='CONVERT TO DATAFLOW'; reason='Separate data, control, state and status; use UML when software structure is primary.'
    else:
        decision='KEEP'; reason='Focused question and useful terminal form; add vector companion after semantic review.'
    audit.append(f'| {ident} | [{path}](../../diagrams/{path}) | {decision} | {reason} |')
(OUT/'diagram-audit.md').write_text('\n'.join(audit)+'\n')
print(f'Audit inventories generated: {len(chapters)} chapters; {sum(row.startswith("| "+str(n)+":") for row in eq for n in range(1,8))} display blocks, 78 diagrams')
