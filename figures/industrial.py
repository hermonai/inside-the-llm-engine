"""Industrial architecture plates: conceptual contracts, not runtime claims."""

BLUE, TEAL, GOLD, INK = '#dcecf7', '#dcefe7', '#f4ead2', '#152b3c'


def validate(s):
    f = s['fixture']
    if s['kind'] == 'industrial-latency':
        # Rendered numbers are an intentionally fixed teaching fixture. Reject
        # even internally consistent edits until both drawing and trace change.
        assert f == {'arrival_ms':0, 'token_ms':[120,145,180,205],
                     'finish_ms':215, 'ttft_ms':120, 'itl_ms':[25,35,25],
                     'tpot_ms':85/3, 'e2e_ms':215}
        times = f['token_ms']
        assert times == sorted(times) and len(times) > 1
        assert times[0] - f['arrival_ms'] == f['ttft_ms']
        assert [b-a for a, b in zip(times, times[1:])] == f['itl_ms']
        assert (times[-1]-times[0])/(len(times)-1) == f['tpot_ms']
        assert f['finish_ms']-f['arrival_ms'] == f['e2e_ms']
    elif s['kind'] == 'industrial-memory':
        assert f == {'layers':2, 'tokens':4, 'kv_heads':1, 'head_dim':2,
                     'bytes':4, 'shared_blocks':1, 'private_blocks':[1,1],
                     'block_bytes':64}
        assert 2*f['layers']*f['tokens']*f['kv_heads']*f['head_dim']*f['bytes'] == 128
        assert f['shared_blocks'] + sum(f['private_blocks']) == 3
        assert 3*f['block_bytes'] == 192 < 2*128
    else:
        assert f['scope'] == 'reference architecture; not a Hermon implementation claim'


def render(s, d):
    k = s['kind']
    if k == 'industrial-stack':
        # Three swimlanes; vertical placement is ownership, not wall-clock time.
        for x, title in [(45,'SERVING SYSTEM'),(355,'MODEL SEMANTICS'),(665,'EXECUTION ENGINE')]:
            d.text(x, 202, title, 18)
        d.box(45,222,270,101,'Request + policy\nTemplate, limits, identity\nQueue / admission',BLUE,17)
        d.box(355,222,270,101,'Typed model program\nIDs → E[V,D] lookup\nX[T,D] → decoder layers',TEAL,17)
        d.box(665,222,290,101,'Executable plan\nGraph / layout / kernels\nCPU or accelerator',BLUE,17)
        d.arrow(315,263,355,263); d.arrow(625,263,665,263)
        d.text(325,249,'IDs',13); d.text(629,249,'ops',13)
        d.box(45,380,270,105,'Sample + stream\nConstraints and RNG\nStop, cancel, or continue',BLUE,17)
        d.box(355,380,270,105,'Output scores\nLast hidden state [D]\nLM head → logits [V]',TEAL,17)
        d.box(665,380,290,105,'Persistent KV state\nRead prior K,V; write new\nLogical IDs ≠ block IDs',GOLD,17)
        d.line(672,387,948,387); d.line(672,478,948,478)
        d.arrow(810,323,810,380); d.text(824,352,'read / write',15)
        d.arrow(788,380,788,323)
        d.arrow(730,323,730,350); d.arrow(730,350,490,350); d.arrow(490,350,490,380)
        d.text(520,344,'hidden activations',15)
        d.arrow(355,429,315,429); d.text(319,412,'z',15)
        d.arrow(80,380,80,323,True); d.text(94,354,'next ID / next iteration',15)
        d.text(45,524,'Optional: prefix reuse avoids work; speculation proposes work for verification.',17)
        d.text(45,552,'Cross-cutting: correctness, resource limits, observability, security, distribution.',17)
        d.text(45,580,'Solid = data / execution    Dashed = continuation control    Double rule = state',15)
    elif k == 'industrial-ownership':
        d.box(45,214,285,109,'MODEL INSTANCE\nImmutable weights + config\nShared across requests',GOLD,17)
        d.box(355,214,290,109,'SEQUENCE STATE\nPosition, RNG, constraints\nLogical KV block handles',TEAL,17)
        d.box(670,214,285,109,'SCHEDULER\nQueues + runnable set\nAdmission and token budget',BLUE,17)
        d.arrow(670,270,645,270,True)
        d.box(45,403,285,109,'BACKEND EXECUTOR\nBuffers + graph resources\nIn-flight completion events',BLUE,17)
        d.box(355,403,290,109,'KV BLOCK MANAGER\nPhysical pages + refcounts\nShared prefix, private tail',GOLD,17)
        d.line(362,410,638,410); d.line(362,505,638,505)
        d.box(670,403,285,109,'CONNECTION / STREAM\nOutput queue + disconnect\nOne terminal outcome',TEAL,17)
        d.arrow(185,323,185,403); d.text(199,366,'borrow weights',14)
        d.arrow(500,323,500,403); d.text(515,365,'reference pages',14)
        d.arrow(810,403,810,323,True)
        d.text(823,358,'cancel',14)
        d.text(823,378,'signal',14)
        d.arrow(330,455,355,455,True)
        d.text(45,546,'Release rule: retire handles only after dependent device work has completed.',18)
        d.text(45,574,'A disconnect is not permission to free a shared prefix or an in-flight buffer.',17)
    elif k == 'industrial-latency':
        d.text(45,207,'A synthetic client trace • four output tokens • one clock • milliseconds',18)
        # Scale deliberately linear, including the short terminal tail.
        x = lambda t: 75+4*t
        d.arrow(75,285,943,285)
        for t,label in [(0,'arrive'),(120,'token 1'),(145,'2'),(180,'3'),(205,'4'),(215,'done')]:
            d.line(x(t),268,x(t),302)
            d.text(x(t),252,label,15,anchor='middle')
            d.text(x(t),323,t,15,anchor='middle')
        d.line(x(0),353,x(120)-5,353,width=3); d.text(285,380,'TTFT = 120 ms',18,anchor='middle')
        d.line(x(120)+5,353,x(205),353,width=3); d.text(725,380,'ITLs = 25, 35, 25 ms',18,anchor='middle')
        d.box(45,407,438,106,'PER-REQUEST MEAN\nTPOT = (205 − 120) / (4 − 1)\n= 28.33 ms per output interval',TEAL,18)
        d.box(507,407,448,106,'END-TO-END\nE2E = 215 − 0 = 215 ms\nIncludes the 10 ms terminal tail',GOLD,18)
        d.text(45,548,'TTFT includes client-visible queueing, prompt work and transport—not just prefill.',17)
        d.text(45,577,'Inter-token latency is not stream-chunk latency. TPOT is undefined for N < 2.',17)
    elif k == 'industrial-memory':
        d.text(45,204,'01  Count disjoint physical allocations, once, on a named device',18)
        # Equal widths represent categories, not measured byte proportions.
        d.box(45,225,910,62,'',BLUE)
        for x,label in [(45,'Weights'),(200,'KV pages'),(355,'Activations'),(510,'Workspace'),(665,'Graph state'),(810,'Runtime')]:
            if x > 45: d.line(x,225,x,287)
            d.text(x+14,262,label,16)
        d.text(45,314,'Schematic categories, not a measured stacked bar. Allocator slack is additional.',16)
        d.box(45,345,438,159,'02  Uniform dense-attention example\nKV bytes = 2 × L × T × Hkv × dh × b\nL=2, T=4, Hkv=1, dh=2, b=4\nOne sequence: 128 valid KV bytes\nNo padding or metadata in this formula',TEAL,16)
        d.box(507,345,448,159,'03  Two sequences, three physical blocks',GOLD,16)
        d.text(539,405,'A',17); d.text(916,405,'B',17)
        d.box(687,387,90,42,'S',TEAL,17)
        d.box(563,436,90,42,'A tail',TEAL,15)
        d.box(811,436,90,42,'B tail',TEAL,15)
        d.arrow(555,401,687,407); d.arrow(909,401,777,407)
        d.arrow(546,411,598,436); d.arrow(923,411,866,436)
        d.text(732,461,'shared S',15,anchor='middle')
        d.text(526,495,'3 × 64 B = 192 B; logical sum = 256 B',16)
        d.text(45,546,'Reserved page capacity ≥ valid KV bytes; page tables and refcounts also cost memory.',16)
        d.text(45,576,'MLA, recurrent state, sliding windows and heterogeneous layers need other formulas.',16)


def text_lines(s):
    maps = {
        'industrial-stack': [
            'SERVING ── IDs / work ──▶ MODEL PROGRAM ── ops ──▶ EXECUTOR',
            'EXECUTOR ══ read/write ══ KV STATE; EXECUTOR ── hidden ──▶ LM HEAD',
            'LM HEAD ── logits ──▶ SAMPLER ── token ──▶ STREAM',
            'SAMPLER ┄┄ next ID / next iteration ┄┄▶ SERVING',
            'Prefix reuse and speculative proposals are optional, not serial stages.'],
        'industrial-ownership': [
            'MODEL owns weights ── borrowed by ──▶ EXECUTOR',
            'SCHEDULER ┄┄ selects ┄┄▶ SEQUENCE owns logical handles',
            'SEQUENCE ── references ──▶ KV MANAGER owns physical pages',
            'STREAM disconnect ┄┄▶ retire sequence ┄┄▶ wait for work completion',
            'completion ┄┄▶ release references; free only at zero references'],
        'industrial-latency': [
            'arrival 0 ──▶ token 1:120 ──▶ 2:145 ──▶ 3:180 ──▶ 4:205 ──▶ done:215',
            'TTFT = 120 ms; ITLs = 25, 35, 25 ms; TPOT = 85/3 ms; E2E = 215 ms',
            'Measure output tokens on one clock; network chunks are not necessarily tokens.'],
        'industrial-memory': [
            'ONE DEVICE ══ weights + KV + live activations + workspace + graph + runtime',
            'A logical pages ──▶ shared S ◀── B logical pages',
            'A logical pages ──▶ private A; B logical pages ──▶ private B',
            'Physical set {S,A,B}: 3 × 64 B = 192 B, not 2 × 128 B = 256 B',
            'Dense uniform KV example: 2 × 2 × 4 × 1 × 2 × 4 = 128 valid bytes.']}
    return maps[s['kind']]
