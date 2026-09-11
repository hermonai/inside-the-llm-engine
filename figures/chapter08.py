"""Projection, ownership and head-geometry plates; numbers come from the oracle."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/reference/python'))
from chapter08_qkv_oracle import fixture
from chapter07 import panel, cells

BLUE, GREEN, GOLD = '#dcecf7', '#dcefe7', '#f4ead2'


def render(s, d, frame=None):
    f = fixture(); kind = s['kind'][5:]
    if kind == 'journey':
        panel(d,50,205,260,95,'01 / Residual',['one token: x̂ [D]'],GREEN)
        for i,(name,width) in enumerate([('Q','Pq'),('K','Pkv'),('V','Pkv')]):
            y=205+i*126
            panel(d,400,y,255,95,'02 / '+name+' projection',['W'+name+' ['+width+',D]'],GOLD)
            panel(d,730,y,215,95,'03 / '+name+' heads',['[H,Dh] / F32'],GREEN)
            d.line(310,252,352,252); d.line(352,252,352,y+48)
            d.arrow(352,y+48,400,y+48); d.arrow(655,y+48,730,y+48)
        d.text(50,588,'Three learned maps consume the same input; no token-to-token mixing occurs here.',19)
    elif kind == 'trace':
        d.text(50,210,'ONE WEIGHT ROW / q[0] = Wq[0,:] · x',21)
        cells(d,160,238,[1,0,2,0],150,GOLD); d.text(50,270,'weight',19)
        cells(d,160,303,f['input'],150,GREEN); d.text(50,335,'input',19)
        for i in range(4): d.arrow(235+150*i,351,235+150*i,388)
        cells(d,160,390,[1,0,6,0],150,BLUE); d.text(50,422,'product',19)
        panel(d,50,476,410,100,'Ordered F32 sum',['0 + 1 + 0 + 6 + 0 = 7'])
        panel(d,515,476,430,100,'Entire Q / four weight rows',['q_flat = [7, 2, -2, -3]'],GREEN)
        d.arrow(460,526,515,526)
    elif kind == 'heads':
        d.text(50,209,'FLAT OWNER / four F32 elements',21)
        cells(d,120,239,[7,2,-2,-3],185)
        for i in range(4): d.text(212+i*185,315,f'{i} / {4*i}',19,anchor='middle')
        d.text(120,347,'element offset / byte displacement',18)
        d.arrow(305,364,305,403,True); d.arrow(675,364,675,403,True)
        panel(d,120,407,350,103,'Head 0 / [7, 2]',['shape [2]; offset 0; stride 1'],GREEN)
        panel(d,510,407,350,103,'Head 1 / [-2, -3]',['shape [2]; offset 2; stride 1'],GREEN)
        d.text(50,563,'Head-major Q [2,2], strides [2,1]. Dashed: borrow, not copy or computation.',19)
    elif kind == 'owners':
        panel(d,50,205,390,128,'MODEL / immutable weights',['Wq, Wk, Wv own parameters','shared by calls; never overwritten'],GOLD)
        panel(d,550,205,395,128,'CALL / immutable views',['borrow weights and x̂','all shapes checked first'])
        d.arrow(550,270,440,270,True); d.text(448,244,'borrows',16)
        for i,name in enumerate(['Q','K','V']):
            x=50+310*i
            panel(d,x,414,275,105,name+' / fresh owner',['Vec payload + metadata','heads borrow this owner'],GREEN)
            d.arrow(710,333,x+135,414)
        d.text(50,568,'into_parts moves ownership. Dropping a borrower does not free its activation.',19)
    elif kind == 'geometry':
        # Group brackets are future attention associations, not projection copies.
        for row,(label,hkv,groups) in enumerate([('MHA',4,[[0],[1],[2],[3]]),
            ('GQA',2,[[0,1],[2,3]]),('MQA',1,[[0,1,2,3]])]):
            y=205+125*row
            d.text(50,y+28,label,23)
            d.text(50,y+59,f'Hkv = {hkv}',18)
            for h in range(4): d.box(240+170*h,y,125,38,f'Q head {h}',GREEN,18)
            for g,qs in enumerate(groups):
                center=302+170*sum(qs)/len(qs)
                for h in qs: d.arrow(302+170*h,y+38,center,y+69,True)
                d.box(center-68,y+71,136,35,f'K{g} and V{g}',GOLD,17)
        d.text(50,593,'Dashed links: future association. Query heads remain distinct in every row.',17)
    elif kind == 'sequence':
        panel(d,50,205,430,163,'ONE TOKEN / implemented',['x̂ [D]','Wq [Pq,D] times x̂ → q [Pq]','three GEMVs, independent outputs','no position argument'],GREEN)
        panel(d,515,205,430,163,'T TOKEN ROWS / derived',['X̂ [T,D]','Qflat = X̂ Wqᵀ [T,Pq]','reuse weights across token rows','sequence operator is future work'])
        panel(d,50,419,895,154,'Do not confuse axis permutation with reshape',
            ['Contiguous [T,H,Dh] strides: [H×Dh, Dh, 1].',
             'Permuted [H,T,Dh] strides: [Dh, H×Dh, 1]; usually non-contiguous.',
             'A reshape to [H,T,Dh] keeps flat order but changes which token a value belongs to.'],GOLD)
    elif kind == 'packing':
        panel(d,50,205,430,137,'SEPARATE / teaching',['three W owners, three GEMVs','Q, K, V have independent owners','head formation moves no payload'],GREEN)
        panel(d,515,205,430,137,'BUNDLED / Hermon PREVIEW',['three W tensors, one shared input','three mul_mat nodes in one graph','one graph is not one fused kernel'])
        d.text(50,400,'CONCATENATED WEIGHT / derived representation; no teaching candidate',19)
        for x,w,name in [(50,440,'Q width Pq'),(490,220,'K width Pkv'),(710,235,'V width Pkv')]:
            d.box(x,430,w,65,name,GOLD,20)
        d.text(50,535,'Flat output starts: Q at 0; K at Pq; V at Pq + Pkv (element offsets).',19)
        d.text(50,566,'GQA split widths are unequal. Splitting the result into three equal thirds is wrong.',18)
    elif kind == 'cost':
        d.text(50,209,'ANALYTICAL / D = 4096, Hq = 32, Dh = 128; bias-free',21)
        for x,label in [(50,'Geometry'),(255,'Parameters'),(490,'FLOPs / token'),(755,'KV / token / layer')]:
            d.text(x,252,label,18)
        for row,(name,c) in enumerate(zip(['MHA / 32','GQA / 8','MQA / 1'],f['costs'])):
            y=278+75*row
            d.box(50,y,895,62,'',GREEN if row==1 else BLUE)
            for x,value in [(66,name),(255,f'{c["parameters"]:,}'),(490,f'{c["flops"]:,}'),
                            (755,f'{c["kv_bytes_per_token_layer_f16"]:,} B')]:
                d.text(x,y+38,value,21)
        panel(d,50,528,895,63,'KV uses F16 here; teaching activations are F32. Counts are not measured speed.',())
    elif kind == 'boundary':
        for row,position in enumerate([0,7]):
            y=211+144*row
            panel(d,50,y,300,105,f'External label / position {position}',['same x̂ and same weights'],GREEN)
            panel(d,435,y,230,105,'Raw projection',['no position input'])
            panel(d,750,y,195,105,'Same Q/K/V',['fresh owners'],GREEN)
            d.arrow(350,y+52,435,y+52); d.arrow(665,y+52,750,y+52)
        panel(d,50,524,895,65,'NEXT / RoPE adds an explicit position-dependent transformation to Q and K.',(),GOLD)
    elif kind == 'source':
        panel(d,50,205,430,177,'CURRENT / default runtime',['BatchedRuntime → llama.cpp','model graph: build_qkv','separate or fused W representation','then RoPE; then attention'])
        panel(d,515,205,430,177,'PREVIEW / opt-in paged',['normalize token rows','project_bundle → tensor bridge','copy input once; build 3 matmuls','copy Q/K/V back; then RoPE'],GOLD)
        panel(d,50,429,895,149,'Representation is a contract, not a name',
            ['Teaching W [P,D] equals GGML logical dimensions ne=[D,P].',
             'GGML heads: ne=[Dh,H,T]; chapter notation: [T,H,Dh].',
             'Hermon 2a3fd521 / llama.cpp 389ff61d; inspected 2026-09-11.'],GREEN)
    else:
        raise ValueError(kind)


def text_lines(scene):
    # Accessible text description, deliberately not a character-box graph.
    return [scene['purpose'], *scene['contract'],
            'Raw fixture: '+str(fixture()['raw']),
            'Each head borrows its activation owner. No position or attention executes.']
