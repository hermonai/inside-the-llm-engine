"""Position geometry as native vector primitives, driven by the complex oracle."""
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/reference/python'))
from chapter09_rope_oracle import fixture
from chapter07 import panel, cells

BLUE, GREEN, GOLD = '#dcecf7', '#dcefe7', '#f4ead2'
INK, ACCENT, MUTED = '#152b3c', '#237c69', '#465c6b'


def curve(d, points, color=INK, dashed=False, width=2):
    for (x,y),(xx,yy) in zip(points,points[1:]):
        d.line(round(x,3),round(y,3),round(xx,3),round(yy,3),dashed,color,width)


def circle(d,cx,cy,r):
    curve(d,[(cx+r*math.cos(t*math.pi/60),cy-r*math.sin(t*math.pi/60)) for t in range(121)],MUTED,width=1)


def plane(d,cx,cy,r,angle,label):
    circle(d,cx,cy,r)
    d.arrow(cx-r-18,cy,cx+r+25,cy); d.arrow(cx,cy+r+18,cx,cy-r-25)
    d.arrow(cx,cy,cx+r,cy)
    d.arrow(cx,cy,cx+r*math.cos(angle),cy-r*math.sin(angle))
    curve(d,[(cx+40*math.cos(angle*t/40),cy-40*math.sin(angle*t/40)) for t in range(41)],ACCENT,width=3)
    d.text(cx-r,cy+r+50,label,18)


def render(s,d,frame=None):
    f=fixture(); kind=s['kind'][5:]
    if kind=='journey':
        for i,name in enumerate(['Q','K','V']):
            y=205+i*120
            panel(d,50,y,235,91,'Raw '+name,['[H,Dh] / F32'],GREEN)
            panel(d,395,y,235,91,'Rotate at p' if name!='V' else 'Unchanged',
                  ['explicit pairing + R' if name!='V' else 'no RoPE in this model'],GOLD if name!='V' else GREEN)
            panel(d,740,y,205,91,'Positioned '+name if name!='V' else 'Original V',['same shape / owner'],GREEN)
            d.arrow(285,y+45,395,y+45); d.arrow(630,y+45,740,y+45)
        d.text(50,591,'The logical position is an input. No attention scores or KV cache exist in this chapter.',18)
    elif kind=='plane':
        plane(d,250,384,130,1,'Positive angle: counterclockwise')
        d.text(407,391,'a',19); d.text(244,222,'b',19)
        panel(d,540,215,405,130,'Basis vectors determine the map',
              ['(1,0) becomes (cos φ, sin φ)','(0,1) becomes (−sin φ, cos φ)'],BLUE)
        panel(d,540,381,405,176,'Position 1 / first plane',
              ['φ = 1 radian, not 1 degree','[1,2] becomes',
               f'[{f["adjacent"][0]:.6f}, {f["adjacent"][1]:.6f}]','Squared length stays 5'],GREEN)
    elif kind=='frequencies':
        for x,angle,label in [(235,1,'Plane 0 / ω = 1 rad/token'),(700,.1,'Plane 1 / ω = 0.1 rad/token')]:
            plane(d,x,363,108,angle,label)
        d.text(50,551,'R = 4; base θ = 100; position p = 1. Both clocks see the same token position.',20)
        d.text(50,583,'Periods: 2π ≈ 6.283 tokens and 20π ≈ 62.832 tokens; circles are unit basis orbits.',18)
    elif kind=='relative':
        cx,cy,r=255,378,138
        circle(d,cx,cy,r)
        for a,label in [(.4,'q at pω'),(1.5,'k at nω')]:
            xx,yy=cx+r*math.cos(a),cy-r*math.sin(a)
            d.arrow(cx,cy,xx,yy); d.text(xx+12,yy-8,label,18)
        curve(d,[(cx+68*math.cos(.4+1.1*t/40),cy-68*math.sin(.4+1.1*t/40)) for t in range(41)],ACCENT,width=3)
        d.text(60,562,'Illustrative basis rays; the angle gap is (n − p)ω.',18)
        panel(d,540,215,405,130,'Undo the query rotation',
              ['R(pω)ᵀ R(nω) = R((n − p)ω)','A common shift cancels'],BLUE)
        panel(d,540,390,405,167,'Independent four-channel trace',
              ['p = 2; n = 5; θ = 100',f'rotated dot = {f["relative_lhs"]:.6f}',
               f'relative dot = {f["relative_rhs"]:.6f}','Reversing n − p changes the score'],GREEN)
    elif kind=='pairing':
        for y,label,values,pairs in [(224,'ADJACENT',['a0','b0','a1','b1'],[(0,1),(2,3)]),
                                      (410,'SPLIT HALF',['a0','a1','b0','b1'],[(0,2),(1,3)])]:
            d.text(50,y+28,label,20)
            cells(d,270,y,values,155,GREEN)
            for pair,(i,j) in enumerate(pairs):
                yy=y+80+25*pair
                d.line(347+155*i,y+48,347+155*i,yy)
                d.line(347+155*i,yy,347+155*j,yy,dashed=bool(pair))
                d.line(347+155*j,yy,347+155*j,y+48)
            d.text(270,y-18,'physical coordinate order',18)
        d.text(50,590,'The permutation must agree with projected channels. Norm preservation cannot detect a mismatch.',17)
    elif kind=='partial':
        d.text(50,214,'ODD HEAD WIDTH Dh = 5 / even rotary prefix R = 4',22)
        cells(d,90,248,[1,2,3,4,'−0.0'],164,GREEN)
        for i in range(4): d.arrow(172+i*164,296,172+i*164,371)
        d.arrow(828,296,828,371,True)
        cells(d,90,374,[f'{v:.3f}' for v in f['adjacent'][:4]]+['−0.0'],164,BLUE)
        panel(d,90,479,630,96,'Rotate only the first four channels',['ω uses R = 4, not Dh = 5'],GOLD)
        panel(d,746,479,190,96,'Tail / identity',['same F32 bits'],GREEN)
    elif kind=='transaction':
        for x,title,body,color in [(50,'01 / Validate',['shape and position','all input values','finite phases'],BLUE),
                                  (365,'02 / Preflight',['read every pair','check F32 outputs','write nothing'],GOLD),
                                  (680,'03 / Commit',['read both old values','then write the pair','tail untouched'],GREEN)]:
            panel(d,x,235,270,175,title,body,color)
        d.arrow(320,320,365,320); d.arrow(635,320,680,320)
        panel(d,50,467,585,107,'Any typed error before commit',['entire owner payload remains bit-identical'],GOLD)
        panel(d,680,467,270,107,'Scratch / O(R)',['F64 sin/cos per plane'],BLUE)
    elif kind=='positions':
        labels=[('A',127,0,9),('B',3,1,2),('C',48,2,7)]
        for x,label in [(50,'Request'),(260,'Logical p'),(485,'Batch row'),(720,'Physical block')]: d.text(x,219,label,21)
        for i,(name,p,row,block) in enumerate(labels):
            y=247+i*85; d.box(50,y,895,62,'',GREEN if i==0 else BLUE)
            for x,value in [(70,name),(280,p),(510,row),(755,block)]: d.text(x,y+40,value,24)
        d.text(50,548,'RoPE consumes 127, 3, 48. Sorting batch rows or relocating blocks does not change p.',19)
        d.text(50,581,'Scheduling and paging shown as derived contracts, not implemented runtime features.',18)
    elif kind=='scaling':
        # Plot phase for the slower plane: linear functions, not model-quality data.
        x0,y0,w,h=115,505,490,255
        d.arrow(x0,y0,x0+w+20,y0); d.arrow(x0,y0,x0,y0-h-15)
        for p in [0,8,16,24,32]:
            xx=x0+w*p/32; d.line(xx,y0,xx,y0+5); d.text(xx,y0+27,p,17,anchor='middle')
        for phase in [1,2,3]:
            yy=y0-h*phase/3.2; d.line(x0-5,yy,x0,yy); d.text(x0-16,yy+5,phase,17,anchor='end')
        d.text(55,222,'φ / rad',19); d.text(330,570,'position / tokens',19)
        for slope,color,dash in [(.1,INK,False),(.025,ACCENT,True),(.01,MUTED,False)]:
            curve(d,[(x0,y0),(x0+w,y0-h*(32*slope)/3.2)],color,dash,3)
        panel(d,665,223,285,135,'Plane 1 / R = 4', ['θ = 100: slope 0.1','p/4: slope 0.025','θ = 10000: slope 0.01'])
        panel(d,665,402,285,153,'Plane 0 distinguishes', ['base change: ω0 = 1','interpolation: ω0 = 1/4','These are not equivalent'],GOLD)
    elif kind=='source':
        panel(d,50,205,430,177,'CURRENT / default Batched',
              ['llama.cpp model graph','Q/K → ggml_rope_ext','GGML CPU supports pair modes','model/runtime scaling parameters'],BLUE)
        panel(d,515,205,430,177,'PREVIEW / opt-in paged',
              ['Hermon rope_inplace','adjacent pairs, F32 phases','optional per-pair divisors','Q/K rotation before KV write'],GOLD)
        panel(d,50,429,895,145,'TEACHING / deliberately narrower',
              ['Standard RoPE, explicit adjacent or split-half prefix, F64 phases and F32 output.',
               'Checked errors, immutable reference or exclusive in-place owner; no model-quality claim.',
               'Hermon 2a3fd521 / llama.cpp 389ff61d; inspected 2026-09-12.'],GREEN)
    else: raise ValueError(kind)


def text_lines(s):
    return [s['purpose'],*s['contract'],'Independent adjacent result: '+str(fixture()['adjacent']),
            'Position rotates Q and K; V stays unchanged. No attention or cache executes.']
