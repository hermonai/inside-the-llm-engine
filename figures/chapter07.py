"""Chapter 7 plates: checked lookup, ownership and a two-pass numerical trace."""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/reference/python'))
from chapter07_visual_oracle import fixture

INK, BLUE, GREEN, GOLD = '#152b3c', '#dcecf7', '#dcefe7', '#f4ead2'


def panel(d, x, y, w, h, title, lines=(), fill=BLUE):
    d.box(x,y,w,h,'',fill)
    d.text(x+18,y+30,title,21)
    for i,line in enumerate(lines): d.text(x+18,y+64+27*i,line,18)


def cells(d, x, y, values, width=100, fill=GREEN):
    for i,value in enumerate(values):
        label = f'{value:.6g}' if isinstance(value,float) else str(value)
        d.box(x+i*width,y,width,48,'',fill)
        d.text(x+(i+.5)*width,y+31,label,21,anchor='middle')


def render(s,d,frame=None):
    f=fixture(); kind=s['kind'][5:]
    if kind=='journey':
        panel(d,50,203,260,106,'01 / Identity',['token 1 is an index','not an activation'],GOLD)
        panel(d,365,203,260,106,'02 / Lookup',['E [3,4] → x [4]','copy row 1 into x'])
        panel(d,680,203,265,106,'03 / RMSNorm',['x [4], w [4], epsilon','new output y [4]'])
        d.arrow(310,257,365,257); d.arrow(625,257,680,257)
        panel(d,50,358,895,96,'The same four values survive the explanation',
              ['x = [1, -2, 3, -4]   →   r ≈ 0.365148   →   y ≈ [0.365, -0.365, 2.191, 1.461]'],GREEN)
        panel(d,50,491,895,93,'Next / three learned projections',
              ['Q/K/V, position, attention and KV state are not implemented by this chapter.'])
        d.arrow(810,309,810,358)
        d.arrow(500,454,500,491,True)
    elif kind=='lookup':
        d.text(50,211,'MODEL / E [3,4], F32, strides [4,1]',21)
        for row in range(3):
            cells(d,95,237+row*58,f['table'][row*4:(row+1)*4],95,GOLD)
            d.text(54,269+row*58,str(row),21)
        d.box(93,293,384,52,'','none')
        d.text(540,241,'REQUEST / token 1',21)
        d.arrow(477,319,558,319)
        cells(d,565,294,f['x'],93)
        d.text(565,375,'x [4] / fresh canonical owner',19)
        d.text(50,451,'Selected row addresses: logical column j = 0, 1, 2, 3',20)
        cells(d,95,476,[f'{o} / {b}' for o,b in zip(f['offsets'],f['bytes'])],185,BLUE)
        d.text(95,556,'element offset / byte displacement; base = 0; no table scan',18)
    elif kind=='sequence':
        d.text(50,209,'IDs [T] = [1, 0, 1]'); d.text(515,209,'X [T,D] = [3,4] / one owner')
        for i,t in enumerate(f['tokens']):
            d.box(70,242+i*88,245,58,f'position {i}: token {t}',GOLD)
            d.arrow(315,271+i*88,510,271+i*88)
            cells(d,515,247+i*88,f['sequence'][4*i:4*i+4],100)
        panel(d,50,523,895,62,'Repeated IDs copy equal values into different output cells.',(),BLUE)
    elif kind=='owners':
        panel(d,50,210,365,122,'MODEL / OwnedTensor',['E: immutable parameters [3,4]','w: immutable learned gain [4]'],GOLD)
        panel(d,570,210,375,122,'CALL / TensorView',['borrows parameter storage','cannot outlive its owner'])
        d.arrow(570,270,415,270,True); d.text(427,250,'borrows',17)
        panel(d,50,405,365,143,'REQUEST / x owner',['lookup copies row 1','mutate x[0] = 99','E[1,0] remains 1'],GREEN)
        panel(d,570,405,375,143,'REQUEST / y owner',['RMSNorm returns new storage','x and w remain unchanged','drop each owner independently'],GREEN)
        d.arrow(237,332,237,405); d.text(255,377,'copy',18)
        d.arrow(415,480,570,480); d.text(429,455,'normalize',17)
        d.text(50,583,'Solid: data transformation / materialization. Dashed: immutable lifetime dependency.',16)
    elif kind=='passes':
        labels=['01 / Square','02 / Reduce','03 / Reciprocal RMS','04 / Scale and write']
        d.text(50,207,'PASS 1 / read all x before producing any y',20)
        for i,label in enumerate(labels):
            xx=50+(i%2)*465; yy=231+(i//2)*173
            fill=GOLD if frame==i else BLUE
            lines=[['x = [1, -2, 3, -4]','x² = [1, 4, 9, 16]'],
                   ['partial sums = [1, 5, 14, 30]','mean = 30 / 4 = 7.5'],
                   ['epsilon = 0.00001','r = 1 / √(7.5 + epsilon)',f'r ≈ {f["inverse"]:.6f}'],
                   ['PASS 2 / read x and w', 'w = [1, 0.5, 2, -1]','yᵢ = (xᵢ × r) × wᵢ']][i]
            panel(d,xx,yy,430,149,label,lines,fill)
        d.arrow(480,304,515,304)
        d.line(730,380,730,392); d.line(730,392,260,392); d.arrow(260,392,260,404)
        d.arrow(480,478,515,478)
        d.text(50,589,'The shared denominator includes x[3]; y[0] cannot be finalized after reading only x[0].',17)
    elif kind=='epsilon':
        panel(d,50,206,430,151,'Defined / zero vector',['x = [0,0,0,0]; epsilon > 0','r = 1 / √epsilon','y = [0,0,0,0]'],GREEN)
        panel(d,515,206,430,151,'Rejected / configuration',['epsilon = 0 or negative','epsilon = NaN or infinity','→ InvalidEpsilon'],GOLD)
        d.text(50,397,'Analytical output scale relative to alpha = 1',21)
        left,top,width,height=95,421,700,132
        for value in [0,0.5,1]:
            y=top+(1-value)*height
            d.line(left,y,left+width,y,True,color='#c2cdd4')
            d.text(78,y+5,str(value),15,anchor='end')
        d.line(left,top,left,top+height); d.line(left,top+height,left+width,top+height)
        points=[]
        for i in range(201):
            alpha=10**(-8+10*i/200)
            ratio=alpha*math.sqrt(7.5+1e-5)/math.sqrt(alpha*alpha*7.5+1e-5)
            points.append((left+width*i/200,top+(1-ratio)*height))
        for a,b in zip(points,points[1:]): d.line(*a,*b,color='#137c69',width=3)
        for exponent,label in [(-8,'10⁻⁸'),(-5,'10⁻⁵'),(-3,'10⁻³'),(0,'1'),(2,'10²')]:
            x=left+(exponent+8)/10*width
            d.text(x,579,label,17,anchor='middle')
        crossover=math.sqrt(1e-5/7.5)
        x=left+(math.log10(crossover)+8)/10*width
        d.line(x,top,x,top+height,True)
        d.text(813,449,'epsilon',18); d.text(813,476,'crossover',18)
        d.text(813,503,'≈ 0.00115',18)
        d.text(813,579,'alpha / log axis',15)
    elif kind=='range':
        d.text(50,211,'F32 arithmetic / alternating [m, -m] / epsilon = 10⁻⁵',20)
        for i,(label,result) in enumerate(zip(['10⁻²⁰','10⁻¹⁰','1','10¹⁰','10²⁰'],f['stress'])):
            yy=239+53*i
            d.box(50,yy,175,44,'m = '+label,GOLD,19)
            d.arrow(225,yy+22,280,yy+22)
            d.box(280,yy,665,44,result+' square/reduction' if result=='finite' else 'ERROR / '+result,GREEN if result=='finite' else GOLD,19)
        panel(d,50,525,895,65,'Different failure: [10¹⁹,10¹⁹,10¹⁹,10¹⁹] has finite squares but a non-finite sum.',())
    elif kind=='cost':
        d.text(50,210,'Analytical payload model / F32 / not measured HBM or RAM traffic',19)
        for i,(label,detail,fill) in enumerate([
            ('Read x / pass 1','4D bytes: reduce squares',GREEN),
            ('Read x / pass 2','4D bytes: apply shared r',GREEN),
            ('Read w / pass 2','4D bytes: learned gain',GOLD),
            ('Write y / pass 2','4D bytes: new owner',GREEN)]):
            xx=50+(i%2)*465; yy=245+(i//2)*123
            panel(d,xx,yy,430,100,label,[detail],fill)
        panel(d,50,514,895,77,'Total RMSNorm payload ≈ 16D bytes; lookup alone ≈ 8D bytes.',
              ['Caches, allocation, fusion and write allocation change physical transfers.'])
    elif kind=='contrast':
        panel(d,50,210,430,166,'RMSNorm / no centering',['x = [4,4,4,4]; gain = [1,1,1,1]','mean(x²) = 16','yᵢ = 4 / √(16 + epsilon)','≈ 1, but not exactly 1'],GREEN)
        panel(d,515,210,430,166,'LayerNorm / center first',['same x; gain 1; bias 0','mean(x) = 4','x − mean(x) = [0,0,0,0]','output = [0,0,0,0]'])
        panel(d,50,427,895,135,'Learned gain is not a second normalization',['Our mixed-sign fixture uses gain [1,0.5,2,-1].','Its output RMS is about 1.342, not 1. Negative gain can flip a sign.','Neither operator is coordinate clipping; this comparison is not a speed benchmark.'],GOLD)
    elif kind=='source':
        panel(d,50,208,430,166,'CURRENT / Hermon default',['Dispatcher selects Batched','worker → Context::decode_batch','llama graph → GGML operators','backend → typed execution'])
        panel(d,515,208,430,166,'PREVIEW / explicit paged path',['tensor_row_f32 copies host row','Rust rms_norm applies gain','model construction checks epsilon','not the default serving route'],GOLD)
        panel(d,50,419,895,158,'The same equation does not imply the same numerical implementation',
              ['Teaching: strided views; F32 products and sequential F32 sum; typed errors.','Pinned GGML CPU: contiguous inner rows; F32 product then double sum.','Storage conversion, graph fusion and device kernels have separate contracts.','Hermon 2a3fd521 / llama.cpp 389ff61d; inspected 2026-09-10.'],GREEN)
    else:
        raise ValueError(kind)


def text_lines(scene):
    f=fixture()
    return [scene['purpose'], *scene['contract'],
            'token 1 ──▶ select E[1,:] ══▶ owned x [4] ══▶ RMSNorm ══▶ owned y [4]',
            'Model owns E and w. Request owns x and y; no output aliases parameters.',
            'x = '+str(f['x']), 'squares = '+str(f['squares']),
            'partial sums = '+str(f['sums']), 'y = '+str(f['output']),
            'element offsets = '+str(f['offsets']), 'byte displacements = '+str(f['bytes'])]
