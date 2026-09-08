"""Chapter 6 plates: geometry consumes equations and recorded measurements."""
import csv
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ch06_oracle',ROOT/'code/reference/python/chapter06_visual_oracle.py')
oracle=importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)

def fixture():
    return oracle.derive()

def measured(name,header):
    text=(ROOT/'research/benchmarks'/name).read_text()
    block=text.split(header+'\n',1)[1].split('```',1)[0]
    lines=[line.split(' #')[0] for line in block.strip().splitlines()]
    return list(csv.DictReader([header,*lines]))

def number(v):
    return f'{v:g}'

def matrix(d,x,y,values,rows,cols,selected=None,w=64,h=44):
    for i in range(rows):
        for j in range(cols):
            chosen=selected is not None and selected(i,j)
            d.box(x+j*w,y+i*h,w,h,number(values[i*cols+j]),'#f4ead2' if chosen else '#dcefe7',18)
            if chosen: d.line(x+j*w+5,y+(i+1)*h-5,x+(j+1)*w-5,y+(i+1)*h-5,width=3)

def panel(d,x,y,w,h,title,lines):
    d.box(x,y,w,h,title+'\n'+'\n'.join(lines),'#dcecf7',18)

def render(scene,d,frame=None):
    f=fixture(); kind=scene['kind'].removeprefix('ch06-')
    w,x,b,y,c=f['w'],f['x'],f['b'],f['y'],f['c']
    if kind=='engine':
        labels=[('01  Equation','y = W x'),('02  Representation','shape → strides → offsets'),('03  Reference loop','loads → products → sum → store'),('04  Reuse','loop order → tiles'),('05  Kernel / hardware','scalar today; CPU/GPU later'),('06  Model output','projection → bias → logits')]
        for idx,(title,body) in enumerate(labels):
            xx=50+(idx%2)*470; yy=205+(idx//2)*124
            panel(d,xx,yy,425,91,title,[body])
            if idx%2==0: d.arrow(xx+425,yy+45,xx+465,yy+45)
        d.text(50,590,'Ch. 5 tensor storage → Ch. 6 operators → Ch. 7 embedding / normalization',17)
    elif kind=='dot':
        d.text(50,210,'Same reduction index k enters each multiply')
        for p in range(4):
            xx=70+p*218
            d.box(xx,234,90,45,number(w[p]),'#f4ead2',20)
            d.box(xx+105,234,90,45,number(x[p]),'#dcefe7',20)
            d.arrow(xx+45,279,xx+90,321); d.arrow(xx+150,279,xx+107,321)
            d.text(xx+82,347,'×',24)
            d.arrow(xx+99,352,xx+99,366)
            d.box(xx+35,369,130,46,number(f['products'][p]),'#dcecf7',21)
            d.text(xx+63,444,'k = '+str(p),17)
        panel(d,70,483,850,83,'Reduce products in increasing k',['2 + (-2) + (-3) + 0 = '+number(y[0])+'; one scalar, not four outputs'])
    elif kind=='row':
        d.text(50,211,'Row 0: W[0,:] '+str(w[:4])+'; x '+str(x),18)
        for p in range(4):
            yy=244+p*78
            fill='#f4ead2' if frame==p else '#dcefe7'
            previous=0 if p==0 else f['sums'][p-1]
            d.box(60,yy,875,61,f'k={p}:  {number(w[p])} × {number(x[p])} = {number(f["products"][p])}     sum: {number(previous)} → {number(f["sums"][p])}',fill,21)
        d.text(60,590,'After k=3, store y[0] = '+number(y[0])+'. This output is written once.',18)
    elif kind=='gemv':
        d.text(50,208,'W [3,4] • immutable'); matrix(d,50,235,w,3,4)
        d.text(405,208,'x [4] • shared across rows'); matrix(d,410,235,x,4,1)
        d.text(710,208,'y [3] • new owner'); matrix(d,760,235,y,3,1)
        for i in range(3):
            d.text(505,264+44*i,'W row '+str(i)+' · x',18)
            d.arrow(680,258+44*i,745,258+44*i)
        panel(d,50,470,885,91,'One reduction per output row',['W is read row-wise; x is logically reused three times.','No input is mutated; the kernel returns canonical y.'])
    elif kind=='memory':
        d.text(50,208,'01  Logical W [3,4], strides [4,1], base 0'); matrix(d,50,232,w,3,4,lambda i,j:i==1)
        panel(d,420,238,515,115,'02  Follow W[1,2]',['element offset = 1×4 + 2 = 6','byte displacement = 4×6 = 24; value = -1'])
        d.text(50,408,'03  Physical storage (each cell is one F32 element)')
        matrix(d,50,435,w,1,12,lambda i,j:j==6,w=74,h=44)
        for p in range(12): d.text(58+p*74,510,str(p)+' / '+str(4*p),15)
        d.text(50,552,'Labels below cells: element offset / byte displacement; not allocator addresses.',17)
    elif kind=='gemm':
        d.text(50,209,'A = W [3,4]'); matrix(d,50,233,w,3,4,lambda i,j:i==0)
        d.text(402,209,'B [4,2]'); matrix(d,405,233,b,4,2,lambda i,j:j==1)
        d.text(754,209,'C [3,2]'); matrix(d,755,233,c,3,2,lambda i,j:i==0 and j==1)
        d.line(50,255,35,255); d.line(35,255,35,414)
        d.line(35,414,178,414); d.arrow(178,414,178,430)
        d.arrow(501,410,501,430)
        panel(d,50,432,885,57,'C[0,1] = 1×1 + (-2)×(-1) + 3×2 + 0×0 = '+number(c[1]),[])
        d.text(60,508,'Column view: C[:,j] = A B[:,j]; B[:,0] is the same x as GEMV.',18)
        d.text(60,548,'Outer-product view: sum A[:,k] B[k,:] over k = 0,1,2,3.',18)
    elif kind=='loops':
        d.text(50,208,'IJK: keep C[0,0], vary k'); d.text(525,208,'IKJ: keep A[0,k], sweep j')
        for p in range(4):
            yy=235+p*72; fill='#f4ead2' if frame==p else '#dcefe7'
            d.box(50,yy,420,57,f'k={p}: A offset {p}; B offset {p*2}',fill,19)
            d.box(525,yy,420,57,f'k={p}: B offsets {p*2},{p*2+1}; C offsets 0,1',fill,18)
        panel(d,50,541,895,55,'Same 24 contributions; different visitation and immediate reuse',[])
    elif kind=='tiles':
        d.text(50,208,'A [5,7]: tiles [4,4]'); matrix(d,50,232,list(range(35)),5,7,lambda i,j:i>=4 or j>=4,w=41,h=37)
        d.text(414,208,'B [7,3]: tiles [4,2]'); matrix(d,415,232,list(range(21)),7,3,lambda i,j:i>=4 or j>=2,w=45,h=37)
        d.text(730,208,'C [5,3]: tiles [4,2]'); matrix(d,735,232,list(range(15)),5,3,lambda i,j:i>=4 or j>=2,w=48,h=37)
        d.text(50,459,'Cells show flat offsets, not values.',17)
        d.text(50,526,'Last tile: i 4..5, k 4..7, j 2..3; all half-open. Underlines mark edge regions.',17)
        d.text(50,566,'ii → kk → jj selects tiles; i → k → j updates C from independent A and B inputs.',17)
    elif kind=='hierarchy':
        labels=['Main memory: whole problem','Last-level cache: shared working regions','Private cache: nearby rows / tiles','Registers: A scalar + partial C','Execution units: multiply / add']
        for idx,label in enumerate(labels):
            yy=204+73*idx; panel(d,60,yy,490,53,label,[])
            if idx<4: d.arrow(305,yy+53,305,yy+71)
        panel(d,600,218,345,140,'Spatial locality',['IKJ consumes adjacent','B / C addresses before','advancing the row.'])
        panel(d,600,389,345,140,'Temporal locality',['One A scalar feeds j values.','Tiling seeks to reuse data','before eviction.'])
        d.text(60,593,'Conceptual hierarchy; cache topology and transfer costs depend on the actual processor.',16)
    elif kind=='hardware':
        panel(d,50,205,425,166,'CPU concept',['macro tiles → micro tiles','pack if profitable → vector registers','lane products → accumulator vectors','horizontal reduction when required','FMA may round only once'])
        panel(d,520,205,425,166,'GPU concept',['device/global memory → tiles','threadgroup/block shares work','local/shared storage when used','registers → parallel MAC / matrix ops','barriers order cooperative reuse'])
        d.text(50,422,'Four conceptual lanes:'); matrix(d,50,445,f['products'],1,4,w=90)
        d.arrow(427,468,506,468); d.text(530,475,'reduce lanes → '+number(y[0]),22)
        d.text(50,554,'CPU SIMD width and GPU groups differ. Not every multiplication runs simultaneously.',17)
        d.text(50,583,'Production explanation only: ENGINE-2 adds no intrinsics, threads or accelerator backend.',17)
    elif kind=='crossover':
        rows=measured('chapter-06-blocked-matmul.md','size,repetitions,ijk_ns,blocked_ns,speedup')
        d.text(50,211,'Measured direct-time / blocked-time; greater than 1 means blocked won.',18)
        origin=218; scale=255
        for idx,row in enumerate(rows):
            ratio=int(row['ijk_ns'])/int(row['blocked_ns']); yy=245+59*idx
            d.text(55,yy+29,'size '+row['size'],19)
            d.box(origin,yy,ratio*scale,41,'','#dcefe7' if ratio>=1 else '#f4ead2')
            d.text(origin+ratio*scale+12,yy+28,f'{ratio:.2f}×',18)
        d.line(origin+scale,231,origin+scale,548,True); d.text(origin+scale-12,577,'1×',18)
        d.text(50,598,'Apple M1 record, 2026-09-03; tile 32. No universal crossover is implied.',16)
    elif kind=='throughput':
        rows=measured('chapter-06-gemv-vs-gemm.md','n,repetitions,kernel,median_ns,gflops,ideal_flop_per_byte')
        d.text(50,210,'Recorded effective GFLOP/s, fixed weights [512,512]')
        for idx,row in enumerate(rows):
            yy=248+89*idx; value=float(row['gflops'])
            d.text(50,yy+29,'N='+row['n'],20); d.box(175,yy,value*105,48,'','#dcefe7')
            d.text(190+value*105,yy+30,number(value),20)
            d.text(175,yy+71,'ideal '+row['ideal_flop_per_byte']+' FLOP/byte; median '+row['median_ns']+' ns',16)
        panel(d,50,531,895,60,'N=8 lost throughput despite more potential reuse; causes were not isolated.',[])
    elif kind=='production':
        panel(d,50,205,425,214,'TEACHING / safe Rust',['OwnedTensor / TensorView','F32, element strides, explicit base','checked GEMV / GEMM','scalar products, F32 accumulation','fresh canonical output','reference or explicit blocked call'])
        panel(d,520,205,425,214,'GGML / pinned source',['type + ne[] + nb[] byte strides','packed weights need type traits','ggml_mul_mat creates graph node','backend selects eligible kernel','CPU vector dot or GPU paths','result layout follows GGML axes'])
        panel(d,50,464,895,113,'Invariant, not identical representation',['same intended contraction after axis translation; stated numerical contract','Teaching W [M,K] ↔ GGML weight ne=[K,M]; row-batched inputs [N,K].','Packing and quantization change physical representation, not the logical roles.'])
    elif kind=='source':
        panel(d,50,204,430,103,'CURRENT default Hermon',['Dispatcher → BatchedRuntime','Context::decode_batch → llama_decode'])
        panel(d,530,204,415,103,'PREVIEW / LIBRARY bridge',['paged → matmul_bundle_with_session','checked Rust → tensor_bridge.cpp'])
        d.arrow(265,307,265,348); d.arrow(735,307,735,348)
        panel(d,50,350,895,95,'Pinned llama.cpp / GGML: graph node is not an executed kernel',['Default: model graph → ggml_mul_mat → configured CPU / Metal / CUDA','Optional host bridge: ggml_mul_mat → session_compute (CPU only)'])
        panel(d,50,475,895,112,'Separate verification pattern: ENGINE-2 never calls the production kernels',['candidate result → finite elementwise error → tolerance','pass → benchmark; fail → reject. No speed claim without workload evidence.'])
    else: raise ValueError(kind)

def text_lines(scene):
    f=fixture()
    lines=[scene['purpose'],*scene['contract'],'','W [3,4] = '+str(f['w']), 'x [4] = '+str(f['x']), 'y [3] = '+str(f['y']), 'B [4,2] = '+str(f['b']), 'C [3,2] = '+str(f['c']), 'row 0 products = '+str(f['products']), 'row 0 partial sums = '+str(f['sums'])]
    if scene['kind']=='ch06-loops':
        lines+=['Trace tuple: i,j,k,A_offset,B_offset,C_offset','IJK '+str(f['ijk'][:4]),'IKJ '+str(f['ikj'][:4])]
    if scene['kind']=='ch06-tiles': lines+=['Half-open tile ranges '+str(f['tails'])]
    if scene['kind']=='ch06-crossover':
        lines += [str(row) for row in measured('chapter-06-blocked-matmul.md','size,repetitions,ijk_ns,blocked_ns,speedup')]
    if scene['kind']=='ch06-throughput':
        lines += [str(row) for row in measured('chapter-06-gemv-vs-gemm.md','n,repetitions,kernel,median_ns,gflops,ideal_flop_per_byte')]
    return lines
