"""Chapter 5 geometry consumes the shared, Rust/oracle-verified fixture."""
import json
from itertools import product


def fixture(scene, root):
    data=json.loads((root/scene['fixture']['source']).read_text())
    for name in ('source','transpose','copy','reshape','slice'):
        state=data[name]
        offsets=[state['base']+sum(i*s for i,s in zip(index,state['strides'])) for index in product(*(range(n) for n in state['shape']))]
        assert offsets==state['offsets']
        storage=data['copy']['values'] if name=='copy' else data['source']['values']
        assert [storage[i] for i in offsets]==state['values']
    return data


def matrix(d,x,y,state):
    rows,cols=state['shape']
    for i in range(rows): d.cells(x,y+48*i,state['values'][i*cols:(i+1)*cols],68)


def render(scene,d,root):
    f=fixture(scene,root); k=scene['kind']
    if k=='tensor-transpose':
        d.text(60,208,'01  Original coordinates'); matrix(d,60,235,f['source'])
        d.text(560,208,'02  Swap axes, preserve values'); matrix(d,560,235,f['transpose'])
        d.arrow(325,285,513,285); d.text(335,264,'metadata only',17)
        d.text(60,377,'A[1,0] = 4',18); d.text(730,314,'Aᵀ[0,1] = 4',18)
        d.text(60,439,'03  Both views borrow this one buffer')
        d.cells(60,460,f['source']['values'],110)
        d.text(60,550,'Transpose logical offsets: '+str(f['transpose']['offsets']),19)
    elif k=='tensor-copy':
        d.text(60,208,'01  Existing source buffer'); d.cells(60,234,f['source']['values'],110)
        d.text(60,326,'02  Read source offsets in logical order')
        d.cells(60,350,f['transpose']['offsets'],110,'#f4ead2')
        d.arrow(808,365,808,449); d.text(742,420,'copy',18)
        d.text(60,443,'03  New canonical buffer'); d.cells(60,465,f['copy']['values'],110)
        d.text(60,550,'C[0,1] = 4; writing C[0,1] = 40 leaves A[1,0] = 4.',19)
    elif k=='tensor-reshape':
        d.text(60,205,'01  Reshape [3,2]'); matrix(d,60,235,f['reshape'])
        d.text(550,205,'02  Transpose [3,2]'); matrix(d,550,235,f['transpose'])
        d.text(235,270,'R[0,1] = 2',20); d.text(725,270,'Aᵀ[0,1] = 4',20)
        d.text(60,420,'strides [2,1]',19); d.text(550,420,'strides [1,3]',19)
        d.box(60,461,870,100,'03  Transpose → reshape_view([6])\nNonContiguous: the API rejects this request; it does not copy.', '#f4ead2',19)
    elif k=='tensor-slice':
        d.text(60,210,'01  Original A'); matrix(d,60,238,f['source'])
        d.arrow(335,283,520,283); d.text(335,256,'columns 1..3',17)
        d.text(550,210,'02  S = A[:,1:3]'); matrix(d,550,238,f['slice'])
        d.text(60,389,'03  Keep full storage; base shifts to element 1')
        d.cells(60,414,f['source']['values'],110)
        for i in range(6): d.text(81+110*i,492,('read '+str(i)) if i in f['slice']['offsets'] else 'skip',16)
        d.box(60,523,870,57,'max offset 5 → minimum backing length 6; logical element count is only 4', '#f4ead2',17)
    elif k=='tensor-ownership':
        d.box(40,203,345,150,'«struct» OwnedTensor\nshape: Vec<usize>\nstrides: Vec<usize>\ndata: Vec<f32>', '#f4ead2',18)
        d.box(685,223,265,112,'«owned buffer»\ninitialized F32 elements\none Vec allocation', '#f4ead2',17)
        d.items.append('<polygon points="385,275 397,268 409,275 397,282" fill="#152b3c"/>')
        d.line(409,275,685,275); d.text(447,253,'data owns',18); d.text(655,267,'1',17)
        d.box(40,416,390,145,"«struct» TensorView<'a>\nstorage: &'a [f32]\nshape, strides: Vec<usize>\nbase_offset: usize", '#dcefe7',18)
        d.box(550,416,400,145,"«struct» TensorViewMut<'a>\nstorage: &'a mut [f32]\nshape, strides: Vec<usize>\ncomplete canonical slice", '#dcecf7',18)
        d.arrow(390,416,690,335,True); d.text(428,375,'shared borrow',16)
        d.arrow(820,416,820,335,True); d.text(835,382,'exclusive',16)
    elif k=='tensor-lifetime':
        for y,title,body,color in [(205,'01  Shared phase','let shared = owner.view();\nread shared[1,2] → 6; this is the last use of shared','#dcefe7'),(327,'02  Exclusive phase','let writable = owner.view_mut();\nwrite writable[1,2] = 60; end the exclusive use','#f4ead2'),(449,'03  Shared phase again','owner.view()[1,2] → 60\nUsing the old shared view after the write request would be rejected.','#dcefe7')]:
            d.text(60,y,title); d.box(60,y+18,875,75,body,color,18)
        d.arrow(500,298,500,333); d.arrow(500,420,500,455)
    elif k=='tensor-production':
        d.box(40,205,430,238,'EDUCATIONAL / Tensor Substrate\nshape [2,3], element strides [3,1]\nbase measured in F32 elements\nOwnedTensor owns Vec<f32>\nTensorView borrows the allocation\ntranspose changes metadata\nmaterialize creates owned F32', '#dcefe7',18)
        d.box(530,205,430,238,'GGML / pinned representation\nne[]: dimension lengths\nnb[]: byte strides, type/block aware\nbuffer and data describe storage\nview_src / view_offs describe views\npacked dtype is not one F32 per slot\nbackend residency constrains access', '#dcecf7',18)
        d.box(40,475,920,101,'Hermon tensor_row_f32: validate host tensor → convert row → owned Vec<f32>\nThe model keeps packed weights; the caller owns only the converted row.\nTensorSession owns mutable scratch; exclusive calls serialize its access.', '#f4ead2',17)
    else: raise ValueError('unknown Chapter 5 plate '+k)


def text_lines(scene,root):
    f=fixture(scene,root); name=scene['kind'].removeprefix('tensor-')
    if name=='ownership': return ['OwnedTensor ◆── data: Vec<f32> ── owns allocation', "TensorView<'a> ┄┄▷ shared &[f32]; owns shape/stride/base metadata", "TensorViewMut<'a> ┄┄▷ exclusive &mut [f32]; owns shape/stride metadata", 'Views are alternatives when exclusive use overlaps; no inheritance.']
    if name=='lifetime': return ['shared read 6 ──▶ last shared use ──▶ exclusive write 60', '                                           │', '                                           ▼', '                               later shared read observes 60']
    if name=='production': return ['F32 element strides ── unit boundary ──▷ GGML byte/block-aware strides', 'packed model row ══ convert ══▶ caller-owned F32 row', 'CURRENT default: batched; PREVIEW: paged; bridge API: LIBRARY']
    states={'transpose':['source','transpose'],'copy':['transpose','copy'],'reshape':['reshape','transpose'],'slice':['source','slice']}[name]
    lines=[]
    for state in states:
        item=f[state]
        lines += [state.upper()+': shape '+str(item['shape'])+'; strides '+str(item['strides'])+'; base '+str(item['base']), '  logical values '+str(item['values']), '  storage offsets '+str(item['offsets']), '']
    return lines+['source allocation [1,2,3,4,5,6] remains shared by the views', 'Only COPY owns a newly materialized payload.']
