#!/usr/bin/env python3
"""Independent F32 equation, address and traversal trace for publication."""
from pathlib import Path
import json
import struct

ROOT=Path(__file__).resolve().parents[3]
def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]

def derive():
    f=json.loads((ROOT/'code/reference/fixtures/chapter06-visual.json').read_text())
    m,k,n=f['shape']; w,x,b=f['w'],f['x'],f['b']
    assert len(w)==m*k and len(x)==k and len(b)==k*n
    assert b[::n]==x
    def trace(left,right):
        products=[f32(a*v) for a,v in zip(left,right)]
        acc=0.; sums=[]
        for value in products:
            acc=f32(acc+value); sums.append(acc)
        return products,sums
    rows=[trace(w[i*k:(i+1)*k],x) for i in range(m)]
    cells=[trace(w[i*k:(i+1)*k],b[j::n]) for i in range(m) for j in range(n)]
    ijk=[[i,j,p,i*k+p,p*n+j,i*n+j] for i in range(m) for j in range(n) for p in range(k)]
    ikj=[[i,j,p,i*k+p,p*n+j,i*n+j] for i in range(m) for p in range(k) for j in range(n)]
    tails=[]
    tm,tk,tn=f['tail_shape']; bm,bk,bn=f['tail_block']
    for ii in range(0,tm,bm):
        for kk in range(0,tk,bk):
            for jj in range(0,tn,bn):
                tails.append([ii,min(ii+bm,tm),kk,min(kk+bk,tk),jj,min(jj+bn,tn)])
    return dict(f, y=[s[-1] for _,s in rows], c=[s[-1] for _,s in cells],
                products=rows[0][0], sums=rows[0][1],
                w_strides=[k,1], b_strides=[n,1], c_strides=[n,1],
                w_offsets=list(range(m*k)), w_bytes=[4*i for i in range(m*k)],
                ijk=ijk,ikj=ikj,tails=tails)

if __name__=='__main__':
    print(json.dumps(derive(),indent=2))
