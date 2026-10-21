import numpy as np, json
from scipy.optimize import minimize
import z3lm
from z3lm import obs_vec
from nufit_parser import NuFIT52
nf=NuFIT52()
x23,xdma,xdcp,_=nf.proj3d(); x13a,x12a,_=nf.proj2d("T13/T12 projection"); x13b,xdmsb,_=nf.proj2d("T13/DMS projection")
def inr(y):
    s12,s13,s23,dc,dm21,dm31=y
    return (min(x23)<=s23<=max(x23) and min(xdma)<=dm31/1e-3<=max(xdma) and min(xdcp)<=dc<=max(xdcp)
            and min(x13a)<=s13<=max(x13a) and min(x12a)<=s12<=max(x12a) and min(xdmsb)<=np.log10(dm21)<=max(xdmsb))
def blk(M):
    y=obs_vec(M)
    return nf.chi2_block(y) if inr(y) else 1e6
def build(p12,p13,p23,r13=1/3):
    e12=np.exp(1j*np.radians(p12)); e13=np.exp(1j*np.radians(p13)); e23=np.exp(1j*np.radians(p23))
    def f(q):
        mm,d1,d2,d3,a12,a23=q
        return mm*np.array([[d1,a12*e12,r13*a12*e13],[a12*e12,d2,a23*e23],[r13*a12*e13,a23*e23,d3]],dtype=complex)
    return f
ref=np.array([0.031592,-0.1579,0.8182,1.0,0.31579,0.6459])
pats=[(0,90,180),(0,120,180),(240,0,180),(0,120,90),(240,240,180)]
rng=np.random.default_rng(7)
out={}
for p in pats:
    b=build(*p); best=None
    starts=[ref]+[ref*(1+0.3*rng.standard_normal(6)) for _ in range(24)]
    for s in starts:
        r=minimize(lambda q: blk(b(q)), s, method='Powell',
                   options=dict(xtol=1e-9,ftol=1e-11,maxiter=20000))
        if best is None or r.fun<best.fun: best=r
    out[str(p)]=float(best.fun)
    print(p, '%.3f'%best.fun, np.round(best.x,4), flush=True)
json.dump(out,open('phase_scan_block.json','w'),indent=1)
