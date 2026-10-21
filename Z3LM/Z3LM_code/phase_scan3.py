import numpy as np, json
from scipy.optimize import minimize, differential_evolution
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
B=[(0.01,0.07),(-0.9,0.9),(0.0,1.4),(0.0,1.6),(0.05,0.9),(0.05,1.2)]
pats=[(0,90,180),(0,120,180),(240,0,180),(0,120,90),(240,240,180)]
out={}
def bounded(f):
    def g(q):
        for (lo,hi),v in zip(B,q):
            if not (lo <= v <= hi):
                return 1e6
        return f(q)
    return g
for p in [(0,120,90),(240,240,180)]:
    b=bounded(build(*p)); best=(1e9,None)
    sols=[]
    for seed in (0,1,2):
        r=differential_evolution(lambda q: blk(b(q)),B,seed=seed,maxiter=250,popsize=18,
                                 tol=1e-9,polish=True,updating='immediate')
        sols.append(r.x); sols.append(r.x*(1+0.05*np.random.default_rng(seed).standard_normal(6)))
        if r.fun<best[0]: best=(r.fun,r.x)
    for s in sols:
        r=minimize(lambda q: blk(b(q)), s, method='Powell', options=dict(xtol=1e-10,ftol=1e-12,maxiter=20000))
        if r.fun<best[0]: best=(r.fun,r.x)
    out[str(p)]=float(best[0])
    print(p,'chi2=%.3f'%best[0],np.round(best[1],4),flush=True)
d=json.load(open('phase_scan_block.json'))
d.update(out)
json.dump(d,open('phase_scan_block.json','w'),indent=1)
