import numpy as np, z3lm
from z3lm import M6, obs_vec, obs_full
from nufit_parser import NuFIT52
from scipy.optimize import minimize_scalar, minimize
nf=NuFIT52()
Q=np.array([1.0,-3/19,9/11,1.0,6/19,135/209])
x23,xdma,xdcp,_=nf.proj3d(); x13a,x12a,_=nf.proj2d("T13/T12 projection"); x13b,xdmsb,_=nf.proj2d("T13/DMS projection")
def inr(y):
    s12,s13,s23,dc,dm21,dm31=y
    return (min(x23)<=s23<=max(x23) and min(xdma)<=dm31/1e-3<=max(xdma) and min(xdcp)<=dc<=max(xdcp)
            and min(x13a)<=s13<=max(x13a) and min(x12a)<=s12<=max(x12a) and min(xdmsb)<=np.log10(dm21)<=max(xdmsb))
def blk(M):
    y=obs_vec(M)
    return nf.chi2_block(y) if inr(y) else 1e6
r=minimize_scalar(lambda m: blk(M6(Q)*m), bounds=(0.029,0.035), method='bounded', options=dict(xatol=1e-9))
print('m0=%.6f chi2=%.4f'%(r.x,r.fun))
o=obs_full(M6(Q)*r.x)
for k in ['s12','s13','s23','delta','dm21','dm31','J','m1','m2','m3','sum_m','mbb']:
    print('  %-5s %.6g'%(k,o[k]))
# dchi2=1 interval
g=np.linspace(r.x-0.002,r.x+0.002,4001); c=np.array([blk(M6(Q)*m) for m in g])
lo=g[c<=r.fun+1].min(); hi=g[c<=r.fun+1].max()
ol=obs_full(M6(Q)*lo); oh=obs_full(M6(Q)*hi)
print('m0 1sig: [%.5f, %.5f]  sum:[%.4f,%.4f] mbb:[%.4f,%.4f]'%(lo,hi,ol['sum_m'],oh['sum_m'],ol['mbb'],oh['mbb']))
np.save('/tmp/m0ref.npy', np.array([r.x,r.fun]))
