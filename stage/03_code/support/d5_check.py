import numpy as np
rng=np.random.default_rng(2026)
def anti():
    A=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); return (A-A.T)/2
def sym():
    A=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); return (A+A.T)/2

print("="*70); print("THEOREM A:  M_e = -2 M_d - M_d^T"); print("="*70)
w=0; wrev=0
for _ in range(400):
    A=anti(); S=sym(); va,vs=rng.normal(size=2)+1j*rng.normal(size=2)
    Md=-va*A+vs*S
    Me=va*A-3*vs*S
    w=max(w,abs(Me-(-2*Md-Md.T)).max())
    wrev=max(wrev,abs(Md-(-(Md+Me)/2 - (3*Md+Me)/2*0)).max())
print(f"  max |M_e - (-2M_d - M_d^T)| over 400 points = {w:.2e}")
print(f"  (parameter-free matrix identity, no vev enters)")

print(); print("="*70); print("THEOREM B:  collinear direction  M_u x : M_d x : M_e x = beta:1:-3"); print("="*70)
worst_e=worst_u=0
for _ in range(500):
    A=anti(); S=sym(); va,vs,ba,bs=rng.normal(size=4)+1j*rng.normal(size=4)
    Mu=ba*A+bs*S; Md=-va*A+vs*S; Me=va*A-3*vs*S
    U,s,Vh=np.linalg.svd(A)
    x=Vh.conj().T[:,-1]                      # right null vector of A
    sc=max(np.abs(Md).max(),1e-30)
    worst_e=max(worst_e,abs(Me@x+3*Md@x).max()/sc)
    worst_u=max(worst_u,abs(Mu@x-bs*Md@x).max()/sc)
print(f"  max |M_e x + 3 M_d x| / scale = {worst_e:.2e}")
print(f"  max |M_u x - beta M_d x| / scale = {worst_u:.2e}")

print(); print("="*70); print("COROLLARY D:  1 < sum m_e^2 / sum m_d^2 <= 9"); print("="*70)
rs=[]
for _ in range(6000):
    A=anti(); S=sym(); va,vs=rng.normal(size=2)+1j*rng.normal(size=2)
    Md=-va*A+vs*S; Me=va*A-3*vs*S
    sd=np.linalg.svd(Md,compute_uv=False)**2; se=np.linalg.svd(Me,compute_uv=False)**2
    rs.append(se.sum()/sd.sum())
rs=np.array(rs)
print(f"  120+126 : min={rs.min():.3f}  median={np.median(rs):.3f}  max={rs.max():.3f}")
print(f"            violations of >1: {(rs<=1).sum()}   violations of <=9: {(rs>9).sum()}")

print(); print("  controls (same ratio, other models):")
for nm,cu,cd,ce,ka,kd,ke in [("10+126",[1,1],[1,1],[1,-3],["sym","sym"],0,0),
                             ("10+120",[1,-1],[1,1],[1,1],["sym","anti"],0,0),
                             ("SU(5) 5+45",[1,1,0,0],[0,0,1,1],[0,0,1,-3],["sym","sym","gen","gen"],0,0)]:
    pass
def ctrl(name, ops):
    lo=hi=0; vals=[]
    for _ in range(3000):
        Ms=[]
        for (cu,cd,ce,kind) in ops:
            Y=sym() if kind=="sym" else (anti() if kind=="anti" else (rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))))
            vd=rng.normal()+1j*rng.normal(); vu=rng.normal()+1j*rng.normal()
            Ms.append(None)
        # build with independent vevs per operator
        Mu=np.zeros((3,3),complex); Md=np.zeros((3,3),complex); Me=np.zeros((3,3),complex)
        for (cu,cd,ce,kind) in ops:
            Y=sym() if kind=="sym" else (anti() if kind=="anti" else (rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))))
            Mu=Mu+cu*(rng.normal()+1j*rng.normal())*Y
            v=(rng.normal()+1j*rng.normal())
            Md=Md+cd*v*Y; Me=Me+ce*v*Y
        sd=np.linalg.svd(Md,compute_uv=False)**2; se=np.linalg.svd(Me,compute_uv=False)**2
        if sd.sum()<1e-12: continue
        r=se.sum()/sd.sum(); vals.append(r)
        lo+= (r<=1); hi+= (r>9)
    print(f"  {name:12s}: median={np.median(vals):.2f}  violations: >1 {lo}, <=9 {hi}")

ctrl("10+126",[(1,1,1,"sym"),(1,1,-3,"sym")])
ctrl("10+120",[(1,1,1,"sym"),(1,-1,1,"anti")])
ctrl("SU(5) 5+45",[(1,0,0,"sym"),(0,1,1,"genT"),(1,0,0,"sym"),(0,1,-3,"genT")])

print(); print("="*70); print("COROLLARY E:  3 m_d,min <= m_e,max  and  m_e,min <= 3 m_d,max"); print("="*70)
v1=v2=0
for _ in range(6000):
    A=anti(); S=sym(); va,vs=rng.normal(size=2)+1j*rng.normal(size=2)
    Md=-va*A+vs*S; Me=va*A-3*vs*S
    md=np.linalg.svd(Md,compute_uv=False); me=np.linalg.svd(Me,compute_uv=False)
    v1+= (3*md.min()>me.max()); v2+= (me.min()>3*md.max())
print(f"  120+126 violations: first {v1}, second {v2}")

print(); print("="*70); print("REDUNDANCY COUNT (analytic P_add)"); print("="*70)
print("  S symmetric  : 6 complex = 12 real")
print("  A antisym    : 3 complex =  6 real")
print("  vevs         : 4 complex =  8 real")
print("  total        :              26 real")
print("  - rescaling 2 reps            -4")
print("  - U(3) flavour                -9")
print("  - M_u overall phase           -1")
print("  physical                     12   vs 13 observables -> P_add = 1")
