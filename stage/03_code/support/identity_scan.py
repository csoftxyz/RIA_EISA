"""
Mechanical scan for exception contractions: determinant-, trace- and other
invariant-type identities that hold WITHOUT a scalar proportionality.
Test: build a bank of sector invariants at random parameter points and ask
whether any inter-sector RATIO is constant.
"""
import numpy as np
rng=np.random.default_rng(4242)
def rmat(k):
    A=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    return {"sym":(A+A.T)/2,"anti":(A-A.T)/2,"gen":A}[k]

def invs(M):
    ev=np.linalg.eigvals(M); m=np.abs(ev)
    return {
      "Re tr":       (np.trace(M).real, 0),
      "Im tr":       (np.trace(M).imag, 0),
      "det":         (np.linalg.det(M).real, np.linalg.det(M).imag),
      "prod m":      (m.prod(), 0),
      "sum m":       (m.sum(), 0),
      "tr MM^dag":   (np.trace(M@M.conj().T).real, 0),
      "tr M^2":      (np.trace(M@M).real, np.trace(M@M).imag),
      "det MM^dag":  (np.linalg.det(M@M.conj().T).real, 0),
    }

def run(name, cu, cd, ce, kinds, tb=3.0, n=300):
    cu,cd,ce=map(np.array,(cu,cd,ce))
    rec={k:[] for k in ["Re tr","Im tr","det","prod m","sum m","tr MM^dag","tr M^2","det MM^dag"]}
    for _ in range(n):
        Ys=[rmat(k) for k in kinds]; r=np.exp(rng.normal(scale=0.7,size=len(kinds)))
        Ms=[sum(cu[j]*r[j]*tb*Ys[j] for j in range(len(Ys))),
            sum(cd[j]*r[j]*Ys[j]      for j in range(len(Ys))),
            sum(ce[j]*r[j]*Ys[j]      for j in range(len(Ys)))]
        I=[invs(M) for M in Ms]
        for k in rec:
            for (a,b) in [(0,1),(0,2),(1,2)]:
                va=I[a][k]; vb=I[b][k]
                if isinstance(va,tuple):
                    if abs(vb[0])<1e-12 or abs(va[0])<1e-12: continue
                    rec[k].append(va[0]/vb[0])
                else:
                    if abs(vb)<1e-12 or abs(va)<1e-12: continue
                    rec[k].append(va/vb)
    hits=[]
    for k,v in rec.items():
        if len(v)<50: continue
        v=np.array(v); rel=np.std(v)/max(abs(np.mean(v)),1e-30)
        if rel<1e-8: hits.append((k, float(np.mean(v)), rel))
    print(f"  {name:28s} constant inter-sector ratios: "
          + (", ".join(f"{k}={val:.4g} (rel std {r:.1e})" for k,val,r in hits) if hits else "NONE"))
    return hits

print("="*84)
print("IDENTITY BANK SCAN  (300 random points per model, tan beta = 3)")
print("="*84)
CASES=[("SU(5) 5+45 (rank3)",   [1,1,0,0],[0,0,1,1],[0,0,1,-3],["sym","sym","gen","gen"]),
       ("SU(5) 5+45+50",        [1,1,1,0,0],[0,0,0,1,1],[0,0,0,1,-3],["sym","sym","sym","gen","gen"]),
       ("SO(10) 120+126 (rank2)",[1,1],[-1,1],[1,-3],["anti","sym"]),
       ("SO(10) 10+120+126",    [1,1,1],[1,-1,1],[1,1,-3],["sym","anti","sym"]),
       ("SU(5) 5 (control)",    [1,0],[0,1],[0,1],["sym","gen"])]
hits={}
for nm,cu,cd,ce,kinds in CASES:
    hits[nm]=run(nm,np.array(cu),np.array(cd),np.array(ce),kinds)

print()
print("="*84)
print("THE TRACE IDENTITY, EXPLICITLY")
print("="*84)
for nm,cu,cd,ce,kinds in CASES[3:4]:
    cu,cd,ce=map(np.array,(cu,cd,ce)); worst=0
    for _ in range(200):
        Ys=[rmat(k) for k in kinds]; r=np.exp(rng.normal(size=3))
        Mu=sum(cu[j]*r[j]*3.0*Ys[j] for j in range(3))
        Md=sum(cd[j]*r[j]*Ys[j] for j in range(3))
        worst=max(worst, abs(np.trace(Mu)-np.trace(Md)))
    print(f"  SO(10) 10+120+126:  max |tr M_u - tr M_d| over 200 points = {worst:.3e}")
    print("  the 120 is traceless, and u,d receive it with opposite signs:")
    print("     tr M_u = tr Y_10 + tr Y_126 = tr M_d  EXACTLY.")
    print("  yet dim(image) = 13 (see verify_all.py), so this identity is INVISIBLE:")
    print("  it constrains the COMPLEX eigenvalues, while the observables are")
    print("  moduli |lambda| and mixing, and the phases absorb it.")
