"""
Corrected census. Three changes relative to model.py, all forced by numerics:
  (1) proportionality M_d = k M_e with k FIXED by CG gives 3 relations; but M_u = t M_d
      with t = tan(beta) FREE gives only 2 (three equal ratios = two conditions).
  (2) the splitting test must include the TRANSPOSE route: if c_e ~ (c_d * T) then
      M_e ~ M_d^T and the spectra coincide -> excluded.
  (3) P_add is not identically zero: the antisymmetric 120 leaves a residue.
      Values below are taken from the verified numerical rank (rank Dpi).
"""
from itertools import combinations
from fractions import Fraction as F

OPS = {   # (name, c_u, c_d, c_e, symmetry T of the Yukawa product)
 "SU(5)": {"5":  [("10.10.5",1,0,0,+1),("10.5b.5b",0,1,1,+1)],
           "45": [("10.10.45",1,0,0,+1),("10.5b.45b",0,1,-3,+1)],
           "50": [("10.10.50",1,0,0,+1)]},
 "SO(10)":{"10": [("16.16.10",1,1,1,+1)],
           "120":[("16.16.120",1,-1,1,-1)],
           "126":[("16.16.126",1,1,-3,+1)]},
 "E6":    {"27": [("27.27.27",1,1,1,+1)]},
 "PS":    {"(1,2,2)": [("FFc.h",1,1,1,+1)],
           "(15,2,2)":[("FFc.15",1,1,-3,+1)]},
}
SIMPLE = {"SU(5)","SO(10)","E6"}
# P_add from the numerical Jacobian rank: P_add = 13 - rank Dpi
PADD = {("SO(10)","120+126"): 1, ("SO(10)","10+120"): 3}

def prop(a,b):
    a=[F(x) for x in a]; b=[F(x) for x in b]
    if all(x==0 for x in a) or all(x==0 for x in b): return False
    i=next(k for k,x in enumerate(a) if x!=0)
    if b[i]==0: return False
    k=F(b[i],a[i])
    return all(y==k*x for x,y in zip(a,b))

def evaluate(g,reps):
    ops=[o for r in reps for o in OPS[g][r]]
    cu=[o[1] for o in ops]; cd=[o[2] for o in ops]; ce=[o[3] for o in ops]; T=[o[4] for o in ops]
    fed = any(cu) and any(cd) and any(ce)
    single = len(ops)==1
    direct = prop(cd,ce)
    cdt = [a*b for a,b in zip(cd,T)]
    trans  = prop(cdt,ce)
    excluded = (not fed) or direct or trans
    splitting = fed and not excluded
    # counting
    if not fed:            mass=ckm=0
    elif excluded:
        mass = 3 + (2 if single else 0)
        ckm  = 4 if single else 0
    else:                  mass=ckm=0
    gauge = 2 if g in SIMPLE else 0
    p_add = PADD.get((g,"+".join(reps)),0) if splitting else 0   # excluded: relations already in mass
    P = mass+ckm+gauge+1+p_add
    return dict(grp=g, model="+".join(reps), nops=len(ops), fed=fed, single=single,
                direct=direct, trans=trans, splitting=splitting,
                mass=mass, ckm=ckm, gauge=gauge, charge=1, p_add=p_add, P=P,
                baseline=gauge+1)

res=[]
for g,irr in OPS.items():
    for k in range(1,len(irr)+1):
        for ch in combinations(list(irr),k): res.append(evaluate(g,ch))

if __name__=="__main__":
    print(f"{'model':26s}{'ops':>4}{'mass':>5}{'CKM':>5}{'gauge':>6}{'chg':>4}{'Padd':>5}{'P':>4}  status")
    print("-"*76)
    for e in sorted(res,key=lambda x:(-x["P"],x["grp"],x["model"])):
        st = "splitting" if e["splitting"] else ("excluded"+(" (transpose)" if e["trans"] and not e["direct"] else ""))
        print(f"{e['grp']+' '+e['model']:26s}{e['nops']:>4}{e['mass']:>5}{e['ckm']:>5}"
              f"{e['gauge']:>6}{e['charge']:>4}{e['p_add']:>5}{e['P']:>4}  {st}")
    S=[e for e in res if e["splitting"]]
    print(f"\n  total {len(res)} | splitting {len(S)} | excluded {len(res)-len(S)}")
    print(f"  P_add values among splitting: {sorted({e['p_add'] for e in S})}")
    print(f"  P_ex among splitting: {sorted({e['P'] for e in S})}")
    print(f"  splitting models: {[e['grp']+' '+e['model'] for e in S]}")
