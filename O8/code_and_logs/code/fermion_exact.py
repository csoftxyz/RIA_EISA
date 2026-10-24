#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Octahedral minimal-fermion model — EXACT computation, no Monte Carlo
=====================================================================
Geometry: octahedron K_{2,2,2}: 6 vertices (axis 0,1,2 x sign +-), 12 edges, 8 triangular faces.
Gauge: U(1) on edges, Wilson S_g = beta sum_f cos(theta_f), theta_f = face curvature.

Fermions (zero-parameter): one complex fermion per vertex, mass m, hopping t=1:
    D = m*I + t * Gamma .* U,   Gamma = unique Z3(T)-equivariant antisymmetric orientation
    (edges forward axis a -> a+1 mod 3), U_ij = e^{i theta_ij}, theta antisymmetric.
KEY STRUCTURAL FACT: octahedron has ODD cycles (triangles) => no bipartite gamma5 grading;
    det(m + D) is complex for m>0.  A real, positive fermion weight therefore REQUIRES the
    conjugate pair  W = det(m+D) det(m-D) = |det|^2  >= 0  (det(m+D)* = det(m-D)).
    => fermionic multiplicity 2 is forced by non-bipartiteness of the octahedron.
    (this is a lattice-chirality statement, NOT a derivation of dim g1 = 4)
At m=0:  det D is real and  W = det(D)^2.

Exact integration over the gauge field (7-dim after gauge fixing) is done analytically:
    det D = sum over permutation cycle-covers  c_sigma * exp(i nu_sigma . theta_link)
    e^{beta cos theta_f} = sum_{n_f} I_{n_f}(beta) e^{i n_f theta_f}
    per-link integration imposes  A n = -nu  =>  n = n0(nu) + k*(1,...,1)
    Z = sum_sigma c_sigma * sum_k prod_f I_{n0_f + k}(beta)          [EXACT]
Z3 sectors (flux charge mod 3, the paper's operational Z3 character):
    sector r: terms with (sum_f n0_f) + 8 k = r (mod 3).
"""
import numpy as np
from scipy.special import iv as besseli
from itertools import permutations
from collections import defaultdict

# ---------------- geometry ----------------
verts = [(a,s) for a in range(3) for s in (1,-1)]
vid   = {v:i for i,v in enumerate(verts)}
edges = []                                   # forward-ordered (u -> v with axis(u)+1=axis(v) mod 3)
for i,u in enumerate(verts):
    for j,v in enumerate(verts):
        if j<=i: continue
        a,b = u[0],v[0]
        if a==b: continue
        edges.append((u,v) if b==(a+1)%3 else (v,u))
E=len(edges); edge_id={frozenset((u,v)):k for k,(u,v) in enumerate(edges)}
def th(u,v):
    return ((1 if v[0]==(u[0]+1)%3 else -1), edge_id[frozenset((u,v))])
faces=[[(0,sx),(1,sy),(2,sz)] for sx in (1,-1) for sy in (1,-1) for sz in (1,-1)]
F=len(faces)
Araw=np.zeros((E,F),int)                      # raw traversal sign per face (all-forward ordering)
for fi,fc in enumerate(faces):
    for k in range(3):
        s,e=th(fc[k],fc[(k+1)%3]); Araw[e,fi]=s
assert np.all(np.abs(Araw).sum(axis=1)==2)
# Consistent orientations: each link must enter its two adjacent faces with OPPOSITE signs
# (orientable surface, S^2).  Solve o_f in {+-1} by BFS over the dual (cube) graph.
adj=[set() for _ in range(F)]
for e in range(E):
    fs=[f for f in range(F) if Araw[e,f]!=0]
    assert len(fs)==2
    adj[fs[0]].add(fs[1]); adj[fs[1]].add(fs[0])
o=np.zeros(F,int); o[0]=1
from collections import deque
dq=deque([0]); seen={0}
while dq:
    f=dq.popleft()
    for g in adj[f]:
        e=[e for e in range(E) if Araw[e,f]!=0 and Araw[e,g]!=0][0]
        og = -o[f]*Araw[e,f]*Araw[e,g]        # require o_f*A_f = -o_g*A_g on shared edge
        if g in seen:
            assert o[g]==og, "inconsistent orientation (non-orientable?)"
        else:
            o[g]=og; seen.add(g); dq.append(g)
assert len(seen)==F
A=Araw*o[np.newaxis,:]                        # A[link,face]
assert np.all(A.sum(axis=1)==0)               # each link: +1 in one face, -1 in the other
assert np.array_equal(A@np.ones(F,int), np.zeros(E,int))
assert np.linalg.matrix_rank(A)==F-1

# Gamma: T-equivariant antisymmetric orientation matrix
Gm=np.zeros((6,6))
for k,(u,v) in enumerate(edges):
    Gm[vid[u],vid[v]]=1.0; Gm[vid[v],vid[u]]=-1.0

# ---------------- determinant expansion ----------------
def expansion(m, t=1.0):
    """list of (coeff, nu[12]) for det(m I + t Gamma.U), complex phases tracked via nu"""
    n=6; out=[]
    adj=[[j for j in range(n) if j!=i and verts[j][0]!=verts[i][0]] for i in range(n)]
    for sig in permutations(range(n)):
        perm=list(sig)
        if any(perm[i]!=i and perm[i] not in adj[i] for i in range(n)): continue
        seen=[False]*n; sg=1
        for i in range(n):
            if seen[i]: continue
            j=i; L=0
            while not seen[j]: seen[j]=True; j=perm[j]; L+=1
            if L%2==0: sg=-sg
        nfix=sum(1 for i in range(n) if perm[i]==i)
        coeff=sg*(m**nfix)*(t**(n-nfix))
        if coeff==0: continue
        nu=np.zeros(E,int)
        gs=1.0
        for i in range(n):
            j=perm[i]
            if i==j: continue
            s,e=th(verts[i],verts[j]); nu[e]+=s; gs*=s
        out.append((coeff*gs,nu))
    agg=defaultdict(float)
    for c,nu in out: agg[tuple(nu)]+=c
    return [(c,np.array(nu,int)) for nu,c in agg.items()]

def expansion_minus(m,t=1.0):
    n=6; out=[]
    adj=[[j for j in range(n) if j!=i and verts[j][0]!=verts[i][0]] for i in range(n)]
    for sig in permutations(range(n)):
        perm=list(sig)
        if any(perm[i]!=i and perm[i] not in adj[i] for i in range(n)): continue
        seen=[False]*n; sg=1
        for i in range(n):
            if seen[i]: continue
            j=i; L=0
            while not seen[j]: seen[j]=True; j=perm[j]; L+=1
            if L%2==0: sg=-sg
        nfix=sum(1 for i in range(n) if perm[i]==i)
        coeff=sg*(m**nfix)*(t**(n-nfix))*((-1)**(n-nfix))
        if coeff==0: continue
        nu=np.zeros(E,int)
        gs=1.0
        for i in range(n):
            j=perm[i]
            if i==j: continue
            s,e=th(verts[i],verts[j]); nu[e]+=s; gs*=s
        out.append((coeff*gs,nu))
    agg=defaultdict(float)
    for c,nu in out: agg[tuple(nu)]+=c
    return [(c,np.array(nu,int)) for nu,c in agg.items()]

_rows=None
def solve_face_charges(nu):
    """A n = -nu over the integers; return integral n0 (kernel = all-ones -> gauge n0[0]=0)."""
    global _rows
    import sympy as sp
    if _rows is None:
        # pick 7 independent rows of A
        rows=[]
        for l in range(E):
            trial=rows+[l]
            if np.linalg.matrix_rank(A[trial])==len(trial): rows=trial
            if len(rows)==F-1: break
        Asub = sp.Matrix(A[np.ix_(rows, range(1,F))].tolist())   # drop column 0
        assert Asub.det()!=0
        _rows=(rows,Asub)
    rows,Asub=_rows
    rhs = sp.Matrix((-nu[rows]).tolist())
    sol = Asub.inv()*rhs
    n0=[0]+[int(sp.nsimplify(sol[i])) for i in range(F-1)]
    n0=np.array(n0,dtype=int)
    assert np.array_equal(A@n0,-nu), (nu.tolist(),(A@n0+nu).tolist())
    return n0
    assert np.array_equal(A @ n0, -nu), (nu.tolist(), (A @ n0 + nu).tolist())
    return n0

def In_table(beta,N=70):
    return besseli(np.arange(N+1), beta)   # tab[j] = I_j(beta)

def series_full(n0,tab,beta,N=70):
    """sum_k prod_f I_{n0_f+k}(beta) over all k (sectors resolved by caller)"""
    return series_sector(n0,tab,N)          # returns per-sector array; total = .sum()

def series_sector(n0,tab,N=70):
    out=np.zeros(3)
    Q0=int(n0.sum())%3
    for k in range(-N,N+1):
        idx=np.abs(n0+k)
        if idx.max()>N: continue
        v=np.prod(tab[idx])
        out[(Q0+8*k)%3]+=v                  # 8 faces -> 8k mod 3 = 2k mod 3
    return out

_cache={}
def sector_sum(nu,tab,N=70):
    key=tuple(nu)
    if key not in _cache: _cache[key]=solve_face_charges(np.array(nu,int))
    return series_sector(_cache[key],tab,N)

def Z_total_sectors(m,beta,Nf_pair=1,N=70):
    """Weight = |det(m+D)|^{2*Nf_pair}? no: W = [det(m+D)det(m-D)]^{Nf_pair}, one pair = Nf_pair=1.
    Returns Zg (pure gauge) and sector array for the fermionic weight."""
    tab=In_table(beta,N)
    # pure gauge
    Zg=0.0
    for k in range(-N,N+1):
        Zg+=tab[abs(k)]**8
    P=expansion(m); Mm=expansion_minus(m)
    Zs=np.zeros(3)
    if Nf_pair==1:
        for c1,nu1 in P:
            s1=sector_sum(nu1,tab,N)
            Zs+=c1*s1
        for c2,nu2 in Mm:
            s2=sector_sum(nu2,tab,N)
            Zs+=c2*s2
        # weight det(m+D)*det(m-D) = product of the two expansions -> need joint sum:
    # joint: W = sum_{s1 in P} sum_{s2 in Mm} c1 c2 (nu1+nu2)
    Zs=np.zeros(3)
    for c1,nu1 in P:
        for c2,nu2 in Mm:
            Zs+=c1*c2*sector_sum(nu1+nu2,tab,N)
    if Nf_pair==2:
        # square the weight: (sum_{s1,s2}...)^2 = sum over 4-tuples: do via merging pairs first
        agg=defaultdict(float)
        for c1,nu1 in P:
            for c2,nu2 in Mm:
                agg[tuple(nu1+nu2)]+=c1*c2
        pairlist=[(c,np.array(nu,int)) for nu,c in agg.items()]
        Zs=np.zeros(3)
        for c1,nu1 in pairlist:
            for c2,nu2 in pairlist:
                Zs+=c1*c2*sector_sum(nu1+nu2,tab,N)
    return Zg,Zs

if __name__=="__main__":
    np.set_printoptions(precision=6,suppress=True)
    print("octahedron: V=6 E=%d F=%d; |det Gamma| = %.6f"%(E,F,np.abs(np.linalg.det(Gm))))
    ev=np.linalg.eigvals(Gm)
    print("free Dirac spectrum at flat config (m=0): eigenvalues of Gamma:", np.round(ev,6))
    print("  |lambda|:", np.round(np.abs(ev),6))
    b=1.0
    Zg,Zs=Z_total_sectors(0.0,b,Nf_pair=1)
    print(f"\nbeta=1, m=0, |det|^2 weight:")
    print(f"  Z_g = {Zg:.6f}  (check sum I_n^8 = 6.622435)")
    tot=Zs.sum()
    print(f"  Z_ferm = {tot:.6f}   <|det|^2> = {tot/Zg:.6f}")
    print(f"  sectors frac: f0={Zs[0]/tot:.6f} f1={Zs[1]/tot:.6f} f2={Zs[2]/tot:.6f}")
    # pure gauge sectors:
    Zg2,Zs_pure=Z_total_sectors(0.0,b,Nf_pair=0) if False else (None,None)
    # (quick pure-gauge sector numbers from earlier run: f0=0.996857 f1=f2=0.001572)
    print("  PURE gauge sectors (known): f0=0.996857 f1=f2=0.001572")
