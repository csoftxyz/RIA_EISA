#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
algebra_census.py - u(4)-invariant tensor census for g0 (+) g1 (+) g2
with g0 = u(4), g1 = 4, g2 = 4bar (parabolic grading of sl(5)).
Counts are dimensions of invariant tensors (numerical null spaces).
The mixed bracket [g1,g2] is treated separately in verify_jacobi.py.
"""
import numpy as np
from itertools import permutations

def su4_gens():
    L=[]
    for a in range(4):
        for b in range(a+1,4):
            M=np.zeros((4,4),complex); M[a,b]=M[b,a]=1; L.append(M)
    for a in range(4):
        for b in range(a+1,4):
            M=np.zeros((4,4),complex); M[a,b]=1j; M[b,a]=-1j; L.append(M)
    for k in range(1,4):
        M=np.zeros((4,4),complex)
        for i in range(k): M[i,i]=1
        M[k,k]=-k; M*=np.sqrt(2/(k*(k+1))); L.append(M)
    return L
GEN15 = su4_gens()
GEN16 = GEN15+[np.eye(4,dtype=complex)]
def antirep(X): return -np.conj(X)
def null_dim(rows,tol=1e-7):
    A=np.concatenate(rows); u,s,vh=np.linalg.svd(A)
    return len(vh[s<tol*max(s.max(),1e-30)])

print("="*74); print("A. INVARIANT TENSOR CENSUS   (u(4)-invariant dims; Z3-charge parity in brackets)")
print("="*74)
# ---- A1: two-leg bilinear / sesquilinear ----
def two_legs(repA,repB,label,conjA=False):
    rb=[]; rh=[]
    for X in GEN16:
        XA=repA(X); XB=repB(X)
        rb.append(np.kron(np.eye(4),XA.T)+np.kron(XB.T,np.eye(4)))
        rh.append(np.kron(np.eye(4),np.conj(XA.T))-np.kron(XB.T,np.eye(4)))
    print(f"  {label:34s} bilinear={null_dim(rb)}  Hermitian={null_dim(rh)}")
two_legs(lambda X:X,       lambda X:X,       "g1(x)g1  [4(x)4,  grade 2]")
two_legs(antirep,          antirep,          "g2(x)g2  [4b(x)4b, grade 1]")
two_legs(lambda X:X,       antirep,          "g1(x)g2  [4(x)4b, grade 0]")
# ---- A2: three-leg trilinear (charge-zero products only) ----
def three_legs(reps,label):
    d0,d1,d2=4,4,4
    rows=[]
    for X in GEN16:
        X0=np.array(reps[0](X)); X1=np.array(reps[1](X)); X2=np.array(reps[2](X))
        A=np.kron(np.eye(d2),np.kron(np.eye(d1),X0.T))
        B=np.kron(np.eye(d2),np.kron(X1.T,np.eye(d0)))
        C=np.kron(X2.T,np.kron(np.eye(d1),np.eye(d0)))
        rows.append(A+B+C)
    print(f"  {label:34s} u(4)-invariant trilinear dim = {null_dim(rows)}")
three_legs([lambda X:X,      lambda X:X,      lambda X:X],    "g1(x)g1(x)g1 [4.4.4,  grade 0]")
three_legs([lambda X:X,      lambda X:X,      antirep],       "g1(x)g1(x)g2 [4.4.4b, grade 1]")
three_legs([lambda X:X,      antirep,         antirep],       "g1(x)g2(x)g2 [4.4b.4b,grade 2]")
three_legs([antirep,         antirep,         antirep],       "g2(x)g2(x)g2 [4b.4b.4b,gr 0]")
# ---- A3: bracket targets ----
pairs=[(a,b) for a in range(4) for b in range(a+1,4)]
def rep_lam2(X,bar=False):
    d=len(pairs); M=np.zeros((d,d),complex)
    Y = antirep(X) if bar else X
    for i,(a,b) in enumerate(pairs):
        for j,(c,d2) in enumerate(pairs):
            M[i,j]=Y[a,c]*Y[b,d2]-Y[a,d2]*Y[b,c]
    return M
n1=null_dim([np.kron(np.eye(4),rep_lam2(X).T)+np.kron(antirep(X).T,np.eye(6)) for X in GEN16])
n2=null_dim([np.kron(np.eye(4),rep_lam2(X,True).T)+np.kron(np.array(X).T,np.eye(6)) for X in GEN16])
print(f"  Hom(Lambda^2 g1 -> g2)              = {n1}   => [F,F] = 0 FORCED")
print(f"  Hom(Lambda^2 g2 -> g1)              = {n2}   => [z,z] = 0 FORCED")
print("  Hermitian mass forms: g1: 1 (above); unique up to real scale.")
