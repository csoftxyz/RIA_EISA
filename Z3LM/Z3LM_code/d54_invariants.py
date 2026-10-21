#!/usr/bin/env python3
"""Delta(27) / Delta(54) invariant-operator census for the neutrino texture.

Question (route A'): can the modulus ratios of the Z3LM texture be *fixed*
by the CG structure of Delta(54), i.e. without tuning continuous couplings?

Method: build the group as 3x3 monomial matrices, then solve for the space of
invariant tensors T with

    (L L) ~ 3 (x) 3  contracted with flavon(s) in 3 or 3bar,

i.e. T_{ij,k} invariant under rho(g) (x) rho(g) (x) conj(rho(g)).
dim of that space = number of independent operators = number of free
couplings in the model.  Same for the two-flavon case T_{ij,kl}.

If dim >= 2 the moduli are linear combinations of flavon components with
free coefficients, so no ratio is predicted: the "derivation" would only
relabel five tuned ratios as five tuned couplings/VEVs.

Run: python3 d54_invariants.py
"""
import itertools
import numpy as np
import sympy as sp

w = sp.exp(2 * sp.pi * sp.I / 3)

# ---------------------------------------------------------------- generators
a = sp.diag(1, w, w ** 2)
b = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
c = w * sp.eye(3)


def closure(gens, limit=200):
    elems = [sp.eye(3)]
    frontier = [sp.eye(3)]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = sp.simplify(x * g)
                if not any(sp.simplify(y - e).is_zero_matrix for e in elems):
                    new.append(y)
                    elems.append(y)
        frontier = new
        if len(elems) > limit:
            break
    return elems


def inv_dim(elems, n=3, conj_last=False, symmetric=False):
    """dim of invariant tensors with n indices (holomorphic by default)."""
    idx = list(itertools.product(range(3), repeat=n))
    pos = {k: i for i, k in enumerate(idx)}
    N = len(idx)
    rows = []
    for g in elems:
        r = np.array(g.tolist(), dtype=complex)
        rc = np.conj(r)
        M = np.zeros((N, N), dtype=complex)
        for k in idx:
            for kp in idx:
                val = 1.0
                for a_i in range(n - 1):
                    val *= r[k[a_i], kp[a_i]]
                val *= (rc if conj_last else r)[k[-1], kp[-1]]
                if abs(val) > 1e-12:
                    M[pos[k], pos[kp]] += val
        M -= np.eye(N)
        rows.append(M.real)
        rows.append(M.imag)
    A = np.vstack([r for r in rows if np.abs(r).max() > 1e-12])
    u, s, vh = np.linalg.svd(A)
    rank = int((s > 1e-8).sum())
    return N - rank


G27 = closure([a, b, c])
print(f"|Delta(27)| = {len(G27)}")

# Delta(54): add the Z2 element that sends (x1,x2,x3) -> (x1, x3, x2)-like
d = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
G54 = closure([a, b, c, d])
print(f"|Delta(54)| = {len(G54)}")

for name, G in (("Delta(27)", G27), ("Delta(54)", G54)):
    print(f"\n{name}:")
    print(f"  dim 3(x)3(x)3    (LL phi, phi in 3)   : {inv_dim(G, 3)}")
    print(f"  dim 3(x)3(x)3bar (LL phi^dag)         : {inv_dim(G, 3, conj_last=True)}")
    print(f"  dim 3(x)3bar(x)3 (L Lbar phi)          : {inv_dim(G, 3, conj_last=True)}")

# explicit one-flavon textures: M_ij(v) = sum_k T_ijk v_k
print("\nExplicit invariant tensors for Delta(27) (one flavon):")
idx = list(itertools.product(range(3), repeat=3))
pos = {k: i for i, k in enumerate(idx)}
N = len(idx)
rows = []
for g in G27:
    r = np.array(g.tolist(), dtype=complex)
    M = np.zeros((N, N), dtype=complex)
    for k in idx:
        for kp in idx:
            val = r[k[0], kp[0]] * r[k[1], kp[1]] * r[k[2], kp[2]]
            if abs(val) > 1e-12:
                M[pos[k], pos[kp]] += val
    M -= np.eye(N)
    rows += [M.real, M.imag]
A = np.vstack([x for x in rows if np.abs(x).max() > 1e-12])
u, s, vh = np.linalg.svd(A)
null = vh[int((s > 1e-8).sum()):]
print(f"  null-space dimension = {len(null)}")
for n_i, vec in enumerate(null):
    T = vec.reshape((3, 3, 3))
    print(f"  basis {n_i}:")
    for i in range(3):
        print("   ", np.round(T[i], 4))
