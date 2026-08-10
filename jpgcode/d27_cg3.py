#!/usr/bin/env python3
"""
d27_cg3.py -- Correct CG analysis of Delta(27): 3x3 = 3bar+3bar+3bar.

Uses orthonormal symmetric/antisymmetric bases of C^3 x C^3.
The antisymmetric part (3-dim) is the epsilon copy (one 3bar).
The symmetric part (6-dim) contains two 3bar copies; we extract both
CG tensors and study the resulting neutrino mass matrix structure.
"""
import numpy as np

w = np.exp(2j * np.pi / 3)
I = ((0, 1, 2), (0, 0, 0))


def mul(g, h):
    p, phi = g
    q, psi = h
    return (tuple(q[p[i]] for i in range(3)),
            tuple((phi[i] + psi[p[i]]) % 3 for i in range(3)))


def to_mat(g):
    p, ph = g
    M = np.zeros((3, 3), complex)
    for i in range(3):
        M[i, p[i]] = np.exp(2j * np.pi * ph[i] / 3)
    return M


A = ((0, 1, 2), (0, 1, 2))
B = ((1, 2, 0), (0, 0, 0))


def make_group(gens):
    G = [I] + [g for g in gens if g != I]
    changed = True
    while changed:
        changed = False
        for g in list(G):
            for h in list(G):
                p = mul(g, h)
                if p not in G:
                    G.append(p)
                    changed = True
    return G


G = make_group([A, B])
r3 = {g: to_mat(g) for g in G}
r3bar = {g: to_mat(g).conj() for g in G}


def char3(g):
    return np.trace(r3[g])


def char3bar(g):
    return np.trace(r3bar[g])


def projector(rep, target_char):
    d = int(round(target_char(I).real))
    P = np.zeros_like(next(iter(rep.values())), dtype=complex)
    for g in G:
        P += np.conj(target_char(g)) * rep[g]
    P *= d / len(G)
    return P


def tensor9_rep():
    return {g: np.kron(r3[g], r3[g]) for g in G}


# ---- orthonormal bases of sym (6) and antisym (3) subspaces ----
S = np.zeros((9, 6), complex)      # sym basis (columns)
A_ = np.zeros((9, 3), complex)     # antisym basis (columns)
# tensor index n = 3*i + j  (e_i x e_j)
diag = [(0, 0), (1, 1), (2, 2)]
off = [(0, 1), (0, 2), (1, 2)]
for c, (i, j) in enumerate(diag):
    S[3 * i + j, c] = 1.0
for c, (i, j) in enumerate(off):
    S[3 * i + j, 3 + c] = 1.0 / np.sqrt(2)
    S[3 * j + i, 3 + c] = 1.0 / np.sqrt(2)
for c, (i, j) in enumerate(off):
    A_[3 * i + j, c] = 1.0 / np.sqrt(2)
    A_[3 * j + i, c] = -1.0 / np.sqrt(2)


def sub_rep(rep9, basis):
    """Representation matrices restricted to a subspace."""
    out = {}
    for g in G:
        out[g] = basis.conj().T @ rep9[g] @ basis
    return out


rep9 = tensor9_rep()
symR = sub_rep(rep9, S)
asymR = sub_rep(rep9, A_)

# check characters
chi_sym = np.trace(symR[I])
print("dim sym =", chi_sym, " dim antisym =", np.trace(asymR[I]))

# verify antisym is a single 3bar: character
from collections import Counter
chi_as = np.array([np.trace(asymR[g]) for g in G])
ip = sum(chi_as * np.array([np.conj(char3bar(g)) for g in G])) / 27
print("antisym x 3bar inner product =", round(ip.real, 3), "(1 -> single 3bar)")

# symmetric part: 3bar multiplicity
chi_sy = np.array([np.trace(symR[g]) for g in G])
ip2 = sum(chi_sy * np.array([np.conj(char3bar(g)) for g in G])) / 27
print("sym x 3bar inner product =", round(ip2.real, 3), "(2 -> two 3bar copies)")

# ---- separate the two 3bar copies in the sym space ----
# We need a Hermitian operator that distinguishes them.  The identity
# operator on the 3bar-isotypic component is 6-dim; the two copies are
# NOT separated by group elements.  We use the projector P3b and then
# find a basis in which the "standard" 3bar embedding (defined by the
# first copy, e.g. the one containing e11) is block-diagonal.
P3b = projector(symR, char3bar)
print("\nrank P3b (sym) =", np.linalg.matrix_rank(P3b))
# P3b acts as identity on all 6 dims (two copies).  Choose the copy
# containing the vector e11 (the (1,1) direction).  Its orbit under G
# spans the first copy.  Project e11 onto the 3bar-isotypic space
# (it already is) and take its orbit.
e11 = S[:, 0]
# orbit of e11 under the symmetric representation
orbit_vectors = []
for g in G:
    orbit_vectors.append(symR[g] @ (S.conj().T @ e11))
O = np.array(orbit_vectors).T        # 6 x 27
Uo, So, _ = np.linalg.svd(O)
print("singular values of e11 orbit:", np.round(So[:6], 4))
V1 = Uo[:, :3]                        # 3-dim subspace: copy 1
# copy 2 = orthogonal complement within the isotypic space (all of sym)
V2 = Uo[:, 3:6]
print("copy1 3-dim? ", V1.shape, " copy2:", V2.shape)

# verify each is invariant
def is_invariant(V):
    for g in G:
        if not np.allclose(symR[g] @ V, V @ (V.conj().T @ symR[g] @ V)):
            return False
    return True

print("copy1 invariant:", is_invariant(V1), " copy2 invariant:", is_invariant(V2))

# ---- CG tensors: C[k, i, j] from V (6 x 3), k = 3bar index ----
def cg_from_V(V):
    """V: 6x3 in sym-basis; return C[k,i,j] with C = V (columns k)."""
    C = np.zeros((3, 3, 3), complex)
    # reconstruct 9-dim embedding: S (9x6) @ V (6x3) = 9x3
    E = S @ V
    for k in range(3):
        for i in range(3):
            for j in range(3):
                C[k, i, j] = E[3 * i + j, k]
    return C


C1 = cg_from_V(V1)
C2 = cg_from_V(V2)
for name, C in [("copy1", C1), ("copy2", C2)]:
    print(f"\n=== CG {name}: C^k_{{ij}} ===")
    for k in range(3):
        print(f"  k={k}:")
        print(np.round(C[k], 4))

# ---- neutrino mass matrix from Weinberg with one flavon phi ~ 3 ----
# M_{ij} = <phi>_k ( a C1^k_{ij} + b C2^k_{ij} )
# phi VEV aligned along (1, w, w^2) or (1, w^2, w).
print("\n=== Weinberg matrix with <phi> = v(1,w,w^2), M = a*C1 + b*C2 ===")
vphi = np.array([1, w, w ** 2])
for (a, b, lbl) in [(1, 0, "a=1,b=0 (only copy1)"),
                    (0, 1, "a=0,b=1 (only copy2)"),
                    (1, 1, "a=1,b=1"),
                    (1, -1, "a=1,b=-1"),
                    (1, 0.5, "a=1,b=0.5")]:
    M = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            M[i, j] = np.sum(vphi * (a * C1[:, i, j] + b * C2[:, i, j]))
    print(f"\n  {lbl}")
    print(np.round(M, 4))
    print("  M12/M13 =", np.round(M[0, 1] / M[0, 2], 4) if abs(M[0, 2]) > 1e-9 else "inf")
    print("  M23/M13 =", np.round(M[1, 2] / M[0, 2], 4) if abs(M[0, 2]) > 1e-9 else "inf")

np.savez('/tmp/cg_full.npz', C1=C1, C2=C2)
