#!/usr/bin/env python3
"""
d27_cg2.py -- Complete Clebsch-Gordan analysis of Delta(27).

Goal: extract the CG tensors that enter the neutrino mass matrix in a
model with L ~ 3, H ~ 1, one flavon phi ~ 3 (or two flavons).

Key results to establish:
  * 3 x 3 = 3bar + 3bar + 3bar;  the antisymmetric 3-dim part is
    contracted with the invariant epsilon tensor (epsilon_{ijk}).
  * The symmetric 6-dim part splits into TWO 3bar copies; we extract
    their CG tensors C1^k_{ij}, C2^k_{ij} (k = 3bar index, i,j sym).
  * Then, with <phi>_k = v (1, w, w^2), the Weinberg matrix is
        M_{ij} = v_k [ a C1^k_{ij} + b C2^k_{ij} ]
    and we read off M12 : M13 : M23 and test whether M13/M12 = 1/3
    can arise from the CG structure alone.
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


def sym_basis():
    """Basis of the 6-dim symmetric tensor space (3x3 sym)."""
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    # embedding E[i,j] -> basis index
    idx = {}
    for n, (i, j) in enumerate(pairs):
        idx[(i, j)] = n
        idx[(j, i)] = n
    return pairs, idx


def sym_rep():
    """6-dim symmetric representation matrices (induced from r3 x r3)."""
    pairs, idx = sym_basis()
    R = {}
    for g in G:
        M = np.zeros((6, 6), complex)
        r = r3[g]
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        M[idx[(a, b)], idx[(c, d)]] += r[a, c] * r[b, d]
        R[g] = M
    return R, pairs


# ---- decompose symmetric 6-dim rep into 3bar copies ----
symR, pairs = sym_rep()
P3b = projector(symR, char3bar)
print("rank of 3bar projector on sym space:", np.linalg.matrix_rank(P3b),
      "(expect 6: two copies of 3bar)")

# Diagonalize P3b to find the two 3-dim invariant subspaces.
# P3b is a projection (P^2 = P), eigenvalues 1 (x6) and 0 (x0).
# The two copies are separated by diagonalizing a commuting operator:
# use rho(B) restricted to the image.
Im = P3b @ np.eye(6)
U, S, _ = np.linalg.svd(P3b)
E = U[:, :6]                     # basis of image (6-dim)
# matrix of B in this basis
MB = E.conj().T @ symR[B] @ E
evB, vecB = np.linalg.eig(MB)
print("eigenvalues of B on sym-3bar space:", np.round(evB, 4))
# B on 3bar has eigenvalues {1, w^2, w}? compute:
print("eigenvalues of r3bar[B]:", np.round(np.linalg.eigvals(r3bar[B]), 4))
# group eigenvectors of MB by distinct eigenvalues (each 3-fold)
groups = {}
for i, e in enumerate(evB):
    key = (round(e.real, 6), round(e.imag, 6))
    groups.setdefault(key, []).append(i)
print("multiplicities:", {k: len(v) for k, v in groups.items()})

copies = []
for key, inds in groups.items():
    if len(inds) != 3:
        continue
    V = vecB[:, inds]
    # orthonormalize columns
    Q, _ = np.linalg.qr(V)
    copies.append(E @ Q)
print(f"found {len(copies)} invariant 3-dim copies")

# ---- extract CG tensors C^k_{ij} from each copy ----
def copy_to_cg(Ecopy):
    """Ecopy: 6 x 3 matrix; column k = CG coefficients of 3bar_k in
    the symmetric basis.  Return C[k, i, j]."""
    pairs, idx = sym_basis()
    C = np.zeros((3, 3, 3), complex)
    for k in range(3):
        for (i, j), n in idx.items():
            C[k, i, j] = Ecopy[n, k]
    return C


print("\n=== CG tensors of the two symmetric 3bar copies ===")
for ci, Ec in enumerate(copies):
    C = copy_to_cg(Ec)
    print(f"\n--- copy {ci+1}: C^k_{{ij}} (k=row, ij=matrix) ---")
    for k in range(3):
        print(f"  k={k}:")
        print(np.round(C[k], 4))
    np.savez(f'/tmp/cg_copy{ci}.npz', C=C)

# ---- antisymmetric 3bar: epsilon ----
eps = np.zeros((3, 3, 3), complex)
for i in range(3):
    for j in range(3):
        for k in range(3):
            eps[i, j, k] = ((1.0 if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                             else (-1.0 if (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)] else 0.0)))
print("\n=== antisymmetric contraction (epsilon) ===")
print("epsilon_{ijk} (k=row):")
for k in range(3):
    print(np.round(eps[k], 4))
