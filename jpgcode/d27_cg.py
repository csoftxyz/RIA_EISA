#!/usr/bin/env python3
"""
d27_cg.py -- Clebsch-Gordan coefficients of Delta(27) in the
monomial basis A=diag(1,w,w^2), B=cyclic shift.

Computes, by explicit projection:
  3 x 3bar = sum of the nine singlets (no 8-dim irrep exists)
  3 x 3    = 3bar + 3bar + 3bar
"""
import numpy as np
from itertools import permutations, product

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
print(f"|Delta(27)| = {len(G)}")

r3 = {g: to_mat(g) for g in G}
r3bar = {g: to_mat(g).conj() for g in G}


def tensor_reps(rA, rB):
    return {g: np.kron(rA[g], rB[g]) for g in G}


def projector(rep, target_char):
    d = int(round(target_char(I).real))
    P = np.zeros_like(next(iter(rep.values())), dtype=complex)
    for g in G:
        P += np.conj(target_char(g)) * rep[g]
    P *= d / len(G)
    return P


def char3(g):
    return np.trace(r3[g])


def char3bar(g):
    return np.trace(r3bar[g])


def char1(g):
    return 1.0 + 0j


def show_basis(basis, label):
    print(f"\n=== {label} ===")
    for j in range(basis.shape[1]):
        v = basis[:, j]
        nz = np.where(np.abs(v) > 1e-6)[0]
        s = " + ".join(
            f"({v[i]:+.4f}{v[i].imag:+.4f}i)" f" e{i}"
            for i in nz)
        print(f"  v{j}: {s}")


# ---------- 3 x 3bar = nine singlets ----------
rep33b = tensor_reps(r3, r3bar)
Q1, _ = projector(rep33b, char1), None
# singlet projector
P1 = projector(rep33b, char1)
H1 = P1.conj().T @ P1
ev, evc = np.linalg.eigh(H1)
v1 = evc[:, -1]
print("\n=== 3 x 3bar singlet (normalized) ===")
print("  v =", np.round(v1, 5))

# 8 projector
def char8(g):
    return abs(char3(g)) ** 2 - 1
P8 = projector(rep33b, char8)
H8 = P8.conj().T @ P8
ev8, evc8 = np.linalg.eigh(H8)
Q8 = evc8[:, -8:]
show_basis(Q8, "3 x 3bar -> 8 (basis)")

# ---------- 3 x 3 = 3bar + 3bar + 3bar ----------
rep33 = tensor_reps(r3, r3)
P3b = projector(rep33, char3bar)
print(f"\n=== 3 x 3 -> 3bar: rank of projector = {np.linalg.matrix_rank(P3b)} (expect 9) ===")
# The isotypic space is all of C^9. Need to separate into 3 copies.
# Strategy: the three copies are the three columns of the 'embedding'
# characterized by how A acts. In the 3bar rep, A = diag(1,w^2,w)
# (conjugate of diag(1,w,w^2)).  In 3x3, A acts as kron(A,A).
# The 9-dim space splits under A into eigenspaces with eigenvalues
# a_i a_j = w^{i+j} (i,j = 0,1,2 exponents of A's eigenvalues 1,w,w^2)
# each of which appears with multiplicity 3 (the three columns j for
# fixed i+j mod 3).  The 3bar copy must have A-eigenvalues (1,w^2,w),
# i.e. exponents (0,2,1) mod 3. So the copy with exponent pattern
# {0,2,1} is picked by A; B mixes the copies.
print("Eigen-decomposition of kron(A,A):")
AA = np.kron(r3[A], r3[A])
evAA, evcAA = np.linalg.eig(AA)
# group by eigenvalue
groups = {}
for i, e in enumerate(evAA):
    key = round(e.real, 6), round(e.imag, 6)
    groups.setdefault(key, []).append(i)
for key, inds in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    print(f"  eigenvalue {np.round(complex(*key),4)}: multiplicity {len(inds)}")


# =====================================================================
# 3 x 3 -> 3bar + 3bar + 3bar : explicit CG construction
# =====================================================================
# Strategy: the three 3bar copies are the invariant subspaces of the
# 9-dim space.  Since P3bar = identity on all of C^9 (rank 9), we
# separate the copies using the cyclic generator B.  In the 3bar rep,
# B (a real permutation) has eigenvalues {1, w, w^2} in some order.
# In the tensor rep, the copies are distinguished by the eigenvalue
# of the 'charge' operator Q = log_A / (2pi i/3) i.e. the A-exponent
# pattern.  We build the three copies explicitly from the invariant
# cubic tensors of Delta(27).

# Known Delta(27) invariant tensors:
#   (i)   delta_{ij} delta_{kl} type singlet of 3 x 3bar   -> v1 above
#   (ii)  the fully antisymmetric epsilon_{ijk} (3x3x3 -> 1)
#   (iii) the 'cubic' contraction: (phi phi phi) singlet with
#         coefficients c_k = sum_i eps ... -- need to find.
# The 3x3 -> 3bar CG's are obtained from the invariant
#   T_{ijk} (phi_i phi_j) psi_k   with psi ~ 3bar, T invariant under G.
# Equivalently, the embedding (3bar -> 3 x 3) is given by a tensor
# C^k_{ij} with C^k = (C^k)^T (symmetric part) and C^k antisymmetric
# parts, such that sum_{ij} C^k_{ij} phi_i phi_j transforms as 3bar_k.

rep33 = tensor_reps(r3, r3)

def find_invariant_rank3():
    """Find all invariant tensors T_{ijk} (i,j,k in {0,1,2}) with
    T_{ijk} (phi_i phi_j) psi_k invariant, i.e. T transforms as
    (3 x 3 x 3bar) -> 1.  Solve for the null space of
    sum over constraints T = r3(g)_{ii'} r3(g)_{jj'} r3bar(g)_{kk'} T."""
    # 27 unknowns; constraints from 27 group elements (redundant)
    M = []
    for g in G:
        R = r3[g]
        Rb = r3bar[g]
        row = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    row.append(R[i,0]*R[j,1]*Rb[k,2])  # placeholder
        # correct: constraint T_{ijk} = sum_{i'j'k'} R_{ii'} R_{jj'} Rbar_{kk'} T_{i'j'k'}
        M.append([R[i,0]*R[j,1]*Rb[k,2] for i in range(3) for j in range(3) for k in range(3)])
    # build actual linear system: T - rho(g) T = 0
    A = []
    for g in G:
        R = r3[g]
        Rb = r3bar[g]
        rho = np.einsum('ia,jb,kc->ijkabc', R, R, Rb).reshape(27, 27)
        A.append(np.eye(27) - rho)
    A = np.concatenate(A)
    U, S, Vt = np.linalg.svd(A)
    null = Vt.T[:, -1]  # one null vector
    return null

null = find_invariant_rank3()
T = null.reshape(3, 3, 3)
print("\n=== invariant T_{ijk} (3x3x3bar -> 1) ===")
print(np.round(T, 4))
