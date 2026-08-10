#!/usr/bin/env python3
"""
d27_seesaw.py -- Check whether Type-I seesaw with Delta(27) can
produce a13/a12 = 1/3 (the one remaining minimal possibility).

Setup: L ~ 3, N_R ~ 3bar, H ~ 1.
  m_D = y <H> * (singlet channel of 3 x 3bar)   -> proportional to 1_3
  M_N = y_N <phi> * (CG of 3bar x 3bar -> 3)    -> rank-1 structure
  m_nu = - m_D M_N^-1 m_D^T
Also tests: N_R ~ 3 (instead of 3bar), two N flavors, and flavons in
both 3 and 3bar.
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


def singlet_channel(rA, rB):
    """Embedding of the singlet in rA x rB (as a matrix acting on the
    tensor product).  Returns the 1-dim invariant direction."""
    rep = {g: np.kron(rA[g], rB[g]) for g in G}
    P1 = projector(rep, lambda g: 1.0)
    # P1 is rank 1: take its eigenvector
    H = P1.conj().T @ P1
    ev, evc = np.linalg.eigh(H)
    return evc[:, -1]


def cg_copies(rA, rB, target_char):
    """All embedding copies of target_char inside rA x rB.
    Returns list of C[k, i, j] with C = 3x3 matrices indexed k."""
    rep = {g: np.kron(rA[g], rB[g]) for g in G}
    P = projector(rep, target_char)
    r = np.linalg.matrix_rank(P)
    U, S, _ = np.linalg.svd(P)
    E = U[:, :r]
    Hc = np.zeros((r, r), complex)
    for g in G:
        Hc += np.random.randn() * (E.conj().T @ rep[g] @ E)
    ev, vec = np.linalg.eigh(Hc)
    copies = []
    for i in range(r // 3):
        V = vec[:, 3 * i:3 * i + 3]
        Q, _ = np.linalg.qr(E @ V)
        C = np.zeros((3, 3, 3), complex)
        for k in range(3):
            for i2 in range(3):
                for j2 in range(3):
                    C[k, i2, j2] = Q[3 * i2 + j2, k]
        copies.append(C)
    return copies


# ---- 1. m_D: singlet channel of 3 x 3bar ----
s = singlet_channel(r3, r3bar)
mD = s.reshape(3, 3)
print("=== m_D (singlet of 3 x 3bar) ===")
print(np.round(mD, 4))
print("-> m_D proportional to identity?",
      np.allclose(mD / mD[0, 0], np.eye(3)))

# ---- 2. M_N: 3bar x 3bar -> 3 (for N ~ 3bar) ----
copies_N = cg_copies(r3bar, r3bar, char3)
print(f"\n=== M_N channel: 3bar x 3bar -> 3: {len(copies_N)} copies ===")

vacs = {"(1,1,1)": np.array([1, 1, 1], complex),
        "(1,w,w2)": np.array([1, w, w ** 2], complex),
        "(1,w2,w)": np.array([1, w ** 2, w], complex),
        "(1,0,0)": np.array([1, 0, 0], complex)}

print(f"\n{'VEV':<10}{'copy':>5} {'ratio13/12 of m_nu':>22} {'|m12|':>8}{'|m13|':>8}{'|m23|':>8}")
for vname, v in vacs.items():
    for ci, C in enumerate(copies_N):
        MN = np.zeros((3, 3), complex)
        for k in range(3):
            MN += v[k] * C[k]
        if abs(np.linalg.det(MN)) < 1e-9:
            continue
        mn = -mD @ np.linalg.inv(MN) @ mD.T
        m12, m13, m23 = abs(mn[0, 1]), abs(mn[0, 2]), abs(mn[1, 2])
        if m12 < 1e-9:
            continue
        print(f"{vname:<10}{ci + 1:>5} {m13 / m12:22.4f} {m12:8.4f}{m13:8.4f}{m23:8.4f}")

# ---- 3. N ~ 3 (M_N: 3 x 3 -> 3bar) ----
print("\n=== N ~ 3 variant (3 x 3 -> 3bar for M_N) ===")
copies_N3 = cg_copies(r3, r3, char3bar)
for vname, v in vacs.items():
    for ci, C in enumerate(copies_N3):
        MN = np.zeros((3, 3), complex)
        for k in range(3):
            MN += v[k] * C[k]
        if abs(np.linalg.det(MN)) < 1e-9:
            continue
        mn = -mD @ np.linalg.inv(MN) @ mD.T
        m12, m13, m23 = abs(mn[0, 1]), abs(mn[0, 2]), abs(mn[1, 2])
        if m12 < 1e-9:
            continue
        print(f"{vname:<10}{ci + 1:>5} {m13 / m12:22.4f} {m12:8.4f}{m13:8.4f}{m23:8.4f}")
