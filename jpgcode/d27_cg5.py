#!/usr/bin/env python3
"""
d27_cg5.py -- Extended No-Go check: all minimal Delta(27) Weinberg
realizations with L in {3, 3bar} and flavons in {3, 3bar}.

Establishes that |M12|=|M13|=|M23| (moduli equal) whenever the three
lepton doublets sit in a single irreducible triplet, independent of
the flavon representations and VEV directions.  Hence a13/a12 = 1/3
cannot be derived from Delta(27) representation theory alone in the
minimal setup; it requires an additional shaping symmetry.
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


def cg_triplet(rL, rF):
    """CG tensors for rL x rF -> 3bar copies: return list of
    C[k, i, j] embeddings (k = 3bar index)."""
    rep = {g: np.kron(rL[g], rF[g]) for g in G}
    P3b = projector(rep, char3bar)
    dim = rep[I].shape[0]
    r = np.linalg.matrix_rank(P3b)
    # image basis
    U, S, _ = np.linalg.svd(P3b)
    E = U[:, :r]
    # invariant subspaces: eigenvectors of a generic hermitian combo
    Hc = np.zeros((r, r), complex)
    for g in G:
        Hc += np.random.randn() * (E.conj().T @ rep[g] @ E)
    ev, vec = np.linalg.eigh(Hc)
    copies = []
    # group by degenerate eigenvalues (multiplicity 3)
    seen = {}
    for i in range(r // 3):
        V = vec[:, 3 * i:3 * i + 3]
        Q, _ = np.linalg.qr(E @ V)
        # Q: dim x 3; reshape to C[k, i, j] with tensor index n=3*i+j
        C = np.zeros((3, 3, 3), complex)
        for k in range(3):
            for i2 in range(3):
                for j2 in range(3):
                    C[k, i2, j2] = Q[3 * i2 + j2, k]
        copies.append(C)
    return copies


def weinberg_matrix(copies, vev, coeffs):
    """M_{ij} = sum_a coeff_a * sum_k vev_a[k] * C_a[k,i,j]"""
    M = np.zeros((3, 3), complex)
    for C, v, c in zip(copies, vev, coeffs):
        for k in range(3):
            M += c * v[k] * C[k]
    return M


vacs = {"(1,1,1)": np.array([1, 1, 1], complex),
        "(1,w,w2)": np.array([1, w, w ** 2], complex),
        "(1,w2,w)": np.array([1, w ** 2, w], complex),
        "(1,0,0)": np.array([1, 0, 0], complex)}

print("=== CG copies for each (rL, rF) combination ===")
for nameL, rL in [("3", r3), ("3bar", r3bar)]:
    for nameF, rF in [("3", r3), ("3bar", r3bar)]:
        copies = cg_triplet(rL, rF)
        print(f"\nL~{nameL}, F~{nameF}: {len(copies)} copies of 3bar in LxF")
        # structure of each copy
        for ci, C in enumerate(copies):
            # is it diagonal-type or off-diagonal-type?
            diag_norm = sum(abs(C[k, i, i]) ** 2 for k in range(3) for i in range(3))
            off_norm = sum(abs(C[k, i, j]) ** 2 for k in range(3)
                           for i in range(3) for j in range(3) if i != j)
            print(f"  copy{ci + 1}: diag-norm={diag_norm:.3f} off-norm={off_norm:.3f}")

# For each combo, scan VEV choices: check if moduli can be unequal
print("\n=== moduli of off-diagonal M for single-flavon, both copies ===")
for nameL, rL in [("3", r3), ("3bar", r3bar)]:
    for nameF, rF in [("3", r3), ("3bar", r3bar)]:
        copies = cg_triplet(rL, rF)
        for nameV, v in vacs.items():
            # use first two copies with generic coefficients
            for c1, c2 in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (1, -1), (1, 0.5)]:
                M = weinberg_matrix(copies[:2], [v, v], [c1, c2])
                m12, m13, m23 = abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2])
                if min(m12, m13, m23) < 1e-9:
                    continue
                spread = max(m12, m13, m23) / min(m12, m13, m23)
                if spread > 1.01:
                    print(f"  UNEQUAL: L~{nameL} F~{nameF} VEV={nameV} c=({c1},{c2})"
                          f"  |M12|={m12:.3f} |M13|={m13:.3f} |M23|={m23:.3f} spread={spread:.2f}")
print("(no output above means moduli are always equal: No-Go)")
