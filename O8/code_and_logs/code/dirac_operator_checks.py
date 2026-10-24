#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dirac_operator_checks.py

Rigour checks for the minimal Dirac operator

    D_{ij} = m delta_ij + t Gamma_ij U_ij ,   U_ij = exp(i theta_ij),
    theta_ji = -theta_ij ,

with Gamma the unique Z3(T)-equivariant antisymmetric orientation of K_{2,2,2}
(edges directed along x -> y -> z -> x).

Checks
  1. Gamma is real antisymmetric; (Gamma U)^dagger = -Gamma U for every U(1)
     configuration.  Hence D is anti-Hermitian off the mass term and
     det(m + D)^* = det(m - D).
  2. det(m + D) is complex for generic U and m > 0, so a real positive weight
     requires the conjugate pair |det(m + D)|^2.
  3. No Z2 grading: Gamma anticommutes with no diagonal sign matrix, and the
     graph is non-bipartite.  (A grading would give a two-colouring.)
  4. At U = 1: rank Gamma = 2, four zero modes, det(m + Gamma) = m^4(m^2+12).
  5. For random U the Dirac spectrum has a gap: no exact zero modes away from
     the flat connection.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "logs", "dirac_operator_checks.json")

verts = [(a, s) for a in range(3) for s in (1, -1)]
vid = {v: i for i, v in enumerate(verts)}
V = 6
edges = []
for i, u in enumerate(verts):
    for j, v in enumerate(verts):
        if j > i and u[0] != v[0]:
            edges.append((i, j))

G = np.zeros((V, V))
for i, u in enumerate(verts):
    for j, v in enumerate(verts):
        if j > i and u[0] != v[0]:
            s = 1 if (u[0] + 1) % 3 == v[0] else -1
            G[i, j] = s
            G[j, i] = -s


def random_U(rng):
    theta = np.zeros((V, V))
    for i, j in edges:
        t = rng.uniform(0, 2 * np.pi)
        theta[i, j] = t
        theta[j, i] = -t
    return np.exp(1j * theta)


def is_bipartite():
    adj = [[] for _ in range(V)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    color = [-1] * V
    color[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if color[w] < 0:
                color[w] = 1 - color[u]
                stack.append(w)
            elif color[w] == color[u]:
                return False
    return True


def has_diagonal_grading():
    for mask in range(1 << V):
        x = np.array([1 if (mask >> k) & 1 else -1 for k in range(V)], float)
        if np.allclose(x[:, None] * G + G * x[None, :], 0):
            return True, x
    return False, None


def main():
    res = {}
    rng = np.random.default_rng(7)
    print("== 1. anti-Hermiticity and conjugation ==")
    res["Gamma_antisymmetric"] = bool(np.allclose(G.T, -G))
    ok_anti = True
    ok_conj = True
    complex_det = True
    for _ in range(200):
        U = random_U(rng)
        M = G * U
        ok_anti &= np.allclose(M.conj().T, -M)
        for m in (0.3, 1.0):
            D = m * np.eye(V) + M
            ok_conj &= np.allclose(np.linalg.det(D).conjugate(),
                                   np.linalg.det(m * np.eye(V) - M))
            complex_det &= abs(np.linalg.det(D).imag) > 1e-8
        W = (np.linalg.det(m * np.eye(V) + M)
             * np.linalg.det(m * np.eye(V) - M))
        assert W.real >= -1e-10
    print("  (Gamma U)^dagger = -Gamma U for 200 random U:", ok_anti)
    print("  det(m+D)^* = det(m-D):", ok_conj)
    print("  det(m+D) complex for generic U:", complex_det)
    res.update(antihermitian=bool(ok_anti), conjugation=bool(ok_conj),
               det_complex=bool(complex_det))

    print("\n== 1b. massless operator D0 = Gamma o U ==")
    real0 = True
    nz = 0
    for _ in range(200):
        U = random_U(rng)
        D0 = G * U
        real0 &= abs(np.linalg.det(D0).imag) < 1e-10
        nz += int(abs(np.linalg.det(D0)) > 1e-8)
    print("  det D0 real for 200 random U:", real0)
    print("  det D0 nonzero for", nz, "/ 200 random U")
    print("  det D0 at the flat connection:", np.linalg.det(G))
    res.update(detD0_real=bool(real0), detD0_nonzero=nz)

    print("\n== 2. no Z2 grading ==")
    has_g, x = has_diagonal_grading()
    bip = is_bipartite()
    print("  diagonal sign matrix anticommuting with Gamma:", has_g)
    print("  graph bipartite:", bip)
    res.update(diagonal_grading=bool(has_g), bipartite=bool(bip))

    print("\n== 3. flat connection ==")
    ev = np.linalg.eigvals(G)
    sv = np.linalg.svd(G, compute_uv=False)
    print("  singular values:", np.round(sv, 12))
    print("  rank:", np.linalg.matrix_rank(G))
    print("  det(Gamma):", round(float(np.linalg.det(G)), 12))
    det_func = {}
    for m in (0.0, 0.3, 1.0, 2.0):
        d = np.linalg.det(m * np.eye(V) + G)
        pred = m ** 4 * (m * m + 12)
        det_func[str(m)] = [float(d), float(pred)]
        print(f"  det({m} 1 + Gamma) = {d:.12f}   m^4(m^2+12) = {pred:.12f}")
    res.update(rank=int(np.linalg.matrix_rank(G)), singular=[float(s) for s in sv],
               det=float(round(np.linalg.det(G), 12)), det_function=det_func)

    print("\n== 4. spectrum away from the flat connection ==")
    mins = []
    for _ in range(400):
        U = random_U(rng)
        D = G * U
        mins.append(np.min(np.abs(np.linalg.eigvals(D))))
    mins = np.array(mins)
    q = np.percentile(mins, [0, 5, 50, 95, 100])
    print("  min |eigenvalue| of Gamma U: min/q5/median/q95/max =",
          np.round(q, 5))
    print("  fraction below 1e-6:", float((mins < 1e-6).mean()))
    res["gap_quantiles"] = [float(x) for x in q]

    ok = (res["Gamma_antisymmetric"] and ok_anti and ok_conj and complex_det
          and not has_g and not bip and res["rank"] == 2
          and abs(res["det"]) < 1e-12)
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    res["pass"] = bool(ok)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
