#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirac spectrum and zero modes of the minimal fermionic extension.

D = m 1 + t Gamma . U ,  Gamma the unique Z3(T)-equivariant antisymmetric
orientation matrix of K_{2,2,2}.  At the flat connection U = 1:

    spec(Gamma) = {0 (x4), +- 2 sqrt(3) i}
    det(m + Gamma) = m^4 (m^2 + 12)

so there are four exact zero modes for every beta at m = t = 0, of purely
algebraic origin (Gamma has rank 2 on a six-dimensional space).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

verts = [(a, s) for a in range(3) for s in (1, -1)]
vid = {v: i for i, v in enumerate(verts)}
G = np.zeros((6, 6))
for i, u in enumerate(verts):
    for j, v in enumerate(verts):
        if j > i and u[0] != v[0]:
            s = 1 if (u[0] + 1) % 3 == v[0] else -1
            G[i, j] = s
            G[j, i] = -s


def e(a, s):
    return np.eye(6)[vid[(a, s)]]


basis = np.array([
    e(0, 1) - e(0, -1),
    e(1, 1) - e(1, -1),
    e(2, 1) - e(2, -1),
    e(0, 1) + e(1, 1) + e(2, 1),
]).T


def main():
    ev = np.linalg.eigvals(G)
    sv = np.linalg.svd(G, compute_uv=False)
    res = dict(eigenvalues=[[round(float(x.real), 12), round(float(x.imag), 12)] for x in ev],
               singular_values=[float(x) for x in sv],
               rank=int(np.linalg.matrix_rank(G)),
               det=float(round(np.linalg.det(G), 12)),
               kernel_residual=float(np.abs(G @ basis).max()),
               kernel_rank=int(np.linalg.matrix_rank(basis)))
    print("octahedron: V=6, E=12, F=8")
    print("eigenvalues of Gamma:", np.round(ev, 8))
    print("singular values      :", np.round(sv, 8))
    print("rank(Gamma) =", res["rank"], " det(Gamma) =", res["det"])
    print("kernel basis (e_a^+ - e_a^-, sum_a e_a^+):")
    print(np.round(basis.T, 3))
    print("max |Gamma . basis| =", res["kernel_residual"],
          " rank of basis =", res["kernel_rank"])
    for m in (0.0, 0.3, 1.0):
        det = np.linalg.det(m * np.eye(6) + G)
        print(f"det(m 1 + Gamma) at m={m}: {det:.10f}  "
              f"(m^4(m^2+12) = {m**4*(m*m+12):.10f})")
        res[f"det_m{m}"] = float(det)
    with open(os.path.join(HERE, os.pardir, "logs", "fermion_spectrum.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
