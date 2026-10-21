#!/usr/bin/env python3
"""
d27_cg4.py -- No-Go analysis: can a13/a12 = 1/3 arise from Delta(27)
representation theory in a minimal L~3 Weinberg model?

Structure of 3 x 3 = 3bar + 3bar + 3bar (established in d27_cg3.py):
  * copy1 (diagonal):      C1^k_{ij} = delta_{ik} delta_{jk}
  * copy2 (off-diagonal):  C2^k_{ij} = (delta_{ik} delta_{jl}
                                         + delta_{il} delta_{jk})/sqrt2
  * antisym (epsilon):     A^k_{ij}  = eps_{ijk}

With L ~ 3, H ~ 1 and n flavons phi_a ~ 3 with VEVs v_a, the Weinberg
matrix is a linear combination of the CGs contracted with the VEVs:
    M = sum_a [ x_a C1(v_a) + y_a C2(v_a) ]
We test ALL aligned VEV directions allowed by Delta(27) (orbits of
(1,1,1), (1,w,w^2), (1,0,0) types) and ask whether
|M13|/|M12| = 1/3 is achievable with the phase pattern of the paper
(M12 real+, M13 ~ w, M23 real-).
"""
import numpy as np
from itertools import product

w = np.exp(2j * np.pi / 3)

# CG tensors (from d27_cg3.py, orthonormal convention)
C1 = np.zeros((3, 3, 3), complex)   # diagonal copy
C2 = np.zeros((3, 3, 3), complex)   # off-diagonal copy
for k in range(3):
    C1[k, k, k] = 1.0
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            # C2^k_{ij} nonzero iff {i,j} = complement of k
            if {i, j} == {x for x in range(3) if x != k}:
                C2[k, i, j] = 1.0 / np.sqrt(2)

print("C1 diagonal copy:", np.round(C1[:, 0, 0], 3), np.round(C1[:, 1, 1], 3), np.round(C1[:, 2, 2], 3))
print("C2 structure: C2^k_{ij} nonzero for {i,j}=comp(k):")
for k in range(3):
    print(f"  k={k}:", np.round(C2[k], 3))


def build_M(vevs, coeffs, which):
    """vevs: list of 3-vectors; coeffs: list of (x,y) per flavon;
    which: 'C1','C2' or 'both'."""
    M = np.zeros((3, 3), complex)
    for v, (x, y), wch in zip(vevs, coeffs, which):
        if wch in ('C1', 'both'):
            for k in range(3):
                M += x * v[k] * C1[k]
        if wch in ('C2', 'both'):
            for k in range(3):
                M += y * v[k] * C2[k]
    return M


# allowed VEV orbits of Delta(27) (projective): representatives
v111 = np.array([1, 1, 1], complex)
v1ww2 = np.array([1, w, w ** 2], complex)
v1w2w = np.array([1, w ** 2, w], complex)
v100 = np.array([1, 0, 0], complex)
v010 = np.array([0, 1, 0], complex)
v001 = np.array([0, 0, 1], complex)
vacs = [("(1,1,1)", v111), ("(1,w,w2)", v1ww2), ("(1,w2,w)", v1w2w),
        ("(1,0,0)", v100), ("(0,1,0)", v010), ("(0,0,1)", v001)]

print("\n=== single flavon, both CG channels ===")
print(f"{'VEV':<12}{'x':>5}{'y':>5} {'|M12|':>8}{'|M13|':>8}{'|M23|':>8}"
      f" {'|M13|/|M12|':>10}  phases(M12,M13,M23)")
for name, v in vacs:
    for x, y in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (1, 0.5), (1, -1)]:
        M = build_M([v], [(x, y)], ['both'])
        m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
        if abs(m12) < 1e-9 or abs(m13) < 1e-9:
            continue
        ratio = abs(m13) / abs(m12)
        ph = np.degrees(np.angle([m12, m13, m23]))
        flag = " <== 1/3!" if abs(ratio - 1 / 3) < 1e-6 else ""
        print(f"{name:<12}{x:>5}{y:>5} {abs(m12):8.4f}{abs(m13):8.4f}{abs(m23):8.4f}"
              f" {ratio:10.4f}  {np.round(ph,0)}{flag}")

print("\n=== two flavons (different VEVs), search for 1/3 ===")
best = []
for n1, v1 in vacs:
    for n2, v2 in vacs:
        for x1, y1 in [(1, 0), (0, 1), (1, 1)]:
            for x2, y2 in [(1, 0), (0, 1), (1, 1), (1, -1), (1, 0.5)]:
                M = build_M([v1, v2], [(x1, y1), (x2, y2)], ['both', 'both'])
                m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
                if abs(m12) < 1e-9 or abs(m13) < 1e-9:
                    continue
                r = abs(m13) / abs(m12)
                if abs(r - 1 / 3) < 0.05:
                    best.append((r, n1, n2, x1, y1, x2, y2, M))
print(f"configurations with |M13|/|M12| in [0.28,0.38]: {len(best)}")
for r, n1, n2, x1, y1, x2, y2, M in best[:10]:
    m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
    ph = np.degrees(np.angle([m12, m13, m23]))
    print(f"  ratio={r:.3f}  VEV=({n1},{n2}) coeff=({x1},{y1},{x2},{y2})"
          f"  |M|={np.round([abs(m12),abs(m13),abs(m23)],3)}"
          f"  phases={np.round(ph,1)}")

# check phase pattern of the paper: M12 real+, M13~w, M23 real-
print("\n=== does any config reproduce the paper phase pattern? ===")
print("paper: arg(M12)=0, arg(M13)=120, arg(M23)=180 (M23 real-)")
target_ph = np.array([0, 120, 180])
n_match = 0
for n1, v1 in vacs:
    for n2, v2 in vacs:
        for x1 in [1, -1]:
            for y1 in [0, 1, -1]:
                for x2 in [1, -1]:
                    for y2 in [0, 1, -1, 0.5, -0.5]:
                        M = build_M([v1, v2], [(x1, y1), (x2, y2)], ['both', 'both'])
                        m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
                        if min(abs(m12), abs(m13), abs(m23)) < 1e-7:
                            continue
                        ph = np.degrees(np.angle([m12, m13, m23])) % 360
                        if np.allclose(np.sort(ph), np.sort(target_ph), atol=3):
                            n_match += 1
                            print(f"  MATCH: VEV=({n1},{n2}) coeff=({x1},{y1},{x2},{y2})"
                                  f"  ratio={abs(m13)/abs(m12):.4f}")
print(f"total phase-pattern matches: {n_match}")
