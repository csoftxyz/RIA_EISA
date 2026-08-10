#!/usr/bin/env python3
"""
source_hunt.py -- Source reconnaissance + combined structural test.

Tests:
  T1: fix d1=-a12/2 AND d2=(9/11)d3 AND a13=a12/3  -> 3 params
      (m0, a12, d3).  chi2 should stay ~0.02-0.1 if all real.
  T2: same + a23 fixed to candidate rationals:
      a23/a12 = 2 (chi2~1.8 known), 45/22, 2.0465, 41/20, 92/45
  T3: source matching: which lattice/algebra invariants reproduce
      1/2, 9/11, 1/3, 2.0465?
      invariant library: PLB NF dims {3,4,6,24,44}, algebra dims
      {12,4,3,19}, |Z3|=3, |S3|=6, C2=4/3, R_norm=1/4, D=1/9,
      orbit sizes {2,6,6,3}, 44, 27 (Delta27 order), etc.
"""
import sys
import numpy as np
from itertools import combinations, product
from scipy.optimize import differential_evolution
from scipy.linalg import eigh
sys.path.insert(0, '/root/.openclaw/workspace/paper/code')
from nufit_parser import NuFIT52, table_available

OMEGA = np.exp(2j * np.pi / 3)


def takagi_all(M):
    w2, V = eigh(M @ M.conj().T)
    idx = np.argsort(w2)
    m = np.sqrt(np.maximum(w2[idx], 0.0))
    U = V[:, idx].conj()
    temp = U.T @ M @ U
    U *= np.exp(-0.5j * np.angle(np.diag(temp)))[None, :]
    d = np.diag(U.T @ M @ U)
    for i in range(3):
        if np.real(d[i]) < 0:
            U[:, i] *= 1j
    return U, m


def obs_vec(M):
    U, m = takagi_all(M)
    t12 = np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
    t23 = np.arctan2(abs(U[1, 2]), abs(U[2, 2]))
    t13 = np.arcsin(min(max(abs(U[0, 2]), 0), 1))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    Ustd = U.copy()
    Ustd[0, :] *= np.exp(-1j * np.angle(Ustd[0, 0]))
    Ustd[:, 1] *= np.exp(-1j * np.angle(Ustd[0, 1]))
    Ustd[:, 2] *= np.exp(-1j * np.angle(Ustd[2, 2]))
    delta = (-np.angle(Ustd[0, 2])) % (2 * np.pi)
    den = (np.cos(t12) * np.sin(t12) * np.cos(t23) * np.sin(t23)
           * (np.cos(t13) ** 2) * np.sin(t13))
    if np.sin(delta) * np.clip(J / den, -1, 1) < 0:
        delta = (np.pi - delta) % (2 * np.pi)
    d = np.degrees(delta)
    if d > 180:
        d -= 360
    return np.array([np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2,
                     d, m[1] ** 2 - m[0] ** 2, m[2] ** 2 - m[0] ** 2])


nf = NuFIT52() if table_available() else None


def mk(ratio_d1, ratio_d2, ratio_a13, ratio_a23):
    """Build matrix builder with structural relations fixed."""
    def M(q):
        m0, a12, d3 = q
        d1 = -ratio_d1 * a12
        d2 = ratio_d2 * d3
        a13 = ratio_a13 * a12
        a23 = ratio_a23 * a12
        return m0 * np.array([[d1, a12, a13 * OMEGA],
                              [a12, d2, -a23],
                              [a13 * OMEGA, -a23, d3]], dtype=complex)
    return M


def fit3(builder, label, seeds=10):
    b3 = [(0.005, 0.10), (0.1, 1.0), (0.5, 2.5)]
    def fo(q):
        try:
            y = obs_vec(builder(q))
        except Exception:
            return 1e6
        return nf.chi2_1d_sum(y) if not np.any(np.isnan(y)) else 1e6
    best = None
    for seed in range(seeds):
        r = differential_evolution(fo, b3, seed=seed, maxiter=900,
                                   tol=1e-12, polish=True,
                                   updating='immediate')
        if best is None or r.fun < best.fun:
            best = r
    y = obs_vec(builder(best.x))
    print(f"  {label:34s} chi2={best.fun:8.3f}  "
          f"(m0={best.x[0]:.5f} a12={best.x[1]:.4f} d3={best.x[2]:.4f})  "
          f"s23={y[2]:.3f} delta={y[3]:.0f}")
    return best.fun


print("=== T1: 3-parameter texture with structural relations ===")
print("relations: d1=-a12/2, d2=(9/11)d3, a13=a12/3, a23 free? -> a23 needs fixing too")
print("NOTE: with d1,d2,a13 fixed we still have a23 free -> 4 params.")
print("Test A: 4 params (m0,a12,d3,a23):")
b4 = [(0.005, 0.10), (0.1, 1.0), (0.5, 2.5), (0.5, 1.5)]


def M4(q):
    m0, a12, d3, a23 = q
    return m0 * np.array([[-0.5 * a12, a12, (a12 / 3) * OMEGA],
                          [a12, (9 / 11) * d3, -a23],
                          [(a12 / 3) * OMEGA, -a23, d3]], dtype=complex)


def fo4(q):
    try:
        y = obs_vec(M4(q))
    except Exception:
        return 1e6
    return nf.chi2_1d_sum(y) if not np.any(np.isnan(y)) else 1e6


best4 = None
for seed in range(12):
    r = differential_evolution(fo4, b4, seed=seed, maxiter=1000,
                               tol=1e-12, polish=True, updating='immediate')
    if best4 is None or r.fun < best4.fun:
        best4 = r
y4 = obs_vec(M4(best4.x))
print(f"  4-param (d1=-a12/2, d2=9d3/11, a13=a12/3): chi2={best4.fun:.3f}")
print(f"    a23/a12 = {best4.x[3]/best4.x[1]:.6f}  (best-fit 2.0465)")
print(f"    s23={y4[2]:.3f} delta={y4[3]:.1f}")

print("\n=== T2: fix a23 too -> 3-param textures ===")
fit3(mk(0.5, 9 / 11, 1 / 3, 2.0), "a23=2*a12")
fit3(mk(0.5, 9 / 11, 1 / 3, 2.0465), "a23=2.0465*a12 (best)")
fit3(mk(0.5, 9 / 11, 1 / 3, 45 / 22), "a23=(45/22)*a12")
fit3(mk(0.5, 9 / 11, 1 / 3, 41 / 20), "a23=(41/20)*a12")
fit3(mk(0.5, 9 / 11, 1 / 3, 92 / 45), "a23=(92/45)*a12")
fit3(mk(0.5, 9 / 11, 1 / 3, 2 + 1 / 21), "a23=(2+1/21)*a12")

print("\n=== T3: source matching ===")
nums = [1, 2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 19, 24, 27, 32, 36, 44,
        54, 72, 108, 144, 216, 324]
targets = {"1/2": 0.5, "9/11": 9 / 11, "1/3": 1 / 3, "2.0465": 2.046525}
# ratios of products of small invariants
inv_small = [1, 2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 19, 24, 27, 36, 44, 54, 72]
print("\nsearching a/b with a,b in", inv_small, "within 0.15%:")
for tname, tval in targets.items():
    hits = []
    for a in inv_small:
        for b in inv_small:
            if b == 0:
                continue
            r = a / b
            if abs(r - tval) / tval < 0.0015:
                hits.append(f"{a}/{b}={r:.6f}")
    print(f"  {tname:8s} (target {tval:.6f}): {hits if hits else 'no simple ratio'}")

print("\n=== T3b: squared/cubic combos (dim g's, |Z3|, orbits) ===")
g = {"g0": 12, "g1": 4, "g2": 3}
orb = {"Dem": 2, "Hyb": 6, "Root": 6, "Flv": 3}
NF = {"Dem": 4, "Hyb": 24, "Root": 6, "Flv": 3}
special = {
    "R_norm": 1 / 4, "D": 1 / 9, "C2": 4 / 3, "1/|Z3|": 1 / 3,
    "|Z3|^2/|S3|": 9 / 6, "|Z3|^2/(|g0|-1)": 9 / 11,
    "NF_Dem/NF_Hyb": 4 / 24, "NF_Root/NF_Hyb": 6 / 24,
    "NF_Dem/NF_Flv": 4 / 3, "|S3|/|Z3|^2": 6 / 9,
    "|g0|/|g0|+": None,
}
for name, v in special.items():
    if v is None:
        continue
    for tname, tval in targets.items():
        if abs(v - tval) / tval < 0.002:
            print(f"  {name} = {v:.6f} ~ {tname} (target {tval:.6f})")
