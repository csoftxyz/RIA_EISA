#!/usr/bin/env python3
"""
z3lm_3param.py -- The 3-parameter Z3LM texture: full prediction table.

Structural relations (from reconnaissance, all data-supported):
    d1  = -(1/2) a12
    d2  = (9/11) d3
    a13 = (1/3)  a12
    a23 = r23 * a12,  r23 in {45/22, 41/20, 92/45} (all chi2<0.05)

Free params: (m0, a12, d3)  ->  3 parameters.

Predictions vs the 6-parameter best fit (lower octant):
    sin2 t12, sin2 t23, sin2 t13, delta, dm21, dm31,
    J, m1, m2, m3, sum m, m_bb, ordering.
"""
import sys
import numpy as np
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


def obs_full(M):
    U, m = takagi_all(M)
    t12 = np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
    t23 = np.arctan2(abs(U[1, 2]), abs(U[2, 2]))
    t13 = np.arcsin(min(max(abs(U[0, 2]), 0), 1))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    s12, c12 = np.sin(t12), np.cos(t12)
    s23, c23 = np.sin(t23), np.cos(t23)
    s13, c13 = np.sin(t13), np.cos(t13)
    D = c12 * s12 * c23 * s23 * c13 ** 2 * s13
    Um1 = abs(U[1, 0]) ** 2
    cosd = float(np.clip((Um1 - s12 ** 2 * c23 ** 2
                          - c12 ** 2 * s23 ** 2 * s13 ** 2)
                         / (2 * s12 * c12 * s23 * c23 * s13), -1, 1))
    sind = float(np.clip(J / D, -1, 1))
    d = np.degrees(np.arctan2(sind, cosd))
    if d >= 180:
        d -= 360
    ms = np.sort(m)
    Ue = U[:, np.argsort(m)]
    mbb = abs(Ue[0, 0] ** 2 * ms[0] + Ue[0, 1] ** 2 * ms[1]
              + Ue[0, 2] ** 2 * ms[2])
    return dict(s12=np.sin(t12) ** 2, s13=np.sin(t13) ** 2,
                s23=np.sin(t23) ** 2, delta=d,
                dm21=m[1] ** 2 - m[0] ** 2, dm31=m[2] ** 2 - m[0] ** 2,
                J=J, m1=ms[0], m2=ms[1], m3=ms[2], sum_m=ms.sum(),
                mbb=mbb, ordering="NO" if ms[2] > ms[1] else "IO")


nf = NuFIT52() if table_available() else None


def fit3(r23, label, seeds=12):
    def M(q):
        m0, a12, d3 = q
        return m0 * np.array([[-0.5 * a12, a12, (a12 / 3) * OMEGA],
                              [a12, (9 / 11) * d3, -r23 * a12],
                              [(a12 / 3) * OMEGA, -r23 * a12, d3]],
                             dtype=complex)

    def fo(q):
        try:
            y = np.array(list(obs_full(M(q)).values())[:6])
        except Exception:
            return 1e6
        s12, s13, s23, dc, dm21, dm31 = (obs_full(M(q))["s12"],
                                          obs_full(M(q))["s13"],
                                          obs_full(M(q))["s23"],
                                          obs_full(M(q))["delta"],
                                          obs_full(M(q))["dm21"],
                                          obs_full(M(q))["dm31"])
        return nf.chi2_1d_sum(np.array([s12, s13, s23, dc, dm21, dm31]))

    b3 = [(0.005, 0.10), (0.1, 1.0), (0.5, 2.5)]
    best = None
    for seed in range(seeds):
        r = differential_evolution(fo, b3, seed=seed, maxiter=900,
                                   tol=1e-12, polish=True,
                                   updating='immediate')
        if best is None or r.fun < best.fun:
            best = r
    o = obs_full(M(best.x))
    print(f"\n=== 3-param texture: a23 = ({r23})*a12  [{label}] ===")
    print(f"chi2_min = {best.fun:.3f}  (6-param reference: 1.33)")
    print(f"free params: m0={best.x[0]:.5f} eV  a12={best.x[1]:.5f}  "
          f"d3={best.x[2]:.5f}")
    print(f"  sin2 t12 = {o['s12']:.5f}   (6p: 0.3050)")
    print(f"  sin2 t23 = {o['s23']:.5f}   (6p: 0.4500)")
    print(f"  sin2 t13 = {o['s13']:.6f}   (6p: 0.02230)")
    print(f"  delta    = {o['delta']:.2f} deg   (6p: -83.6)")
    print(f"  dm21     = {o['dm21']:.4e}   (6p: 7.413e-5)")
    print(f"  dm31     = {o['dm31']:.4e}   (6p: 2.505e-3)")
    print(f"  J        = {o['J']:+.5f}   (6p: -0.0332)")
    print(f"  m1,m2,m3 = {o['m1']:.4f}, {o['m2']:.4f}, {o['m3']:.4f} eV")
    print(f"  sum m    = {o['sum_m']:.4f} eV   (6p: 0.0697)")
    print(f"  m_bb     = {o['mbb']:.5f} eV   (6p: 0.0050)")
    print(f"  ordering = {o['ordering']}")
    return best.fun, o


print("=== 3-PARAMETER Z3LM TEXTURE: PREDICTION TABLE ===")
for r23, label in [(45 / 22, "45/22 = 2.0455"),
                   (41 / 20, "41/20 = 2.05"),
                   (92 / 45, "92/45 = 2.0444"),
                   (2.0465, "best-fit 2.0465")]:
    fit3(r23, label)
