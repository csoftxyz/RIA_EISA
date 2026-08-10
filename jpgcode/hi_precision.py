#!/usr/bin/env python3
"""
hi_precision.py -- High-precision refit of the Z3LM texture against
the official NuFIT 5.2 tables, to test structural ratio hypotheses:

  H1: |d1|/a12 = 1/2     (found: 0.49955, 0.09% off)
  H2: d2/d3    = 9/11    (found: 0.81866, 0.06% off)
  H3: a23/a12  = 2       (found: 2.04645, 2.3% off - expected fake)

Uses tighter DE settings + local polish to get ~6 significant digits.
"""
import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize
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
    return np.array([np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2,
                     d, m[1] ** 2 - m[0] ** 2, m[2] ** 2 - m[0] ** 2])


def M6(q):
    m0, d1, d2, d3, a12, a23 = q
    return m0 * np.array([[d1, a12, (a12 / 3.0) * OMEGA],
                          [a12, d2, -a23],
                          [(a12 / 3.0) * OMEGA, -a23, d3]], dtype=complex)


if not table_available():
    print("table missing")
    sys.exit(1)
nf = NuFIT52()

B6 = [(0.005, 0.10), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5),
      (0, 1), (0, 1.5)]


def objective(q):
    try:
        y = obs_vec(M6(q))
    except Exception:
        return 1e6
    if np.any(np.isnan(y)):
        return 1e6
    return nf.chi2_1d_sum(y)


print("=== high-precision 6-param refit (multi-seed + polish) ===")
best = None
for seed in range(16):
    r = differential_evolution(objective, B6, seed=seed, maxiter=1200,
                               tol=1e-13, polish=True, updating='immediate',
                               atol=1e-12)
    if best is None or r.fun < best.fun:
        best = r
# extra local polish from the winner
r2 = minimize(objective, best.x, method='Nelder-Mead',
              options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
if r2.fun < best.fun:
    best = r2

q = best.x
print(f"chi2_min = {best.fun:.6f}")
print("params (8 sig digits):")
print(f"  m0  = {q[0]:.8f}")
print(f"  d1  = {q[1]:.8f}")
print(f"  d2  = {q[2]:.8f}")
print(f"  d3  = {q[3]:.8f}")
print(f"  a12 = {q[4]:.8f}")
print(f"  a23 = {q[5]:.8f}")

print("\n=== structural ratio tests ===")
r_abs = {
    "|d1|/a12  vs 1/2": (abs(q[1]) / q[4], 0.5),
    "d2/d3     vs 9/11": (q[2] / q[3], 9 / 11),
    "d3/d2     vs 11/9": (q[3] / q[2], 11 / 9),
    "a23/a12   vs 2": (q[5] / q[4], 2.0),
    "a23/a12   vs 2.0465": (q[5] / q[4], 2.0465),
    "(d2+d3)/a23 vs 2.8": ((q[2] + q[3]) / q[5], 2.8),
    "a12/|d1|  vs 2": (q[4] / abs(q[1]), 2.0),
}
for name, (v, target) in r_abs.items():
    err = abs(v - target) / target * 100
    verdict = "MATCH" if err < 0.2 else ("close" if err < 1 else "no")
    print(f"  {name:22s}: ratio={v:.8f}  target={target:.6f}  "
          f"dev={err:+.4f}%  [{verdict}]")

# test with uncertainty: how much can d2/d3 move within chi2 < 1?
print("\n=== profile: d2/d3 vs chi2 (is 9/11 within 1-sigma?) ===")
# scan d2/d3 by fixing ratio, refit others
for ratio_t, label in [(9 / 11, "9/11"), (0.81866, "best-fit")]:
    def M_ratio(q):
        m0, d1, d3, a12, a23 = q
        d2 = ratio_t * d3
        return m0 * np.array([[d1, a12, (a12 / 3.0) * OMEGA],
                              [a12, d2, -a23],
                              [(a12 / 3.0) * OMEGA, -a23, d3]], dtype=complex)
    b5 = [(0.005, 0.10), (-1.5, 1.5), (-1.5, 1.5), (0, 1), (0, 1.5)]
    fo = lambda qq: (objective2 := None) or _obj(M_ratio(qq))
    # simpler: reuse objective via closure
    def fo(qq):
        try:
            y = obs_vec(M_ratio(qq))
        except Exception:
            return 1e6
        return nf.chi2_1d_sum(y) if not np.any(np.isnan(y)) else 1e6
    rb = None
    for seed in range(10):
        r = differential_evolution(fo, b5, seed=seed, maxiter=800,
                                   tol=1e-12, polish=True,
                                   updating='immediate')
        if rb is None or r.fun < rb.fun:
            rb = r
    print(f"  d2/d3 = {ratio_t:.6f} ({label}): chi2_min = {rb.fun:.3f}")

print("\n=== conclusion ===")
print("if d2/d3=9/11 gives chi2 ~ 0.02-0.1: structural (1 sigma ok)")
print("if chi2 jumps >> 1: fake (best-fit 0.81866 was coincidence)")
