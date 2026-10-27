#!/usr/bin/env python3
"""
T5 -- Marginality: the geometric-mean estimator is exact only for the marginal power law.

A charge estimator built from the mesh alone cannot know the exponent of the potential it is
measuring.  With the flux radius fixed by geometry, rho_n^2 = r_n r_{n+1}, the recovered
coefficient for a trial potential r^-p is

    C_p(n) = (V_n - V_N) / (r_n^-p - r_N^-p),        V_n - V_N = (1/4pi)(1/r_n - 1/r_N).

C_p is independent of n (exact) if and only if p = 1.  For p != 1 the recovered coefficient
depends on n and on the mesh ratio q: a geometric estimator manufactures running for every
non-marginal power.  This is the quantitative form of "estimator defined by geometry alone
<=> zero anomalous dimension".
"""
import json
import math
import numpy as np

OUT = {}
print("=" * 84)
print("T5: recovered coefficient C_p(n) for trial potentials r^-p, geometric-mean flux radius")
print("    exact <=> C_p independent of n and of q")
print(f"\n{'p':>6} {'q':>9} {'mean C_p':>14} {'max_n |C_p/<C> - 1|':>22} {'verdict':>12}")
res = {}
for p in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    for q in [1.05, math.sqrt(3), 2.0]:
        r = 1.0 * q ** np.arange(0, 40)
        rho = np.sqrt(r[:-1] * r[1:])
        d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
        V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
        num = V - V[-1]
        den = r ** (-p) - r[-1] ** (-p)
        good = np.abs(den) > 1e-14
        C = num[good] / den[good]
        spread = float(np.max(np.abs(C / np.mean(C) - 1)))
        res[f"p={p},q={q:.6f}"] = spread
        verdict = "exact" if spread < 1e-12 else "running"
        print(f"{p:6.2f} {q:9.5f} {np.mean(C):14.9f} {spread:22.3e} {verdict:>12}")

# amplitude ratio between adjacent shells: A_{n+1}/A_n = q^p for p != 1, = 1 for p = 1
print("\n  shell-to-shell amplitude ratio A_{n+1}/A_n  (should be q^(p-1) for p != 1, 1 for p = 1):")
print(f"  {'p':>5} {'q':>8} {'measured ratio':>16} {'q^(p-1)':>12} {'1':>8}")
for p in [0.5, 1.0, 1.5, 2.0]:
    for q in [1.2, math.sqrt(3), 2.0]:
        r = q ** np.arange(0.0, 30.0)
        rho = np.sqrt(r[:-1] * r[1:])
        d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
        V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
        with np.errstate(invalid="ignore", divide="ignore"):
            A = (V - V[-1]) / (r ** (-p) - r[-1] ** (-p))
        ratio = float(np.median(A[1:-2] / A[0:-3]))
        print(f"  {p:5.2f} {q:8.4f} {ratio:16.9f} {q**(p-1):12.6f} {1.0:8.1f}")

OUT["spread"] = res
with open("logs/t5_marginality.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote logs/t5_marginality.json")
