#!/usr/bin/env python3
"""Refined look-elsewhere scan: rational Z3-locked textures vs Z3LM.

Statistic: official NuFIT 5.2 joint (3D + 2D) chi-square block,
optimised over the single scale m0.  chi2_block is the closest analogue
of a true chi-square available from the release tables.
"""
import sys
from math import gcd
import numpy as np
from scipy.optimize import minimize_scalar
import z3lm
from nufit_parser import NuFIT52

NF = NuFIT52()
R0 = np.array([0.5, 9/11, 45/22, 1/3, 6/19])
NAMES = ['|d1|/a12', 'd2/d3', 'a23/a12', 'a13/a12', 'a12/d3']


def obs_of(r, m0):
    r1, r2, r3, r4, r5 = r
    d3, a12 = 1.0, r5
    q = np.array([1.0, -r1 * a12, r2 * d3, d3, a12, r3 * a12])
    M = z3lm.M6(q) * m0
    M[0, 2] = M[2, 0] = r4 * a12 * m0 * z3lm.OMEGA
    return z3lm.obs_vec(M)


def chi2(r):
    def f(m0):
        o = obs_of(r, m0)
        return NF.chi2_block(np.array([o[0], o[1], o[2], o[3], o[4], o[5]]))
    res = minimize_scalar(f, bounds=(0.025, 0.045), method='bounded',
                          options=dict(xatol=1e-7))
    return res.fun, res.x


def rationals_near(r, pct, Qmax):
    lo, hi = r * (1 - pct), r * (1 + pct)
    out = []
    for q in range(1, Qmax + 1):
        for p in range(max(1, int(np.floor(lo * q))), int(np.ceil(hi * q)) + 1):
            if gcd(p, q) != 1:
                continue
            v = p / q
            if lo <= v <= hi:
                out.append((p, q, v))
    return out


def main():
    Qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    pct = float(sys.argv[2]) / 100.0 if len(sys.argv) > 2 else 0.03
    c0, m0 = chi2(R0)
    print(f"Qmax={Qmax} window=+-{pct*100:.0f}%")
    print(f"chi2_block(Z3LM) = {c0:.3f}  (m0 = {m0:.6f} eV)")

    lists = [rationals_near(R0[i], pct, Qmax) for i in (0, 1, 2, 4)]
    print("rational candidates per ratio:", [len(l) for l in lists])
    res = []
    for p1, q1, v1 in lists[0]:
        for p2, q2, v2 in lists[1]:
            for p3, q3, v3 in lists[2]:
                for p5, q5, v5 in lists[3]:
                    r = (v1, v2, v3, 1/3, v5)
                    if np.allclose(r, R0):
                        continue
                    c, _ = chi2(r)
                    res.append((c, (p1, q1), (p2, q2), (p3, q3), (p5, q5)))
    res.sort(key=lambda t: t[0])
    n = len(res)
    print(f"\nscanned {n} alternative rational textures")
    for thr in (c0, 2.0, 2.5, 4.0):
        k = sum(1 for t in res if t[0] <= thr)
        print(f"  chi2 <= {thr:5.2f}: {k:4d} ({100*k/n:.1f}%)")
    print("\nbest 20 alternatives:")
    for c, *rs in res[:20]:
        print(f"  chi2={c:6.3f}  " + "  ".join(f"{p}/{q}" for p, q in rs))
    with open('rational_scan_refined.txt', 'w') as f:
        f.write(f"chi2(Z3LM)={c0:.4f}\n")
        for c, *rs in res[:200]:
            f.write(f"{c:.4f}\t" + "\t".join(f"{p}/{q}" for p, q in rs) + "\n")


if __name__ == '__main__':
    main()
