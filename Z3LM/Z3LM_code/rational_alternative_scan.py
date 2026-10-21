#!/usr/bin/env python3
"""How many OTHER rational Z3-locked textures fit the data as well as Z3LM?

Reviewer 1's central objection: the five ratios were read off a numerical
best fit, so the "one-parameter" claim is a post-hoc rationalisation.
The sharpest way to test this is a look-elsewhere count:

  enumerate Z3-locked textures whose moduli are small-denominator rationals
  p/q (q <= Q) inside +-3% of the Z3LM values, profile the overall scale m0,
  and count how many reach chi2 <= chi2_Z3LM.

If that number is O(1) the texture is a unique discrete selection.
If it is large, the rational pattern carries no information and the
paper must be repositioned as a parametrisation, not a prediction.

Run: python3 rational_alternative_scan.py [QMAX] [PCT]
"""
import sys
from math import gcd
import numpy as np
import z3lm
from nufit_parser import NuFIT52, table_available

NF = NuFIT52() if table_available() else None
M0GRID = np.linspace(0.020, 0.055, 141)

OBS = np.array([0.305, 0.0223, 0.450, -130.0, 7.41e-5, 2.505e-3])
ERR = np.array([0.010, 0.0006, 0.018, 27.5, 0.21e-5, 0.025e-3])

R0 = np.array([0.5, 9 / 11, 45 / 22, 1 / 3, 6 / 19])   # Z3LM values


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


def chi2_of(r, m0=None):
    """Official NuFIT-5.2 1D-projection sum, minimised over m0."""
    r1, r2, r3, r4, r5 = r
    d3, a12 = 1.0, r5
    q = np.array([1.0, -r1 * a12, r2 * d3, d3, a12, r3 * a12])
    best = np.inf
    for m0 in M0GRID:
        M = z3lm.M6(q) * m0
        M[0, 2] = M[2, 0] = r4 * a12 * m0 * z3lm.OMEGA
        o = z3lm.obs_vec(M)
        c = NF.chi2_1d_sum(np.array([o[0], o[1], o[2], o[3], o[4], o[5]]))
        if c < best:
            best = c
    return best


def chi2_of_old(r, m0=None):
    """chi2 for a ratio tuple r = (r1,r2,r3,r4,r5); m0 profiled."""
    r1, r2, r3, r4, r5 = r
    d3 = 1.0
    a12 = r5 * d3
    q = np.array([1.0, -r1 * a12, r2 * d3, d3, a12, r3 * a12])
    # a13 = a12/3 is fixed by the texture family (r4 = 1/3 by construction)
    M = z3lm.M6(q)
    M[0, 2] = M[2, 0] = r4 * a12 * z3lm.OMEGA
    o = z3lm.obs_vec(M)          # s12,s13,s23,delta,dm21,dm31 (m0 = 1)
    # m0 profile: only the two dm^2 scale
    mass_obs = o[4:6]
    best = np.inf
    for m0 in (1.0,):
        d21 = mass_obs[0] * m0 ** 2
        d31 = mass_obs[1] * m0 ** 2
        s = (0.305 - o[0]) / 0.010
        c = ((o[0] - OBS[0]) / ERR[0]) ** 2 + ((o[1] - OBS[1]) / ERR[1]) ** 2 \
            + ((o[2] - OBS[2]) / ERR[2]) ** 2 + ((o[3] - OBS[3]) / ERR[3]) ** 2
        # optimal m0 for the two mass-squared differences (single scale)
        num = OBS[4] * mass_obs[0] / ERR[4] ** 2 + OBS[5] * mass_obs[1] / ERR[5] ** 2
        den = mass_obs[0] ** 2 / ERR[4] ** 2 + mass_obs[1] ** 2 / ERR[5] ** 2
        if den <= 0:
            continue
        m02 = num / den
        if m02 <= 0:
            continue
        c += ((mass_obs[0] * m02 - OBS[4]) / ERR[4]) ** 2 \
             + ((mass_obs[1] * m02 - OBS[5]) / ERR[5]) ** 2
        best = min(best, c)
    return best


def main():
    Qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    pct = float(sys.argv[2]) / 100.0 if len(sys.argv) > 2 else 0.03

    lists = [rationals_near(R0[i], pct, Qmax) for i in (0, 1, 2, 4)]
    print(f"Qmax={Qmax}  window=+-{pct*100:.0f}%  "
          f"candidates per ratio: {[len(l) for l in lists]}")

    base = chi2_of(R0)
    print(f"chi2(Z3LM) = {base:.3f}\n")

    results = []
    for p1, q1, v1 in lists[0]:
        for p2, q2, v2 in lists[1]:
            for p3, q3, v3 in lists[2]:
                for p5, q5, v5 in lists[3]:
                    c = chi2_of((v1, v2, v3, 1 / 3, v5))
                    results.append((c, (p1, q1), (p2, q2), (p3, q3), (p5, q5)))
    results.sort(key=lambda t: t[0])
    n = len(results)
    better = sum(1 for r in results if r[0] <= base)
    print(f"scanned {n} rational textures")
    print(f"with chi2 <= chi2(Z3LM) = {base:.3f}: {better} "
          f"({100.0*better/max(n,1):.1f}%)")
    print(f"with chi2 < 2 : {sum(1 for r in results if r[0] < 2.0)}")
    print(f"with chi2 < 4 : {sum(1 for r in results if r[0] < 4.0)}")
    print("\nbest 15:")
    for c, *rs in results[:15]:
        print(f"  chi2={c:7.3f}   " + "  ".join(f"{p}/{q}" for p, q in rs))
    np.save('rational_scan.npy', np.array([(r[0],) + tuple(x for pq in r[1:] for x in pq)
                                           for r in results]))


if __name__ == '__main__':
    main()
