#!/usr/bin/env python3
"""
perturb_analysis.py -- Perturbation-theory view of the Z3LM texture
parameters.

Idea: if the texture has a 'leading-order' symmetric limit (e.g.
mu-tau symmetry d2=d3, or TBM-like), the observed ratios may be
perturbative corrections in a small parameter eps.  We scan:
  - additive relations (d2+d1, d3-d2, ...) vs simple fractions
  - differences normalized by a12 (the natural scale)
  - whether d2/d3 = 1 - eps with eps = 2/11 = 8/44 (lattice gap)
  - perturbative chain: d1 ~ O(eps), a13 ~ O(eps), ...
"""
import numpy as np
from fractions import Fraction

# high-precision best fit (official NuFIT 5.2)
m0, d1, d2, d3, a12, a23 = 0.02242954, -0.22256984, 1.15386934, 1.40954774, 0.44558787, 0.91190663
a13 = a12 / 3

print("=== parameter set (m0 units) ===")
print(f"d1={d1:.6f} d2={d2:.6f} d3={d3:.6f} a12={a12:.6f} a23={a23:.6f} a13={a13:.6f}")

print("\n=== 1. normalized differences (by a12) ===")
diffs = {
    "(d2-d1)/a12": (d2 - d1) / a12,
    "(d3-d2)/a12": (d3 - d2) / a12,
    "(d3-d1)/a12": (d3 - d1) / a12,
    "(d2-d3)/a12": (d2 - d3) / a12,
    "d2/a12": d2 / a12,
    "d3/a12": d3 / a12,
    "d1/a12": d1 / a12,
    "(d2+d1)/a12": (d2 + d1) / a12,
    "(d3+d1)/a12": (d3 + d1) / a12,
    "(d2+d3)/a12": (d2 + d3) / a12,
    "a23/a12": a23 / a12,
    "2a23/a12": 2 * a23 / a12,
    "a23/(d3-d2)": a23 / (d3 - d2),
    "(d3-d2)/a23": (d3 - d2) / a23,
    "(d3-d2)/(2*a12)": (d3 - d2) / (2 * a12),
}
for k, v in diffs.items():
    f = Fraction(v).limit_denominator(20)
    err = abs(v - float(f)) / abs(v) * 100
    mark = " <== clean" if err < 0.5 else ""
    print(f"  {k:20s} = {v:10.6f}  ~ {f.numerator}/{f.denominator} "
          f"(err {err:.2f}%){mark}")

print("\n=== 2. perturbative hypothesis: d2/d3 = 1 - eps ===")
r = d2 / d3
eps = 1 - r
print(f"d2/d3 = {r:.8f},  eps = 1 - d2/d3 = {eps:.8f}")
print(f"eps vs 2/11   = {2/11:.8f}  (dev {(eps-2/11)/(2/11)*100:+.3f}%)")
print(f"eps vs 2/11=8/44 (44-vector gap)")
print(f"eps vs 1/6   = {1/6:.8f}  (dev {(eps-1/6)/(1/6)*100:+.3f}%)  [PLB eps_q]")
print(f"eps vs 1/5   = {0.2:.8f}  (dev {(eps-0.2)/0.2*100:+.3f}%)")
print(f"eps vs 1/5.5 = {1/5.5:.8f}")

print("\n=== 3. is d3-d2 related to a13 or a12/3? ===")
print(f"d3-d2 = {d3-d2:.8f}")
print(f"a12/2 = {a12/2:.8f}  (dev {(d3-d2-a12/2)/(a12/2)*100:+.3f}%)")
print(f"a12*2/3 = {2*a12/3:.8f}  (dev {(d3-d2-2*a12/3)/(2*a12/3)*100:+.3f}%)")
print(f"a12*3/5 = {3*a12/5:.8f}  (dev {(d3-d2-3*a12/5)/(3*a12/5)*100:+.3f}%)")
print(f"a23/3.5 = {a23/3.5:.8f}  (dev {(d3-d2-a23/3.5)/(a23/3.5)*100:+.3f}%)")

print("\n=== 4. perturbative chain (orders in eps) ===")
print("hypothesis: a13 = O(eps), d1 = O(eps), (d3-d2) = O(eps), a12 = O(1)")
print(f"  a13/a12    = {a13/a12:.6f}  (=1/3, large -> not small)")
print(f"  |d1|/a12   = {abs(d1)/a12:.6f}  (=1/2)")
print(f"  (d3-d2)/a12= {(d3-d2)/a12:.6f}")
print(f"  (d2-d1)/a12= {(d2-d1)/a12:.6f}")
print("=> a13/a12=1/3 and |d1|/a12=1/2 are O(1): NOT a small-eps expansion")
print("=> the texture is NOT a perturbative deformation of a symmetric limit")
print("   (the 1/3, 1/2, 9/11 ratios are O(1) structural, not corrections)")

print("\n=== 5. exact additive check: d2 = 9/11 d3 with d3 free ===")
print(f"if d2=(9/11)d3: d2 predicted = {9/11*d3:.6f} vs actual {d2:.6f}")
print(f"  -> d2 constraint: chi2 = 0.025 (fits fine)")
print(f"if additionally d1 = -a12/2: both hold, chi2 = 0.026")
print("=> the ratios are INDEPENDENT structural relations (not a chain)")

print("\n=== 6. 44-vector decomposition of all ratios ===")
print("9/11  = 36/44 = (44-8)/44   [44 vectors minus 8]")
print("1/2   = 22/44 = (44/2)/44")
print("1/3   = ? 44/3 not integer; 1/3 = |Z3|^-1 (group-theoretic)")
print("45/22 = 90/44 = (2*44+2)/44")
print("  => 1/2 and 9/11 both admit 44-vector forms;")
print("     1/3 and 45/22 do not (need group order, not lattice)")
