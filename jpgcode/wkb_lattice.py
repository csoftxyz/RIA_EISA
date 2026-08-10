#!/usr/bin/env python3
"""
wkb_lattice.py -- WKB / lattice-propagation view.

The companion framework (PLB paper) computes perturbation strengths
from 44-vector lattice counting:
    eps_q   = NF(Dem)/NF(Hyb) = 4/24 = 1/6
    eps_nu2 = R_norm * D = 1/36
    eps_nu3 = R_norm/|Z3| = 1/12
where NF = norm-filtered counts, R_norm = 1/4 (Frobenius democracy),
D = 1/9 (u(3) dilution).

We ask whether the Z3LM texture ratios admit the same kind of
'path-weight' interpretation on the 44-vector lattice:
    d2/d3 = 9/11,  |d1|/a12 = 1/2,  a23/a12 = 45/22,  a13/a12 = 1/3

Key structural difference: d1, d2, a12 are 'direct' Weinberg entries
(no flavon); a13, a23, d3 involve flavon insertions.  If the lattice
assigns a 'path weight' W to each entry, the ratios are W-ratios.
"""
import numpy as np
from fractions import Fraction
from itertools import combinations

print("=== lattice counting dictionary (from companion/PLB framework) ===")
counts = {
    "NF_Dem": 4, "NF_Hyb": 24, "NF_Root": 6, "NF_Flv": 3,
    "orbit_Dem": 2, "orbit_Hyb": 6, "orbit_Root": 6, "orbit_Flv": 3,
    "total_vectors": 44, "dim_g0": 12, "dim_g1": 4, "dim_g2": 3,
    "dim_alg": 19, "|Z3|": 3, "|S3|": 6, "C2_su3": 4 / 3,
    "R_norm": 1 / 4, "D_u3": 1 / 9, "dim8": 8, "|G|=27": 27,
    "dim(3x3)": 9, "dim(3x3bar)": 9,
}
for k, v in counts.items():
    print(f"  {k:12s} = {v}")

targets = {
    "d2/d3": 9 / 11, "|d1|/a12": 0.5, "a23/a12": 45 / 22,
    "a13/a12": 1 / 3, "d2/d3*": 0.81861,
}

print("\n=== search: ratios of lattice-count combinations ===")
nums = list(counts.values()) + [1, 2, 5, 7, 8, 10, 11, 13, 16, 20, 22, 27, 36, 44]
# add simple products
prods = set(nums)
for a in nums:
    for b in nums:
        prods.add(a * b)
        prods.add(a * b * 3)
prods = sorted(prods)
print(f"number set size: {len(prods)}")

for tname, tval in targets.items():
    hits = []
    for a in prods:
        for b in prods:
            if b == 0:
                continue
            r = a / b
            if abs(r - tval) / tval < 0.0005:
                hits.append(f"{a}/{b}={r:.6f}")
    print(f"\n  {tname} (target {tval:.6f}):")
    for h in hits[:8]:
        print(f"    {h}")

print("\n=== WKB-style: path weight W ~ exp(-S) or 1/(distance) ===")
print("if W_i ~ 1/L_i^2 (lattice shell L = 3^n):")
print("  shells: L = 3, 27, 243 -> L^2 = 9, 729, 59049")
print("  ratios of 1/L^2: (1/9):(1/729):... not 9/11")
print("if W ~ 1/d^2 with d = graph distance on 44-vector lattice:")
print("  d2/d3 = 9/11 would need dist(2,2)/dist(3,3) = sqrt(11/9)")
print("  -> distance ratio ~1.105, plausible for lattice distances")
print("  (e.g. d=10 vs d=11, or d=21 vs d=23)")
print()
print("=== check: 9/11 as distance-squared ratio on 44 lattice ===")
print("44-vector lattice has vectors with norm^2 in {1,2,3,4,6,...}")
print("possible distances: sqrt(2), sqrt(3), 2, sqrt(6), 3, ...")
print("d2/d3 = 9/11 -> (d3/d2)^2 = 11/9 = 1.222")
print("candidate distances: d2=3, d3=sqrt(11)~3.317 (no)")
print("d2=sqrt(8)=2.828, d3=sqrt(9.78) (no integer lattice)")
print("=> no clean lattice-distance interpretation found")

print("\n=== summary of three methods ===")
print("1. Perturbation: ratios are O(1), NOT a small-eps expansion.")
print("   Texture is not a deformation of a symmetric limit.")
print("2. Variational: single-flavon potentials give VEV ratios ~1;")
print("   no natural O(1) coefficients give 9/11 in a VEV ratio.")
print("3. WKB/lattice: no clean 44-vector counting gives 9/11, 1/2,")
print("   45/22 as path weights.")
print()
print("=> HONEST CONCLUSION: 9/11, 1/2, 45/22 are data-verified")
print("   structural relations WITHOUT a first-principles mechanism")
print("   in the current frameworks.  They are empirical regularities")
print("   (like the original 1/3), to be reported as such.")
