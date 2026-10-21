#!/usr/bin/env python3
"""
recon_scan.py -- Parameter-vs-lattice reconnaissance.

Question: can the PLB lattice idea (44-vector orbits, NF dimension
ratios, Frobenius norms) fix the SIX parameters of the current Z3LM
texture, killing the free parameters?

Z3LM best-fit parameters (official NuFIT 5.2, lower octant):
    m0 = 0.02243 eV
    d1 = -0.2226,  d2 = 1.1539,  d3 = 1.4095
    a12 = 0.4456,  a23 = 0.9119,  a13 = a12/3 = 0.1485

PLB lattice invariants:
    NF dims: Dem=4, Hyb=24, Root=6   (44-vector lattice, S3xZ2 orbits)
    R_norm = 1/4 (Frobenius democracy ratio)
    D = 1/dim u(3) = 1/9 (dilution)
    eps_nu2 = R_norm*D = 1/36
    eps_nu3 = R_norm/|Z3| = 1/12
    eps_q   = sqrt(R_norm*D) = 1/6
    dims g0=12, g1=4, g2=3 (19D algebra)
    C2(SU3,3) = 4/3
    lambda = 73/324 = (2/9)(1+1/72)

We look for simple rational ratios among the texture parameters and
check whether they match the lattice invariant set.
"""
import numpy as np
from fractions import Fraction

m0, d1, d2, d3, a12, a23 = 0.02243, -0.2226, 1.1539, 1.4095, 0.4456, 0.9119
a13 = a12 / 3

print("=== Z3LM best-fit parameters ===")
print(f"m0={m0}, d1={d1}, d2={d2}, d3={d3}, a12={a12}, a23={a23}, a13={a13}")

print("\n=== Dimensionless ratios among texture parameters ===")
ratios = {
    "a23/a12": a23 / a12,
    "a13/a12": a13 / a12,
    "d2/d1": d2 / d1,
    "d3/d1": d3 / d1,
    "d3/d2": d3 / d2,
    "d2/d3": d2 / d3,
    "a12/|d1|": a12 / abs(d1),
    "a23/|d1|": a23 / abs(d1),
    "|d1|/a12": abs(d1) / a12,
    "(d2-d1)/(d3-d1)": (d2 - d1) / (d3 - d1),
    "d2+d3": d2 + d3,
    "(d2+d3)/a23": (d2 + d3) / a23,
    "m0*sqrt": None,
}
for k, v in ratios.items():
    if v is None:
        continue
    # nearest simple fraction (denominator <= 12)
    f = Fraction(v).limit_denominator(12)
    print(f"  {k:20s} = {v:10.5f}   ~ {f.numerator}/{f.denominator} = {float(f):.5f}"
          f"   (err {abs(v-float(f))/abs(v)*100:.2f}%)")

print("\n=== PLB lattice invariant set (for matching) ===")
inv = {
    "1/2": 0.5, "1/3": 1/3, "1/4": 0.25, "1/6": 1/6, "1/8": 0.125,
    "1/9": 1/9, "1/12": 1/12, "1/16": 1/16, "1/18": 1/18, "1/24": 1/24,
    "1/27": 1/27, "1/32": 1/32, "1/36": 1/36, "1/54": 1/54, "1/72": 1/72,
    "1/108": 1/108, "1/144": 1/144, "1/216": 1/216, "1/324": 1/324,
    "2/9": 2/9, "4/3": 4/3, "73/324": 73/324, "3/4": 0.75, "1/48": 1/48,
    "3/2": 1.5, "2/3": 2/3, "5/6": 5/6, "7/6": 7/6, "4/9": 4/9, "8/9": 8/9,
}
print("  invariants:", sorted(inv.keys()))

print("\n=== Match search: which texture ratios hit lattice invariants? ===")
all_ratios = {
    "a23/a12": a23 / a12,
    "d2/d3": d2 / d3,
    "d3/d2": d3 / d2,
    "d2/d1": d2 / d1,
    "d3/d1": d3 / d1,
    "|d1|/a12": abs(d1) / a12,
    "a12/a23": a12 / a23,
    "a23/a13": a23 / a13,
    "(d2+d3)/a23": (d2 + d3) / a23,
    "(d2+d3)/(2*a23)": (d2 + d3) / (2 * a23),
    "d1+a12+d3": d1 + a12 + d3,
    "(d2-d1)/(d3-d1)": (d2 - d1) / (d3 - d1),
}
for name, v in all_ratios.items():
    if abs(v) < 1e-9:
        continue
    # check reciprocal too
    hits = []
    for iname, iv in inv.items():
        if abs(v - iv) / abs(v) < 0.02:
            hits.append(f"{iname} (err {abs(v-iv)/abs(v)*100:.2f}%)")
        if abs(1 / v - iv) / abs(1 / v) < 0.02:
            hits.append(f"1/{iname} (reciprocal)")
    if hits:
        print(f"  {name:20s} = {v:10.5f}  ->  {', '.join(hits)}")
    else:
        print(f"  {name:20s} = {v:10.5f}  ->  (no lattice invariant within 2%)")

print("\n=== Raw fractional approximations (denominator <= 60) ===")
for name, v in all_ratios.items():
    if abs(v) < 1e-9:
        continue
    f = Fraction(v).limit_denominator(60)
    print(f"  {name:20s} = {v:10.5f}  ~  {f.numerator}/{f.denominator} "
          f"(err {abs(v-float(f))/abs(v)*100:.2f}%)")

print("\n=== Mass ratios (absolute scale m0) ===")
m1, m2, m3 = 0.0076, 0.0115, 0.0506
print(f"  m2/m1 = {m2/m1:.4f} ~ {Fraction(m2/m1).limit_denominator(12)}")
print(f"  m3/m2 = {m3/m2:.4f} ~ {Fraction(m3/m2).limit_denominator(12)}")
print(f"  m3/m1 = {m3/m1:.4f} ~ {Fraction(m3/m1).limit_denominator(12)}")
print(f"  m0/m1 = {m0/m1:.4f} ~ {Fraction(m0/m1).limit_denominator(12)}")
print(f"  m0/m2 = {m0/m2:.4f} ~ {Fraction(m0/m2).limit_denominator(12)}")
