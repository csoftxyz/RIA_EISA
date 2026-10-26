#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_predictions.py
=====================
Every number that appears in the paper, recomputed from the closed formulae.

Inputs (external, quoted):
  PDG 2024         |V_us| = 0.22500(67)
  NuFIT 5.2 (NO)   sin^2 th12 = 0.304(12), sin^2 th13 = 0.02225(60),
                   sin^2 th23 = 0.450(17)   [from the official dchi2 table
                   v52_clean_3d.txt shipped in ../nufit_official],
                   delta_CP = 230 deg (best fit of that table; published 232)
  NuFIT 5.2        dm2_21 = 7.42e-5 eV^2, |dm2_31| = 2.505e-3 eV^2

Internal (derived in the paper):
  eps_q = 1/6, C2 = 4/3, lambda0 = 2/9, lambda = 73/324
  a = eps_q^2 = 1/36, beta = sqrt(2/3) lambda0 = 2 sqrt6/27, delta = 240 deg
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- external
PDG_VUS, PDG_VUS_E = 0.22500, 0.00067
NUFIT = dict(
    s12=0.304, s12e=0.012,
    s13=0.02225, s13e=0.00060,
    s23=0.450, s23e=0.017,
    dcp=230.0, dcpe=35.0,
    dm21=7.42e-5, dm31=2.505e-3,
)
VUD_EXP, VUD_E = 0.97373, 0.00031

# ---------------------------------------------------------------- internal
lam0 = 2.0 / 9.0
lam = 73.0 / 324.0
epsq = 1.0 / 6.0
C2 = 4.0 / 3.0
a = 1.0 / 36.0
beta = np.sqrt(2.0 / 3.0) * lam0
delta = np.deg2rad(240.0)

# ---------------------------------------------------------------- TBM
nu1 = np.array([2, -1, -1]) / np.sqrt(6)
nu2 = np.array([1, 1, 1]) / np.sqrt(3)
nu3 = np.array([0, 1, -1]) / np.sqrt(2)
e1, e2, e3 = np.eye(3)
b = beta * np.exp(1j * delta)
A = 1 + a * a
B = A + abs(b) ** 2

# orthonormal triple: w1 normalised by sqrt(B), w2 by sqrt(A),
# w3 = normalised orthogonal complement of span{w1,w2}
w1 = (nu1 + a * nu2 + b * nu3) / np.sqrt(B)
w2 = (nu2 - a * nu1) / np.sqrt(A)
nu3p = nu3 - (np.conj(b) / B) * (nu1 + a * nu2 + b * nu3)
w3 = nu3p / np.linalg.norm(nu3p)

U = np.array([[e1 @ w1, e1 @ w2, e1 @ w3],
              [e2 @ w1, e2 @ w2, e2 @ w3],
              [e3 @ w1, e3 @ w2, e3 @ w3]], dtype=complex)

s13 = abs(U[0, 2]) ** 2
s12 = abs(U[0, 1]) ** 2 / (1 - s13)
s23 = abs(U[1, 2]) ** 2 / (1 - s13)
# PDG convention J = s12 c12 s23 c23 s13 c13^2 sin(delta); the phase of U_e3 is -delta
J = -np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))

# closed forms
Xi = 2 / np.sqrt(6) + a / np.sqrt(3)
kap = 1 / np.sqrt(6) - a / np.sqrt(3)
f13 = beta ** 2 * Xi ** 2 / (A * B)
f12 = (1 - np.sqrt(2) * a) ** 2 / (3 * A) / (1 - f13)
f23 = (A * A / 2 + beta ** 2 * kap ** 2 + np.sqrt(2) * A * beta * kap * np.cos(delta)) \
    / (A * B - beta ** 2 * Xi ** 2)

# neutrinos
m2 = np.sqrt(NUFIT["dm21"])
m3 = np.sqrt(NUFIT["dm31"])
sum_m = m2 + m3
mbeta = np.sqrt(s12 * m2 ** 2 + s13 * m3 ** 2)
mee_lo = abs(s12 * m2 - s13 * m3)
mee_hi = s12 * m2 + s13 * m3

# Cabibbo-derived
Vud = np.sqrt(1 - lam ** 2)
Vcb_over_lam = None  # not predicted

rows = []


def add(name, val, exp, err, fmt="{:.5f}"):
    pull = (val - exp) / err
    rows.append(dict(name=name, value=val, exp=exp, err=err, pull=pull))
    return pull


p13 = add(r"$\sin^2\theta_{13}$", s13, NUFIT["s13"], NUFIT["s13e"])
p12 = add(r"$\sin^2\theta_{12}$", s12, NUFIT["s12"], NUFIT["s12e"])
p23 = add(r"$\sin^2\theta_{23}$", s23, NUFIT["s23"], NUFIT["s23e"])
pdcp = add(r"$\delta_{CP}$ [deg]", 240.0, NUFIT["dcp"], NUFIT["dcpe"])
plam = add(r"$\lambda=|V_{us}|$", lam, PDG_VUS, PDG_VUS_E)
pvud = add(r"$|V_{ud}|$ (CKM unitarity)", Vud, VUD_EXP, VUD_E, )

lines = []
lines.append("=" * 78)
lines.append("PREDICTIONS  (zero continuous parameters in the mixing sector)")
lines.append("=" * 78)
lines.append(f"eps_q = 1/6, C2 = 4/3, lambda0 = 2/9, lambda = 73/324 = {lam:.8f}")
lines.append(f"a = 1/36, beta = 2 sqrt6/27 = {beta:.7f}, delta = 240 deg")
lines.append("")
lines.append(f"{'observable':<32}{'predicted':>12}{'experiment':>16}{'pull':>9}")
lines.append("-" * 78)
labels = [r"sin^2 theta_13", r"sin^2 theta_12", r"sin^2 theta_23",
          "delta_CP [deg]", "lambda = |V_us|", "|V_ud| (unitarity)"]
for lab, r in zip(labels, rows):
    lines.append(f"{lab:<32}{r['value']:>12.6f}{r['exp']:>10.5f}+-{r['err']:<5.5f}"
                 f"{r['pull']:>+9.2f}")
lines.append("")
lines.append(f"J_CP = {J:+.5f}   (experiment -0.0280 +- 0.0015 -> pull "
             f"{(J+0.0280)/0.0015:+.2f})")
lines.append(f"sum m_nu = {sum_m*1e3:.1f} meV,  m_beta = {mbeta*1e3:.2f} meV,  "
             f"|m_ee| in [{mee_lo*1e3:.2f}, {mee_hi*1e3:.2f}] meV")
lines.append("")
lines.append("closed forms vs explicit construction:")
lines.append(f"  s13: {s13:.12f} vs {f13:.12f}  (diff {s13-f13:+.2e})")
lines.append(f"  s12: {s12:.12f} vs {f12:.12f}  (diff {s12-f12:+.2e})")
lines.append(f"  s23: {s23:.12f} vs {f23:.12f}  (diff {s23-f23:+.2e})")
lines.append("")
lines.append("sum rule  sin^2 th23 = 1/2 + s13 cos(delta)/sqrt2:")
lines.append(f"  {0.5 + np.sqrt(s13)*np.cos(delta)/np.sqrt(2):.6f}  (exact {s23:.6f})")
lines.append("")
lines.append("arithmetic checks:")
lines.append(f"  lambda0*(1+1/72) = {lam0*(1+1/72):.12f}  = 73/324 = {73/324:.12f}")
lines.append(f"  gcd(73,324) = {np.gcd(73,324)};  324 = 2^2*3^4;  v_3(324) = 4")
lines.append(f"  16/729 = {16/729:.8f};  (4/9)*lambda0^2 = {4/9*lam0**2:.8f}")
lines.append("")
lines.append("refuted alternatives (same data):")
alt = [
    ("eps_nu3 = 1/12 -> s13", (1 / 12) ** 2 * 2 / 3 / (1 + (1 / 12) ** 2), 0.00060),
    ("[1/46,1/44] (interval)", 0.5 * (1 / 46 + 1 / 44), 0.00060),
    ("TM2: 1/(3 cos^2 th13)", 1 / (3 * (1 - NUFIT["s13"])), NUFIT["s12e"]),
    ("sin^2 th12 = 1/3 - lambda/9", 1 / 3 - lam / 9, NUFIT["s12e"]),
]
for name, val, err in alt:
    ref = NUFIT["s13"] if "s13" in name or "interval" in name else NUFIT["s12"]
    lines.append(f"  {name:<34} {val:>10.6f}  pull {(val-ref)/err:>+7.1f}")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(ROOT, "logs", "verify_predictions.log"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
with open(os.path.join(ROOT, "logs", "verify_predictions.json"), "w", encoding="utf-8") as f:
    json.dump(dict(rows=rows, J=J, sum_m=sum_m, mbeta=mbeta,
                   mee=[mee_lo, mee_hi], s13=s13, s12=s12, s23=s23), f, indent=1)
