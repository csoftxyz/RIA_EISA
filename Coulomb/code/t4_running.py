#!/usr/bin/env python3
"""
T4 -- q-universality, and the anatomy of spurious running.

(1) With the geometric-mean rule the recovered charge is 1 to machine precision for every mesh
    ratio q, from q = 1+1e-4 to q = 10.  Nothing in the result refers to the group that selected q.
(2) With the left-endpoint rule the extracted charge is exactly q_k.  Under refinement
    q_k = 2^(1/k) this is an apparent coupling
        alpha_eff(k) = alpha q_k = alpha (1 + ln2/k + ...),
    a fake flow with a 1/k tail.  Fitting alpha(1 + c/k) to the first k_max points and
    extrapolating returns a wrong charge; the bias is reported.
"""
import json
import math
import numpy as np

OUT = {}
R_MAX = 1e6


def coeff(q, rule, n_min=8):
    r = 1.0 * q ** np.arange(int(math.ceil(math.log(R_MAX) / math.log(q))) + 1)
    if isinstance(rule, str):
        rho = {"geometric_mean": np.sqrt(r[:-1] * r[1:]),
               "left": r[:-1],
               "arithmetic": 0.5 * (r[:-1] + r[1:])}[rule]
    else:
        rho = rule
    d = (r[1:] - r[:-1]) / (4.0 * math.pi * rho ** 2)
    V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
    ref = (1.0 / r - 1.0 / r[-1]) / (4.0 * math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (V - V[-1]) / ref
    good = np.isfinite(ratio) & (ref > 0)
    return float(np.median(ratio[good]))


print("=" * 84)
print("(1) universality in q     (geometric-mean rule)")
print(f"{'q':>12} {'charge coefficient':>22} {'|c - 1|':>14}")
U = {}
for q in [1.0001, 1.001, 1.01, 1.05, 1.2, math.sqrt(3), 1.5, 2.0, 3.0, 5.0, 10.0]:
    c = coeff(q, "geometric_mean")
    U[f"{q:.6g}"] = c
    print(f"{q:12.6f} {c:22.15f} {abs(c-1):14.3e}")

print()
print("=" * 84)
print("(2) spurious running, refinement family q_k = 2^(1/k)")
print("    exact rule: alpha_eff/alpha = 1 identically;   left-endpoint rule: = q_k")
print(f"\n{'k':>3} {'q_k':>10} {'left rule':>14} {'1+ln2/k':>13} {'exact rule':>18}")
rows = []
for k in range(1, 9):
    q = 2 ** (1 / k)
    cl = coeff(q, "left")
    cg = coeff(q, "geometric_mean")
    rows.append({"k": k, "q": q, "left": cl, "geom": cg})
    print(f"{k:3d} {q:10.6f} {cl:14.9f} {1+math.log(2)/k:13.9f} {cg:18.13f}")

print("\n  Fit alpha_eff(k) = alpha (1 + c/k) to the first k_max points (truth: alpha = 1):")
print(f"  {'k_max':>6} {'fitted c':>11} {'alpha_inferred':>17} {'bias':>12} {'bias / 1e-4':>13}")
EX = {}
for kmax in [2, 3, 4, 6, 8]:
    kk = np.arange(1.0, kmax + 1.0)
    y = np.array([r["left"] for r in rows[:kmax]])
    X = np.vstack([np.ones_like(kk), 1.0 / kk]).T
    (a, c), *_ = np.linalg.lstsq(X, y, rcond=None)
    EX[kmax] = {"c": float(c), "alpha": float(a), "bias": float(a - 1)}
    print(f"{kmax:6d} {c:11.6f} {a:17.12f} {a-1:12.3e} {abs(a-1)/1e-4:13.1f}")

print(f"\n  analytic fake slope ln2 = {math.log(2):.10f}; fitted c at k_max=8 = {EX[8]['c']:.6f}")
print(f"  at k_max = 4 (a typical published refinement study) the inferred charge is wrong by "
      f"{abs(EX[4]['bias']):.2e},")
print(f"  i.e. {abs(EX[4]['bias'])/4e-5:.0f} times the 1e-4 level at which such agreements are typically quoted.")

OUT = {"universality": U, "refinement": rows, "extrapolation": EX, "ln2": math.log(2)}
with open("logs/t4_running.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote logs/t4_running.json")
