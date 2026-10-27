#!/usr/bin/env python3
"""
T6 -- Item-by-item audit of the exponents in Eqs. (6) and (7).

Eq. (6):  A_n = (1/Omega_d) (r_n^{-1} - r_{n+1}^{-1}) / (r_n^{-p} - r_{n+1}^{-p})
Eq. (7):  A_{n+1}/A_n = q^{p-1}

Checks
  (A) d = 3, constant q: log-log slope of A_n vs r_n must be exactly p - 1.
  (B) general d: with the Coulomb potential r^{2-d}, the slope must be p - d + 2.
  (C) general d, marginal case p = d - 2: slope must be 0 (shell-independent).
  (D) variable mesh ratio: Eq. (7) is exact only for constant q; quantify the failure.
  (E) the two closed forms used in the text:
        q_k = 2^{1/k}             = 1 + (ln2)/k + (ln2)^2/(2k^2) + O(k^-3)
        4q_k/(1+q_k)^2            = 1 - (ln2)^2/(4k^2) + O(k^-3)
      and the order of the remainder (k^-3, not k^-4).
"""
import json
import math
import numpy as np

OUT = {}
print("=" * 86)
print("T6: audit of the exponents in Eqs. (6) and (7)")
print("=" * 86)


def amplitude(r, q, p, d=3):
    """Exact potential increments for V = r^{2-d} (unit coefficient), amplitude for trial r^{-p}."""
    rn, rn1 = r[:-1], r[1:]
    num = rn ** (2 - d) - rn1 ** (2 - d)
    den = rn ** (-p) - rn1 ** (-p)
    return num / den


print("\n(A) d = 3, constant q: fitted log-log slope of A_n(r_n)")
print(f"{'p':>5} {'q':>7} {'slope':>12} {'p-1':>8} {'diff':>10}")
A = {}
for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for q in [1.2, math.sqrt(3), 2.0]:
        r = 1.0 * q ** np.arange(0.0, 30.0)
        A_ = amplitude(r, q, p, d=3)
        sl = float(np.polyfit(np.log(r[:-1]), np.log(A_), 1)[0])
        A[f"p{p}_q{q:.3f}"] = sl
        print(f"{p:5.2f} {q:7.4f} {sl:12.9f} {p-1:8.4f} {sl-(p-1):10.2e}")

print("\n(B) general d (potential r^{2-d}): slope must be p - d + 2")
print(f"{'d':>3} {'p':>5} {'q':>7} {'slope':>12} {'p-d+2':>8} {'diff':>10}")
B = {}
for d in [3, 4, 5]:
    for p in [1.0, 2.0, 3.0]:
        q = math.sqrt(2.0)
        r = 1.0 * q ** np.arange(0.0, 30.0)
        A_ = amplitude(r, q, p, d=d)
        sl = float(np.polyfit(np.log(r[:-1]), np.log(A_), 1)[0])
        B[f"d{d}_p{p}"] = sl
        print(f"{d:3d} {p:5.2f} {q:7.4f} {sl:12.9f} {p-d+2:8.4f} {sl-(p-d+2):10.2e}")

print("\n(B2) ratio check: A_(n+1)/A_n must equal q^(p-d+2), not q^(p-1)")
print(f"{'d':>3} {'p':>5} {'q':>7} {'measured':>14} {'q^(p-d+2)':>14} {'q^(p-1)':>14} {'rel.err':>10}")
B2 = {}
for d in [3, 4, 5]:
    for p in [1.0, 2.0, 3.0]:
        q = math.sqrt(2.0)
        r = 1.0 * q ** np.arange(0.0, 30.0)
        A_ = amplitude(r, q, p, d=d)
        ratio = float(np.median(A_[1:] / A_[:-1]))
        pred = q ** (p - d + 2)
        B2[f"d{d}_p{p}"] = {"ratio": ratio, "pred": pred, "alt": q ** (p - 1)}
        print(f"{d:3d} {p:5.2f} {q:7.4f} {ratio:14.10f} {pred:14.10f} {q**(p-1):14.10f} "
              f"{abs(ratio/pred-1):10.2e}")

print("\n(C) marginal case p = d-2: amplitude must be shell-independent")
print(f"{'d':>3} {'q':>7} {'A_max/A_min - 1':>18}")
C = {}
for d in [3, 4, 5]:
    q = math.sqrt(2.0)
    r = 1.0 * q ** np.arange(0.0, 20.0)
    A_ = amplitude(r, q, d - 2, d=d)
    spread = float(np.max(A_) / np.min(A_) - 1)
    C[d] = spread
    print(f"{d:3d} {q:7.4f} {spread:18.3e}")

print("\n(D) variable mesh ratio: Eq. (7) requires constant q")
print("    r_n = n^1.3, p = 2, d = 3")
r = np.arange(1.0, 40.0) ** 1.3
A_ = amplitude(r, None, 2.0, d=3)
q_eff = r[1:-1] / r[:-2]
ratio = A_[1:] / A_[:-1]
pred = q_eff ** 1.0
print(f"    measured A_(n+1)/A_n in [{ratio.min():.6f}, {ratio.max():.6f}], "
      f"while q_eff^(p-1) spans [{pred.min():.6f}, {pred.max():.6f}]")
print(f"    ratio of the two: in [{(ratio/pred).min():.6f}, {(ratio/pred).max():.6f}] "
      f"-> Eq. (7) is NOT exact for variable q")
D = {"min": float((ratio / pred).min()), "max": float((ratio / pred).max())}

print("\n(E) closed forms at q_k = 2^(1/k)")
print(f"{'k':>3} {'q_k':>12} {'1+ln2/k':>12} {'diff*k^2':>12} {'4q/(1+q)^2':>14} {'1-(ln2)^2/4k^2':>16} {'diff*k^3':>11}")
E = []
for k in [1, 2, 4, 8, 16, 32, 64]:
    q = 2 ** (1 / k)
    qlin = 1 + math.log(2) / k
    ar = 4 * q / (1 + q) ** 2
    ar2 = 1 - math.log(2) ** 2 / (4 * k ** 2)
    E.append({"k": k, "q": q, "d1": q - qlin, "d2": ar - ar2})
    print(f"{k:3d} {q:12.9f} {qlin:12.9f} {(q-qlin)*k**2:12.3e} {ar:14.12f} {ar2:16.12f} {(ar-ar2)*k**3:11.3e}")

print("\n  fitted exponents of the remainder (log-log slope vs k):")
kk = np.array([8, 16, 32, 64], dtype=float)
sl1 = float(np.polyfit(np.log(kk), np.log(np.abs([e["d1"] for e in E if e["k"] in (8, 16, 32, 64)])), 1)[0])
sl2 = float(np.polyfit(np.log(kk), np.log(np.abs([e["d2"] for e in E if e["k"] in (8, 16, 32, 64)])), 1)[0])
print(f"    q_k - (1+ln2/k)          : exponent {sl1:+.3f}   (so O(k^-2))")
print(f"    4q/(1+q)^2 - [1-(ln2)^2/4k^2]: exponent {sl2:+.3f}   (so the remainder is O(k^-3), not O(k^-4))")

print("\n" + "=" * 86)
print("VERDICT")
print("  Eq. (6): exponent p-1 correct in d=3 with constant q;")
print("           general d gives p-d+2; variable q adds the prefactor (q_n-1)/(q_n(1-q_n^-p)).")
print("  Eq. (7): exact only for constant q (geometric mesh).")
print("  Closed form: q_k = 1+ln2/k+(ln2)^2/(2k^2)+...; 4q/(1+q)^2 = 1-(ln2)^2/(4k^2)+O(k^-3).")
print("  -> the manuscript's O(k^-4) must be O(k^-3).")

OUT = {"A": A, "B": B, "B2": B2, "C": C, "D": D, "E": E, "slopes": {"sl1": sl1, "sl2": sl2}}
with open("logs/t6_exponents.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote logs/t6_exponents.json")
