#!/usr/bin/env python3
"""
T3 -- (a) The charge is exact in the discrete theory (divergence theorem);
      (b) the continuum-style estimator is biased on a cubic lattice;
      (c) the geometric-mean rule is exact on ARBITRARY refinement sequences, not just
          geometric ones.

(a) 3D cubic lattice, L^3 periodic box, unit source at the origin, V = lattice Green's function.
    For any region S containing the origin, sum_{edges crossing dS} (V_in - V_out) = 1.
    Finite-box neutrality requires adding back |S|/L^3 (uniform compensating background);
    this is done explicitly rather than hidden.
(b) The estimator Q(R) = 4 pi R^2 [Vbar(R) - Vbar(R+1)] on the same data: biased, oscillating.
(c) Same test on r_n = n^1.5, on r_n = exp(sqrt(n)), and on a random monotone sequence:
    exactness survives, because the criterion is per-shell and refers to no group.
"""
import json
import math
import numpy as np

OUT = {}
L = 96
print("=" * 84)
print(f"T3a: cubic lattice {L}^3, discrete Laplacian Green's function")
k = 2 * np.pi * np.fft.fftfreq(L)
KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
lap = 4 * (np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2 + np.sin(KZ / 2) ** 2)
lap[0, 0, 0] = 1.0
src = np.zeros((L, L, L))
src[0, 0, 0] = 1.0
V = np.real(np.fft.ifftn(np.fft.fftn(src) / lap))

idx = np.indices((L, L, L))
idx = np.where(idx > L // 2, idx - L, idx)
R = np.sqrt((idx ** 2).sum(axis=0))

print("\n(a) enclosed charge from the discrete flux through d{|x| <= n}")
print(f"{'n':>4} {'sites':>7} {'raw flux':>16} {'+|S|/L^3':>14} {'charge':>16}")
div = {}
for n in [2, 3, 5, 8, 12]:
    S = R <= n
    flux = 0.0
    for ax, s in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        flux += float(np.sum(V[S] - np.roll(V, -s, ax)[S]))
    corr = flux + S.sum() / L ** 3
    div[n] = {"raw": flux, "corr": corr, "sites": int(S.sum())}
    print(f"{n:4d} {S.sum():7d} {flux:16.12f} {S.sum()/L**3:14.9f} {corr:16.12f}")

print("\n(b) continuum-style estimator Q(R) = 4 pi R^2 [Vbar(R) - Vbar(R+1)] on the cubic lattice")
print(f"{'R':>4} {'Vbar(R)':>14} {'Q(R)':>12} {'Q(R)/1 - 1':>13}")
rows = []
for Rv in range(3, 16):
    m1, m2 = np.abs(R - Rv) < 0.5, np.abs(R - (Rv + 1)) < 0.5
    v1, v2 = float(V[m1].mean()), float(V[m2].mean())
    Q = 4 * math.pi * Rv ** 2 * (v1 - v2)
    rows.append({"R": Rv, "V": v1, "Q": Q})
    print(f"{Rv:4d} {v1:14.9f} {Q:12.6f} {Q-1:13.3e}")
Qs = np.array([r["Q"] for r in rows])
print(f"\n  mean Q over R=8..15 : {Qs[5:].mean():.4f}   (spread {Qs[5:].std():.4f})")
print("  -> an O(10%) bias with shell-structure oscillation, at R/a <= 15: not a small correction")

print("\n(c) exactness on arbitrary refinement sequences (geometric-mean flux radius)")
print(f"{'sequence':>26} {'n_shells':>9} {'max |Q/Q_exact - 1|':>22}")


def check(r):
    rho = np.sqrt(r[:-1] * r[1:])
    d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
    Vv = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
    ref = (1.0 / r - 1.0 / r[-1]) / (4 * math.pi)
    ratio = (Vv - Vv[-1]) / ref
    good = np.isfinite(ratio) & (ref > 0)
    return float(np.max(np.abs(ratio[good] - 1)))


n = np.arange(1, 400, dtype=float)
tests = {
    "r_n = n^1.5": n ** 1.5,
    "r_n = n^0.6": n ** 0.6,
    "r_n = exp(sqrt(n))": np.exp(np.sqrt(n)),
    "r_n = n^(1+0.3 sin n)": n ** (1 + 0.3 * np.sin(n)),
    "random monotone (log)": np.exp(np.cumsum(np.random.default_rng(7).uniform(0.01, 0.5, 399))),
}
C = {}
for name, r in tests.items():
    e = check(r)
    C[name] = e
    print(f"{name:>26} {len(r)-1:9d} {e:22.3e}")

OUT = {"divergence": {str(k): v for k, v in div.items()}, "estimator": rows,
       "estimator_mean_Q_R8_15": float(Qs[5:].mean()), "estimator_std": float(Qs[5:].std()),
       "arbitrary_meshes": C}
with open("logs/t3_lattice.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote logs/t3_lattice.json")
