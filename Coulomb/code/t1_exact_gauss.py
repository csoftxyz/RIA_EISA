#!/usr/bin/env python3
"""
T1 -- Exactness criterion for the discrete Gauss law on a geometric mesh.

Theorem under test (Theorem 1):
  Shells [r_n, r_{n+1}], n = 0..N-1, r_{n+1}/r_n = q > 1, flux sphere of shell n at radius rho_n,
  unit charge at the origin, discrete Gauss law

      V_n - V_{n+1} = (r_{n+1} - r_n) / (4 pi rho_n^2).

  Then  V_n - V_N = (1/4pi)(1/r_n - 1/r_N)  for every n and every q
  if and only if  rho_n^2 = r_n r_{n+1}.

Verified:
  (A) geometric-mean rule: exact to machine precision for every q;
  (B) left-endpoint rule rho=r_n: coefficient exactly q (spurious charge renormalisation);
  (C) arithmetic-mean rule: coefficient 4q/(1+q)^2;
  (D) family rho = lambda*sqrt(r r'): coefficient exactly 1/lambda^2 (necessity);
  (E) random per-shell perturbations: residual ~ O(eps), no cancellation.
"""
import json
import math
import numpy as np

OUT = {}
R_MAX = 1e6


def mesh(q, r0=1.0, r_max=R_MAX):
    n = int(math.ceil(math.log(r_max / r0) / math.log(q)))
    return r0 * q ** np.arange(n + 1)


def coeff(q, rule, n_res=0):
    """Recovered charge coefficient from the (boundary-anchored) discrete potential."""
    r = mesh(q)
    if isinstance(rule, str):
        if rule == "geometric_mean":
            rho = np.sqrt(r[:-1] * r[1:])
        elif rule == "left":
            rho = r[:-1]
        elif rule == "right":
            rho = r[1:]
        elif rule == "arithmetic":
            rho = 0.5 * (r[:-1] + r[1:])
        else:
            raise ValueError(rule)
    else:
        rho = np.asarray(rule, dtype=float)
    d = (r[1:] - r[:-1]) / (4.0 * math.pi * rho ** 2)
    V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
    ref = (1.0 / r - 1.0 / r[-1]) / (4.0 * math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (V - V[-1]) / ref
    good = np.isfinite(ratio) & (ref > 0)
    return r, V, ref, ratio, good


print("=" * 80)
print("(A) geometric-mean rule  rho_n = sqrt(r_n r_{n+1})")
print(f"{'q':>10} {'n_shells':>9} {'max rel. deviation from 1':>28}")
A = {}
for q in [1.0001, 1.01, 1.05, 1.2, math.sqrt(3), 1.5, 2.0, 3.0, 5.0]:
    r, V, ref, ratio, good = coeff(q, "geometric_mean")
    e = float(np.max(np.abs(ratio[good] - 1)))
    A[f"{q:.6g}"] = e
    print(f"{q:10.6f} {len(r)-1:9d} {e:28.3e}")

print()
print("=" * 80)
print("(B) left-endpoint  rho=r_n  [prediction: coefficient = q]")
print(f"{'q':>10} {'coefficient':>18} {'q':>14} {'abs diff':>12}")
B = {}
for q in [1.01, 1.05, 1.2, math.sqrt(3), 1.5, 2.0, 3.0]:
    r, V, ref, ratio, good = coeff(q, "left")
    c = float(np.median(ratio[good]))
    B[f"{q:.6g}"] = c
    print(f"{q:10.6f} {c:18.12f} {q:14.9f} {abs(c-q):12.2e}")

print()
print("=" * 80)
print("(C) arithmetic mean  rho=(r_n+r_{n+1})/2  [prediction: coefficient = 4q/(1+q)^2]")
print(f"{'q':>10} {'coefficient':>18} {'4q/(1+q)^2':>16} {'abs diff':>12}")
C = {}
for q in [1.01, 1.05, 1.2, math.sqrt(3), 1.5, 2.0, 3.0]:
    r, V, ref, ratio, good = coeff(q, "arithmetic")
    c = float(np.median(ratio[good]))
    cf = 4 * q / (1 + q) ** 2
    C[f"{q:.6g}"] = c
    print(f"{q:10.6f} {c:18.12f} {cf:16.12f} {abs(c-cf):12.2e}")

print()
print("=" * 80)
print("(D) necessity: rho_n = lambda*sqrt(r_n r_{n+1})  [prediction: coefficient = lambda^-2]")
q = math.sqrt(3.0)
r = mesh(q)
base = np.sqrt(r[:-1] * r[1:])
D = {}
print(f"{'lambda':>8} {'coefficient':>18} {'lambda^-2':>16} {'abs diff':>12}")
for lam in [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 2.0]:
    rr, V, ref, ratio, good = coeff(q, lam * base)
    c = float(np.median(ratio[good]))
    D[f"{lam}"] = c
    print(f"{lam:8.3f} {c:18.12f} {1/lam**2:16.12f} {abs(c-1/lam**2):12.2e}")

print()
print("=" * 80)
print("(E) random per-shell perturbations  rho_n = sqrt(r_n r_{n+1})(1+delta_n)")
print(f"{'seed':>5} {'eps':>8} {'max|delta|':>12} {'max rel. deviation':>22} {'ratio to eps':>14}")
E = []
for seed in range(3):
    for eps in [1e-2, 1e-3, 1e-4]:
        rg = np.random.default_rng(seed)
        delta = eps * rg.uniform(-1, 1, size=len(base))
        rr, V, ref, ratio, good = coeff(q, base * (1 + delta))
        e = float(np.max(np.abs(ratio[good] - 1)))
        E.append({"seed": seed, "eps": eps, "err": e})
        print(f"{seed:5d} {eps:8.0e} {np.max(np.abs(delta)):12.3e} {e:22.3e} {e/eps:14.3f}")

print()
print("=" * 80)
print("SUMMARY")
print(f"  geometric-mean rule : max deviation from exact Coulomb = {max(A.values()):.3e}  (machine precision)")
print(f"  left-endpoint rule  : coefficient = q  -> spurious renormalisation factor 1+(q-1)")
print(f"  arithmetic-mean rule: coefficient = 4q/(1+q)^2, max |coeff-1| = "
      f"{max(abs(v-1) for v in C.values()):.3e}")
print(f"  lambda family       : coefficient = 1/lambda^2, unique lambda=1 gives 1")
print(f"  random perturbations: deviation/eps in [{min(x['err']/x['eps'] for x in E):.2f}, "
      f"{max(x['err']/x['eps'] for x in E):.2f}], no cancellation")

OUT = {"A_geometric_mean": A, "B_left": B, "C_arithmetic": C, "D_lambda": D, "E_random": E,
       "mesh_sizes": {f"{q:.6g}": len(mesh(q)) for q in [1.0001, 1.01, 1.05, 1.2, math.sqrt(3), 1.5, 2.0, 3.0, 5.0]}}
with open("logs/t1_exact_gauss.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote logs/t1_exact_gauss.json")
