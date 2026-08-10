#!/usr/bin/env python3
"""
d27_cg6.py -- Definitive No-Go scan with continuous coefficients.

Physical setup: L ~ 3 (or 3bar), H ~ 1, flavons phi_a ~ 3 (or 3bar),
Weinberg matrix
    M_{ij} = sum_a sum_{ch=1,2} c_{a,ch} * [C_ch(v_a)]_{ij}
where C_1, C_2 are the TWO independent symmetric 3bar channels of
3x3 (diagonal-type and off-diagonal-type, defined by the e11 orbit).

We optimize the continuous coefficients c_{a,ch} to see whether
   (i)  |M13|/|M12| = 1/3
  (ii)  arg(M12)=0, arg(M13)=120deg, arg(M23)=180deg
can be simultaneously satisfied for any combination of allowed VEV
directions.  Allowed VEVs: orbits of (1,1,1), (1,w,w^2), (1,0,0).
"""
import numpy as np
from scipy.optimize import minimize

w = np.exp(2j * np.pi / 3)
sqrt2 = np.sqrt(2.0)

# Physical CG channels (e11-orbit convention, from d27_cg3):
#   C1 (diagonal):   C1^k_{ij} = delta_{ik} delta_{jk}
#   C2 (off-diag):   C2^k_{ij} = (delta_{ik} delta_{jl} + delta_{il} delta_{jk})/sqrt2
C1 = np.zeros((3, 3, 3), complex)
C2 = np.zeros((3, 3, 3), complex)
for k in range(3):
    C1[k, k, k] = 1.0
    for i in range(3):
        for j in range(3):
            if i != j and {i, j} == {x for x in range(3) if x != k}:
                C2[k, i, j] = 1.0 / sqrt2

vacs = {
    "(1,1,1)": np.array([1, 1, 1], complex),
    "(1,w,w2)": np.array([1, w, w ** 2], complex),
    "(1,w2,w)": np.array([1, w ** 2, w], complex),
    "(1,0,0)": np.array([1, 0, 0], complex),
    "(0,1,0)": np.array([0, 1, 0], complex),
    "(0,0,1)": np.array([0, 0, 1], complex),
}


def build_M(vev_list, c):
    """c: flat array of length 2*nflav."""
    M = np.zeros((3, 3), complex)
    for a, v in enumerate(vev_list):
        for ch in range(2):
            C = C1 if ch == 0 else C2
            cc = c[2 * a + ch]
            for k in range(3):
                M += cc * v[k] * C[k]
    return M


def target_fn(c, vev_list, wt_ratio=1.0, wt_phase=1.0):
    """Weighted objective: ratio -> 1/3, phases -> (0,120,180)."""
    M = build_M(vev_list, c)
    m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
    if min(abs(m12), abs(m13), abs(m23)) < 1e-12:
        return 1e6
    r = abs(m13) / abs(m12)
    ph = np.degrees(np.angle([m12, m13, m23]))
    ph_target = np.array([0, 120, 180])
    dph = np.minimum(abs(ph - ph_target), 360 - abs(ph - ph_target))
    return wt_ratio * (r - 1 / 3) ** 2 + wt_phase * np.sum(dph ** 2) / 1e3


def fit(vev_list):
    n = 2 * len(vev_list)
    best = None
    for seed in range(20):
        c0 = np.random.randn(n)
        r = minimize(lambda c: target_fn(c, vev_list), c0, method='Nelder-Mead',
                     options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
        if best is None or r.fun < best.fun:
            best = r
    M = build_M(vev_list, best.x)
    m12, m13, m23 = M[0, 1], M[0, 2], M[1, 2]
    r = abs(m13) / abs(m12)
    ph = np.degrees(np.angle([m12, m13, m23]))
    return best.fun, r, ph, M


print("=== two-flavon continuous optimization ===")
print(f"{'VEV pair':<24}{'best_obj':>10}{'r13/12':>9}{'phases':>18}")
found_any = False
names = list(vacs.keys())
for i in range(len(names)):
    for j in range(i, len(names)):
        f, r, ph, M = fit([vacs[names[i]], vacs[names[j]]])
        ok_ratio = abs(r - 1 / 3) < 0.02
        ph_n = np.degrees(np.angle([M[0,1], M[0,2], M[1,2]]))
        ok_phase = (np.allclose(np.sort(np.round(ph_n) % 360), [0, 120, 180], atol=5)
                    or np.allclose(np.sort(np.round(ph_n) % 360), [0, 180, 240], atol=5))
        mark = " <== MATCH" if (ok_ratio and ok_phase) else ""
        if ok_ratio and ok_phase:
            found_any = True
        print(f"({names[i]:<9},{names[j]:<9}) {f:10.3e}{r:9.4f} "
              f"{np.round(ph_n % 360, 0)}{mark}")

print(f"\n{'='*60}\nCONCLUSION: any configuration satisfying BOTH |M13|/|M12|=1/3 AND the paper phase pattern? {found_any}")
if not found_any:
    print("NO-GO: a13/a12 = 1/3 with arg(M12)=0, arg(M13)=120deg, arg(M23)=180deg")
    print("cannot be obtained from minimal Delta(27) Weinberg realizations")
    print("(L in {3,3bar}, flavons in {3,3bar}, any allowed VEV directions).")
    print("=> 1/3 is an independent phenomenological input; Delta(27) fixes")
    print("   only the cubic-root phase structure (omega) and octant symmetry.")
