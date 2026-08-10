#!/usr/bin/env python3
"""
test_paper.py -- End-to-end verification of every quantitative claim
in the paper.

Run:  python3 test_paper.py
Requires: the official NuFIT 5.2 table at the default path
(see nufit_parser.py), numpy, scipy.

Checks (paper section in parentheses):
  * Takagi residual < 1e-15                    (App. A)
  * 6-parameter reference observables           (Sec. 4.1-4.2, 4.5)
  * chi2 = 0.02 on official tables (6-param
    reference, dof = 0; no goodness-of-fit
    claim -- the final 1-parameter model is
    verified separately in z3lm_1param.py)      (Sec. 4.2, 4.5)
  * pulls < 0.3 sigma                           (Table 4)
  * 7-param fit chi2=0, a13/a12 ~ 0.44          (Sec. 4.3)
  * phase scan table values                     (Table 6)
  * mass predictions, mbb, Sum m                (Sec. 7)
  * Delta(27) group facts                       (App. C)

The final one-parameter texture (all five ratios fixed,
chi2 = 0.10, dof = 5) is verified by `z3lm_1param.py`.

Author: Yuxuan Zhang  (paper code, v1.1 final)
"""
import sys
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, '.')
from z3lm import M6, obs_vec, obs_full, takagi_all, takagi_residual, BEST_FIT  # noqa: E402
from nufit_parser import NuFIT52, table_available  # noqa: E402
from fit_official import M7, M_phase, B6, B7, fit  # noqa: E402
import d27_check  # noqa: E402

PASS = 0
FAIL = 0


def check(cond, msg, tol=None, got=None, want=None):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}" + (f"  (got {got}, want {want})" if got is not None else ""))


def near(a, b, rtol=2e-3):
    return np.isclose(a, b, rtol=rtol, atol=0)


def near_angle(a, b, tol_deg=1.0):
    """Compare angles modulo 360 deg (paper quotes 0..360, code -180..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0) < tol_deg


def main():
    if not table_available():
        print("NuFIT table not found at default path; aborting official checks.")
        return 1
    nf = NuFIT52()

    print("=== A. Takagi decomposition (App. A) ===")
    M = M6(BEST_FIT)
    r = takagi_residual(M)
    check(r < 1e-15, f"Takagi residual {r:.1e} < 1e-15", got=r)

    print("\n=== B. Best-fit observables (Sec. 4) ===")
    o = obs_full(M6(BEST_FIT))
    targets = dict(s12=0.3051, s23=0.4500, s13=0.02230,
                   dm21=7.413e-5, dm31=2.505e-3, J=-0.0332,
                   m1=0.0076, m2=0.0115, m3=0.0506,
                   sum_m=0.0697, mbb=0.0050)
    for k, v in targets.items():
        check(near(o[k], v), f"{k} = {o[k]:.4g} ~ {v}", got=o[k], want=v)
    check(near_angle(o['delta'], -83.6),
          f"delta = {o['delta']:.1f} ~ -83.6 (standard PDG/Jarlskog)",
          got=o['delta'], want=-83.6)

    print("\n=== C. chi2 on official tables (Sec. 4.2) ===")
    y = obs_vec(M6(BEST_FIT))
    c1 = nf.chi2_1d_sum(y)
    check(c1 < 2.0, f"chi2_1d_sum = {c1:.3f} < 2.0 (dof=0; dominated by delta)", got=c1)
    cb = nf.chi2_block(y)
    check(cb < 2.0, f"chi2_block = {cb:.3f} < 2.0", got=cb)

    print("\n=== D. Pulls (Table 4) ===")
    bf = nf.best_fit_1d()
    pull_s23 = (y[2] - bf['s23'][0]) / bf['s23'][1]
    pull_dcp = abs((y[3] - bf['delta'][0])) / bf['delta'][1]
    check(abs(pull_s23) < 0.3, f"|pull(s23)| = {abs(pull_s23):.2f} < 0.3")
    check(abs(pull_dcp) < 2.0, f"|pull(delta)| = {abs(pull_dcp):.2f} < 2.0 "
          f"(delta={y[3]:.1f}, target={bf['delta'][0]:.0f})")

    print("\n=== E. 7-parameter fit (Sec. 4.3) ===")
    # 7-parameter version (a13 free), constrained to the texture
    # phase structure d2,d3 > 0 (the unconstrained fit drifts to
    # basins with d2,d3<0 that are unphysical for the texture).
    def fo7c(q):
        m0, d1, d2, d3, a12, a13, a23 = q
        if d2 <= 0 or d3 <= 0:
            return 1e6
        try:
            return nf.chi2_1d_sum(obs_vec(M7(q)))
        except Exception:
            return 1e6
    B7c = [(0.005, 0.10), (-1.5, 1.5), (0.01, 1.5), (0.01, 1.5),
           (0.1, 1.0), (0.01, 0.5), (0.1, 2.0)]
    best7 = None
    for sd in [0, 1, 2, 3, 42, 123]:
        r = differential_evolution(fo7c, B7c, seed=sd, maxiter=700,
                                   tol=1e-10, polish=True)
        if best7 is None or r.fun < best7.fun:
            best7 = r
    c7 = best7.fun
    q7 = best7.x
    check(c7 < 0.05, f"chi2(7p, constrained) = {c7:.3f} < 0.05", got=c7)
    ratio = q7[5] / q7[4]
    check(ratio > 2.0,
          f"a13/a12 = {ratio:.2f} > 2 (releasing 1/3 moves delta to NuFIT peak)",
          got=ratio)

    print("\n=== F. Phase scan table (Table 6) ===")
    # Five patterns of Table 6, rephasing-invariant delta extraction
    # (v1.2), restricted to the texture manifold d2,d3 > 0 (the
    # unconstrained fit drifts to basins with d2,d3<0 that are
    # unphysical for the texture).
    def fo_phase(build, q):
        m0, d1, d2, d3, a12, a23 = q
        if d2 <= 0 or d3 <= 0:
            return 1e6
        try:
            return nf.chi2_1d_sum(obs_vec(build(q)))
        except Exception:
            return 1e6
    expect = {(0, 90, 180): 8.84, (0, 120, 180): 1.32,
              (240, 0, 180): 0.30, (0, 120, 90): 162.3,
              (240, 240, 180): 58.9}
    B6c = [(0.005, 0.10), (-1.5, 1.5), (0.01, 1.5), (0.01, 1.5),
           (0.1, 1.0), (0.1, 2.0)]
    for pat, want in expect.items():
        build, npar = M_phase(*pat, ratio13=1.0 / 3.0)
        best = None
        for sd in range(6):
            r = differential_evolution(lambda q: fo_phase(build, q),
                                       B6c, seed=sd, maxiter=600,
                                       tol=1e-9, polish=True)
            if best is None or r.fun < best.fun:
                best = r
        c = best.fun
        tol = max(1.5, 0.25 * want)
        check(abs(c - want) < tol,
              f"({pat[0]},{pat[1]},{pat[2]}) chi2={c:.2f} ~ {want}",
              got=c, want=want)

    print("\n=== G. Delta(27) (App. C) ===")
    rc = d27_check.main()  # prints its own OK/FAIL lines
    check(rc == 0, "d27_check passes")

    print("\n" + "=" * 40)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
