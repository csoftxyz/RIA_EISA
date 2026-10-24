#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_all.py -- reproduce and assert every headline number of the manuscript.

Run:  python3 code/verify_all.py
Exit code 0 iff all checks pass.
"""
import importlib.util
import os
import sys
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40

spec = importlib.util.spec_from_file_location("eg", os.path.join(HERE, "exact_gauge.py"))
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)

checks = []


def check(name, value, target, rtol=1e-9, atol=0.0):
    ok = abs(value - target) <= max(atol, rtol * abs(target))
    checks.append((name, ok, value, target))
    return ok


def main():
    # --- pure gauge, F = 8 ------------------------------------------------
    check("Z(1) = sum I_n^8", float(eg.ZF(8)), 6.6224348046478, 1e-12)
    check("<e^{i theta_f}>(1)", float(eg.single_face(8)), 0.4488851447705, 1e-11)
    check("sigma_oct(1)", float(-mp.log(eg.single_face(8))), 0.8009882263, 1e-9)
    sinf = float(-mp.log(eg.In(1) / eg.In(0)))
    check("sigma_inf(1)", sinf, 0.8065623460, 1e-9)
    check("deficit / sigma_inf", (sinf - float(-mp.log(eg.single_face(8)))) / sinf,
          0.0069, 0.02)

    # --- exact Wilson loops ----------------------------------------------
    vals = {A: float(eg.wilson_split(8, A)) for A in (1, 2, 3, 4)}
    for A, tgt in [(1, 0.4488851448), (2, 0.2066157684),
                   (3, 0.1063616040), (4, 0.0791731320)]:
        check(f"W(A={A})", vals[A], tgt, 1e-9)
    # --- finite-volume family --------------------------------------------
    for F, tgt in [(4, 0.6828077378), (6, 0.7792268789), (8, 0.8009882263),
                   (12, 0.8063397593), (20, 0.8065619950)]:
        check(f"sigma_F(F={F})", float(-mp.log(eg.single_face(F))), tgt, 2e-8)

    # --- specific heat ----------------------------------------------------
    check("c(1)", float(eg.specific_heat(mp.mpf(1), 8)), 2.9326556, 1e-6)
    check("c(2.05) peak", float(eg.specific_heat(mp.mpf("2.05"), 8)), 5.74522, 1e-4)

    # --- Villain ----------------------------------------------------------
    beta_star = 4 / mp.pi
    check("beta_* = 4/pi", float(beta_star), 1.2732395447351627, 1e-12)
    F = 8
    zv1 = (2 * mp.pi) ** (-4) * mp.jtheta(3, 0, mp.e ** -4)

    def ZV(b):
        return (2 * mp.pi * b) ** (-mp.mpf(F) / 2) * mp.jtheta(
            3, 0, mp.e ** (-mp.mpf(F) / (2 * b)))

    C = mp.sqrt(2 * mp.pi / F) * beta_star ** F * mp.mpf(1) ** (mp.mpf(1) / 2 - F)
    check("Villain duality residual", float(abs(ZV(1) / (C * ZV(beta_star ** 2)) - 1)),
          0.0, atol=1e-30)

    # --- fermions ---------------------------------------------------------
    verts = [(a, s) for a in range(3) for s in (1, -1)]
    vid = {v: i for i, v in enumerate(verts)}
    G = np.zeros((6, 6))
    for i, u in enumerate(verts):
        for j, v in enumerate(verts):
            if j > i and u[0] != v[0]:
                s = 1 if (u[0] + 1) % 3 == v[0] else -1
                G[i, j] = s
                G[j, i] = -s
    check("rank Gamma", float(np.linalg.matrix_rank(G)), 2.0, 0.0)
    check("det Gamma", float(np.linalg.det(G)), 0.0, atol=1e-12)
    check("det(m=1 + Gamma)", float(np.linalg.det(np.eye(6) + G)), 13.0, 1e-12)
    check("det(m=0.3 + Gamma)", float(np.linalg.det(0.3 * np.eye(6) + G)), 0.097929, 1e-6)

    # --- finite-size asymptotic -------------------------------------------
    r = eg.In(1) / eg.In(0)
    sig = -mp.log(r)
    i21 = eg.In(2) / eg.In(1)
    bracket = 1 + i21 * r - 2 * r ** 2
    pref = mp.e ** (2 * sig) * bracket
    check("asymptotic prefactor", float(pref), 3.556548, 1e-6)
    ex8 = -mp.log(eg.single_face(8))
    as8 = sig - r ** 6 * bracket
    check("asym vs exact at F=8 (rel)", float(abs(as8 - ex8) / ex8), 0.0, atol=1e-4)
    check("asym vs exact at F=12 (rel)", float(abs((sig - r ** 10 * bracket)
          - (-mp.log(eg.single_face(12)))) / (-mp.log(eg.single_face(12)))), 0.0,
          atol=1e-7)

    # --- fermionic average ------------------------------------------------
    spec2 = importlib.util.spec_from_file_location(
        "fe", os.path.join(HERE, "fermion_exact.py"))
    fe = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fe)
    Zg, Zs = fe.Z_total_sectors(0.0, 1.0, Nf_pair=1)
    check("<|det D0|^2>", float(Zs.sum() / Zg), 53.180808, 1e-7)

    # --- auxiliary scripts (subprocess verdicts) ---------------------------
    import subprocess
    for script, need in (("dirac_operator_checks.py", "PASS"),
                         ("triangulation_uniqueness.py", "PASS")):
        p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           capture_output=True, text=True)
        ok = (p.returncode == 0 and need in p.stdout)
        checks.append((f"{script} verdict", ok, float(ok), 1.0))
        if not ok:
            print(p.stdout, p.stderr)

    # --- report -----------------------------------------------------------
    width = max(len(c[0]) for c in checks)
    nfail = 0
    print("=" * (width + 34))
    print(f"{'check':<{width}}  {'value':>16}  status")
    print("-" * (width + 34))
    for name, ok, val, tgt in checks:
        nfail += (not ok)
        print(f"{name:<{width}}  {val:>16.10g}  {'PASS' if ok else 'FAIL'}")
    print("-" * (width + 34))
    print(f"{len(checks) - nfail}/{len(checks)} checks passed")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
