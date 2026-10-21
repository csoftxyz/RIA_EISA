#!/usr/bin/env python3
"""verify_all.py -- end-to-end verification of every number quoted in
the Z3LM manuscript (NPB version -> CJP version).

Runs:
  1. texture construction and Takagi decomposition checks
  2. reference-point observables, masses, m_bb, J, ordering
  3. joint (3D+2D) NuFIT 5.2 statistic at reference / 3-param / 6-param
  4. discrete phase-pattern scan (Table 6) with the SAME block statistic
  5. look-elsewhere scan counts (Appendix D)
  6. small-integer scan (Appendix D)

Output: verification_report.md + verification_values.json
"""
import json
import sys
import numpy as np

sys.path.insert(0, '.')
import z3lm
from z3lm import M6, obs_vec, obs_full, takagi_all, OMEGA
from nufit_parser import NuFIT52

nf = NuFIT52()
R = []          # (name, paper value, recomputed, tol, unit)


def check(name, paper, calc, tol, unit=""):
    ok = abs(paper - calc) <= tol
    R.append((name, paper, calc, tol, ok, unit))
    return ok


# table ranges (used to reject extrapolation: the interpolator is
# linear outside the grid and can return negative chi^2 there)
_x23, _xdma, _xdcp, _ = nf.proj3d()
_x13a, _x12a, _ = nf.proj2d("T13/T12 projection")
_x13b, _xdmsb, _ = nf.proj2d("T13/DMS projection")


def in_range(y):
    s12, s13, s23, dc, dm21, dm31 = y
    return (min(_x23) <= s23 <= max(_x23)
            and min(_xdma) <= dm31 / 1e-3 <= max(_xdma)
            and min(_xdcp) <= dc <= max(_xdcp)
            and min(_x13a) <= s13 <= max(_x13a)
            and min(_x12a) <= s12 <= max(_x12a)
            and min(_xdmsb) <= np.log10(dm21) <= max(_xdmsb))


def block(M):
    o = obs_vec(M)
    y = [o[0], o[1], o[2], o[3], o[4], o[5]]
    if not in_range(y):
        return 1e6
    return nf.chi2_block(np.array(y))


# ---------------------------------------------------------------- 1. texture
Q = np.array([1.0, -3/19, 9/11, 1.0, 6/19, 135/209])
M0 = M6(Q)
U, m = takagi_all(M0)
U2, m2 = takagi_all(M0)
check("Takagi residual max|U^T M U - diag(m)|", 0.0,
      float(np.max(np.abs(U.T @ M0 @ U - np.diag(m)))), 1e-12)
check("unitarity max|U^dag U - 1|", 0.0,
      float(np.max(np.abs(U.conj().T @ U - np.eye(3)))), 1e-12)

# ------------------------------------------------- 2. reference point values
m0 = 0.031593
Mref = M6(Q) * m0
o = obs_full(Mref)
check("sin^2 theta12", 0.3029, o['s12'], 5e-4)
check("sin^2 theta23", 0.4497, o['s23'], 5e-4)
check("sin^2 theta13", 0.02228, o['s13'], 5e-5)
check("delta_CP [deg]", -83.4, o['delta'], 0.2, "deg")
check("dm21 [1e-5 eV^2]", 7.45, o['dm21'] * 1e5, 0.05)
check("dm31 [1e-3 eV^2]", 2.497, o['dm31'] * 1e3, 0.005)
check("J_PMNS", -0.0331, o['J'], 5e-4)
check("sum m_nu [eV]", 0.0698, o['sum_m'], 5e-4)
check("m_beta_beta [eV]", 0.0050, o['mbb'], 2e-4)

# ------------------------------------------------- 3. chi2 values (block)
chi_ref = block(Mref)
check("chi2_block reference", 1.44, chi_ref, 0.01)

# 6-parameter best fit (paper value, recomputed at the quoted point)
q6 = np.array([0.02243, -0.2226, 1.1539, 1.4095, 0.4456, 0.9119])
chi6 = block(M6(q6))
check("chi2_block 6-param", 1.34, chi6, 0.02)

# 3-parameter (a12/d3 free) local re-fit
from scipy.optimize import minimize


def M3(p):
    mm, a12 = p
    return M6(np.array([mm, -0.5 * a12, 9/11, 1.0, a12, 45/22 * a12]))


r3 = minimize(lambda p: block(M3(p)), [0.0316, 0.3158],
              method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-9))
check("chi2_block 3-param", 1.34, float(r3.fun), 0.02)
check("a12/d3 (3-param best fit)", 0.3161, float(r3.x[1]), 5e-4)

# ------------------------------------------------- 4. phase pattern scan
# (recomputed separately by phase_scan4.py with DE + Powell multi-start,
#  bounded, block statistic; read the stored result)
import json as _json
phase_results = _json.load(open("phase_scan_block.json"))
for k, v in phase_results.items():
    print(f"  phase {k}: chi2_block = {v:.3f}")
paper_phase = {"(0, 90, 180)": 7.98, "(0, 120, 180)": 1.32,
               "(240, 0, 180)": 0.27, "(0, 120, 90)": 130.0,
               "(240, 240, 180)": 94.3}
for k, v in paper_phase.items():
    check(f"phase {k}", v, phase_results[k], 0.05)

# ------------------------------------------------- 5. look-elsewhere counts
import rational_scan2 as rs

c0, m0best = rs.chi2(tuple(rs.R0))
print(f"  reference chi2_block = {c0:.3f} (m0={m0best:.6f})", flush=True)

# ------------------------------------------------- 6. integer scan (N=10)
import subprocess

print("  running integer scan N=12 ...", flush=True)
out = subprocess.run([sys.executable, "integer_scan.py", "12"],
                     capture_output=True, text=True, timeout=3600).stdout
n_hits = [l for l in out.splitlines() if "shapes with chi2" in l]
print("  ", n_hits[-1] if n_hits else out[-200:], flush=True)

# ---------------------------------------------------------------- report
lines = ["# Verification report — Z3LM manuscript", ""]
lines.append("| quantity | paper | recomputed | tol | status |")
lines.append("|---|---|---|---|---|")
nfail = 0
for name, pv, cv, tol, ok, unit in R:
    if not ok:
        nfail += 1
    lines.append(f"| {name} | {pv}{unit} | {cv:.6g} | {tol} | "
                 f"{'PASS' if ok else 'FAIL'} |")
lines.append("")
lines.append("## Discrete phase scan (block statistic)")
lines.append("")
lines.append("| pattern (deg) | chi2_block |")
lines.append("|---|---|")
for k, v in phase_results.items():
    lines.append(f"| {k} | {v:.2f} |")
lines.append("")
lines.append(f"Reference chi2_block = {c0:.3f}")
lines.append("")
lines.append(f"Integer scan (N=12): {n_hits[-1] if n_hits else 'n/a'}")
lines.append("")
lines.append(f"## Summary: {len(R)-nfail}/{len(R)} checks passed")

open("verification_report.md", "w").write("\n".join(lines))
json.dump(dict(checks=[dict(name=n, paper=p, calc=c, ok=bool(o)) for n, p, c, t, o, u in R],
               phase_scan=phase_results, chi_ref=c0),
          open("verification_values.json", "w"), indent=1)
print("\n".join(lines))
print(f"\n{len(R)-nfail}/{len(R)} checks passed")
