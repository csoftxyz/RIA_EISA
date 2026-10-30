#!/usr/bin/env python3
"""Assert every headline number of the manuscript against logs/cpc_analysis.json.

Cards are labelled by the LaTeX label of the object they verify
(tab:bias, tab:radius, tab:exp, tab:lattice, tab:stab, fig:flow, ...), so the
output stays aligned with the PDF even when floats are reordered.

Exit status 0 if all checks pass.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(ROOT, "logs", "cpc_analysis.json")))

fails = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<56} got {got:.12g}  want {want:.12g}")
    if not ok:
        fails.append(name)


print("tab:bias -- closed-form bias (d = 3)")
A = D["A_bias"]
for q, ref in (("1.200000", 0.991736), ("1.732051", 0.928203), ("2.000000", 0.888889)):
    check(f"arithmetic mean at q={q}", A[q]["arith"], ref, 1e-6)
check("left endpoint at q=2", A["2.000000"]["left"], 2.0, 1e-12)
check("right endpoint at q=2", A["2.000000"]["right"], 0.5, 1e-12)
check("geometric rule at q=5", A["5.000000"]["geo"], 1.0, 1e-15)

print("  deep-mesh sweep (text: 'one part in 10^15', 2e5 shells)")
check("worst telescoping residual", A["deep_mesh_worst"], 0.0, 1e-15)
check("shells reached", float(A["deep_mesh_shells"]), 200000.0, 0.5)

print("tab:radius -- sampling radius and the naive formula")
B = D["B_radius"]
check("d=4 exact radius, q=1.1", B["d4_q1.1"]["exact"], 1.0484, 5e-4)
check("d=4 naive radius, q=1.1", B["d4_q1.1"]["naive"], 0.8321, 5e-4)
check("d=5 naive radius, q=1.1", B["d5_q1.1"]["naive"], 0.7963, 5e-4)
check("d=6 naive radius, q=1.1", B["d6_q1.1"]["naive"], 0.7939, 5e-4)
check("threshold q*, d=3", B["threshold_d3"], 1.0, 1e-4)
check("threshold q*, d=4 (golden ratio)", B["threshold_d4"], 1.6180, 5e-4)
check("threshold q*, d=5", B["threshold_d5"], 1.8393, 5e-4)
check("threshold q*, d=6", B["threshold_d6"], 1.9276, 5e-4)
assert B["d4_q1.1"]["exact_inside"] and not B["d4_q1.1"]["naive_inside"]
check("d=4 potential difference, exact radius", B["d4_check"]["V_exact_rho"], B["d4_check"]["V_exact"], 1e-15)
check("d=4 potential difference, naive radius (factor 2 off)",
      B["d4_check"]["V_naive_rho"] / B["d4_check"]["V_exact"], 2.0, 1e-12)

print("tab:exp -- log-log slopes of A_n vs r_n")
C = D["C_scaling"]
for key, pred in (("d3_p0.5", -0.5), ("d3_p1.0", 0.0), ("d3_p1.5", 0.5), ("d3_p2.0", 1.0),
                  ("d4_p1.0", -1.0), ("d4_p2.0", 0.0), ("d5_p1.0", -2.0), ("d5_p3.0", 0.0)):
    check(f"slope {key}", C[key]["slope"], pred, 1e-9)

print("tab:lattice -- lattice estimators on 96^3")
L = D["D_lattice"]
for n in ("2", "3", "5", "8", "12"):
    check(f"discrete flux charge, n={n}", L["flux"][n]["corr"], 1.0, 1e-10)
w = L["window_8_15"]
check("left-endpoint mean, R=8..15", w["left_mean"], 0.9002, 1e-3)
check("left-endpoint spread, R=8..15", w["left_std"], 0.0212, 1e-3)
check("exact-radius mean, R=8..15", w["exact_mean"], 0.9818, 1e-3)
check("exact-radius spread, R=8..15", w["exact_std"], 0.0285, 1e-3)
check("mean endpoint factor, R=8..15", w["factor_mean"], 0.9171, 1e-3)
check("factor x exact mean reproduces Q_left mean", w["product_check"], w["left_mean"], 1e-3)
check("endpoint factor at R=15", L["estimator"]["15"]["endpoint_factor"], 0.9375, 1e-12)

print("tab:stab -- stability of the radius formula (d = 4)")
E = D["E_stability"]
check("difference form error at q-1=1e-4", E["1e-04"]["difference"], 6.4e-14, 1e-14)
check("difference form error at q-1=1e-8", E["1e-08"]["difference"], 5.5e-10, 5e-11)
check("factored form error at q-1=1e-2", E["1e-02"]["factored"], 6.1e-17, 5e-17)
check("factored form error at q-1=1e-8", E["1e-08"]["factored"], 1.7e-17, 5e-17)

print("tab:window -- extrapolation windows")
F = D["F_windows"]
check("alpha_inferred, k<=4", F["window_(1, 2, 3, 4)"]["alpha"], 0.8957, 1e-3)
check("alpha_inferred, k=1,2,3,4,6,8", F["window_(1, 2, 3, 4, 6, 8)"]["alpha"], 0.9322, 1e-3)
check("alpha_inferred, k=3,4,6,8", F["window_(3, 4, 6, 8)"]["alpha"], 0.9876, 1e-3)
check("alpha_inferred, k=4,6,8", F["window_(4, 6, 8)"]["alpha"], 0.9912, 1e-3)
check("closed-form bias q_k-1 at k=4", F["qk"]["4"]["left_bias"], 0.189207, 1e-6)

print("Sec. 6.5 -- arbitrary refinement sequences")
for k, v in D["G_sequences"].items():
    check(f"telescoping residual, {k}", abs(v), 0.0, 1e-15)

print()
if fails:
    print(f"FAILED {len(fails)} check(s):", ", ".join(fails))
    sys.exit(1)
print(f"all checks passed ({len(D)} result blocks)")
