"""Analytic origin of the single-S weaker-hierarchy preference (breakthrough B).

In the S=I congruence basis, single-S gives Mnu = vL I - (1/vR) MD MD^T with
MD = aA + bI (A antisymmetric).  Since A^T = -A the cross term cancels:
    MD MD^T = b^2 I - a^2 A^2.
A 3x3 antisymmetric A has A^2 eigenvalues {0, lambda, lambda} (one zero + a
repeated pair, the Pfaffian structure det A = 0).  Hence MD MD^T has a repeated
eigenvalue, and so does Mnu: the single-S neutrino matrix ALWAYS has only TWO
distinct eigenvalues (the pencil repeated root).  Its three Takagi masses are
therefore drawn from a two-parameter spectral family.

Breaking single-S (MR = vR I + S', S' independent symmetric) removes the repeated
root: Mnu generically has THREE distinct eigenvalues, a three-parameter family
that can realise a stronger hierarchy.

Observation (120+126 only): single-S Mnu has 2 distinct eigenvalues (the pencil
repeated root), broken-S has 3.  IMPORTANT CORRECTION (self-audit): this 3->2
reduction is NOT the mechanism of the weaker-hierarchy preference.  The control
content 10+126 has 3 distinct eigenvalues (no repeated root) yet shows the SAME
preference, and the preference persists in pure type-I (vL=0).  The preference is
caused by the single-S structure of M_R, not by the repeated root or antisymmetry.
This script records the 3->2 spectral observation as a property of 120+126, not as
an explanation of the general preference.

Verified: single-S 120+126 distinct-eigenvalue count = 2 (always), broken-S = 3.
"""
import numpy as np
import strict_obs as S


def n_distinct_eigs(M, tol=1e-6):
    e = np.linalg.eigvals(M)
    e = np.sort(np.abs(e))
    if e[-1] < 1e-12:
        return 1
    gaps = np.diff(e)
    return int((gaps > tol * e[-1]).sum()) + 1


def verify(n=600, seed=2):
    rng = np.random.default_rng(seed)
    s_cnt = []; b_cnt = []
    a2_zero = 0; a2_repeat = 0
    for _ in range(n):
        A = S.anti(rng.normal(size=6)); Sp = S.sym(rng.normal(size=12))
        # A^2 structure: one zero eigenvalue + repeated pair
        eA2 = np.linalg.eigvals(A @ A)
        if np.min(np.abs(eA2)) < 1e-8:
            a2_zero += 1
        gaps = [abs(eA2[i] - eA2[j]) for i in range(3) for j in range(i + 1, 3)]
        if min(gaps) < 1e-6 * max(np.max(np.abs(eA2)), 1e-12):
            a2_repeat += 1
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
        if abs(vR) < 1e-8:
            continue
        vL = kappa * su * sd / vR
        a = v1u - 3 * v15u; b = -3 * su
        MD = a * A + b * np.eye(3)
        Mnu_s = vL * np.eye(3) - MD @ np.linalg.inv(vR * np.eye(3)) @ MD.T
        Mnu_b = vL * np.eye(3) - MD @ np.linalg.inv(vR * np.eye(3) + Sp) @ MD.T
        s_cnt.append(n_distinct_eigs(Mnu_s))
        b_cnt.append(n_distinct_eigs(Mnu_b))
    s_cnt = np.array(s_cnt); b_cnt = np.array(b_cnt)
    return {"n": len(s_cnt),
            "singleS_distinct_mean": float(s_cnt.mean()),
            "singleS_always_two": bool(np.all(s_cnt == 2)),
            "brokenS_distinct_mean": float(b_cnt.mean()),
            "brokenS_generic_three": bool(np.mean(b_cnt == 3) > 0.95),
            "A2_has_zero_frac": a2_zero / n,
            "A2_has_repeat_frac": a2_repeat / n,
            "observation": "single-S 120+126 Mnu always has 2 distinct eigenvalues (pencil repeated root, "
                       "from A^2 eigs {0,lambda,lambda}); broken-S has 3. NOTE: this 3->2 reduction is "
                       "NOT the mechanism of the weaker-hierarchy preference -- 10+126 has 3 distinct "
                       "eigenvalues (no repeated root) yet the same preference, which persists in pure "
                       "type-I. The preference is caused by single-S M_R structure, not antisymmetry."}


if __name__ == "__main__":
    r = verify()
    print("SINGLE-S SPECTRAL OBSERVATION, 120+126 only (n=%d)" % r["n"])
    print("  A^2 has zero eigenvalue frac : %.3f" % r["A2_has_zero_frac"])
    print("  A^2 has repeated pair frac   : %.3f" % r["A2_has_repeat_frac"])
    print("  single-S distinct eigs mean  : %.3f  always-two=%s" % (r["singleS_distinct_mean"], r["singleS_always_two"]))
    print("  broken-S distinct eigs mean  : %.3f  generic-three=%s" % (r["brokenS_distinct_mean"], r["brokenS_generic_three"]))
    print("  OBSERVATION:", r["observation"])
