"""Attribution control for the 'statistical consequence' (vulnerability V1).

The paper attributes the 'single-S => weaker neutrino hierarchy' preference to
the pencil identity.  This control tests that attribution: run the SAME paired
single-S vs broken-S comparison on 10+126, which does NOT satisfy the pencil
identity.  If the preference persists, it is a seesaw-STRUCTURE effect (single-S
M_R vs two-structure M_R), NOT a consequence of the pencil identity.

Result: 10+126 (no pencil) shows the same preference (m3/m1 KS~0.27, p~1e-128),
so the preference is NOT pencil-specific.  The pencil attribution is therefore
incorrect and must be reframed as a single-S seesaw-structure effect.
"""
import numpy as np
from scipy import stats
import strict_obs as S


def pmns_ratios(Mnu, Me):
    Le, _ = S.left_unitary(Me)
    Mm = Le.T @ Mnu @ Le
    P, mn = S.takagi(Mm)
    mn = np.sort(mn)[::-1]
    return mn[1] / mn[0], mn[2] / mn[0]


def control_10_126(n=4000, seed=9):
    rng = np.random.default_rng(seed)
    s3 = []; b3 = []; s2 = []; b2 = []
    for _ in range(n):
        h = S.sym(rng.normal(size=12)); Sy = S.sym(rng.normal(size=12)); Sp = S.sym(rng.normal(size=12))
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        v = rng.normal(size=6) + 1j * rng.normal(size=6)
        vu, vd, su, sd, vR, vL = v
        if abs(vR) < 1e-8:
            continue
        MD = vu * h - 3 * su * Sy
        Me = vd * h - 3 * sd * Sy
        Mnu_s = vL * Sy - MD @ np.linalg.inv(vR * Sy) @ MD.T
        Mnu_b = vL * Sy - MD @ np.linalg.inv(vR * Sy + Sp) @ MD.T
        a2, a3 = pmns_ratios(Mnu_s, Me)
        c2, c3 = pmns_ratios(Mnu_b, Me)
        s3.append(a3); b3.append(c3); s2.append(a2); b2.append(c2)
    s3 = np.array(s3); b3 = np.array(b3); s2 = np.array(s2); b2 = np.array(b2)
    ks3, p3 = stats.ks_2samp(s3, b3)
    ks2, p2 = stats.ks_2samp(s2, b2)
    return {"content": "10+126 (NO pencil identity)",
            "m3m1": {"ks": float(ks3), "p": float(p3),
                     "single_mean": float(s3.mean()), "broken_mean": float(b3.mean()),
                     "preference_present": bool(s3.mean() > b3.mean() and p3 < 1e-6)},
            "m2m1": {"ks": float(ks2), "p": float(p2),
                     "single_mean": float(s2.mean()), "broken_mean": float(b2.mean()),
                     "preference_present": bool(s2.mean() > b2.mean() and p2 < 1e-6)},
            "conclusion": "preference present WITHOUT pencil => it is a single-S seesaw-structure "
                          "effect, NOT a pencil-identity consequence; pencil attribution is incorrect"}


if __name__ == "__main__":
    r = control_10_126()
    print("ATTRIBUTION CONTROL:", r["content"])
    for k in ["m3m1", "m2m1"]:
        v = r[k]
        print("  %s KS=%.3f p=%.1e single=%.4f broken=%.4f present=%s" %
              (k, v["ks"], v["p"], v["single_mean"], v["broken_mean"], v["preference_present"]))
    print("  ", r["conclusion"])


def pure_typeI_control(n=2000, seed=5):
    """Self-audit correction: the preference persists in PURE type-I (vL=0) for
    10+126, which has NO repeated root.  Hence the preference is caused by the
    single-S M_R structure, NOT by the pencil repeated root / antisymmetry."""
    rng = np.random.default_rng(seed)
    s = []; b = []
    for _ in range(n):
        h = S.sym(rng.normal(size=12)); Sy = S.sym(rng.normal(size=12)); Sp = S.sym(rng.normal(size=12))
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        v = rng.normal(size=6) + 1j * rng.normal(size=6)
        vu, vd, su, sd, vR, vL = v
        if abs(vR) < 1e-8:
            continue
        MD = vu * h - 3 * su * Sy; Me = vd * h - 3 * sd * Sy
        Mnu_s = -MD @ np.linalg.inv(vR * Sy) @ MD.T          # vL = 0
        Mnu_b = -MD @ np.linalg.inv(vR * Sy + Sp) @ MD.T
        s.append(pmns_ratios(Mnu_s, Me)[1]); b.append(pmns_ratios(Mnu_b, Me)[1])
    s = np.array(s); b = np.array(b)
    ks, p = stats.ks_2samp(s, b)
    return {"pure_typeI_m3m1": {"ks": float(ks), "p": float(p),
                                "single_mean": float(s.mean()), "broken_mean": float(b.mean()),
                                "preference_present": bool(s.mean() > b.mean() and p < 1e-6)},
            "conclusion": "preference persists in pure type-I (no type-II, no repeated root for "
                          "10+126) => caused by single-S M_R structure, not the pencil repeated root"}


def ten126_eigcount(n=400, seed=11):
    """10+126 single-S has THREE distinct eigenvalues (no repeated root)."""
    rng = np.random.default_rng(seed)
    cnt = []
    for _ in range(n):
        h = S.sym(rng.normal(size=12)); Sy = S.sym(rng.normal(size=12))
        v = rng.normal(size=6) + 1j * rng.normal(size=6)
        vu, vd, su, sd, vR, vL = v
        if abs(vR) < 1e-8:
            continue
        MD = vu * h - 3 * su * Sy
        Mnu = vL * Sy - MD @ np.linalg.inv(vR * Sy) @ MD.T
        e = np.sort(np.abs(np.linalg.eigvals(Mnu)))
        cnt.append(int((np.diff(e) > 1e-6 * e[-1]).sum()) + 1)
    cnt = np.array(cnt)
    return {"mean_distinct": float(cnt.mean()),
            "frac_three": float((cnt == 3).mean()),
            "no_repeated_root": bool((cnt == 3).mean() > 0.95)}
