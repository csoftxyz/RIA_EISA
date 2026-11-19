"""Mining experiments for the pencil identity (audit-13 follow-up).

Three questions, all numerical and reproducible:

(7) GROUP-THEORETIC ORIGIN / UNIVERSALITY.  The pencil identity disc(Mnu S^-1)=0
    follows from Mnu = c1 S + c2 A S^-1 A with B=A S^-1 traceless rank-2.  The
    rank-2/traceless property comes from A being 3x3 ANTISYMMETRIC (det A=0), but
    the identity itself only needs M_R proportional to a SINGLE symmetric S
    (single-126) plus type-I seesaw.  Hence it should hold for EVERY single-S
    content (120+126, 10+126, 10+120+126), NOT just 120+126, and regardless of
    whether an antisymmetric 120 is present.  We test this across contents.

(6) BREAKING SCALE.  With M_R = vR S + eps S' (single-S broken by a second
    symmetric structure), how fast does the pencil residual grow with eps?
    We measure the scaling law.

(4) STATISTICAL CONSEQUENCE.  Do points satisfying the pencil identity (i.e. all
    single-S points) show a non-uniform distribution in PMNS observable space
    compared to generic non-single-S points?  If yes, the pencil has a statistical
    (not identically-vanishing) observable consequence.

All results are written to pencil_mining_results.json.
"""
import json
import os
import numpy as np
import strict_obs as S


def disc3(M):
    t1 = np.trace(M); t2 = (t1 * t1 - np.trace(M @ M)) / 2; t3 = np.linalg.det(M)
    return t1 * t1 * t2 * t2 - 4 * t2**3 - 4 * t1**3 * t3 - 27 * t3**2 + 18 * t1 * t2 * t3


def pencil_resid(Mnu, Sy):
    return abs(disc3(Mnu @ np.linalg.inv(Sy)))


def rel_resid(Mnu, Sy):
    # scale-invariant residual: |disc| / (typical eigenvalue scale)^6
    M = Mnu @ np.linalg.inv(Sy)
    sc = np.max(np.abs(np.linalg.eigvals(M)))
    return abs(disc3(M)) / max(sc**6, 1e-300)


# ---- (7) universality across single-S contents ----
def universality(n=300):
    rng = np.random.default_rng(2026)
    out = {}
    builders = {
        "120+126": (S.build, 34),
        "10+126": (S.build_10_126, 36),
        "10+120+126": (S.build_10_120_126, 50),
    }
    for name, (fn, npar) in builders.items():
        rr = []
        for _ in range(n):
            p = rng.normal(size=npar)
            Mu, Md, Me, Mnu = fn(p)
            # recover S from the builder: it is the symmetric 126 Yukawa.
            # For each content S is the matrix multiplying vR in M_R.
            if name == "120+126":
                Sy = S.sym(p[6:18])
            elif name == "10+126":
                Sy = S.sym(p[12:24])
            else:
                Sy = S.sym(p[18:30])
            if abs(np.linalg.det(Sy)) < 1e-8:
                continue
            rr.append(rel_resid(Mnu, Sy))
        out[name] = {"median_rel_resid": float(np.median(rr)),
                     "max_rel_resid": float(np.max(rr)),
                     "n": len(rr),
                     "identity_holds": bool(np.median(rr) < 1e-8)}
    # double-126 (two copies) should BREAK it
    rr = []
    for _ in range(n):
        p = rng.normal(size=54)
        Mu, Md, Me, Mnu = S.build_two_copies(p)
        Sy = S.sym(p[6:18])  # first 126
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        rr.append(rel_resid(Mnu, Sy))
    out["double-126 (broken)"] = {"median_rel_resid": float(np.median(rr)),
                                  "max_rel_resid": float(np.max(rr)),
                                  "n": len(rr),
                                  "identity_holds": bool(np.median(rr) < 1e-8)}
    return out


# ---- (6) breaking scale: M_R = vR S + eps S' ----
def breaking_scale(n=200):
    rng = np.random.default_rng(11)
    eps_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    med = {e: [] for e in eps_list}
    for _ in range(n):
        A = S.anti(rng.normal(size=6)); Sy = S.sym(rng.normal(size=12))
        Sp = S.sym(rng.normal(size=12))
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
        vL = kappa * su * sd / vR
        MD = (v1u - 3 * v15u) * A - 3 * su * Sy
        for e in eps_list:
            MR = vR * Sy + e * Sp
            Mnu = vL * Sy - MD @ np.linalg.inv(MR) @ MD.T
            med[e].append(rel_resid(Mnu, Sy))
    rows = [{"eps": e, "median_rel_resid": float(np.median(med[e]))} for e in eps_list]
    # fit power law log(resid) = a + b log(eps) on the small-eps range
    xs = np.log([r["eps"] for r in rows[:4]])
    ys = np.log([max(r["median_rel_resid"], 1e-300) for r in rows[:4]])
    b, a = np.polyfit(xs, ys, 1)
    return {"rows": rows, "power_law_exponent": float(b),
            "interpretation": "residual ~ eps^%.2f at small eps" % b}


# ---- (4) statistical consequence in PMNS space ----
def pmns_stats(n=4000):
    rng = np.random.default_rng(5)
    # single-S (pencil-satisfying) points: 120+126
    # non-single-S (pencil-violating) points: M_R = vR S + S' (extra structure)
    def pmns_obs(Mnu, Me):
        Le, _ = S.left_unitary(Me)
        Mm = Le.T @ Mnu @ Le
        P, mn = S.takagi(Mm)
        mn = np.sort(mn)[::-1]
        return np.array([abs(P[0, 1]), abs(P[0, 2]), abs(P[1, 2]),
                         S.jarlskog(P), mn[1] / mn[0], mn[2] / mn[0]])
    single = []; broken = []
    for _ in range(n):
        A = S.anti(rng.normal(size=6)); Sy = S.sym(rng.normal(size=12))
        Sp = S.sym(rng.normal(size=12))
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
        vL = kappa * su * sd / vR
        MD = (v1u - 3 * v15u) * A - 3 * su * Sy
        Me = (v1d - 3 * v15d) * A - 3 * sd * Sy
        Mnu_s = vL * Sy - MD @ np.linalg.inv(vR * Sy) @ MD.T
        Mnu_b = vL * Sy - MD @ np.linalg.inv(vR * Sy + Sp) @ MD.T
        single.append(pmns_obs(Mnu_s, Me))
        broken.append(pmns_obs(Mnu_b, Me))
    single = np.array(single); broken = np.array(broken)
    labels = ["|Ue2|", "|Ue3|", "|Umu3|", "J_PMNS", "m2/m1", "m3/m1"]
    comp = []
    for i, lab in enumerate(labels):
        comp.append({"observable": lab,
                     "single_mean": float(np.mean(single[:, i])),
                     "single_std": float(np.std(single[:, i])),
                     "broken_mean": float(np.mean(broken[:, i])),
                     "broken_std": float(np.std(broken[:, i])),
                     "mean_shift_in_std": float(abs(np.mean(single[:, i]) - np.mean(broken[:, i])) /
                                                max(np.std(broken[:, i]), 1e-12))})
    return {"n_single": len(single), "n_broken": len(broken), "comparison": comp,
            "note": "single-S vs broken-S at the SAME Yukawas/vevs (paired); "
                    "shift measured in units of the broken-sample std"}


def main():
    res = {"universality": universality(),
           "breaking_scale": breaking_scale(),
           "pmns_stats": pmns_stats()}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pencil_mining_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("=== (7) universality ===")
    for k, v in res["universality"].items():
        print("  %-22s holds=%s median_rel_resid=%.2e" % (k, v["identity_holds"], v["median_rel_resid"]))
    print("=== (6) breaking scale ===")
    for r in res["breaking_scale"]["rows"]:
        print("  eps=%.0e resid=%.2e" % (r["eps"], r["median_rel_resid"]))
    print("  power law:", res["breaking_scale"]["interpretation"])
    print("=== (4) PMNS stats (single vs broken, paired) ===")
    for c in res["pmns_stats"]["comparison"]:
        print("  %-8s single=%.3f broken=%.3f shift=%.2f std" %
              (c["observable"], c["single_mean"], c["broken_mean"], c["mean_shift_in_std"]))


if __name__ == "__main__":
    main()
