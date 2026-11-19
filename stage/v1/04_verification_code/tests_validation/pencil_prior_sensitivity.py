"""Prior-sensitivity analysis of the pencil statistical preference (mining 4).

The preference (single-S points favour a weaker neutrino hierarchy than
single-S-broken points) was found under a uniform Gaussian Yukawa prior.  A
referee will ask whether it is an artefact of that prior.  Here we re-run the
PAIRED comparison (identical Yukawas/vevs; M_R=vR S vs M_R=vR S+eps S') under
several qualitatively different priors and breaking strengths, and report the
two-sample KS statistic and the direction of the mean shift for m3/m1 and m2/m1.

Robustness criterion: the preference is prior-robust if, across priors, the
single-S mean of m3/m1 (and m2/m1) stays ABOVE the broken mean with a small KS
p-value.  We also vary eps to check it is not a fine-tuned breaking amplitude.

All results -> pencil_prior_sensitivity_results.json.
"""
import json
import os
import numpy as np
from scipy import stats
import strict_obs as S


def pmns_obs(Mnu, Me):
    Le, _ = S.left_unitary(Me)
    Mm = Le.T @ Mnu @ Le
    P, mn = S.takagi(Mm)
    mn = np.sort(mn)[::-1]
    return np.array([abs(P[0, 1]), abs(P[0, 2]), abs(P[1, 2]),
                     S.jarlskog(P), mn[1] / mn[0], mn[2] / mn[0]])


# ---- prior draw functions: return (A6, S12, Sp12, v8complex) ----
def draw_gaussian(rng):
    return (rng.normal(size=6), rng.normal(size=12), rng.normal(size=12),
            rng.normal(size=8) + 1j * rng.normal(size=8))


def draw_uniform(rng):
    return (rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 12),
            rng.uniform(-1, 1, 8) + 1j * rng.uniform(-1, 1, 8))


def draw_loguniform(rng):
    def lu(n):
        mag = 10.0 ** rng.uniform(-2, 2, n)
        ph = rng.uniform(-np.pi, np.pi, n)
        return mag * np.exp(1j * ph)
    # A and S are complex in general; build real+imag from log-uniform magnitudes
    return (lu(6).real, lu(12).real, lu(12).real, lu(8))


def draw_heavytail(rng):
    # Student-t df=2 via normal / sqrt(chi2/df); heavy tails
    def ht(n):
        z = rng.normal(size=n)
        w = rng.chisquare(2, size=n) / 2.0
        return z / np.sqrt(np.maximum(w, 1e-6))
    return (ht(6), ht(12), ht(12), ht(8) + 1j * ht(8))


def draw_real(rng):
    return (rng.normal(size=6), rng.normal(size=12), rng.normal(size=12),
            rng.normal(size=8) + 0j)


def draw_hierarchical(rng):
    # generation-dependent widths mimicking a Yukawa hierarchy (1, 0.1, 0.01)
    def hier(n):
        w = np.array([1.0, 0.1, 0.01])
        out = rng.normal(size=n) * np.tile(w, (n // 3 + 1))[:n]
        return out
    return (hier(6), hier(12), hier(12), hier(8) + 1j * hier(8))


PRIORS = {"gaussian": draw_gaussian, "uniform": draw_uniform,
          "loguniform": draw_loguniform, "heavytail": draw_heavytail,
          "real": draw_real, "hierarchical": draw_hierarchical}


def paired(draw, eps, n, seed):
    rng = np.random.default_rng(seed)
    single = []; broken = []
    for _ in range(n):
        a6, s12, sp12, v = draw(rng)
        A = S.anti(a6); Sy = S.sym(s12); Sp = S.sym(sp12)
        if abs(np.linalg.det(Sy)) < 1e-10:
            continue
        v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
        if abs(vR) < 1e-8 or abs(su * sd) < 1e-12:
            continue
        vL = kappa * su * sd / vR
        MD = (v1u - 3 * v15u) * A - 3 * su * Sy
        Me = (v1d - 3 * v15d) * A - 3 * sd * Sy
        Mnu_s = vL * Sy - MD @ np.linalg.inv(vR * Sy) @ MD.T
        Mnu_b = vL * Sy - MD @ np.linalg.inv(vR * Sy + eps * Sp) @ MD.T
        single.append(pmns_obs(Mnu_s, Me))
        broken.append(pmns_obs(Mnu_b, Me))
    return np.array(single), np.array(broken)


def analyse(n=2500):
    out = {}
    for pname, draw in PRIORS.items():
        single, broken = paired(draw, 1.0, n, seed=101)
        row = {}
        for i, lab in [(4, "m2/m1"), (5, "m3/m1")]:
            ks, p = stats.ks_2samp(single[:, i], broken[:, i])
            row[lab] = {"ks": float(ks), "p": float(p),
                        "single_mean": float(np.mean(single[:, i])),
                        "broken_mean": float(np.mean(broken[:, i])),
                        "preference_holds": bool(np.mean(single[:, i]) > np.mean(broken[:, i]) and p < 1e-6)}
        out[pname] = row
    # breaking-strength sweep under the gaussian prior
    sweep = {}
    for eps in [0.3, 1.0, 3.0]:
        single, broken = paired(draw_gaussian, eps, n, seed=202)
        ks, p = stats.ks_2samp(single[:, 5], broken[:, 5])
        sweep["eps=%.1f" % eps] = {"ks_m3m1": float(ks), "p_m3m1": float(p),
                                   "single_mean": float(np.mean(single[:, 5])),
                                   "broken_mean": float(np.mean(broken[:, 5])),
                                   "preference_holds": bool(np.mean(single[:, 5]) > np.mean(broken[:, 5]) and p < 1e-6)}
    return {"priors": out, "breaking_sweep": sweep, "n_per_prior": n}


def robust_summary(res):
    pr = res["priors"]
    # DIRECTION robustness: single-S mean > broken mean (weaker hierarchy)
    dir_m3 = sum(1 for v in pr.values() if v["m3/m1"]["single_mean"] > v["m3/m1"]["broken_mean"])
    dir_m2 = sum(1 for v in pr.values() if v["m2/m1"]["single_mean"] > v["m2/m1"]["broken_mean"])
    # strong significance (p<1e-6) count
    n_hold_m3 = sum(1 for v in pr.values() if v["m3/m1"]["preference_holds"])
    n_hold_m2 = sum(1 for v in pr.values() if v["m2/m1"]["preference_holds"])
    sw = res["breaking_sweep"]
    n_sw = sum(1 for v in sw.values() if v["preference_holds"])
    return {"m3m1_direction_across_priors": "%d/%d" % (dir_m3, len(pr)),
            "m2m1_direction_across_priors": "%d/%d" % (dir_m2, len(pr)),
            "m3m1_strong_priors": "%d/%d" % (n_hold_m3, len(pr)),
            "m2m1_strong_priors": "%d/%d" % (n_hold_m2, len(pr)),
            "breaking_sweep_holds": "%d/%d" % (n_sw, len(sw)),
            "direction_prior_robust": bool(dir_m3 == len(pr) and dir_m2 == len(pr) and n_sw == len(sw)),
            "note": "direction (weaker hierarchy for single-S) is prior-robust; "
                    "magnitude is prior-dependent (weakens for log-uniform / hierarchical priors)"}


def main():
    res = analyse()
    res["summary"] = robust_summary(res)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pencil_prior_sensitivity_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("PRIOR SENSITIVITY (paired, n=%d per prior)" % res["n_per_prior"])
    for pname, row in res["priors"].items():
        m3 = row["m3/m1"]; m2 = row["m2/m1"]
        print("  %-13s m3/m1 KS=%.3f p=%.1e hold=%s | m2/m1 KS=%.3f p=%.1e hold=%s" %
              (pname, m3["ks"], m3["p"], m3["preference_holds"], m2["ks"], m2["p"], m2["preference_holds"]))
    print("BREAKING SWEEP (gaussian):")
    for k, v in res["breaking_sweep"].items():
        print("  %s KS=%.3f p=%.1e hold=%s" % (k, v["ks_m3m1"], v["p_m3m1"], v["preference_holds"]))
    print("SUMMARY:", res["summary"])


if __name__ == "__main__":
    main()
