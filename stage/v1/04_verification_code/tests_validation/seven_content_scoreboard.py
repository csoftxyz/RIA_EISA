"""Seven-content declaration-cost scoreboard + Pfaffian generalisation test.

Uses the SAME Clebsch table (OPS) as the census, so no new physics assumption is
introduced.  For each viable content we build the charged mass matrices from the
operator record (kind, c_u, c_d, c_e, free_de) and fit the six indicative GUT
ratios with Levenberg-Marquardt; the declaration cost is 0 if a generic point fits
(squared log-residual < 0.01), >=1 otherwise (lower bound).

Pfaffian generalisation: the pencil identity needs M_D in span{A, S} with A
ANTISYMMETRIC and M_R proportional to the SAME symmetric S.  We check, for every
content that admits a seesaw, whether this structural condition holds.  SU(5) 45
has an antisymmetric UP Yukawa but its down/lepton Yukawa is general (not the same
S), and SU(5) has no type-I seesaw singlet at renormalisable level, so the pencil
does NOT generalise to it.  The condition is specific to SO(10) 120+126.

SELF-AUDIT FIX (v24 round 3): a general complex matrix ('gen', 18 real dof) must be
represented as complex SYMMETRIC (12) + complex ANTISYMMETRIC (6).  An earlier
version used sym + i*sym, which spans only complex symmetric matrices (rank 12,
verified by Jacobian rank) and underparameterised 'gen'.  The fix restores the full
18 dof and removes an optimiser-sensitive false positive (5+45+50 read cost>=1 at
40 starts under the buggy builder, but cost 0 once 'gen' is correct).  The
declaration-cost verdict is now stable: 5 contents cost 0, 10+120 and 120+126 >=1.
"""
import numpy as np
import viability_fit as V

# Clebsch table identical to census OPS (kind, c_u, c_d, c_e, free_de).
OPS = {
    "SU(5)": {"5": [("sym", 1, 0, 0, False), ("gen", 0, 1, 1, False)],
              "45": [("anti", 1, 0, 0, False), ("gen", 0, 1, -3, False)],
              "50": [("sym", 1, 0, 0, False)]},
    "SO(10)": {"10": [("sym", 1, 1, 1, False)],
               "120": [("anti", 1, 1, -3, True)],
               "126": [("sym", 1, 1, -3, False)]},
    "E6": {"27": [("gen", 1, 1, 1, False)]},
    "PS": {"(1,2,2)": [("gen", 1, 1, 1, False)],
           "(15,2,2)": [("gen", 1, 1, -3, False)]},
}
VIABLE = [("SU(5)", ["5", "45"]), ("SU(5)", ["5", "45", "50"]),
          ("SO(10)", ["10", "120"]), ("SO(10)", ["10", "126"]),
          ("SO(10)", ["120", "126"]), ("SO(10)", ["10", "120", "126"]),
          ("PS", ["(1,2,2)", "(15,2,2)"])]


def build_charged_generic(group, irreps, P):
    """Build Mu, Md, Me matching the paper's vev structure: each bidoublet has an
    up-vev and a down-vev; the lepton entry is the Clebsch weight acting on the
    SAME down-vev (no independent lepton vev).  The 120 (anti, free_de) carries two
    bidoublets: its antisymmetric Y uses (v1u,v15u) for up and (v1d,v15d) for
    down/lepton, i.e. Mu+=(v1u+v15u)A, Md+=(v1d+v15d)A, Me+=(v1d-3v15d)A."""
    n = P.shape[0]
    Mu = np.zeros((n, 3, 3), complex); Md = np.zeros((n, 3, 3), complex); Me = np.zeros((n, 3, 3), complex)
    off = 0
    for ir in irreps:
        for (kind, cu, cd, ce, free_de) in OPS[group][ir]:
            if kind == "sym":
                Y = V.sym_batch(P, off); off += 12
            elif kind == "anti":
                Y = V.anti_batch(P, off); off += 6
            else:
                # general complex matrix = complex symmetric (12) + complex antisymmetric (6) = 18.
                # (sym + i*sym would span only complex symmetric, rank 12 -- a bug fixed in self-audit.)
                Y = V.sym_batch(P, off) + V.anti_batch(P, off + 12); off += 18
            if free_de:
                # 120 antisymmetric bidoublet: two vevs per sector pair (1,15)
                v1u = V.cvec(P, off); v15u = V.cvec(P, off + 2)
                v1d = V.cvec(P, off + 4); v15d = V.cvec(P, off + 6); off += 8
                Mu = Mu + cu * (v1u + v15u) * Y
                Md = Md + cd * (v1d + v15d) * Y
                Me = Me + (v1d + ce * v15d) * Y   # ce=-3 => v1d-3v15d
            else:
                vu = V.cvec(P, off); vd = V.cvec(P, off + 2); off += 4
                Mu = Mu + cu * vu * Y
                Md = Md + cd * vd * Y
                Me = Me + ce * vd * Y
    return Mu, Md, Me


def npar_content(group, irreps):
    off = 0
    for ir in irreps:
        for (kind, cu, cd, ce, free_de) in OPS[group][ir]:
            off += {"sym": 12, "anti": 6, "gen": 18}[kind]
            off += 8 if free_de else 4   # (v1u,v15u,v1d,v15d) vs (vu,vd)
    return off


def fit_content(group, irreps, ntarget=6, n_starts=25, n_iter=250):
    npar = npar_content(group, irreps)
    build = lambda P: build_charged_generic(group, irreps, P)
    tgt = V.TGT5 if ntarget == 5 else V.TGT6
    rng = np.random.default_rng(20261119)
    best = None
    for s in range(n_starts):
        p0 = rng.normal(size=npar) * (1.0 if s % 3 else 0.3)
        c, p = V.lm_fit(build, npar, tgt, p0)
        if best is None or c < best[0]:
            best = (c, p.copy())
    c, p = best
    got = V.ratios_batch(build, p[None, :], ntarget)[0]
    return {"content": "%s %s" % (group, "+".join(irreps)), "npar": npar,
            "best_sq_resid": float(c), "fitted_mtau_mb": float(got[4]),
            "declaration_cost_lower_bound": 0 if c < 0.01 else 1}


def scoreboard(n_starts=25):
    rows = [fit_content(g, ir, n_starts=n_starts) for (g, ir) in VIABLE]
    return {"rows": rows,
            "n_zero": sum(1 for r in rows if r["declaration_cost_lower_bound"] == 0),
            "n_ge1": sum(1 for r in rows if r["declaration_cost_lower_bound"] >= 1)}


# ---- Pfaffian generalisation structural test (COMPUTED, audit 14 B3) ----
# Group-level structural fact (not a per-content choice): only SO(10) embeds a
# renormalisable type-I seesaw singlet (the RH neutrino sits in the 16 of SO(10));
# SU(5)/E6/PS have no gauge-singlet RH neutrino at renormalisable level, so M_R
# cannot be proportional to a symmetric Yukawa S there.
SEESAW_SINGLET = {"SO(10)": True, "SU(5)": False, "E6": False, "PS": False}


def pfaffian_scope():
    """The pencil needs: (i) an antisymmetric Yukawa A, (ii) M_D in span{A,S} with
    S symmetric, (iii) M_R proportional to that same S (type-I seesaw singlet).
    All three conditions are COMPUTED from the OPS Clebsch table plus the
    group-level seesaw-singlet fact; no hand-written per-content verdicts."""
    out = {}
    targets = [("SO(10)", ["120", "126"]), ("SO(10)", ["10", "126"]),
               ("SO(10)", ["10", "120", "126"]), ("SU(5)", ["45"]), ("SU(5)", ["5", "45"])]
    for group, irreps in targets:
        ops = [op for ir in irreps for op in OPS[group][ir]]
        kinds = [op[0] for op in ops]
        antisym_A = "anti" in kinds                      # (i) an antisymmetric Yukawa exists
        seesaw = SEESAW_SINGLET[group]                   # (iii) prerequisite: renormalisable singlet
        # (ii) M_D uses ONLY the anti Yukawa and ONE symmetric Yukawa (the same S
        # that feeds M_R); any extra 'sym' or 'gen' operator adds an independent
        # matrix direction, taking M_D out of span{A,S}.
        md_span = ("anti" in kinds) and (kinds.count("sym") == 1) and ("gen" not in kinds)
        mr_prop_S = seesaw and ("sym" in kinds)
        pencil = bool(antisym_A and md_span and mr_prop_S)
        rec = {"antisym_A": bool(antisym_A), "MD_in_span_A_S": bool(md_span),
               "MR_prop_S": bool(mr_prop_S), "pencil": pencil}
        if group == "SU(5)":
            rec["note"] = ("45 has antisym up Yukawa but no renormalisable type-I seesaw singlet; "
                           "down/lepton 45 is general, not the same S. Pfaffian does not generalise.")
        out["%s %s" % (group, "+".join(irreps))] = rec
    return out


if __name__ == "__main__":
    print("SEVEN-CONTENT DECLARATION-COST SCOREBOARD")
    r = scoreboard()
    for row in r["rows"]:
        print("  %-26s npar=%2d sq.resid=%.4f mtau/mb=%.3f cost>=%d" %
              (row["content"], row["npar"], row["best_sq_resid"], row["fitted_mtau_mb"],
               row["declaration_cost_lower_bound"]))
    print("  zero-cost: %d/7 ; >=1: %d/7" % (r["n_zero"], r["n_ge1"]))
    print("\nPFAFFIAN GENERALISATION SCOPE")
    for k, v in pfaffian_scope().items():
        print("  %-22s pencil=%s  (A=%s span=%s MR~S=%s)" %
              (k, v["pencil"], v["antisym_A"], v["MD_in_span_A_S"], v["MR_prop_S"]))
        if "note" in v:
            print("      ", v["note"])
