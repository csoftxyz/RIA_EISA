"""Declaration-cost scoreboard (mining direction 2).

The Discussion notes that what distinguishes viable contents is "how much each
must DECLARE in order to fit the spectrum -- that number is computable, it has
not been computed."  This script computes a first, well-defined version of it.

Operational definition (generic, no declared structure):
  declaration_cost = 0  if a generic point fits all six indicative GUT-scale
                        charged ratios to squared log-residual < 0.01;
                   >=1  otherwise (the content needs at least one extra declared
                        input -- relaxed GUT boundary, threshold, extra Higgs --
                        to reach the spectrum).

This is a LOWER BOUND on the true declaration cost (a generic-fit failure proves
>=1 declaration is needed; success proves 0 suffice for these ratios).  It breaks
the P_add=0 tie of the census: 10+126 fits with 0 declarations, 120+126 is stuck
at the m_tau/m_b wall (~2.76) and needs >=1.

Targets are indicative GUT-scale ratios (no RG); the verdict is about the tree-
level matching class, consistent with the paper's stated scope.
"""
import numpy as np
import viability_fit as V


def scoreboard(n_starts=40, n_iter=250):
    V.N_STARTS = n_starts
    V.N_ITER = n_iter
    rows = []
    for name in ["120+126", "10+126"]:
        c, got, costs, p = V.run(name, 6)
        cost = 0 if c < 0.01 else 1   # lower bound: >=1 if generic fit fails
        rows.append({"content": name,
                     "best_sq_resid": float(c),
                     "sqrt_abs_log_resid": float(np.sqrt(c)),
                     "fitted_mtau_mb": float(got[4]),
                     "target_mtau_mb": float(V.TGT6[4]),
                     "declaration_cost_lower_bound": cost,
                     "verdict": "0 declarations (generic fit)" if cost == 0
                                else ">=1 declaration (generic fit blocked)"})
    return {"rows": rows,
            "tie_break": "census P_add=0 for both; declaration cost breaks the tie "
                         "(10+126=0, 120+126>=1)",
            "scope": "tree-level matching, indicative GUT ratios, no RG; lower bound"}


if __name__ == "__main__":
    r = scoreboard()
    print("DECLARATION-COST SCOREBOARD (generic fit, lower bound)")
    for row in r["rows"]:
        print("  %-10s cost>=%d  sq.resid=%.5f  m_tau/m_b=%.3f (target %.2f)  %s" %
              (row["content"], row["declaration_cost_lower_bound"], row["best_sq_resid"],
               row["fitted_mtau_mb"], row["target_mtau_mb"], row["verdict"]))
    print("  ", r["tie_break"])
