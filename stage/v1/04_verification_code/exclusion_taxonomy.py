"""Exclusion classification theorem (mining direction 5).

Reframes the census from the weak statement "seven viable contents all have
P_add=0" (near-tautological) to the strong, non-trivial statement: the ELEVEN
excluded contents are excluded by a small set of GROUP-THEORETIC mechanisms,
dominated by the CKM-locking (transpose-degeneracy) mechanism, and the exclusion
of the minimal content survives measured-boundary RG running.

This script tabulates, from the census passed in by the driver (this run's
computation, no re-derivation and no read of a previous results.json), the
exclusion mechanism of every excluded content and counts the mechanism classes.
The mechanism is read off the census flags:
  unfed                : a charged sector receives no Yukawa (rank deficiency)
  free_de (md=me)      : down and lepton matrices forced proportional/equal
  trans (CKM=I)        : transpose degeneracy -> V_CKM diagonal -> excluded
  all_anti (massless)  : every Yukawa antisymmetric -> a massless generation

The theorem: exclusion is not scattered parameter choices but a concentrated
group-theoretic phenomenon (9/11 by CKM locking alone).
"""
import json
import os


def classify(census):
    excluded = [e for e in census if not e["splitting"]]
    viable = [e for e in census if e["splitting"]]
    rows = []
    counts = {"CKM-lock (trans)": 0, "unfed": 0, "md=me (free_de)": 0, "massless (all_anti)": 0}
    for e in excluded:
        mech = []
        if not e["fed"]:
            mech.append("unfed"); counts["unfed"] += 1
        if e["free_de"]:
            mech.append("md=me"); counts["md=me (free_de)"] += 1
        if e["trans"]:
            mech.append("CKM=I"); counts["CKM-lock (trans)"] += 1
        if e["all_anti"]:
            mech.append("massless"); counts["massless (all_anti)"] += 1
        rows.append({"group": e["group"], "model": e["model"],
                     "rankC": e["rankC"], "n_ops": e["n_ops"],
                     "mechanisms": mech,
                     "primary": "CKM-lock" if e["trans"] else ("unfed" if not e["fed"] else "other")})
    n_ckm = sum(1 for r in rows if r["primary"] == "CKM-lock")
    return {"n_excluded": len(excluded), "n_viable": len(viable),
            "excluded_rows": rows, "mechanism_counts": counts,
            "ckm_lock_fraction": "%d/%d" % (n_ckm, len(excluded)),
            "statement": "exclusion is concentrated: %d/%d excluded contents are "
                         "CKM-locked (V_CKM=I) by group theory alone" % (n_ckm, len(excluded)),
            "viable_p_add": [e["p_add"] for e in viable]}


def compute(census, rg):
    """Classify THIS run's census and attach THIS run's measured-boundary RG
    result.  Both are passed in by the driver (audit 14 B2): the classification
    must never read a previous run's results.json, so the master verifier
    reproduces from an empty output directory."""
    out = classify(census)
    # attach RG robustness of the minimal-content exclusion (measured this run)
    out["rg_robust"] = {"gut_mtau_mb": rg["gut_mtau_mb"],
                        "wall": 2.76,
                        "exclusion_survives_running": bool(rg["gut_mtau_mb"] < 2.0)}
    return out


if __name__ == "__main__":
    # Standalone convenience view of the DELIVERED record; the driver instead
    # passes this run's census/RG via compute(census, rg) (audit 14 B2).
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "06_results", "results.json")
    if not os.path.exists(path):
        print("No delivered results.json found. Run verify_prd.py first; this standalone")
        print("view only summarises an existing record (the master verifier passes its")
        print("own census/RG into compute() and needs no such file).")
        raise SystemExit(0)
    res = json.load(open(path))
    r = compute(res["census"], res["exact_relations"]["rg_lowenergy"])
    print("EXCLUSION CLASSIFICATION THEOREM")
    print("  excluded=%d viable=%d" % (r["n_excluded"], r["n_viable"]))
    print("  mechanism counts:", r["mechanism_counts"])
    print("  CKM-lock fraction:", r["ckm_lock_fraction"])
    for row in r["excluded_rows"]:
        print("    %-6s %-14s %s" % (row["group"], row["model"], "+".join(row["mechanisms"])))
    print("  statement:", r["statement"])
    print("  RG robust:", r["rg_robust"])
