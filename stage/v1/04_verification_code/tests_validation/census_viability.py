"""
Adds a PHENOMENOLOGICAL viability diagnosis to the census.

The original census used a structural criterion: "M_d not forced proportional
to M_e".  That is weaker than "can reproduce the observed masses".  Two distinct
failure modes must be separated:

  (F1) RANK / PROPORTIONALITY failure.
       Only one symmetric Yukawa structure, or none:
         {126}  -> M_d = s_d f, M_e = -3 s_d f  => proportional, rank 1.
         {10}   -> M_d = v h,   M_e = v h       => proportional, rank <= 3 but
                   no splitting at all.
         {120}  -> antisymmetric only           => rank 2, two massless families.

  (F2) DIAGONAL-LOCK failure (the phenomenologically fatal one).
       Exactly one symmetric structure AND it comes from the 126, with an
       antisymmetric 120 also present:
         {120,126}: M_d = a A + s_d f, M_e = b A - 3 s_d f.
       Since A has zero diagonal, (M_e)_ii = -3 (M_d)_ii EXACTLY, which pins
       m_tau/m_b near 3 for hierarchical spectra.  Observed value ~1.4-1.7.
       Literature: Bajc-Melfo-Senjanovic-Vissani hep-ph/0510139;
                   Babu-Bajc-Saad 1612.04329 Sec.2.

  (OK)  TWO OR MORE independent symmetric structures, e.g. {10,126}:
         M_d = v h + s_d f,  M_e = v h - 3 s_d f.
       The diagonal relation is relaxed, m_tau/m_b is free, and M_e is NOT
       proportional to f, so the neutrino alignment obstruction of our paper
       does NOT apply.  This is the standard viable renormalizable content.

KEY CONSEQUENCE FOR THE PAPER:
  Our alignment theorem requires a SINGLE symmetric Yukawa.  Contents with two
  symmetric structures escape it.  So the theorem is best read as: a single
  symmetric Yukawa cannot do the job -- either it fails structurally (F1), or it
  is diagonal-locked and excluded (F2).  At least two independent symmetric
  structures are needed, which is exactly what the standard viable model has.

Effective bilinear reps (16, 144) supply an effective symmetric structure and
are counted as such, with the caveat that they are non-renormalizable.
"""
import itertools
import json

REPS = [10, 120, 126, 16, 144]
# reps supplying a symmetric Yukawa structure to the charged sector
SYM_SOURCE = {10: "h (renormalizable)", 126: "f (renormalizable)",
              16: "effective (16 16*)/M", 144: "effective (144 144*)/M"}
ANTI_SOURCE = {120}


def diagnose(content):
    c = set(content)
    sym = [r for r in c if r in SYM_SOURCE]
    anti = [r for r in c if r in ANTI_SOURCE]
    n_sym = len(sym)
    renorm = c <= {10, 120, 126}
    if n_sym == 0:
        return dict(n_sym=0, mode="F1", viable=False, renorm=renorm,
                    reason="antisymmetric only: rank 2, two massless families")
    if n_sym == 1:
        if sym == [126] and anti:
            return dict(n_sym=1, mode="F2", viable=False, renorm=renorm,
                        reason="single symmetric f from 126 plus antisymmetric 120: "
                               "(M_e)_ii = -3(M_d)_ii forces m_tau/m_b ~ 3")
        if sym == [126]:
            return dict(n_sym=1, mode="F1", viable=False, renorm=renorm,
                        reason="M_d proportional to M_e, rank 1")
        if sym == [10]:
            return dict(n_sym=1, mode="F1", viable=False, renorm=renorm,
                        reason="M_d = M_e exactly: no down/lepton splitting")
        return dict(n_sym=1, mode="F1*", viable=False, renorm=renorm,
                    reason="single effective symmetric structure: proportional or "
                           "diagonal-locked, and non-renormalizable")
    # two or more symmetric structures
    return dict(n_sym=n_sym, mode="OK", viable=True, renorm=renorm,
                reason="%d independent symmetric structures: diagonal lock relaxed, "
                       "M_e not proportional to M_nu" % n_sym)


def main():
    rows = []
    for k in range(1, 6):
        for comb in itertools.combinations(REPS, k):
            d = diagnose(comb)
            rows.append(dict(content=list(comb), size=k, **d))
    with open("census_viability.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("=" * 92)
    print("CENSUS WITH PHENOMENOLOGICAL VIABILITY DIAGNOSIS")
    print("=" * 92)
    print("%-22s %5s %6s %8s %7s  %s"
          % ("content", "n_sym", "mode", "viable", "renorm", "reason"))
    for r in rows:
        print("%-22s %5d %6s %8s %7s  %s"
              % ("+".join(map(str, r["content"])), r["n_sym"], r["mode"],
                 "YES" if r["viable"] else "no",
                 "yes" if r["renorm"] else "no", r["reason"][:60]))
    print()
    viable = [r for r in rows if r["viable"]]
    vr = [r for r in viable if r["renorm"]]
    print("viable contents: %d / %d" % (len(viable), len(rows)))
    print("viable AND renormalizable: %d" % len(vr))
    print("minimal viable renormalizable contents:")
    if vr:
        m = min(r["size"] for r in vr)
        for r in vr:
            if r["size"] == m:
                print("   ", "+".join(map(str, r["content"])))
    print()
    print("F2 (diagonal-locked, structurally splitting but EXCLUDED):")
    for r in rows:
        if r["mode"] == "F2":
            print("   ", "+".join(map(str, r["content"])), "|", r["reason"])
    print()
    print("IMPLICATION FOR THE ALIGNMENT THEOREM:")
    print("  the theorem assumes a single symmetric Yukawa.  Every content with")
    print("  n_sym = 1 fails (F1 or F2); every viable content has n_sym >= 2 and")
    print("  therefore does NOT satisfy the theorem's hypothesis.  The theorem is")
    print("  thus a statement that ONE symmetric structure is insufficient, which")
    print("  explains why the standard viable model contains two.")


if __name__ == "__main__":
    main()
