"""GRADE-(a) PROOF: charged-sector observable map of 120+126 is dominant (rank 13).

The rational bi-unitary invariants
  tr(H_f), tr(H_f^2), tr(H_f^3)   (f=u,d,e)          -> 9 masses
  tr(Hu Hd), tr(Hu^2 Hd), tr(Hu Hd^2), tr(Hu^2 Hd^2),
  tr((Hu Hd)^2), tr(Hu Hd Hu^2 Hd^2), tr([Hu,Hd]^3)  -> 4 CKM
are functions of the 13 charged observables ALONE.  Their exact Jacobian (dual
numbers over Q(i), fraction elimination) has rank 13 at a rational point, which
PROVES rank(dObs/dp)=13 (since rank(dInv/dp) <= rank(dObs/dp) <= 13).  Hence the
charged observable map is dominant and P_add=0 exactly (grade a).

IMPORTANT: cross words mixing H_e with H_u or H_d are NOT functions of the 13
charged observables, because the quark and lepton LEFT-handed flavour bases rotate
independently, so the relative orientation of H_e against H_u,H_d is not among the
13 observables (nine masses plus four CKM).  Such cross words are bi-unitary
invariants but not observable functions; including them spuriously gives rank 14.
Only the H_u-H_d cross (CKM) is admissible.  (Right-handed phases leave H_f
invariant and are NOT the reason; an earlier draft's stated reason was wrong.)
"""
import exact_core as E

c = E.c; D = E.D
mm, dag, tr, sb = E.mm, E.dag, E.tr, E.sb
exact_rank = E.exact_rank; rational_point = E.rational_point
comps, dcomps = E.comps, E.dcomps


def charged_invariants(m3):
    Mu, Md, Me = m3
    Hu, Hd, He = mm(Mu, dag(Mu)), mm(Md, dag(Md)), mm(Me, dag(Me))
    out = []
    for Hx in (Hu, Hd, He):
        H2 = mm(Hx, Hx); H3 = mm(H2, Hx)
        out += [tr(Hx), tr(H2), tr(H3)]
    X, Y = Hu, Hd                      # ONLY Hu-Hd cross = CKM
    X2 = mm(X, X); Y2 = mm(Y, Y); XY = mm(X, Y)
    out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)), tr(mm(X2, Y2)),
            tr(mm(XY, XY)), tr(mm(mm(XY, X2), Y2))]
    Cm = sb(XY, mm(Y, X))
    out += [tr(mm(mm(Cm, Cm), Cm))]
    return out


CONTENTS = {"120+126": (E.bld_120_126, 34, 30),
            "10+126": (E.bld_10_126, 36, 30),
            "10+120+126": (E.bld_10_120_126, 50, 42),
            "10+120": (E.bld_10_120, 30, 30)}


def rank_at(seed, content="120+126"):
    bld, npt, nch = CONTENTS[content]
    pt = rational_point(npt, seed)
    def gv(i, j): return c(pt[i], pt[j])
    base = charged_invariants(bld(gv)[:3])
    n = len(comps(base))
    cols = []
    for k in range(nch):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(pt[i], pt[j]), s)
        cols.append(dcomps(charged_invariants(bld(gd)[:3])))
    rows = [[cols[j][i] for j in range(nch)] for i in range(n)]
    return exact_rank(rows)


def all_contents():
    return {ct: [rank_at(s, ct) for s in (2, 5, 7)] for ct in CONTENTS}


if __name__ == "__main__":
    res = all_contents()
    for ct, r in res.items():
        print("%-12s charged exact rank (seeds 2,5,7): %s  grade-a=%s" % (ct, r, all(x == 13 for x in r)))
    print("ALL SO(10) charged grade-(a):", all(all(x == 13 for x in r) for r in res.values()))
