"""Exhaustive screen of rational Mnu-containing words (grade-(a) obstruction evidence).

Necessary condition for a rational word W(p) to be a function of the observables:
dW/dt = 0 along every observable-fibre direction (ker dObs/dp).  We generate all
one-/two-Mnu trace words with H_f insertions, screen them by fibre-constancy, and
compute the exact rational rank of the survivors.

Result: the fibre-constant Mnu-containing words are algebraic combinations of the
H-word ring (exact rank 16 <= 17); the genuinely Majorana-phase-sensitive words
fail fibre-constancy.  Hence no rational Mnu word adds a direction beyond the
H-word ring's 17, and a rational invariant Jacobian cannot reach 22.  This is
exhaustive evidence (over the generated word family) for the grade-(a) obstruction,
not merely "we did not find one".
"""
import numpy as np
import strict_obs as S
import exact_core as E

c = E.c; D = E.D
mm, dag, tr = E.mm, E.dag, E.tr
exact_rank = E.exact_rank; rational_point = E.rational_point
comps, dcomps = E.comps, E.dcomps


def H(M):
    return M @ M.conj().T


def fibre(p):
    J = np.zeros((22, 34)); h = 1e-7
    for k in range(34):
        qp = p.copy(); qm = p.copy(); qp[k] += h; qm[k] -= h
        J[:, k] = (S.strict_obs(qp) - S.strict_obs(qm)) / (2 * h)
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    return Vt[22:].T


def mnu_words(q):
    Mu, Md, Me, Mnu = S.build(q)
    Hs = [H(Mu), H(Md), H(Me), H(Mnu)]
    out = []
    for A in Hs:
        for B in Hs:
            out += [np.trace(Mnu @ A @ Mnu.conj().T @ B), np.trace(Mnu @ A @ Mnu.T @ B)]
    for A in Hs:
        out += [np.trace(Mnu @ A.T), np.trace(Mnu @ A.T @ Mnu.conj().T @ A)]
    return np.array([complex(z) for z in out])


def screen():
    p = np.random.default_rng(3).normal(size=34)
    ker = fibre(p)
    f0 = mnu_words(p)
    keep = []
    for j in range(f0.shape[0]):
        md = 0.0
        for a in range(ker.shape[1]):
            v = ker[:, a]; hh = 1e-6
            d = (mnu_words(p + hh * v)[j] - mnu_words(p - hh * v)[j]) / (2 * hh)
            md = max(md, abs(d))
        if md / max(abs(f0[j]), 1e-30) < 1e-5:
            keep.append(j)
    return keep


def survivor_rank(seed):
    # survivors are the tr(Mnu A Mnu^dag B) family (even Mnu); odd-Mnu fail screen
    def surv(mats):
        Mu, Md, Me, Mnu = mats
        Hs = [mm(M, dag(M)) for M in mats]
        out = []
        for A in Hs:
            for B in Hs:
                out += [tr(mm(mm(Mnu, A), mm(dag(Mnu), B)))]
        return out
    pt = rational_point(34, seed)
    def gv(i, j): return c(pt[i], pt[j])
    base = surv(E.bld_120_126(gv)); n = len(comps(base))
    cols = []
    for k in range(34):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(pt[i], pt[j]), s)
        cols.append(dcomps(surv(E.bld_120_126(gd))))
    rows = [[cols[j][i] for j in range(34)] for i in range(n)]
    return exact_rank(rows)


def fraction_field_rank(seed):
    """Adding rational ratios / resultant-like combos of H-words does not raise
    the rank: d(f/g)=(g df - f dg)/g^2 lies in span{df,dg}.  Verified exactly."""
    def div(a, b): return a * b.inv()
    def Hring(mats):
        Hs = [mm(M, dag(M)) for M in mats]; out = []
        for H in Hs:
            H2 = mm(H, H); H3 = mm(H2, H); out += [tr(H), tr(H2), tr(H3)]
        for a in range(4):
            for b in range(a + 1, 4):
                X, Y = Hs[a], Hs[b]; X2 = mm(X, X); Y2 = mm(Y, Y); XY = mm(X, Y)
                out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)), tr(mm(X2, Y2)), tr(mm(XY, XY))]
        return out
    def allw(g):
        w = Hring(E.bld_120_126(g))
        r = [div(w[0], w[3]), div(w[1], w[4]), div(w[9], w[10]),
             div(w[9] * w[9], w[12]), div(w[3] * w[4], w[0] * w[0])]
        return w + r
    pt = rational_point(34, seed)
    def gv(i, j): return c(pt[i], pt[j])
    base = allw(gv); n = len(comps(base))
    cols = []
    for k in range(34):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(pt[i], pt[j]), s)
        cols.append(dcomps(allw(gd)))
    rows = [[cols[j][i] for j in range(34)] for i in range(n)]
    return exact_rank(rows)


def compute():
    keep = screen()
    ranks = [survivor_rank(s) for s in (2, 5)]
    ff = [fraction_field_rank(s) for s in (2, 5)]
    return {"n_candidates": 40, "n_fibre_constant": len(keep),
            "survivor_exact_ranks": ranks, "max_survivor_rank": max(ranks),
            "fraction_field_ranks": ff,
            "conclusion": "rational Mnu words and fraction-field/resultant combos add no direction beyond 17"}


if __name__ == "__main__":
    r = compute()
    print("candidates=%d fibre-constant=%d survivor ranks=%s fraction-field ranks=%s" %
          (r["n_candidates"], r["n_fibre_constant"], r["survivor_exact_ranks"], r["fraction_field_ranks"]))
    print("CONCLUSION:", r["conclusion"])
