"""
CRISIS CHECK.  The exact rational Jacobian gives rank 17 for 120+126 and 18 for
10+126, while the floating-point run claimed 22.  Both cannot be right.

Two hypotheses:
  (H1) the invariant set I is too small, so rank(dI/dp) < rank(dO/dp);
  (H2) there is a genuine relation, and the floating-point 22 was spurious
       (the Takagi half-angle branch can jump between p+h and p-h).

Decisive tests:
  (a) enrich I until the rank saturates;
  (b) construct the redundancy directions ANALYTICALLY and verify they lie in
      the kernel.  Their number bounds the rank from above by
      n_params - n_redundancy, independent of any invariant set.

Redundancies of the correlated system (all exact):
  R1  U(3) congruence Y -> U Y U^T on every Yukawa simultaneously        9 real
  R2  rescale A -> lam A, all four 120 vevs -> /lam                      2 real
  R3  rescale S -> mu S, su,sd,vR,vL -> /mu                              2 real
  R4  up-type vevs (v1u,v15u,su) -> e^{i phi}, vL -> e^{2 i phi}         1 real
      (Mu -> e^{i phi} Mu leaves H_u fixed; Mnu -> e^{2 i phi} Mnu is a
       global phase of the PMNS matrix, removable by charged-lepton rephasing)
"""
from fractions import Fraction as F
import exact_core as E

C = E.C; D = E.D; c = E.c
mm = E.mm; T = E.T; dag = E.dag; tr = E.tr; sc = E.sc; ad = E.ad; sb = E.sb
herm = E.herm; inv3 = E.inv3; anti3 = E.anti3; sym3 = E.sym3


def words(H_list, maxdeg=4):
    """All traces of words up to a given length, plus commutator cubes."""
    out = []
    Hs = H_list
    n = len(Hs)
    for A in Hs:
        A2 = mm(A, A); A3 = mm(A2, A)
        out += [tr(A), tr(A2), tr(A3)]
    for i in range(n):
        for j in range(i + 1, n):
            X, Y = Hs[i], Hs[j]
            X2 = mm(X, X); X3 = mm(X2, X); Y2 = mm(Y, Y); Y3 = mm(Y2, Y)
            XY = mm(X, Y)
            out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)),
                    tr(mm(X2, Y2)), tr(mm(XY, XY)),
                    tr(mm(X3, Y)), tr(mm(X, Y3)),
                    tr(mm(X3, Y2)), tr(mm(X2, Y3)),
                    tr(mm(XY, mm(X, Y))), tr(mm(mm(XY, XY), XY)),
                    tr(mm(mm(X2, Y), mm(X, Y2))),
                    tr(mm(mm(XY, X2), Y2)), tr(mm(mm(XY, Y2), X2))]
            Cm = sb(XY, mm(Y, X))
            out += [tr(mm(mm(Cm, Cm), Cm)), tr(mm(Cm, X)), tr(mm(Cm, Y)),
                    tr(mm(mm(Cm, Cm), X)), tr(mm(mm(Cm, Cm), Y))]
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                if i == j or i == k:
                    continue
                out += [tr(mm(mm(Hs[i], Hs[j]), Hs[k])),
                        tr(mm(mm(Hs[i], Hs[j]), mm(Hs[k], Hs[k])))]
    return out


def rich_invariants(Mu, Md, Me, Mnu):
    Hu, Hd, He, Hn = herm(Mu), herm(Md), herm(Me), herm(Mnu)
    out = words([Hu, Hd, He, Hn])
    # Majorana-sensitive covariants
    Yb = mm(mm(Mnu, T(He)), dag(Mnu))
    Yd = mm(mm(Mnu, T(Hd)), dag(Mnu))
    Yu = mm(mm(Mnu, T(Hu)), dag(Mnu))
    He2 = mm(He, He); Hn2 = mm(Hn, Hn)
    out += words([Yb, He, Hn])
    out += [tr(mm(Yb, Yd)), tr(mm(Yb, Yu)), tr(mm(Yd, Yu)),
            tr(mm(mm(Yb, Yd), He)), tr(mm(mm(Yb, Yu), Hd))]
    # Mnu itself carries Majorana info through its symmetric structure
    out += [tr(mm(Mnu, dag(Mnu))), tr(mm(mm(Mnu, dag(Mnu)), mm(Mnu, dag(Mnu))))]
    return out


def jac_exact(bld, point, invf):
    n = len(point)

    def gv(i, j): return c(point[i], point[j])
    ncomp = len(E.comps(invf(*bld(gv))))
    cols = []
    for k in range(n):
        def gd(i, j, k=k):
            seed = c(1) if i == k else c(0)
            if j == k: seed = seed + c(0, 1)
            return D(c(point[i], point[j]), seed)
        cols.append(E.dcomps(invf(*bld(gd))))
    rows = [[cols[j][i] for j in range(n)] for i in range(ncomp)]
    return E.exact_rank(rows), rows, ncomp


# ---------------------------------------------------------------- redundancy dirs
def redundancy_dirs_120_126(point):
    """Return explicit tangent vectors (lists of Fractions) for R2, R3, R4.
    U(3) directions are constructed from the generators at the identity."""
    n = len(point)
    dirs = []
    # index map
    # A: 0..5 (3 complex), S: 6..17 (6 complex)
    # vevs: v1u 18,19 ; v15u 20,21 ; v1d 22,23 ; v15d 24,25 ; su 26,27 ; sd 28,29
    #       vR 30,31 ; vL 32,33
    def zero(): return [F(0)] * n
    # R2: A -> (1+eps) A  => delta A_re = A_re, delta A_im = 0 ; vevs -> (1-eps) v
    d = zero()
    for k in range(0, 6, 2): d[k] = point[k]
    for k in (18, 20, 22, 24): d[k] = -point[k]; d[k + 1] = -point[k + 1]
    dirs.append(("R2 real rescale A", d))
    d = zero()
    for k in range(1, 6, 2): d[k] = point[k]
    # lam = 1 + i eps : v -> v/(1+i eps) ~ v(1 - i eps) => delta v_re = +v_im, delta v_im = -v_re
    for k in (18, 20, 22, 24):
        d[k] = point[k + 1]; d[k + 1] = -point[k]
    dirs.append(("R2 imag rescale A", d))
    # R3: S -> (1+eps) S ; su,sd,vR,vL -> /mu
    d = zero()
    for k in range(6, 18, 2): d[k] = point[k]
    for k in (26, 28, 30, 32): d[k] = -point[k]; d[k + 1] = -point[k + 1]
    dirs.append(("R3 real rescale S", d))
    d = zero()
    for k in range(7, 18, 2): d[k] = point[k]
    for k in (26, 28, 30, 32):
        d[k] = point[k + 1]; d[k + 1] = -point[k]
    dirs.append(("R3 imag rescale S", d))
    # R4: up vevs v1u,v15u,su -> e^{i phi} ; vL -> e^{2 i phi}
    d = zero()
    for k in (18, 20, 26):
        d[k] = -point[k + 1]; d[k + 1] = point[k]
    k = 32
    d[k] = -2 * point[k + 1]; d[k + 1] = 2 * point[k]
    dirs.append(("R4 up-phase + vL", d))
    return dirs


def kernel_check(rows, dirs, ncomp, npar):
    print("  redundancy directions in the kernel?")
    allok = True
    for name, d in dirs:
        res = [sum((rows[i][j] * d[j] for j in range(npar)), F(0)) for i in range(ncomp)]
        ok = all(x == 0 for x in res)
        print("    %-24s %s" % (name, "IN KERNEL (exact)" if ok else "NOT in kernel"))
        if not ok: allok = False
    return allok


if __name__ == "__main__":
    print("=" * 74)
    print("ENRICHED INVARIANTS + REDUNDANCY KERNEL CHECK")
    print("=" * 74)
    for name, bld, n in (("120+126", E.bld_120_126, 34),
                         ("10+126", E.bld_10_126, 36),
                         ("10+120+126", E.bld_10_120_126, 50)):
        print()
        print("-" * 74)
        print(name)
        pt = E.rational_point(n, 2)
        r_small, _, nc0 = jac_exact(bld, pt, E.invariants)
        print("  small invariant set : %d components, exact rank %d" % (nc0, r_small))
        r_big, rows, nc1 = jac_exact(bld, pt, rich_invariants)
        print("  rich  invariant set : %d components, exact rank %d" % (nc1, r_big))
        if name == "120+126":
            dirs = redundancy_dirs_120_126(pt)
            kernel_check(rows, dirs, nc1, n)
            print("  analytic redundancy count: U(3) 9 + R2 2 + R3 2 + R4 1 = 14")
            print("  => rank <= %d - 14 = %d" % (n, n - 14))
        if name == "10+126":
            print("  analytic redundancy: U(3) 9 + rescale S10 2 + rescale S126 2 + phase 1 = 14")
            print("  => rank <= %d - 14 = %d" % (n, n - 14))
        if name == "10+120+126":
            print("  analytic redundancy: U(3) 9 + 3 rescalings 6 + phase 1 = 16")
            print("  => rank <= %d - 16 = %d" % (n, n - 16))
    print()
    print("=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    print("  If the enriched rank equals n_params - n_redundancy and that is < 22,")
    print("  the neutrino-extended system DOES carry exact relations, and the")
    print("  floating-point rank 22 was an artefact of the Takagi branch choice.")
