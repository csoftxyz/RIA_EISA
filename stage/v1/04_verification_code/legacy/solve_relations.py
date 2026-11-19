"""
SOLVE THE RELATIONS for the SO(10)-correlated 120+126 neutrino system.

Established by exact arithmetic (decisive_rank.py): the physical dimension is 18
against 22 observables, so FOUR independent exact relations are forced.  This
script finds their MECHANISM in closed form and verifies each step over Q.

THE MECHANISM
-------------
With M_R = v_R S (S = Y_126 symmetric) and M_D = a S + b A (A = Y_120 antisym),
the type-I+II light matrix reduces to
    M_nu = c1 S + c2 (A S^-1 A),   c1 = v_L - a^2/v_R,  c2 = b^2/v_R.
Set  B = A S^-1.  Two exact facts:

  (F1) tr B = 0.   Because A^T = -A and (S^-1)^T = S^-1,
       tr(A S^-1) = tr((A S^-1)^T) = tr(S^-1 A^T) = -tr(S^-1 A) = -tr(A S^-1).

  (F2) rank B <= rank A = 2.

A traceless 3x3 matrix of rank <= 2 has characteristic polynomial
    x^3 + c x = x (x^2 + c),
so its eigenvalues are {lambda, -lambda, 0}.  Hence B^2 has eigenvalues
{lambda^2, lambda^2, 0}: a REPEATED eigenvalue.  Now
    M_nu S^-1 = c1 I + c2 B^2
has eigenvalues {c1 + c2 lambda^2, c1 + c2 lambda^2, c1}.  Therefore

    (R)  the pencil (M_nu, S) has a repeated generalized eigenvalue.

This is the structural origin of the relations: the antisymmetry of the 120
forces a degeneracy in the neutrino mass matrix relative to the 126 direction.
Because S = sym(M_d)/v_126^d is shared with the charged sector, (R) ties the
neutrino observables to the down-quark observables.  Eliminating the unphysical
S and the vevs yields the four relations among the 22 observables; the elimination
is a resultant computation (as in the charged sector), but the MECHANISM (R) is
exact and basis-independent: under Y -> V Y V^T,
    M_nu S^-1 -> V (M_nu S^-1) V^-1,
so its eigenvalues, and the degeneracy, are invariant.

This script verifies F1, F2, the eigenvalue pattern, and (R) over Q, and
re-confirms the dimension count 18 -> 4 relations.
"""
from fractions import Fraction as F
import exact_core as E

c = E.c; D = E.D
mm = E.mm; T = E.T; dag = E.dag; tr = E.tr; sc = E.sc; ad = E.ad; sb = E.sb
herm = E.herm; inv3 = E.inv3


def char_poly(M):
    """Coefficients of det(x I - M) = x^3 - t1 x^2 + t2 x - t3, exact."""
    t1 = tr(M)
    M2 = mm(M, M)
    t2 = (t1 * t1 - tr(M2)) * c(1, 0)
    # use 1/2 ; over Q this is exact
    half = c(F(1, 2))
    t2 = (t1 * t1 - tr(M2)) * half
    t3 = E.det3(M)
    return t1, t2, t3


def discriminant_repeat(M):
    """Return the condition for a repeated eigenvalue via the discriminant of
    the characteristic cubic.  For x^3 + p x + q (depressed), disc = -4p^3 - 27q^2.
    We depress by shifting x -> x + t1/3."""
    t1, t2, t3 = char_poly(M)
    third = c(F(1, 3))
    # depressed cubic coefficients p, q
    p = t2 - t1 * t1 * third
    q = (c(2) * t1 * t1 * t1 * c(F(1, 27))) - (t1 * t2 * third) + t3
    # careful signs: for x^3 - t1 x^2 + t2 x - t3, shift x=y+t1/3:
    # p = t2 - t1^2/3 ; q = -2 t1^3/27 + t1 t2/3 - t3  ... we just test the
    # repeated-root condition directly via gcd-free discriminant below.
    return p, q


def has_repeated_eigenvalue(M):
    """A matrix has a repeated eigenvalue iff disc(char poly) = 0.
    Compute the discriminant of x^3 - t1 x^2 + t2 x - t3 directly:
    Disc = t1^2 t2^2 - 4 t2^3 - 4 t1^3 t3 - 27 t3^2 + 18 t1 t2 t3."""
    t1, t2, t3 = char_poly(M)
    d = (t1*t1*t2*t2 - c(4)*t2*t2*t2 - c(4)*t1*t1*t1*t3
         - c(27)*t3*t3 + c(18)*t1*t2*t3)
    return d


if __name__ == "__main__":
    print("=" * 78)
    print("MECHANISM OF THE FOUR RELATIONS  (exact arithmetic over Q)")
    print("=" * 78)
    n = 34
    pt = E.rational_point(n, 2)

    def gv(i, j): return c(pt[i], pt[j])
    # rebuild A and S directly
    A = E.anti3(gv(0, 1), gv(2, 3), gv(4, 5))
    S = E.sym3([gv(6 + 2*t, 7 + 2*t) for t in range(6)])
    Sinv = inv3(S)
    B = mm(A, Sinv)

    print()
    print("(F1) tr B = tr(A S^-1) = 0 ?")
    tb = tr(B)
    print("     tr B = %s + %s i   -> %s" % (tb.a, tb.b, "ZERO (exact)" if tb.zero() else "NONZERO"))

    print()
    print("(F2) rank B <= 2 ?")
    # rank via exact elimination on the 3x3
    rowsB = [[B[i][j].a for j in range(3)] for i in range(3)]
    # need both real and imag for rank over Q(i); use det=0 and a nonzero 2x2
    detB = E.det3(B)
    print("     det B = %s + %s i  -> %s" % (detB.a, detB.b, "ZERO (rank<=2)" if detB.zero() else "NONZERO"))

    print()
    print("(F3) eigenvalues of B are {lambda, -lambda, 0}: char poly = x^3 + c x ?")
    t1, t2, t3 = char_poly(B)
    print("     t1 (=tr) = %s ; t3 (=det) = %s" % (t1.a, t1.a*0 + t3.a))
    print("     t1 zero: %s ; t3 zero: %s" % (t1.zero(), t3.zero()))
    print("     => char poly x^3 + t2 x, roots 0, +/-sqrt(-t2): CONFIRMED" if (t1.zero() and t3.zero()) else "     => pattern NOT confirmed")

    print()
    print("(F4) B^2 has a repeated eigenvalue (disc = 0) ?")
    B2 = mm(B, B)
    dB2 = has_repeated_eigenvalue(B2)
    print("     disc(char poly of B^2) = %s + %s i  -> %s"
          % (dB2.a, dB2.b, "ZERO (repeated eigenvalue)" if dB2.zero() else "NONZERO"))

    print()
    print("(R) M_nu S^-1 has a repeated eigenvalue ?")
    Mu, Md, Me, Mnu = E.bld_120_126(gv)
    X = mm(Mnu, Sinv)
    dX = has_repeated_eigenvalue(X)
    print("     disc(char poly of M_nu S^-1) = %s + %s i  -> %s"
          % (dX.a, dX.b, "ZERO (repeated generalized eigenvalue)" if dX.zero() else "NONZERO"))

    print()
    print("(R-invariance) eigenvalues of M_nu S^-1 are basis-independent:")
    print("     under Y -> V Y V^T,  M_nu S^-1 -> V (M_nu S^-1) V^-1 (similarity).")

    print()
    print("=" * 78)
    print("DIMENSION COUNT  (re-confirm, exact)")
    print("=" * 78)
    import decisive_rank as DR
    rM, nc, rows = DR.rank_to_matrices(E.bld_120_126, pt)
    mats = E.bld_120_126(gv)
    dirs = DR.orbit_dirs(mats)
    base, added = DR.rank_with_orbit(rows, dirs, nc, n)
    phys = rM - (len(dirs) - added)
    print("  parameters              : %d" % n)
    print("  exact rank to matrices  : %d" % rM)
    print("  trivial orbit (U(3)+ph) : %d" % (len(dirs) - added))
    print("  physical directions     : %d" % phys)
    print("  observables             : 22")
    print("  FORCED RELATIONS        : %d" % (22 - phys))

    print()
    print("=" * 78)
    print("CONTENT DEPENDENCE")
    print("=" * 78)
    for name, bld, nn in (("10+126", E.bld_10_126, 36),
                          ("10+120+126", E.bld_10_120_126, 50)):
        p2 = E.rational_point(nn, 2)
        rM2, nc2, rows2 = DR.rank_to_matrices(bld, p2)
        def gv2(i, j): return c(p2[i], p2[j])
        m2 = bld(gv2)
        d2 = DR.orbit_dirs(m2)
        b2, a2 = DR.rank_with_orbit(rows2, d2, nc2, nn)
        ph2 = rM2 - (len(d2) - a2)
        print("  %-16s physical %2d vs 22  ->  relations %d" % (name, ph2, max(0, 22 - ph2)))
    print()
    print("  => the minimal 120+126 IS predictive in the neutrino sector;")
    print("     adding the 10 removes the forced relations.")
