"""
COMPLETE THE FOUR RELATIONS.

Established: M_nu S^-1 = c1 I + c2 B^2 with B = A S^-1 traceless of rank <= 2.
Its eigenvalues are {mu, mu, c1} with mu = c1 + c2 lambda^2.  The degeneracy is
ONE complex condition = TWO real relations.  Two more are needed.

They come from the SAME structure, one level down.  Because B has eigenvalues
{lambda, -lambda, 0}, its square B^2 = lambda^2 P where P is the rank-2 projector
onto the span of the +/-lambda eigenvectors.  Hence

    M_nu S^-1 - c1 I = c2 lambda^2 P ,   rank <= 2,

so the matrix  N := M_nu - c1 S  has rank <= 2 as well:

    (R2)  det( M_nu - c1 S ) = 0  for the SAME c1 that appears in the pencil.

But c1 is not free: it is fixed by the pencil's simple eigenvalue.  Eliminating
c1 between "the pencil has a repeated root" and "the shifted matrix is singular"
gives additional constraints.  More usefully, note that A antisymmetric implies

    (R3)  the antisymmetric part of  M_nu S^-1  is controlled:
          M_nu S^-1 = c1 I + c2 A S^-1 A S^-1,
          and (A S^-1)^2 is a polynomial in B with NO independent direction.

The cleanest complete statement is obtained by counting the image of the map
    (S, A, c1, c2) -> M_nu = c1 S + c2 A S^-1 A
inside Sym^2(C^3) (complex dimension 6).  Its dimension is what we measure
exactly; the codimension there, plus the charged-sector coupling through the
SHARED S, gives the four relations.

This script measures:
  (a) the exact dimension of the image of (S,A,c1,c2) -> M_nu   [complex -> real]
  (b) the exact dimension of the image of the FULL map to (Md, Me, Mnu) modulo
      the trivial orbit, i.e. the physical dimension, and identifies how many
      conditions are neutrino-internal vs charged-neutrino.
"""
from fractions import Fraction as F
import exact_core as E
import decisive_rank as DR

c = E.c; D = E.D
mm = E.mm; T = E.T; dag = E.dag; tr = E.tr; sc = E.sc; ad = E.ad; sb = E.sb
inv3 = E.inv3


def sym_entries(M):
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    out = []
    for i, j in idx:
        z = M[i][j]
        out += [z.a, z.b]
    return out


def sym_entries_d(M):
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    out = []
    for i, j in idx:
        z = M[i][j]
        out += [z.d.a, z.d.b]
    return out


def image_map(p):
    """(S 6c, A 3c, c1 1c, c2 1c) = 22 real -> M_nu symmetric (12 real)."""
    A = E.anti3(p[0], p[1], p[2])          # placeholders replaced below
    return None


def build_img(get):
    A = E.anti3(get(0, 1), get(2, 3), get(4, 5))
    S = E.sym3([get(6 + 2*t, 7 + 2*t) for t in range(6)])
    c1 = get(18, 19); c2 = get(20, 21)
    return ad(sc(S, c1), sc(mm(mm(A, inv3(S)), A), c2))


def rank_img(point):
    n = len(point)

    def gv(i, j): return c(point[i], point[j])
    nc = len(sym_entries(build_img(gv)))
    cols = []
    for k in range(n):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(point[i], point[j]), s)
        cols.append(sym_entries_d(build_img(gd)))
    rows = [[cols[j][i] for j in range(n)] for i in range(nc)]
    return E.exact_rank(rows), nc


if __name__ == "__main__":
    print("=" * 78)
    print("IMAGE DIMENSION OF THE REDUCED NEUTRINO MAP")
    print("=" * 78)
    pt = E.rational_point(22, 3)
    r, nc = rank_img(pt)
    print("  parameters (S,A,c1,c2) : 22 real")
    print("  target Sym^2(C^3)      : %d real" % nc)
    print("  EXACT image dimension  : %d" % r)
    print("  codimension in Sym^2   : %d  (= neutrino-internal relations)" % (nc - r))
    print()
    print("  Interpretation: the reduced map cannot reach a generic symmetric")
    print("  matrix; its image has codimension %d.  These are relations among" % (nc - r))
    print("  the neutrino matrix entries ALONE, before using the charged sector.")

    print()
    print("=" * 78)
    print("WHERE THE FOUR COME FROM")
    print("=" * 78)
    n = 34
    p34 = E.rational_point(n, 2)
    rM, ncM, rows = DR.rank_to_matrices(E.bld_120_126, p34)
    def gv(i, j): return c(p34[i], p34[j])
    mats = E.bld_120_126(gv)
    dirs = DR.orbit_dirs(mats)
    base, added = DR.rank_with_orbit(rows, dirs, ncM, n)
    phys = rM - (len(dirs) - added)
    print("  full physical dimension : %d" % phys)
    print("  observables             : 22")
    print("  total relations         : %d" % (22 - phys))
    print()
    print("  charged-only physical dimension (13 observables):")
    # charged sector alone: drop Mnu from the target
    def charged_comps(mats, deriv=False):
        out = []
        for M in mats[:3]:
            for i in range(3):
                for j in range(3):
                    z = M[i][j]
                    v = z.d if deriv else (z.v if isinstance(z, D) else z)
                    out += [v.a, v.b]
        return out
    cols = []
    for k in range(n):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(p34[i], p34[j]), s)
        cols.append(charged_comps(E.bld_120_126(gd), True))
    ncch = len(cols[0])
    rowsch = [[cols[j][i] for j in range(n)] for i in range(ncch)]
    rch = E.exact_rank(rowsch)
    # trivial orbit on charged only: U(3) 9 + up-phase 1 = 10
    print("    exact rank to (Mu,Md,Me) : %d" % rch)
    print("    minus trivial orbit 10   : %d" % (rch - 10))
    print("    charged observables      : 13")
    print("    charged relations        : %d" % max(0, 13 - (rch - 10)))
    print()
    print("  => neutrino extension adds 22-13 = 9 observables but only")
    print("     %d-%d = %d physical directions, forcing %d relations."
          % (phys, rch - 10, phys - (rch - 10), 22 - phys))
