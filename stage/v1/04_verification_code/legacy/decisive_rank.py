"""
DECISIVE TEST.  Avoid the invariant-completeness problem entirely.

Compute the exact rank of the map  parameters -> (Mu, Md, Me, Mnu)  as points of
C^{9+9+9+6} = R^{66}.  Call it r_M.

Redundancies act on the matrices as
    Mu -> e^{i phi} V Mu V^T ,  Md -> V Md V^T ,
    Me -> V Me V^T ,            Mnu -> e^{2 i phi} V Mnu V^T ,
with V in U(3).  These leave every physical observable unchanged: singular values
are invariant, CKM is invariant, and the extra phase on Mnu is a global PMNS phase
removed by charged-lepton rephasing.  The group has dimension 9 + 1 = 10, plus the
rescalings of each Yukawa against its vevs.

If the group acts with discrete stabiliser at a generic point, then
    r_M = n_params - dim(redundancy)
and the number of physical directions is r_M - 10 (the U(3)+phase part acts purely
on the matrices, not on observables).  Comparing that with 22 decides the question
WITHOUT needing a complete invariant set:

    physical directions >= 22  =>  no relation is forced by dimension counting;
    physical directions <  22  =>  relations MUST exist.

The rescalings are already inside r_M's kernel, so they are not subtracted twice.
"""
from fractions import Fraction as F
import exact_core as E

c = E.c; D = E.D; mm = E.mm; T = E.T; dag = E.dag; sc = E.sc


def matrix_comps(mats, deriv=False):
    out = []
    for M in mats:
        for i in range(3):
            for j in range(3):
                z = M[i][j]
                v = z.d if deriv else (z.v if isinstance(z, D) else z)
                out += [v.a, v.b]
    return out


def rank_to_matrices(bld, point):
    n = len(point)

    def gv(i, j): return c(point[i], point[j])
    base = bld(gv)
    nc = len(matrix_comps(base))
    cols = []
    for k in range(n):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(point[i], point[j]), s)
        cols.append(matrix_comps(bld(gd), deriv=True))
    rows = [[cols[j][i] for j in range(n)] for i in range(nc)]
    return E.exact_rank(rows), nc, rows


def u3_generators():
    """Nine Hermitian generators of u(3), as exact Gaussian-rational matrices."""
    G = []
    for a in range(3):                       # diagonal, i * E_aa
        M = [[c(0) for _ in range(3)] for _ in range(3)]
        M[a][a] = c(0, 1)
        G.append(M)
    for a in range(3):
        for b in range(a + 1, 3):
            M = [[c(0) for _ in range(3)] for _ in range(3)]
            M[a][b] = c(1); M[b][a] = c(1)   # real symmetric
            G.append(M)
            N = [[c(0) for _ in range(3)] for _ in range(3)]
            N[a][b] = c(0, 1); N[b][a] = c(0, -1)  # imaginary antisymmetric
            G.append(N)
    return G


def orbit_dirs(mats):
    """Tangent vectors of the U(3) congruence plus the up-sector phase,
    expressed in the same 66-component coordinate system."""
    Mu, Md, Me, Mnu = mats
    dirs = []
    for X in u3_generators():                # delta M = X M + M X^T
        dM = []
        for M in (Mu, Md, Me, Mnu):
            dM.append(E.ad(mm(X, M), mm(M, T(X))))
        dirs.append(matrix_comps(dM))
    # up-sector phase: Mu -> i Mu, Mnu -> 2i Mnu, others zero
    i_ = c(0, 1)
    zeroM = [[c(0) for _ in range(3)] for _ in range(3)]
    dM = [sc(Mu, i_), zeroM, zeroM, sc(Mnu, c(0, 2))]
    dirs.append(matrix_comps(dM))
    return dirs


def rank_with_orbit(rows, dirs, nc, n):
    """Rank of the span of (Jacobian rows) together with the orbit directions.
    Orbit directions live in the same R^66; if they are already in the row span
    the action is free along the image, confirming the redundancy count."""
    aug = [list(r) for r in rows]
    # orbit dirs are vectors in R^66, i.e. COLUMNS of the same space as rows?
    # rows are indexed by component; to test membership we append them as extra
    # ROWS of a matrix whose columns are components -> different shape.
    # Instead: check whether each orbit dir lies in the COLUMN space of J,
    # i.e. in the span of {J e_k}.  Column space of J = span of cols.
    # We stored rows[i][j] = d(component i)/d(param j).  Column space is spanned
    # by vectors v_j = (rows[i][j])_i.  Test membership by rank increase.
    cols = [[rows[i][j] for i in range(nc)] for j in range(n)]
    base = E.exact_rank([list(x) for x in cols])
    added = 0
    for d in dirs:
        trial = [list(x) for x in cols] + [list(d)]
        if E.exact_rank(trial) > base + added:
            added += 1
    return base, added


if __name__ == "__main__":
    print("=" * 76)
    print("EXACT RANK OF THE MAP TO MASS MATRICES, AND THE REDUNDANCY ORBIT")
    print("=" * 76)
    cfg = [("120+126", E.bld_120_126, 34, 2),
           ("10+126", E.bld_10_126, 36, 2),
           ("10+120+126", E.bld_10_120_126, 50, 2)]
    for name, bld, n, nrescale in cfg:
        print()
        print("-" * 76)
        print(name)
        pt = E.rational_point(n, 2)
        rM, nc, rows = rank_to_matrices(bld, pt)
        print("  target components            : %d" % nc)
        print("  EXACT rank to matrices  r_M  : %d" % rM)

        def gv(i, j): return c(pt[i], pt[j])
        mats = bld(gv)
        dirs = orbit_dirs(mats)
        base, added = rank_with_orbit(rows, dirs, nc, n)
        print("  U(3)+phase orbit dirs        : %d" % len(dirs))
        print("  of which INDEPENDENT of image: %d  (0 means all inside the image)" % added)
        # physical directions = r_M minus the orbit dimension that acts trivially
        orbit_dim = len(dirs) - added
        phys = rM - orbit_dim
        print("  orbit directions inside image: %d" % orbit_dim)
        print("  physical directions          : %d   (observables 22)" % phys)
        if phys >= 22:
            print("  => dimension permits no relation; consistent with P_add = 0")
        else:
            print("  => FEWER physical directions than observables: RELATIONS EXIST")
            print("     number of forced relations >= %d" % (22 - phys))
    print()
    print("=" * 76)
    print("NOTE")
    print("=" * 76)
    print("  This test needs no complete invariant set: it counts directions in")
    print("  matrix space and removes exactly those that act trivially on observables.")
