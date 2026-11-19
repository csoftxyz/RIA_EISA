import os
"""
CORRECT CODIMENSION MEASUREMENT.

The previous test differentiated the constraints ALONG the model, where they
vanish identically, so the differential is necessarily zero.  That says nothing
about independence.

The right question: regard (E1) and (E2) as functions on the AMBIENT space of
pairs (M_d, M_nu) of 3x3 complex matrices (M_d general, M_nu symmetric), and
compute the rank of their differential AT a point of the model.  That rank is
the number of independent real conditions they impose on the ambient space.

Ambient coordinates: M_d 9 complex = 18 real, M_nu symmetric 6 complex = 12 real
=> 30 real.  Evaluate the constraint differential there.
"""
from fractions import Fraction as F
import sys
sys.path.insert(0, r"D:\Documents\qwen-agent\i15g3v3rg3\论文C21\PRD_submission\04_verification_code")
import exact_core as X

c = X.c; D = X.D
mm = X.mm; tr = X.tr; inv3 = X.inv3; det3 = X.det3


def _k(ref, a, b=0):
    return D(c(a, b)) if isinstance(ref, D) else c(a, b)


def sym_part(M):
    h = _k(M[0][0], F(1, 2))
    return [[(M[i][j] + M[j][i]) * h for j in range(3)] for i in range(3)]


def anti_part(M):
    h = _k(M[0][0], F(1, 2))
    return [[(M[i][j] - M[j][i]) * h for j in range(3)] for i in range(3)]


def disc(M):
    t1 = tr(M)
    t2 = (t1*t1 - tr(mm(M, M))) * _k(M[0][0], F(1, 2))
    t3 = det3(M)
    return (t1*t1*t2*t2 - _k(M[0][0], 4)*t2*t2*t2
            - _k(M[0][0], 4)*t1*t1*t1*t3 - _k(M[0][0], 27)*t3*t3
            + _k(M[0][0], 18)*t1*t2*t3)


def cons(Md, Mn):
    Sd = sym_part(Md); Ad = anti_part(Md)
    Xe = mm(Mn, inv3(Sd))
    d = disc(Xe)
    v = [Ad[1][2], -Ad[0][2], Ad[0][1]]
    z0 = _k(Md[0][0], 0)
    Sv = [sum((Sd[i][k]*v[k] for k in range(3)), z0) for i in range(3)]
    Nv = [sum((Mn[i][k]*v[k] for k in range(3)), z0) for i in range(3)]
    cr = [Nv[1]*Sv[2] - Nv[2]*Sv[1],
          Nv[2]*Sv[0] - Nv[0]*Sv[2],
          Nv[0]*Sv[1] - Nv[1]*Sv[0]]
    return [d] + cr


def split(vals, deriv=False):
    out = []
    for z in vals:
        w = z.d if deriv else (z.v if isinstance(z, D) else z)
        out += [w.a, w.b]
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("CODIMENSION IN THE AMBIENT (M_d, M_nu) SPACE")
    print("=" * 78)
    pt = X.rational_point(34, 2)
    def gv(i, j): return c(pt[i], pt[j])
    Mu, Md0, Me, Mn0 = X.bld_120_126(gv)
    # ambient coordinate list: Md 9 complex then Mn 6 complex (upper triangle)
    base_md = [[Md0[i][j] for j in range(3)] for i in range(3)]
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    base_mn = [Mn0[i][j] for i, j in idx]
    coords = [z for row in base_md for z in row] + base_mn
    namb = 2 * len(coords)          # 18 + 12 = 30 real
    print("  ambient real dimension : %d" % namb)

    def build(get):
        Md = [[get(2*(3*i+j), 2*(3*i+j)+1) for j in range(3)] for i in range(3)]
        zz = [get(18+2*t, 19+2*t) for t in range(6)]   # Mn occupies real slots 18..29
        Mn = [[zz[0], zz[1], zz[2]], [zz[1], zz[3], zz[4]], [zz[2], zz[4], zz[5]]]
        return Md, Mn

    flat = [w for z in coords for w in (z.a, z.b)]   # 30 real numbers
    # get(i,j) reads complex parameter (i,j) -> real indices i and j directly,
    # because coords was flattened as re,im pairs: complex slot t uses 2t,2t+1.
    def gval(i, j): return c(flat[i], flat[j])
    assert len(flat) == namb, (len(flat), namb)
    vals = split(cons(*build(gval)))
    print("  constraint components  : %d" % len(vals))
    print("  all vanish at the model point : %s" % all(x == 0 for x in vals))
    cols = []
    for k in range(namb):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(flat[i], flat[j]), s)
        cols.append(split(cons(*build(gd)), True))
    rows = [[cols[j][i] for j in range(namb)] for i in range(len(vals))]
    r = X.exact_rank(rows)
    print("  EXACT rank of constraint differential : %d" % r)
    print("  => independent real conditions imposed : %d" % r)
    print()
    if r == 4:
        print("  The two explicit relations impose exactly FOUR real conditions,")
        print("  matching 22 - 18 = 4.  The audit's request is fully met.")
    else:
        print("  Count is %d, not 4: the explicit set does not yet exhaust the" % r)
        print("  forced relations, or some components are dependent.")
    # separate the two blocks
    for name, sel in (("E1 disc only", [0, 1]),
                      ("E2 cross only", [2, 3, 4, 5, 6, 7])):
        rr = [[cols[j][i] for j in range(namb)] for i in sel]
        print("    %-14s rank %d" % (name, X.exact_rank(rr)))
