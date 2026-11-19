"""
EXACT WITNESS AND ANALYTIC PROOF.

Response to the second audit, which asks for an analytic argument or an exact
full-rank witness rather than random floating-point sampling.

PART 1 -- analytic proof of constrained surjectivity.
    The reduced map is (S, A, c1, c2) -> c1 S + c2 A S^-1 A.
    At c2 = 0 it is (S, c1) -> c1 S.  Since S ranges over ALL invertible symmetric
    matrices and c1 over C*, the image already contains every invertible symmetric
    matrix, a Zariski-open dense subset of Sym^2(C^3).  The map is therefore
    DOMINANT, and the rank-two term is not needed for the conclusion.
    c2 = b^2/v_R vanishes at b = v1u - 3 v15u = 0, an admissible vev choice.

PART 2 -- exact full-rank witness for the joint system, over Q.
    No floating point is used anywhere.  Parameters are rational, entries are
    Gaussian rationals, and derivatives come from forward-mode dual numbers over
    Q(i).  Rank is decided by fraction Gaussian elimination with no tolerance.

    The physical observables (masses, angles, phases) are irrational, so we do not
    differentiate them directly.  Instead we use POLYNOMIAL bi-unitary invariants
    I, each of which is a function of the 22 observables O alone.  Hence
        rank(dI/dp) <= rank(dO/dp) <= 22.
    Exhibiting rank(dI/dp) = 22 at ONE exact rational point therefore PROVES
    rank(dO/dp) = 22 at that point, and dominance follows.  This is a proof, not
    a sampling argument.
"""
from fractions import Fraction as F

Z = F(0)


# ------------------------------------------------------------ exact Gaussian rationals
class C:
    __slots__ = ("a", "b")

    def __init__(self, a, b=Z):
        self.a = F(a); self.b = F(b)

    def __add__(self, o): return C(self.a + o.a, self.b + o.b)
    def __sub__(self, o): return C(self.a - o.a, self.b - o.b)
    def __mul__(self, o):
        return C(self.a * o.a - self.b * o.b, self.a * o.b + self.b * o.a)
    def __neg__(self): return C(-self.a, -self.b)
    def conj(self): return C(self.a, -self.b)
    def inv(self):
        n = self.a * self.a + self.b * self.b
        if n == 0: raise ZeroDivisionError
        return C(self.a / n, -self.b / n)
    def zero(self): return self.a == 0 and self.b == 0


def c(a, b=0): return C(F(a), F(b))


# ------------------------------------------------------------ dual numbers over Q(i)
class D:
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v = v; self.d = d if d is not None else c(0)

    def __add__(self, o): return D(self.v + o.v, self.d + o.d)
    def __sub__(self, o): return D(self.v - o.v, self.d - o.d)
    def __mul__(self, o): return D(self.v * o.v, self.v * o.d + self.d * o.v)
    def __neg__(self): return D(-self.v, -self.d)
    def conj(self): return D(self.v.conj(), self.d.conj())
    def inv(self):
        iv = self.v.inv()
        return D(iv, -(iv * iv * self.d))


def zero(x): return D(c(0)) if isinstance(x, D) else c(0)
def three(x): return D(c(3)) if isinstance(x, D) else c(3)


# ------------------------------------------------------------ matrix algebra
def mm(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), zero(A[0][0]))
             for j in range(3)] for i in range(3)]


def T(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def dag(A): return [[A[j][i].conj() for j in range(3)] for i in range(3)]
def tr(A): return sum((A[i][i] for i in range(3)), zero(A[0][0]))
def sc(A, s): return [[x * s for x in r] for r in A]
def ad(A, B): return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]
def sb(A, B): return [[A[i][j] - B[i][j] for j in range(3)] for i in range(3)]
def herm(M): return mm(M, dag(M))


def det3(A):
    a, b, cc = A[0]; d, e, f = A[1]; g, h, i = A[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + cc * (d * h - e * g)


def inv3(A):
    idt = det3(A).inv()
    a, b, cc = A[0]; d, e, f = A[1]; g, h, i = A[2]
    cof = [[e * i - f * h, cc * h - b * i, b * f - cc * e],
           [f * g - d * i, a * i - cc * g, cc * d - a * f],
           [d * h - e * g, b * g - a * h, a * e - b * d]]
    return [[cof[r][s] * idt for s in range(3)] for r in range(3)]


def anti3(x, y, z):
    o = zero(x)
    return [[o, x, y], [-x, o, z], [-y, -z, o]]


def sym3(u):
    return [[u[0], u[1], u[2]], [u[1], u[3], u[4]], [u[2], u[4], u[5]]]


# ------------------------------------------------------------ exact rank over Q
def exact_rank(rows):
    m = [[F(x) for x in r] for r in rows]
    nr = len(m); nc = len(m[0]) if nr else 0
    piv = 0
    for col in range(nc):
        sel = next((r for r in range(piv, nr) if m[r][col] != 0), None)
        if sel is None: continue
        m[piv], m[sel] = m[sel], m[piv]
        pv = m[piv][col]
        m[piv] = [x / pv for x in m[piv]]
        for r in range(nr):
            if r != piv and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[piv])]
        piv += 1
        if piv == nr: break
    return piv


# ------------------------------------------------------------ model builders
# Every complex parameter is read by INDEX pair (real index, imag index), so the
# derivative seed is unambiguous.
def bld_120_126(get):
    A = anti3(get(0, 1), get(2, 3), get(4, 5))
    S = sym3([get(6 + 2 * t, 7 + 2 * t) for t in range(6)])
    v1u, v15u, v1d, v15d, su, sd, vR, vL = [get(18 + 2 * t, 19 + 2 * t) for t in range(8)]
    t3 = three(A[0][1]); n3 = -t3
    Mu = ad(sc(A, v1u + v15u), sc(S, su))
    Md = ad(sc(A, v1d + v15d), sc(S, sd))
    Me = ad(sc(A, v1d - t3 * v15d), sc(S, n3 * sd))
    MD = ad(sc(A, v1u - t3 * v15u), sc(S, n3 * su))
    Mnu = sb(sc(S, vL), mm(mm(MD, inv3(sc(S, vR))), T(MD)))
    return Mu, Md, Me, Mnu


def bld_10_126(get):
    S10 = sym3([get(2 * t, 1 + 2 * t) for t in range(6)])
    S126 = sym3([get(12 + 2 * t, 13 + 2 * t) for t in range(6)])
    au, ad_, su, sd, vR, vL = [get(24 + 2 * t, 25 + 2 * t) for t in range(6)]
    t3 = three(S10[0][0]); n3 = -t3
    Mu = ad(sc(S10, au), sc(S126, su))
    Md = ad(sc(S10, ad_), sc(S126, sd))
    Me = ad(sc(S10, ad_), sc(S126, n3 * sd))
    MD = ad(sc(S10, au), sc(S126, n3 * su))
    Mnu = sb(sc(S126, vL), mm(mm(MD, inv3(sc(S126, vR))), T(MD)))
    return Mu, Md, Me, Mnu


def bld_10_120(get):
    S10 = sym3([get(2 * t, 1 + 2 * t) for t in range(6)])
    A = anti3(get(12, 13), get(14, 15), get(16, 17))
    au, ad_, v1u, v15u, v1d, v15d = [get(18 + 2 * t, 19 + 2 * t) for t in range(6)]
    t3 = three(A[0][1])
    Mu = ad(sc(S10, au), sc(A, v1u + v15u))
    Md = ad(sc(S10, ad_), sc(A, v1d + v15d))
    Me = ad(sc(S10, ad_), sc(A, v1d - t3 * v15d))
    return Mu, Md, Me


def bld_10_120_126(get):
    S10 = sym3([get(2 * t, 1 + 2 * t) for t in range(6)])
    A = anti3(get(12, 13), get(14, 15), get(16, 17))
    S = sym3([get(18 + 2 * t, 19 + 2 * t) for t in range(6)])
    au, ad_, v1u, v15u, v1d, v15d, su, sd, vR, vL = [get(30 + 2 * t, 31 + 2 * t) for t in range(10)]
    t3 = three(A[0][1]); n3 = -t3
    Mu = ad(ad(sc(S10, au), sc(A, v1u + v15u)), sc(S, su))
    Md = ad(ad(sc(S10, ad_), sc(A, v1d + v15d)), sc(S, sd))
    Me = ad(ad(sc(S10, ad_), sc(A, v1d - t3 * v15d)), sc(S, n3 * sd))
    MD = ad(ad(sc(S10, au), sc(A, v1u - t3 * v15u)), sc(S, n3 * su))
    Mnu = sb(sc(S, vL), mm(mm(MD, inv3(sc(S, vR))), T(MD)))
    return Mu, Md, Me, Mnu


# ------------------------------------------------------------ polynomial invariants
def invariants(Mu, Md, Me, Mnu):
    """Each entry is a polynomial bi-unitary invariant, hence a function of the
    22 physical observables alone.  Complex entries supply two real components."""
    Hu, Hd, He, Hn = herm(Mu), herm(Md), herm(Me), herm(Mnu)
    out = []
    for H in (Hu, Hd, He, Hn):                      # spectral data of each sector
        H2 = mm(H, H); H3 = mm(H2, H)
        out += [tr(H), tr(H2), tr(H3)]
    for X, Y in ((Hu, Hd), (He, Hn)):               # relative orientation: CKM and PMNS
        X2 = mm(X, X); Y2 = mm(Y, Y); XY = mm(X, Y)
        out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)), tr(mm(X2, Y2)), tr(mm(XY, XY))]
        out += [tr(mm(mm(XY, X2), Y2))]             # CP-odd content
        Cm = sb(XY, mm(Y, X))                       # commutator
        out += [tr(mm(mm(Cm, Cm), Cm))]             # Jarlskog-type
    Yb = mm(mm(Mnu, T(He)), dag(Mnu))               # Majorana-sensitive: conjugation-covariant
    He2 = mm(He, He)
    out += [tr(Yb), tr(mm(Yb, Yb)), tr(mm(Yb, He)), tr(mm(Yb, He2)), tr(mm(Yb, Hn))]
    return out


def comps(vals):
    out = []
    for z in vals:
        v = z.v if isinstance(z, D) else z
        out += [v.a, v.b]
    return out


def dcomps(vals):
    out = []
    for z in vals:
        out += [z.d.a, z.d.b]
    return out


# ------------------------------------------------------------ witness driver
def witness(bld, npar, point, label, target=22):
    print("-" * 74)
    print(label)
    print("  %d rational parameters, exact Q(i) arithmetic, no tolerance" % npar)

    def gv(i, j): return c(point[i], point[j])
    ncomp = len(comps(invariants(*bld(gv))))
    print("  %d real invariant components" % ncomp)
    cols = []
    for k in range(npar):
        def gd(i, j, k=k):
            # z = p_i + i p_j.  d z/d p_i = 1 ; d z/d p_j = i.
            seed = c(1) if i == k else c(0)
            if j == k: seed = seed + c(0, 1)
            return D(c(point[i], point[j]), seed)
        cols.append(dcomps(invariants(*bld(gd))))
    rows = [[cols[j][i] for j in range(npar)] for i in range(ncomp)]
    r = exact_rank(rows)
    ok = (r == target)
    print("  EXACT rank(dI/dp) over Q = %d   (needed %d)" % (r, target))
    print("  %s" % ("==> rank(dO/dp) = 22 PROVEN at this rational point"
                    if ok else "==> not full rank at this point"))
    return r


def rational_point(n, seed):
    """Deterministic small rational point; avoids obvious degeneracies."""
    out = []
    for i in range(n):
        num = ((i * 7 + seed * 3) % 11) - 5
        den = 1 + ((i * 3 + seed) % 4)
        if num == 0: num = 1 + (i % 3)
        out.append(F(num, den))
    return out




# ============================================================ exact relations
def char_poly(M):
    t1 = tr(M)
    t2 = (t1 * t1 - tr(mm(M, M))) * _kk(M[0][0], F(1, 2))
    t3 = det3(M)
    return t1, t2, t3


def disc(M):
    t1, t2, t3 = char_poly(M)
    k = lambda a: _kk(M[0][0], a)
    return (t1*t1*t2*t2 - k(4)*t2*t2*t2 - k(4)*t1*t1*t1*t3
            - k(27)*t3*t3 + k(18)*t1*t2*t3)


def u3_generators():
    G = []
    for a in range(3):
        M = [[c(0) for _ in range(3)] for _ in range(3)]
        M[a][a] = c(0, 1); G.append(M)
    for a in range(3):
        for b in range(a + 1, 3):
            M = [[c(0) for _ in range(3)] for _ in range(3)]
            M[a][b] = c(1); M[b][a] = c(1); G.append(M)
            N = [[c(0) for _ in range(3)] for _ in range(3)]
            N[a][b] = c(0, 1); N[b][a] = c(0, -1); G.append(N)
    return G


def flat_mats(mats, deriv=False):
    out = []
    for M in mats:
        for i in range(3):
            for j in range(3):
                z = M[i][j]
                v = z.d if deriv else (z.v if isinstance(z, D) else z)
                out += [v.a, v.b]
    return out


def rank_to_matrices(bld, point, nsel=None):
    n = len(point)
    def gv(i, j): return c(point[i], point[j])
    base = bld(gv)
    if nsel is not None: base = base[:nsel]
    nc = len(flat_mats(base))
    cols = []
    for k in range(n):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(point[i], point[j]), s)
        m = bld(gd)
        if nsel is not None: m = m[:nsel]
        cols.append(flat_mats(m, True))
    rows = [[cols[j][i] for j in range(n)] for i in range(nc)]
    return exact_rank(rows), nc, rows


def orbit_dirs(mats):
    Mu, Md, Me, Mnu = mats
    dirs = []
    for X in u3_generators():
        dirs.append(flat_mats([ad(mm(X, M), mm(M, T(X))) for M in mats]))
    i_ = c(0, 1)
    zM = [[c(0) for _ in range(3)] for _ in range(3)]
    dirs.append(flat_mats([sc(Mu, i_), zM, zM, sc(Mnu, c(0, 2))]))
    return dirs


def phys_dim(bld, point):
    rM, nc, rows = rank_to_matrices(bld, point)
    def gv(i, j): return c(point[i], point[j])
    mats = bld(gv)
    dirs = orbit_dirs(mats)
    cols = [[rows[i][j] for i in range(nc)] for j in range(len(point))]
    base = exact_rank([list(x) for x in cols])
    added = 0
    for d in dirs:
        if exact_rank([list(x) for x in cols] + [list(d)]) > base + added:
            added += 1
    return rM - (len(dirs) - added), rM


def _kk(ref, a, b=0):
    """Constant matching the entry type (C or D)."""
    return D(c(a, b)) if isinstance(ref, D) else c(a, b)


def sym_part(M):
    h = _kk(M[0][0], F(1, 2))
    return [[(M[i][j] + M[j][i]) * h for j in range(3)] for i in range(3)]


def anti_part(M):
    h = _kk(M[0][0], F(1, 2))
    return [[(M[i][j] - M[j][i]) * h for j in range(3)] for i in range(3)]


def explicit_constraints(Md, Mn):
    """(E1) disc(M_nu [sym M_d]^-1)=0 ; (E2) M_nu v parallel to (sym M_d) v,
    v = kernel direction of anti(M_d).  Both involve only physical matrices."""
    Sd = sym_part(Md); Ad = anti_part(Md)
    d = disc(mm(Mn, inv3(Sd)))
    v = [Ad[1][2], -Ad[0][2], Ad[0][1]]
    z0 = _kk(Md[0][0], 0)
    Sv = [sum((Sd[i][k] * v[k] for k in range(3)), z0) for i in range(3)]
    Nv = [sum((Mn[i][k] * v[k] for k in range(3)), z0) for i in range(3)]
    cr = [Nv[1]*Sv[2] - Nv[2]*Sv[1],
          Nv[2]*Sv[0] - Nv[0]*Sv[2],
          Nv[0]*Sv[1] - Nv[1]*Sv[0]]
    return [d] + cr


def explicit_codimension():
    """Rank of the differential of (E1)-(E2) on the AMBIENT pair space
    (M_d general, M_nu symmetric), real dimension 30, at a model point."""
    pt = rational_point(34, 2)
    def gv(i, j): return c(pt[i], pt[j])
    Mu, Md0, Me, Mn0 = bld_120_126(gv)
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    coords = [z for row in Md0 for z in row] + [Mn0[i][j] for i, j in idx]
    flat = [w for z in coords for w in (z.a, z.b)]
    namb = len(flat)

    def build(get):
        Md = [[get(2*(3*i+j), 2*(3*i+j)+1) for j in range(3)] for i in range(3)]
        zz = [get(18+2*t, 19+2*t) for t in range(6)]
        Mn = [[zz[0], zz[1], zz[2]], [zz[1], zz[3], zz[4]], [zz[2], zz[4], zz[5]]]
        return Md, Mn

    def split(vals, deriv=False):
        out = []
        for z in vals:
            w = z.d if deriv else (z.v if isinstance(z, D) else z)
            out += [w.a, w.b]
        return out

    def gval(i, j): return c(flat[i], flat[j])
    vals = split(explicit_constraints(*build(gval)))
    cols = []
    for k in range(namb):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(flat[i], flat[j]), s)
        cols.append(split(explicit_constraints(*build(gd)), True))
    rows = [[cols[j][i] for j in range(namb)] for i in range(len(vals))]
    return exact_rank(rows), all(x == 0 for x in vals), namb


def run():
    out = {}
    pt = rational_point(34, 2)
    def gv(i, j): return c(pt[i], pt[j])
    A = anti3(gv(0, 1), gv(2, 3), gv(4, 5))
    S = sym3([gv(6 + 2*t, 7 + 2*t) for t in range(6)])
    Si = inv3(S)
    B = mm(A, Si)
    out['trB_zero'] = tr(B).zero()
    out['detB_zero'] = det3(B).zero()
    out['discB2_zero'] = disc(mm(B, B)).zero()
    Mu, Md, Me, Mnu = bld_120_126(gv)
    out['discPencil_zero'] = disc(mm(Mnu, Si)).zero()
    ec = explicit_constraints(Md, Mnu)
    out['E1_E2_vanish'] = all(z.zero() for z in ec)
    codim, allzero, namb = explicit_codimension()
    out['explicit_codimension'] = codim
    out['explicit_ambient_dim'] = namb
    out['explicit_all_vanish'] = allzero
    # DEPRECATED (audit 9): the orbit-based phys_dim() is WITHDRAWN (it gave
    # 18/4/14/34, all wrong).  Observable-space dimensions now come from
    # strict_obs.run() (basis-independent Autonne-Takagi): 22/0 for every
    # content, charged 13.  This module emits only the exact grade-(a)
    # identities and the exact matrix-map rank; no withdrawn dimension numbers.
    r120, _, _ = rank_to_matrices(bld_120_126, pt)
    out['rank_matrices_120_126'] = r120
    out['phys_dim_source'] = 'strict_obs.run(); orbit method withdrawn (audit 9)'
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=1))
