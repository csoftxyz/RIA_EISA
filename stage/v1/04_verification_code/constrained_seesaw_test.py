"""
DECISIVE TEST prompted by the external audit.

The audit objects that in SO(10) the Majorana matrix is NOT free:
    M_R = v_R Y_126
shares the SAME Yukawa that feeds the charged fermions.  Our earlier extended
Jacobian used a free symmetric M_R (12 independent reals) and also set M_D = M_u.
Both are wrong for SO(10).  Correct structure (Babu-Bajc-Saad Eqs. 3.10-3.14):

    M_U   = v10u Y10 + v126u Y126 + (v1u + v15u) Y120
    M_D   = v10d Y10 + v126d Y126 + (v1d + v15d) Y120
    M_E   = v10d Y10 - 3 v126d Y126 + (v1d - 3 v15d) Y120
    M_nuD = v10u Y10 - 3 v126u Y126 + (v1u - 3 v15u) Y120     <-- NOT equal to M_U
    M_R   = v_R Y126                                          <-- NOT free
    M_L   = v_L Y126
    M_N   = M_L - M_nuD M_R^{-1} M_nuD^T

Question: with these correlations imposed, does P_add stay zero?

Algebra first.  For 120+126 (no 10), write a = -3 v126u, b = (v1u - 3v15u):
    M_nuD = a S + b A ,  M_R = v_R S
    (aS+bA) S^-1 (aS+bA)^T = (aS+bA) S^-1 (aS - bA)      [S sym, A antisym]
                           = a^2 S - b^2 A S^-1 A
    => M_N = (v_L - a^2/v_R) S + (b^2/v_R) A S^-1 A
So the image is governed by S and by A S^-1 A.  Note A S^-1 A IS symmetric:
    (A S^-1 A)^T = A^T S^-T A^T = (-A) S^-1 (-A) = A S^-1 A.
But rank(A) = 2, so rank(A S^-1 A) <= 2.  Whether the sum is still surjective
onto symmetric matrices is NOT obvious.  Test it.
"""
import numpy as np

rng = np.random.default_rng(556677)


def anti(p):
    a, b, c = p[0]+1j*p[1], p[2]+1j*p[3], p[4]+1j*p[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(p):
    v = p[:12].reshape(6, 2); z = v[:, 0]+1j*v[:, 1]
    return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)


def sv(M):
    return np.linalg.svd(M, compute_uv=False)


print("="*78)
print("0. ALGEBRAIC IDENTITY CHECK:  M_N = c1 S + c2 A S^-1 A")
print("="*78)
worst = 0.0; symerr = 0.0; ranks = []
for _ in range(300):
    A = anti(rng.normal(size=6)); S = sym(rng.normal(size=12))
    if abs(np.linalg.det(S)) < 1e-6:
        continue
    a = rng.normal()+1j*rng.normal(); b = rng.normal()+1j*rng.normal()
    vR = rng.normal()+1j*rng.normal(); vL = rng.normal()+1j*rng.normal()
    if abs(vR) < 1e-6:
        continue
    MD = a*S + b*A
    MN_direct = vL*S - MD @ np.linalg.inv(vR*S) @ MD.T
    MN_formula = (vL - a*a/vR)*S + (b*b/vR)*(A @ np.linalg.inv(S) @ A)
    worst = max(worst, np.max(np.abs(MN_direct-MN_formula))/max(np.max(np.abs(MN_direct)), 1e-300))
    T = A @ np.linalg.inv(S) @ A
    symerr = max(symerr, np.max(np.abs(T-T.T))/max(np.max(np.abs(T)), 1e-300))
    ranks.append(np.linalg.matrix_rank(T, tol=1e-8))
print("  identity max rel error      = %.2e" % worst)
print("  (A S^-1 A) symmetry error   = %.2e" % symerr)
print("  rank(A S^-1 A) observed     = %s   (A has rank 2)" % sorted(set(ranks)))

print()
print("="*78)
print("1. IS  c1 S + c2 A S^-1 A  SURJECTIVE ONTO SYMMETRIC MATRICES?")
print("="*78)
# Parameter count: S 6 complex, A 3 complex, c1 c2 2 complex = 11 complex = 22 real.
# Target: symmetric 3x3 = 6 complex = 12 real.  Surplus 10 real.  But rank(A S^-1 A)<=2
# could still obstruct.  Measure the Jacobian of (S,A,c1,c2) -> M_N (12 real components).
def unpack_img(p):
    A = anti(p[0:6]); S = sym(p[6:18])
    c1 = p[18]+1j*p[19]; c2 = p[20]+1j*p[21]
    if abs(np.linalg.det(S)) < 1e-8:
        return None
    return c1*S + c2*(A @ np.linalg.inv(S) @ A)


def obs_img(p):
    M = unpack_img(p)
    if M is None:
        return np.full(12, np.nan)
    # 6 complex entries of a symmetric matrix -> 12 reals
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    return np.array([v for i, j in idx for v in (M[i, j].real, M[i, j].imag)])


def rk(f, p, h=1e-6, tol=1e-7):
    f0 = f(p)
    if not np.all(np.isfinite(f0)):
        return None, None
    J = np.zeros((len(f0), len(p)))
    for i in range(len(p)):
        a = p.copy(); a[i] += h
        b = p.copy(); b[i] -= h
        fa, fb = f(a), f(b)
        if not (np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):
            return None, None
        J[:, i] = (fa-fb)/(2*h)
    s = np.linalg.svd(J, compute_uv=False)
    if s[0] == 0:
        return None, None
    return int(np.sum(s > tol*s[0])), s


rs = []; cs = []
for _ in range(10):
    p = rng.normal(size=22)
    r, s = rk(obs_img, p)
    if r is not None:
        rs.append(r)
        if r == 12:
            cs.append(s[11]/s[0])
print("  params 22 real -> symmetric target 12 real")
print("  ranks: %s   (full = 12)" % sorted(set(rs)))
if cs:
    print("  median s_min/s_max at full rank = %.2e" % np.median(cs))
print("  => surjective onto symmetric matrices" if max(rs) == 12 else "  => NOT surjective!")

print()
print("="*78)
print("2. FULL SO(10) 120+126 WITH CORRELATED M_R  (the audit's demand)")
print("="*78)
# params: A 6, S 12, v1u v15u v1d v15d (8), v126u v126d (4), vR vL (4) = 34
N34 = 34


def build34(p):
    A = anti(p[0:6]); S = sym(p[6:18])
    v = p[18:34].reshape(8, 2); v = v[:, 0]+1j*v[:, 1]
    v1u, v15u, v1d, v15d, su, sd, vR, vL = v
    Mu = (v1u+v15u)*A + su*S
    Md = (v1d+v15d)*A + sd*S
    Me = (v1d-3*v15d)*A - 3*sd*S
    MD = (v1u-3*v15u)*A - 3*su*S          # correct Dirac matrix
    if abs(vR) < 1e-10 or abs(np.linalg.det(S)) < 1e-10:
        return None
    MR = vR*S                             # CORRELATED Majorana
    ML = vL*S
    MN = ML - MD @ np.linalg.inv(MR) @ MD.T
    return Mu, Md, Me, MN


def obs20(p):
    out = build34(p)
    if out is None:
        return np.full(20, np.nan)
    Mu, Md, Me, MN = out
    _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)
    V = Vu @ Vd.conj().T
    J = abs(np.imag(V[0, 1]*V[1, 2]*V[0, 2].conj()*V[1, 1].conj()))
    _, Un = np.linalg.eigh(MN @ MN.conj().T)
    _, _, Ve = np.linalg.svd(Me)
    U = Ve @ Un.conj().T
    Jn = abs(np.imag(U[0, 1]*U[1, 2]*U[0, 2].conj()*U[1, 1].conj()))
    return np.concatenate([sv(Mu), sv(Md), sv(Me),
                           [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J],
                           sv(MN), [abs(U[0, 1]), abs(U[0, 2]), abs(U[1, 2]), Jn]])


rs = []; cs = []
for _ in range(10):
    p = rng.normal(size=N34)
    r, s = rk(obs20, p)
    if r is not None:
        rs.append(r)
        if r == 20:
            cs.append(s[19]/s[0])
print("  params %d real, observables 20 (13 charged + 7 nu invariants)" % N34)
print("  ranks: %s   -> P_add = %d" % (sorted(set(rs)), 20-max(rs)))
if cs:
    print("  median s_min/s_max = %.2e" % np.median(cs))

print()
print("="*78)
print("3. ADD THE TWO MAJORANA PHASES -> 22 OBSERVABLES")
print("="*78)
# Takagi: MN = U* D U^dagger.  Majorana phases sit in U relative to the charged
# lepton basis.  Extract them from the symmetric Takagi factor.
def takagi(M):
    # M symmetric complex.  Use M M^dagger eigenbasis then fix phases.
    H = M @ M.conj().T
    d, U = np.linalg.eigh(H)
    order = np.argsort(-d); d = d[order]; U = U[:, order]
    # U^T M U should be diagonal positive after phase fixing
    D = U.T @ M @ U
    ph = np.angle(np.diag(D))
    U = U @ np.diag(np.exp(-1j*ph/2))
    return np.sqrt(np.maximum(d, 0)), U


def obs22(p):
    out = build34(p)
    if out is None:
        return np.full(22, np.nan)
    Mu, Md, Me, MN = out
    _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)
    V = Vu @ Vd.conj().T
    J = abs(np.imag(V[0, 1]*V[1, 2]*V[0, 2].conj()*V[1, 1].conj()))
    mn, Un = takagi(MN)
    _, _, Ve = np.linalg.svd(Me)
    U = Ve @ Un.conj().T
    Jn = abs(np.imag(U[0, 1]*U[1, 2]*U[0, 2].conj()*U[1, 1].conj()))
    # two Majorana phases: rephasing-invariant combinations
    m1 = np.angle(U[0, 0]**2 * U[0, 1].conj()**2)
    m2 = np.angle(U[0, 0]**2 * U[0, 2].conj()**2)
    return np.concatenate([sv(Mu), sv(Md), sv(Me),
                           [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J],
                           mn, [abs(U[0, 1]), abs(U[0, 2]), abs(U[1, 2]), Jn, m1, m2]])


rs = []; cs = []
for _ in range(10):
    p = rng.normal(size=N34)
    r, s = rk(obs22, p)
    if r is not None:
        rs.append(r)
        if r == 22:
            cs.append(s[21]/s[0])
print("  params %d real, observables 22 (13 charged + 9 nu incl. Majorana)" % N34)
print("  ranks: %s   -> P_add = %d" % (sorted(set(rs)), 22-max(rs)))
if cs:
    print("  median s_min/s_max = %.2e" % np.median(cs))

print()
print("="*78)
print("4. OTHER SO(10) CONTENTS WITH CORRELATED M_R")
print("="*78)
# 10+126 : Y10 sym (12), Y126 sym (12), vevs v10u v10d v126u v126d vR vL (12) = 36
def build_10_126(p):
    S10 = sym(p[0:12]); S126 = sym(p[12:24])
    v = p[24:36].reshape(6, 2); v = v[:, 0]+1j*v[:, 1]
    a10u, a10d, su, sd, vR, vL = v
    Mu = a10u*S10 + su*S126
    Md = a10d*S10 + sd*S126
    Me = a10d*S10 - 3*sd*S126
    MD = a10u*S10 - 3*su*S126
    if abs(vR) < 1e-10 or abs(np.linalg.det(S126)) < 1e-10:
        return None
    MN = vL*S126 - MD @ np.linalg.inv(vR*S126) @ MD.T
    return Mu, Md, Me, MN


def mk_obs(builder):
    def f(p):
        out = builder(p)
        if out is None:
            return np.full(20, np.nan)
        Mu, Md, Me, MN = out
        _, _, Vu = np.linalg.svd(Mu); _, _, Vd = np.linalg.svd(Md)
        V = Vu @ Vd.conj().T
        J = abs(np.imag(V[0, 1]*V[1, 2]*V[0, 2].conj()*V[1, 1].conj()))
        _, Un = np.linalg.eigh(MN @ MN.conj().T)
        _, _, Ve = np.linalg.svd(Me)
        U = Ve @ Un.conj().T
        Jn = abs(np.imag(U[0, 1]*U[1, 2]*U[0, 2].conj()*U[1, 1].conj()))
        return np.concatenate([sv(Mu), sv(Md), sv(Me),
                               [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J],
                               sv(MN), [abs(U[0, 1]), abs(U[0, 2]), abs(U[1, 2]), Jn]])
    return f


for name, builder, n in [("10+126", build_10_126, 36)]:
    rs = []
    for _ in range(8):
        r, s = rk(mk_obs(builder), rng.normal(size=n))
        if r is not None:
            rs.append(r)
    print("  %-12s params %d, obs 20, ranks %s -> P_add = %d"
          % (name, n, sorted(set(rs)), 20-max(rs)))

print()
print("="*78)
print("5. CONTROL: WHAT IF M_R WERE FORCED PROPORTIONAL TO THE IDENTITY?")
print("="*78)
# This is a DECLARED structure, outside SO(10).  Show it DOES create relations,
# confirming that the framework detects purchased predictivity.
def build_declared(p):
    A = anti(p[0:6]); S = sym(p[6:18])
    v = p[18:30].reshape(6, 2); v = v[:, 0]+1j*v[:, 1]
    v1u, v15u, v1d, v15d, su, sd = v
    Mu = (v1u+v15u)*A + su*S
    Md = (v1d+v15d)*A + sd*S
    Me = (v1d-3*v15d)*A - 3*sd*S
    MD = (v1u-3*v15u)*A - 3*su*S
    mR = p[30]+1j*p[31]                     # single scale: M_R = mR * Identity
    MN = -(MD @ MD.T)/mR
    return Mu, Md, Me, MN


rs = []
for _ in range(8):
    r, s = rk(mk_obs(build_declared), rng.normal(size=32))
    if r is not None:
        rs.append(r)
print("  M_R = m R * I (declared): params 32, obs 20, ranks %s -> P_add = %d"
      % (sorted(set(rs)), 20-max(rs)))
print("  (a nonzero P_add here would show the machinery detects declared structure)")
print()
print("done.")
