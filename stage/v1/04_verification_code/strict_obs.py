"""Independent implementation of STRICT physical observables (audit 9 check).

Written from scratch, not copied from the auditor's scripts, to test the claim
that the delivered neutrino observables were not basis-independent and that the
true generic observable-space rank is 22 (zero relations), not 19 (three).

Correct construction
--------------------
Charged (13):
  9 singular values of Mu, Md, Me;
  V_CKM = Lu^dag Ld with Lu, Ld the LEFT unitaries from SVD (not Vh);
  3 moduli + 1 Jarlskog.

Neutrino (9):
  Le = left unitary from SVD(Me).
  Go to the charged-lepton mass basis by TRANSPOSE congruence
        Mnu_mass = Le^T Mnu Le          (Weinberg operator transforms as U^T M U)
  Autonne-Takagi:  Mnu_mass = W* D W^dag,  obtained from eigh(Mnu_mass^dag Mnu_mass)
  (note: eigh(Mnu Mnu^dag) would give W*, the WRONG side).
  Fix column phases: z = diag(W^T Mnu_mass W), P = W diag(exp(-i arg z /2)),
  so that P^T Mnu_mass P = diag(m) with m real positive.
  P is then the PMNS matrix.  Observables: 3 masses, 3 moduli, 1 Jarlskog,
  2 Majorana phases phi_i = arg(P_1i^2 P_11*^2), i=2,3 (row-rephasing invariant).

Basis independence (the decisive prerequisite)
---------------------------------------------
Common left flavour rotation R:  M_s -> R M_s,  Mnu -> R* Mnu R^dag.
  Then Le -> R Le and Mnu_mass -> Le^T R^T R* Mnu R^dag R Le = Mnu_mass (invariant,
  since R^T R* = I).  So all 9 neutrino observables must be unchanged.
Unphysical right rotation of Me:  Me -> Me V.  Le unchanged => all 22 unchanged.
"""
import numpy as np


def anti(q):
    a, b, c = q[0] + 1j * q[1], q[2] + 1j * q[3], q[4] + 1j * q[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(q):
    z = [q[2 * t] + 1j * q[2 * t + 1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)


def build(p):
    """120+126, correlated M_R = v_R Y_126, v_L tied (34 real parameters)."""
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(8)]
    v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
    vL = kappa * su * sd / vR
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    MD = (v1u - 3 * v15u) * A - 3 * su * S
    Mnu = vL * S - MD @ np.linalg.inv(vR * S) @ MD.T
    return Mu, Md, Me, Mnu


def build_freevL(p):
    """120+126 with v_L a FREE independent parameter (34 real params: A6+S12+8
    complex vevs, the 8th being v_L itself rather than kappa).
    Used to check that tying v_L does not change the observable dimension."""
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(8)]
    v1u, v15u, v1d, v15d, su, sd, vR, vL = v
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    MD = (v1u - 3 * v15u) * A - 3 * su * S
    Mnu = vL * S - MD @ np.linalg.inv(vR * S) @ MD.T
    return Mu, Md, Me, Mnu


def left_unitary(M):
    U, s, Vh = np.linalg.svd(M)
    order = np.argsort(s)[::-1]
    return U[:, order], s[order]


def takagi(Msym):
    """Return P, m with P^T Msym P = diag(m), m real positive descending."""
    H = Msym.conj().T @ Msym
    w, W = np.linalg.eigh(H)
    order = np.argsort(w)[::-1]
    W = W[:, order]
    z = np.diag(W.T @ Msym @ W)
    phases = np.exp(-0.5j * np.angle(z))
    P = W * phases[None, :]
    m = np.abs(z)
    return P, m


def jarlskog(V):
    return float(np.imag(V[0, 0] * V[1, 1] * np.conj(V[0, 1] * V[1, 0])))


def strict_obs(p):
    Mu, Md, Me, Mnu = build(p)
    out = []
    Lu, mu = left_unitary(Mu)
    Ld, md = left_unitary(Md)
    Le, me = left_unitary(Me)
    out += list(mu) + list(md) + list(me)              # 9
    V = Lu.conj().T @ Ld
    out += [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), jarlskog(V)]   # 4  -> 13
    Mnu_mass = Le.T @ Mnu @ Le                          # transpose congruence
    P, mn = takagi(Mnu_mass)
    out += list(mn)                                     # 3
    out += [abs(P[0, 1]), abs(P[0, 2]), abs(P[1, 2]), jarlskog(P)]   # 4
    out += [float(np.angle(P[0, 1] ** 2 * np.conj(P[0, 0]) ** 2)),
            float(np.angle(P[0, 2] ** 2 * np.conj(P[0, 0]) ** 2))]   # 2  -> 22
    return np.array(out, float)


def selftest(p):
    Mu, Md, Me, Mnu = build(p)
    Le, _ = left_unitary(Me)
    Mm = Le.T @ Mnu @ Le
    P, m = takagi(Mm)
    D = P.T @ Mm @ P
    off = np.linalg.norm(D - np.diag(np.diag(D)))
    uni = np.linalg.norm(P.conj().T @ P - np.eye(3))
    diagres = np.linalg.norm(np.diag(D) - m)
    symres = np.linalg.norm(Mnu - Mnu.T)
    return symres, off, uni, diagres


def rand_unitary(seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


def basis_check(p):
    """Max change of each observable block under physical/unphysical rotations."""
    f0 = strict_obs(p)
    worst_common = 0.0
    for sd in (3, 11, 23):
        R = rand_unitary(sd)
        Mu, Md, Me, Mnu = build(p)
        q = p.copy()
        # rebuild rotated matrices directly by monkeypatching build via closure
        global _ROT
        _ROT = R
        f1 = _obs_rotated(p, R, None)
        worst_common = max(worst_common, float(np.max(np.abs(f1 - f0))))
    worst_right = 0.0
    for sd in (5, 17):
        V = rand_unitary(sd)
        f1 = _obs_rotated(p, None, V)
        worst_right = max(worst_right, float(np.max(np.abs(f1 - f0))))
    return worst_common, worst_right


def _obs_rotated(p, R, Vright):
    Mu, Md, Me, Mnu = build(p)
    if R is not None:
        Mu = R @ Mu; Md = R @ Md; Me = R @ Me
        Mnu = R.conj() @ Mnu @ R.conj().T
    if Vright is not None:
        Me = Me @ Vright
    out = []
    Lu, mu = left_unitary(Mu); Ld, md = left_unitary(Md); Le, me = left_unitary(Me)
    out += list(mu) + list(md) + list(me)
    V = Lu.conj().T @ Ld
    out += [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), jarlskog(V)]
    Mm = Le.T @ Mnu @ Le
    P, mn = takagi(Mm)
    out += list(mn)
    out += [abs(P[0, 1]), abs(P[0, 2]), abs(P[1, 2]), jarlskog(P)]
    out += [float(np.angle(P[0, 1] ** 2 * np.conj(P[0, 0]) ** 2)),
            float(np.angle(P[0, 2] ** 2 * np.conj(P[0, 0]) ** 2))]
    return np.array(out, float)


def rank_at(p, h=1e-6, tol=1e-7):
    f0 = strict_obs(p)
    J = np.zeros((len(f0), len(p)))
    for k in range(len(p)):
        q = p.copy(); q[k] += h
        J[:, k] = (strict_obs(q) - f0) / h
    s = np.linalg.svd(J, compute_uv=False)
    return int((s > tol * s[0]).sum()), s


# ---------------------------------------------------------------- content builders
def build_10_126(p):
    h = sym(p[0:12]); S_ = sym(p[12:24])
    v = [p[24 + 2 * t] + 1j * p[25 + 2 * t] for t in range(6)]
    vu, vd, su, sd, vR, vL = v
    Mu = vu * h + su * S_; Md = vd * h + sd * S_; Me = vd * h - 3 * sd * S_
    MD_ = vu * h - 3 * su * S_
    Mnu = vL * S_ - MD_ @ np.linalg.inv(vR * S_) @ MD_.T
    return Mu, Md, Me, Mnu


def build_10_120_126(p):
    h = sym(p[0:12]); A = anti(p[12:18]); S_ = sym(p[18:30])
    v = [p[30 + 2 * t] + 1j * p[31 + 2 * t] for t in range(10)]
    au, ad_, v1u, v15u, v1d, v15d, su, sd, vR, vL = v
    Mu = au * h + (v1u + v15u) * A + su * S_
    Md = ad_ * h + (v1d + v15d) * A + sd * S_
    Me = ad_ * h + (v1d - 3 * v15d) * A - 3 * sd * S_
    MD_ = au * h + (v1u - 3 * v15u) * A - 3 * su * S_
    Mnu = vL * S_ - MD_ @ np.linalg.inv(vR * S_) @ MD_.T
    return Mu, Md, Me, Mnu


def build_charged(p):
    A = anti(p[0:6]); S_ = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(6)]
    v1u, v15u, v1d, v15d, su, sd = v
    Mu = (v1u + v15u) * A + su * S_
    Md = (v1d + v15d) * A + sd * S_
    Me = (v1d - 3 * v15d) * A - 3 * sd * S_
    return Mu, Md, Me


def build_dflat(p):
    """120+126 with D-flat-like alignment v15 = k*v1 (up and down), v_L tied.
    params: A(6) S(12) v1u v1d su sd vR kappa ku kd (16) = 34."""
    A = anti(p[0:6]); S_ = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(8)]
    v1u, v1d, su, sd, vR, kappa, ku, kd = v
    v15u, v15d = ku * v1u, kd * v1d
    vL = kappa * su * sd / vR
    Mu = (v1u + v15u) * A + su * S_
    Md = (v1d + v15d) * A + sd * S_
    Me = (v1d - 3 * v15d) * A - 3 * sd * S_
    MD_ = (v1u - 3 * v15u) * A - 3 * su * S_
    Mnu = vL * S_ - MD_ @ np.linalg.inv(vR * S_) @ MD_.T
    return Mu, Md, Me, Mnu


def build_two_copies(p):
    """120+126+126' : two independent symmetric Yukawas S,S'; M_R=vR S+vRp S'.
    params: A(6) S(12) Sp(12) + 12 complex vevs (24) = 54."""
    A = anti(p[0:6]); S_ = sym(p[6:18]); Sp = sym(p[18:30])
    v = [p[30 + 2 * t] + 1j * p[31 + 2 * t] for t in range(12)]
    v1u, v15u, v1d, v15d, su, sd, sup, sdp, vR, vRp, vL, vpL = v
    Mu = (v1u + v15u) * A + su * S_ + sup * Sp
    Md = (v1d + v15d) * A + sd * S_ + sdp * Sp
    Me = (v1d - 3 * v15d) * A - 3 * sd * S_ - 3 * sdp * Sp
    MD_ = (v1u - 3 * v15u) * A - 3 * su * S_ - 3 * sup * Sp
    MR = vR * S_ + vRp * Sp
    Mnu = vL * S_ + vpL * Sp - MD_ @ np.linalg.inv(MR) @ MD_.T
    return Mu, Md, Me, Mnu


def charged_obs_strict(p):
    Mu, Md, Me = build_charged(p)
    Lu, mu = left_unitary(Mu); Ld, md = left_unitary(Md); Le, me = left_unitary(Me)
    V = Lu.conj().T @ Ld
    return np.array(list(mu) + list(md) + list(me) +
                    [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), jarlskog(V)], float)


def full_obs_strict(buildfn):
    def f(p):
        Mu, Md, Me, Mnu = buildfn(p)
        Lu, mu = left_unitary(Mu); Ld, md = left_unitary(Md); Le, me = left_unitary(Me)
        V = Lu.conj().T @ Ld
        out = list(mu) + list(md) + list(me) + \
            [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), jarlskog(V)]
        Mm = Le.T @ Mnu @ Le
        P, mn = takagi(Mm)
        out += list(mn) + [abs(P[0, 1]), abs(P[0, 2]), abs(P[1, 2]), jarlskog(P)]
        out += [float(np.angle(P[0, 1] ** 2 * np.conj(P[0, 0]) ** 2)),
                float(np.angle(P[0, 2] ** 2 * np.conj(P[0, 0]) ** 2))]
        return np.array(out, float)
    return f


def measure(fn, npar, seeds=(1, 2, 3, 42, 99), h=1e-6, tol=1e-7):
    ranks = []
    for sd in seeds:
        p = np.random.default_rng(sd).normal(size=npar)
        f0 = fn(p); J = np.zeros((len(f0), npar))
        for k in range(npar):
            q = p.copy(); q[k] += h
            J[:, k] = (fn(q) - f0) / h
        s = np.linalg.svd(J, compute_uv=False)
        ranks.append(int((s > tol * s[0]).sum()))
    return ranks


def run():
    """Single source of truth for observable-space dimensions (strict Takagi)."""
    ch = measure(charged_obs_strict, 30)
    m120 = measure(full_obs_strict(build), 34)
    m10 = measure(full_obs_strict(build_10_126), 36)
    m10120 = measure(full_obs_strict(build_10_120_126), 50)
    g = lambda r: max(set(r), key=r.count)
    return {
        "charged_ranks": ch, "charged_phys": g(ch),
        "ranks_120_126": m120, "phys_120_126": g(m120),
        "relations_120_126": 22 - g(m120),
        "ranks_10_126": m10, "phys_10_126": g(m10),
        "relations_10_126": 22 - g(m10),
        "ranks_10_120_126": m10120, "phys_10_120_126": g(m10120),
        "relations_10_120_126": 22 - g(m10120),
        "method": "strict Autonne-Takagi, basis-independent (audit 9)",
    }


if __name__ == "__main__":
    print("=" * 74)
    print("STRICT TAKAGI OBSERVABLES -- independent implementation")
    print("=" * 74)
    rng = np.random.default_rng(1)
    p = rng.normal(size=34)
    sr, off, uni, dr = selftest(p)
    print("[selftest] ||Mnu-Mnu^T||=%.2e  Takagi offdiag=%.2e  unitarity=%.2e  diag=%.2e"
          % (sr, off, uni, dr))
    wc, wr = basis_check(p)
    print("[basis] max change under common left rotation : %.3e" % wc)
    print("[basis] max change under unphysical Me right   : %.3e" % wr)
    print()
    print("  seed   rank(tol1e-7)   s[21]/s[0]        s[20]/s[0]")
    ranks = []
    for sd in (1, 2, 3, 42, 99, 12345):
        rng = np.random.default_rng(sd)
        q = rng.normal(size=34)
        r, s = rank_at(q)
        ranks.append(r)
        print("  %6d      %2d          %.3e        %.3e"
              % (sd, r, s[21] / s[0], s[20] / s[0]))
    print()
    print("ranks:", ranks)
    print("GENERIC RANK =", max(set(ranks), key=ranks.count))
    print("=> observable relations =", 22 - max(set(ranks), key=ranks.count))
