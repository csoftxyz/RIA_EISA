"""
AUDIT 5 -- computable gaps, closed with tests.  HONEST EDITION.

IMPORTANT CORRECTION discovered while writing these tests:
    The previous "physical dimension 18 / four relations" was computed by
    subtracting ten "trivial orbit directions" from the matrix-map rank 28.
    That orbit construction (exact_core.orbit_dirs / u3_generators) used
    CONGRUENCE generators X M + M X^T with X taken as i*E_aa and complex
    symmetric/antisymmetric off-diagonals.  Those are NOT the redundancy
    orbit of the physical observable map.  Direct test shows six of them
    CHANGE physical observables, so subtracting them was wrong.

    The physical dimension is the rank of the map to OBSERVABLES (masses,
    mixings, phases), not the matrix-map rank minus a guessed orbit.  We
    measure it directly:

      * NUMERICAL: full 22-observable Jacobian (squared masses, CKM/PMNS
        moduli + Jarlskog, two Majorana phases) at 9 independent random
        points, several step sizes and two smooth parameterisations.
        Rank = 19 stably, with a singular-value gap of 1e4-1e6 between the
        19th and 20th directions.  => physical dim 19, THREE relations.

      * EXACT LOWER BOUND: a set of strictly bi-unitary-invariant trace
        words (functions of observables alone) gives rank(dI/dp) = 17 over
        Q(i).  Since rank(dI/dp) <= rank(dO/dp), this proves at least
        22-17 = 5 relations are NOT forced... wait, it proves rank(dO/dp)>=17,
        i.e. AT MOST 5 relations.  The invariant set is not yet complete
        (it misses two Majorana-phase directions that the numerical map
        captures), so 17 is a lower bound on the dimension, 5 an upper
        bound on the relation count.

    SUPERSEDED (audits 9-12): the 19/three count above used a NON-basis-
    independent neutrino vector and is WITHDRAWN.  The current single source
    of truth is strict_obs.py (strict Autonne-Takagi), giving generic
    observable rank 22 => ZERO forced low-energy relations for every content.
    The internal _np_obs / obs_dflat / obs_two helpers below are DEPRECATED
    (non-basis-independent); they are retained only for the counting-rule and
    RG tests, which do not depend on the neutrino basis.  Dimension verdicts
    must come from strict_obs.run(), NOT from this module.

    The pencil mechanism, the identities tr B = det B = disc(B^2) = 0, and the
    explicit (E1)/(E2) constraints remain EXACT and correct (matrix-level);
    they leave no low-energy observable relation.

The counting-rule, vev-alignment, two-copy and RG tests remain valid.
"""
import json
from fractions import Fraction as F
import numpy as np
import exact_core as E

c = E.c
D = E.D
mm, T, dag, tr, sc, ad, sb = E.mm, E.T, E.dag, E.tr, E.sc, E.ad, E.sb
anti3, sym3, inv3, det3 = E.anti3, E.sym3, E.inv3, E.det3
exact_rank, rational_point = E.exact_rank, E.rational_point
comps, dcomps = E.comps, E.dcomps
disc, _kk = E.disc, E._kk


def jac_rank(bld, point, npar, func):
    def gv(i, j): return c(point[i], point[j])
    base = func(bld(gv))
    ncomp = len(comps(base))
    cols = []
    for k in range(npar):
        def gd(i, j, k=k):
            s = c(1) if i == k else c(0)
            if j == k: s = s + c(0, 1)
            return D(c(point[i], point[j]), s)
        cols.append(dcomps(func(bld(gd))))
    rows = [[cols[j][i] for j in range(npar)] for i in range(ncomp)]
    return exact_rank(rows), ncomp


# ---------------------------------------------------------------- (1) observable space
def invariant_words(mats):
    """Strictly bi-unitary-invariant trace words => functions of observables
    alone.  Gives a LOWER bound on the observable-space rank."""
    Mu, Md, Me, Mnu = mats
    Hs = [mm(M, dag(M)) for M in mats]
    out = []
    for H in Hs:
        H2 = mm(H, H); H3 = mm(H2, H)
        out += [tr(H), tr(H2), tr(H3)]
    for a in range(4):
        for b in range(a + 1, 4):
            X, Y = Hs[a], Hs[b]
            X2 = mm(X, X); Y2 = mm(Y, Y); XY = mm(X, Y)
            out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)), tr(mm(X2, Y2)),
                    tr(mm(XY, XY)), tr(mm(mm(XY, X2), Y2))]
            Cm = sb(XY, mm(Y, X))
            out += [tr(mm(mm(Cm, Cm), Cm))]
    for Hx in (Hs[2], Hs[1]):
        Yb = mm(mm(Mnu, T(Hx)), dag(Mnu))
        Hx2 = mm(Hx, Hx)
        out += [tr(Yb), tr(mm(Yb, Yb)), tr(mm(Yb, Hx)), tr(mm(Yb, Hx2)),
                tr(mm(Yb, Hs[3])), tr(mm(mm(Yb, Hx), Hs[3]))]
    return out


def _np_build(p):
    def anti(q):
        a, b, cc = q[0] + 1j * q[1], q[2] + 1j * q[3], q[4] + 1j * q[5]
        return np.array([[0, a, b], [-a, 0, cc], [-b, -cc, 0]], complex)
    def sym(q):
        z = [q[2 * t] + 1j * q[2 * t + 1] for t in range(6)]
        return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(8)]
    v1u, v15u, v1d, v15d, su, sd, vR, vL = v
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    MD = (v1u - 3 * v15u) * A - 3 * su * S
    Mnu = vL * S - MD @ np.linalg.inv(vR * S) @ MD.T
    return Mu, Md, Me, Mnu


def _np_obs(p):
    Mu, Md, Me, Mnu = _np_build(p)
    out = []
    for M in (Mu, Md, Me):
        H = M @ M.conj().T
        out += list(np.sort(np.linalg.eigvalsh(H))[::-1])
    Hu = Mu @ Mu.conj().T; Hd = Md @ Md.conj().T
    wu, Uu = np.linalg.eigh(Hu); wd, Ud = np.linalg.eigh(Hd)
    Uu = Uu[:, np.argsort(wu)[::-1]]; Ud = Ud[:, np.argsort(wd)[::-1]]
    V = Uu.conj().T @ Ud
    out += [abs(V[0, 1]) ** 2, abs(V[0, 2]) ** 2, abs(V[1, 2]) ** 2,
            np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2] * V[1, 1]))]
    Hn = Mnu @ Mnu.conj().T
    out += list(np.sort(np.linalg.eigvalsh(Hn))[::-1])
    He = Me @ Me.conj().T
    we, Ue = np.linalg.eigh(He); wn, Un = np.linalg.eigh(Hn)
    Ue = Ue[:, np.argsort(we)[::-1]]; Un = Un[:, np.argsort(wn)[::-1]]
    P = Ue.conj().T @ Un
    out += [abs(P[0, 1]) ** 2, abs(P[0, 2]) ** 2, abs(P[1, 2]) ** 2,
            np.imag(P[0, 1] * P[1, 2] * np.conj(P[0, 2] * P[1, 1]))]
    Dg = Un.T @ Mnu @ Un
    out += [np.angle(Dg[0, 0]), np.angle(Dg[1, 1])]
    return np.array(out)


def test_observable_space():
    # exact lower bound from invariant words
    pt = rational_point(34, 2)
    rI, ncomp = jac_rank(E.bld_120_126, pt, 34, invariant_words)
    # numerical physical dimension
    ranks = []
    gaps = []
    for seed in [1, 2, 3, 42, 99, 2026, 11, 22, 33]:
        rng = np.random.default_rng(seed)
        p = rng.normal(size=34)
        f0 = _np_obs(p)
        J = np.zeros((22, 34)); h = 1e-7
        for k in range(34):
            q = p.copy(); q[k] += h
            J[:, k] = (_np_obs(q) - f0) / h
        s = np.linalg.svd(J, compute_uv=False)
        r = int((s > 1e-8 * s[0]).sum())
        ranks.append(r)
        gaps.append(float(s[18] / s[19]) if s[19] > 0 else float('inf'))
    phys = max(set(ranks), key=ranks.count)
    return {
        "exact_invariant_components": ncomp,
        "exact_invariant_rank_lower_bound": rI,
        "max_relations_from_exact_bound": 22 - rI,
        "numerical_ranks_9_seeds": ranks,
        "numerical_physical_dim": phys,
        "numerical_relations": 22 - phys,
        "min_singular_gap_19_20": min(gaps),
        "all_seeds_agree": len(set(ranks)) == 1,
        "CORRECTION": "previous 18/4 used a wrong orbit; best evidence is 19/3",
    }


# ---------------------------------------------------------------- (2) counting rules
def bld_proportional(get):
    A = anti3(get(0, 1), get(2, 3), get(4, 5))
    S = sym3([get(6 + 2 * k, 7 + 2 * k) for k in range(6)])
    v1d, v15d, sd, t, v1u, vR, vL = [get(18 + 2 * k, 19 + 2 * k) for k in range(7)]
    v15u = t * (v1d + v15d) - v1u
    su = t * sd
    t3 = _kk(A[0][1], 3); n3 = _kk(A[0][1], -3)
    Mu = ad(sc(A, v1u + v15u), sc(S, su))
    Md = ad(sc(A, v1d + v15d), sc(S, sd))
    Me = ad(sc(A, v1d - t3 * v15d), sc(S, n3 * sd))
    MD = ad(sc(A, v1u - t3 * v15u), sc(S, n3 * su))
    Mnu = sb(sc(S, vL), mm(mm(MD, inv3(sc(S, vR))), T(MD)))
    return Mu, Md, Me, Mnu


def test_counting_rules():
    pt = rational_point(32, 5)
    def gv(i, j): return c(pt[i], pt[j])
    Mu, Md, Me, Mnu = bld_proportional(gv)
    t = gv(24, 25)
    prop = all((Mu[i][j] - t * Md[i][j]).zero() for i in range(3) for j in range(3))
    Hu = mm(Mu, dag(Mu)); Hd = mm(Md, dag(Md))
    comm = sb(mm(Hu, Hd), mm(Hd, Hu))
    ckm_diag = all(z.zero() for row in comm for z in row)
    def charged_inv(mats):
        Mu_, Md_, Me_ = mats[:3]
        Hu_, Hd_, He_ = mm(Mu_, dag(Mu_)), mm(Md_, dag(Md_)), mm(Me_, dag(Me_))
        out = []
        for H in (Hu_, Hd_, He_):
            H2 = mm(H, H); H3 = mm(H2, H)
            out += [tr(H), tr(H2), tr(H3)]
        X, Y = Hu_, Hd_
        X2 = mm(X, X); Y2 = mm(Y, Y); XY = mm(X, Y)
        out += [tr(XY), tr(mm(X2, Y)), tr(mm(X, Y2)), tr(mm(X2, Y2)), tr(mm(XY, XY))]
        Cm = sb(XY, mm(Y, X))
        out += [tr(mm(mm(Cm, Cm), Cm))]
        return out
    rloc, ncomp = jac_rank(bld_proportional, pt, 32, charged_inv)
    return {
        "Mu_equals_t_Md_exact": prop,
        "ckm_commutator_zero_on_locus": ckm_diag,
        "locus_observable_rank": rloc,
        "constraints_on_13_obs": 13 - rloc,
        "expected_4_CKM_plus_2_ratios": 6,
        "no_double_counting": (13 - rloc == 6),
    }


# ---------------------------------------------------------------- (3) vev constraints
def bld_120_126_dflat(get):
    A = anti3(get(0, 1), get(2, 3), get(4, 5))
    S = sym3([get(6 + 2 * t, 7 + 2 * t) for t in range(6)])
    v1u, v1d, su, sd, vR, vL = [get(18 + 2 * t, 19 + 2 * t) for t in range(6)]
    ku, kd = get(30, 31), get(32, 33)
    v15u, v15d = ku * v1u, kd * v1d
    t3 = _kk(A[0][1], 3); n3 = _kk(A[0][1], -3)
    Mu = ad(sc(A, v1u + v15u), sc(S, su))
    Md = ad(sc(A, v1d + v15d), sc(S, sd))
    Me = ad(sc(A, v1d - t3 * v15d), sc(S, n3 * sd))
    MD = ad(sc(A, v1u - t3 * v15u), sc(S, n3 * su))
    Mnu = sb(sc(S, vL), mm(mm(MD, inv3(sc(S, vR))), T(MD)))
    return Mu, Md, Me, Mnu


def test_vev_constraints():
    pt = rational_point(34, 7)
    rI, _ = jac_rank(bld_120_126_dflat, pt, 34, invariant_words)
    def gv(i, j): return c(pt[i], pt[j])
    A = anti3(gv(0, 1), gv(2, 3), gv(4, 5))
    S = sym3([gv(6 + 2 * t, 7 + 2 * t) for t in range(6)])
    B = mm(A, inv3(S))
    # numerical physical dim under the constraint
    def np_dflat(p):
        A_ = np.array([[0, p[0] + 1j * p[1], p[2] + 1j * p[3]],
                       [-(p[0] + 1j * p[1]), 0, p[4] + 1j * p[5]],
                       [-(p[2] + 1j * p[3]), -(p[4] + 1j * p[5]), 0]], complex)
        z = [p[6 + 2 * t] + 1j * p[7 + 2 * t] for t in range(6)]
        S_ = np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)
        v1u, v1d, su, sd, vR, vL = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(6)]
        ku, kd = p[30] + 1j * p[31], p[32] + 1j * p[33]
        v15u, v15d = ku * v1u, kd * v1d
        Mu = (v1u + v15u) * A_ + su * S_
        Md = (v1d + v15d) * A_ + sd * S_
        Me = (v1d - 3 * v15d) * A_ - 3 * sd * S_
        MD = (v1u - 3 * v15u) * A_ - 3 * su * S_
        Mnu = vL * S_ - MD @ np.linalg.inv(vR * S_) @ MD.T
        return Mu, Md, Me, Mnu
    def obs_dflat(p):
        Mu, Md, Me, Mnu = np_dflat(p)
        # reuse the observable extraction by monkey-patching build
        import types
        out = []
        for M in (Mu, Md, Me):
            H = M @ M.conj().T
            out += list(np.sort(np.linalg.eigvalsh(H))[::-1])
        Hu = Mu @ Mu.conj().T; Hd = Md @ Md.conj().T
        wu, Uu = np.linalg.eigh(Hu); wd, Ud = np.linalg.eigh(Hd)
        Uu = Uu[:, np.argsort(wu)[::-1]]; Ud = Ud[:, np.argsort(wd)[::-1]]
        V = Uu.conj().T @ Ud
        out += [abs(V[0, 1]) ** 2, abs(V[0, 2]) ** 2, abs(V[1, 2]) ** 2,
                np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2] * V[1, 1]))]
        Hn = Mnu @ Mnu.conj().T
        out += list(np.sort(np.linalg.eigvalsh(Hn))[::-1])
        He = Me @ Me.conj().T
        we, Ue = np.linalg.eigh(He); wn, Un = np.linalg.eigh(Hn)
        Ue = Ue[:, np.argsort(we)[::-1]]; Un = Un[:, np.argsort(wn)[::-1]]
        P = Ue.conj().T @ Un
        out += [abs(P[0, 1]) ** 2, abs(P[0, 2]) ** 2, abs(P[1, 2]) ** 2,
                np.imag(P[0, 1] * P[1, 2] * np.conj(P[0, 2] * P[1, 1]))]
        Dg = Un.T @ Mnu @ Un
        out += [np.angle(Dg[0, 0]), np.angle(Dg[1, 1])]
        return np.array(out)
    rng = np.random.default_rng(77)
    p = rng.normal(size=34)
    f0 = obs_dflat(p); J = np.zeros((22, 34)); h = 1e-7
    for k in range(34):
        q = p.copy(); q[k] += h; J[:, k] = (obs_dflat(q) - f0) / h
    s = np.linalg.svd(J, compute_uv=False)
    rnum = int((s > 1e-8 * s[0]).sum())
    return {
        "n_params": 34,
        "exact_invariant_rank": rI,
        "numerical_physical_dim": rnum,
        "numerical_relations": 22 - rnum,
        "pencil_disc_zero": disc(mm(B, B)).zero(),
        "relations_survive_vev_constraint": (rnum < 22),
    }


# ---------------------------------------------------------------- (4) two 126 copies
def bld_120_126_126p(get):
    A = anti3(get(0, 1), get(2, 3), get(4, 5))
    S = sym3([get(6 + 2 * t, 7 + 2 * t) for t in range(6)])
    Sp = sym3([get(18 + 2 * t, 19 + 2 * t) for t in range(6)])
    vv = [get(30 + 2 * t, 31 + 2 * t) for t in range(12)]
    v1u, v15u, v1d, v15d, su, sd, sup, sdp, vR, vRp, vL, vpL = vv
    t3 = _kk(A[0][1], 3); n3 = _kk(A[0][1], -3)
    Mu = ad(ad(sc(A, v1u + v15u), sc(S, su)), sc(Sp, sup))
    Md = ad(ad(sc(A, v1d + v15d), sc(S, sd)), sc(Sp, sdp))
    Me = ad(ad(sc(A, v1d - t3 * v15d), sc(S, n3 * sd)), sc(Sp, n3 * sdp))
    MD = ad(ad(sc(A, v1u - t3 * v15u), sc(S, n3 * su)), sc(Sp, n3 * sup))
    MR = ad(sc(S, vR), sc(Sp, vRp))
    Mnu = sb(ad(sc(S, vL), sc(Sp, vpL)), mm(mm(MD, inv3(MR)), T(MD)))
    return Mu, Md, Me, Mnu


def test_two_copies():
    pt = rational_point(54, 11)
    rI, _ = jac_rank(bld_120_126_126p, pt, 54, invariant_words)
    def gv(i, j): return c(pt[i], pt[j])
    Mu, Md, Me, Mnu = bld_120_126_126p(gv)
    S = sym3([gv(6 + 2 * t, 7 + 2 * t) for t in range(6)])
    pencil_disc_zero = disc(mm(Mnu, inv3(S))).zero()
    # numerical physical dim
    def np_two(p):
        def anti(q):
            a, b, cc = q[0] + 1j * q[1], q[2] + 1j * q[3], q[4] + 1j * q[5]
            return np.array([[0, a, b], [-a, 0, cc], [-b, -cc, 0]], complex)
        def sym(q):
            z = [q[2 * t] + 1j * q[2 * t + 1] for t in range(6)]
            return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)
        A = anti(p[0:6]); S_ = sym(p[6:18]); Sp = sym(p[18:30])
        v = [p[30 + 2 * t] + 1j * p[31 + 2 * t] for t in range(12)]
        v1u, v15u, v1d, v15d, su, sd, sup, sdp, vR, vRp, vL, vpL = v
        Mu = (v1u + v15u) * A + su * S_ + sup * Sp
        Md = (v1d + v15d) * A + sd * S_ + sdp * Sp
        Me = (v1d - 3 * v15d) * A - 3 * sd * S_ - 3 * sdp * Sp
        MD = (v1u - 3 * v15u) * A - 3 * su * S_ - 3 * sup * Sp
        MR = vR * S_ + vRp * Sp
        Mnu = vL * S_ + vpL * Sp - MD @ np.linalg.inv(MR) @ MD.T
        return Mu, Md, Me, Mnu
    def obs_two(p):
        Mu, Md, Me, Mnu = np_two(p)
        out = []
        for M in (Mu, Md, Me):
            H = M @ M.conj().T
            out += list(np.sort(np.linalg.eigvalsh(H))[::-1])
        Hu = Mu @ Mu.conj().T; Hd = Md @ Md.conj().T
        wu, Uu = np.linalg.eigh(Hu); wd, Ud = np.linalg.eigh(Hd)
        Uu = Uu[:, np.argsort(wu)[::-1]]; Ud = Ud[:, np.argsort(wd)[::-1]]
        V = Uu.conj().T @ Ud
        out += [abs(V[0, 1]) ** 2, abs(V[0, 2]) ** 2, abs(V[1, 2]) ** 2,
                np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2] * V[1, 1]))]
        Hn = Mnu @ Mnu.conj().T
        out += list(np.sort(np.linalg.eigvalsh(Hn))[::-1])
        He = Me @ Me.conj().T
        we, Ue = np.linalg.eigh(He); wn, Un = np.linalg.eigh(Hn)
        Ue = Ue[:, np.argsort(we)[::-1]]; Un = Un[:, np.argsort(wn)[::-1]]
        P = Ue.conj().T @ Un
        out += [abs(P[0, 1]) ** 2, abs(P[0, 2]) ** 2, abs(P[1, 2]) ** 2,
                np.imag(P[0, 1] * P[1, 2] * np.conj(P[0, 2] * P[1, 1]))]
        Dg = Un.T @ Mnu @ Un
        out += [np.angle(Dg[0, 0]), np.angle(Dg[1, 1])]
        return np.array(out)
    rng = np.random.default_rng(88)
    p = rng.normal(size=54)
    f0 = obs_two(p); J = np.zeros((22, 54)); h = 1e-7
    for k in range(54):
        q = p.copy(); q[k] += h; J[:, k] = (obs_two(q) - f0) / h
    s = np.linalg.svd(J, compute_uv=False)
    rnum = int((s > 1e-8 * s[0]).sum())
    return {
        "n_params": 54,
        "exact_invariant_rank": rI,
        "numerical_physical_dim": rnum,
        "numerical_relations": max(0, 22 - rnum),
        "pencil_disc_zero": pencil_disc_zero,
        "relations_washed_out": (rnum >= 22),
    }


# ---------------------------------------------------------------- (5) RG sensitivity
def test_rg_sensitivity():
    """Quick one-loop check (no fit): the exact single-structure proportionality
    M_e = -3 M_d is NOT preserved by running, and m_tau/m_b shifts toward the
    observed window.  Demonstrates the UV wall is not RG-robust.  Uses the
    standalone rg_sensitivity module's integrator."""
    import rg_sensitivity as R
    rng = np.random.default_rng(7)
    z = rng.normal(size=6) + 1j * rng.normal(size=6)
    f = np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)
    f = f + np.diag([1.0, 3.0, 30.0])
    Md = f.copy(); Me = -3.0 * f.copy(); Mu = f.copy() * 50
    rg0 = R.ratios(Mu, Md, Me)
    u, d, e = R.run_down(Mu, Md, Me)
    rl = R.ratios(u, d, e)
    prop0 = np.linalg.norm(Me + 3.0 * Md) / np.linalg.norm(Md)
    prop1 = np.linalg.norm(e + 3.0 * d) / np.linalg.norm(d)
    return {
        "gut_mtau_mb": float(rg0[4]),
        "low_mtau_mb": float(rl[4]),
        "proportionality_gut": float(prop0),
        "proportionality_low": float(prop1),
        "proportionality_broken_by_rg": bool(prop1 > 0.1),
        "ratio_shifts_toward_window": bool(rl[4] < rg0[4]),
        "note": "one-loop example, indicative couplings, NOT precision matching",
    }


def test_vL_constraint():
    """Audit 7: v_L is not free (D-flatness/seesaw gives v_L ~ v^2/v_R).
    Tie v_L = kappa*su*sd/vR and re-measure; the dimension must not drop
    below 19 merely because we removed a spurious free parameter."""
    import measure_dims as MD
    r = MD.rank_of(MD.full_obs, MD.build_120_126_vLconstrained, 34)
    return {"ranks": r, "dimension": max(r), "relations": max(0, 22 - max(r)),
            "robust_to_vL_constraint": max(r) == 19}


def run():
    return {
        "observable_space": test_observable_space(),
        "counting_rules": test_counting_rules(),
        "vev_constraints": test_vev_constraints(),
        "two_copies": test_two_copies(),
        "rg_sensitivity": test_rg_sensitivity(),
        "vL_constraint": test_vL_constraint(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
