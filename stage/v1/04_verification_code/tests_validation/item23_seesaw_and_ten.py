"""
ITEM 2: is the alignment obstruction robust against the seesaw mechanism?
ITEM 3: does adding a 10 remove the m_tau/m_b obstruction?

ITEM 2 -- analytic, then verified.
In 120+126 there is only ONE symmetric Yukawa f.  Both
    M_D = c_D v_R f      and      M_R = v_R f
are proportional to it.  Hence
    M_nu^I  = M_D M_R^{-1} M_D^T = c_D^2 v_R f ,
    M_nu^II = v_L f ,
    M_nu^mixed = (v_L - c_D^2 v_R) f .
ALL THREE are proportional to f.  So the charged-lepton alignment obstruction
cannot be evaded by choosing type-I, type-II, or mixed dominance.  This is a
stronger statement than the type-II-only result in the paper.

ITEM 3 -- adding a 10 supplies a SECOND symmetric matrix h, so
    M_d = v10 h + a A + s_d S ,   M_e = v10 h + b A - 3 s_d S ,
    M_e + 3 M_d = 4 v10 h + (b+3a) A ,
and the diagonal relation becomes (M_e)_ii + 3(M_d)_ii = 4 v10 h_ii, which is
no longer zero.  The m_tau/m_b ~ 3 prediction is thereby relaxed.  We test
numerically whether the observed window becomes reachable.
"""
import numpy as np

rng = np.random.default_rng(90210)


def anti(p):
    a, b, c = p[0]+1j*p[1], p[2]+1j*p[3], p[4]+1j*p[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(p):
    z = [p[2*t]+1j*p[2*t+1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)


sv = lambda M: np.linalg.svd(M, compute_uv=False)

TGT = np.array([0.022, 0.0011, 0.062, 0.00029, 1.60])
NAMES = ["m_s/m_b", "m_d/m_b", "m_mu/m_tau", "m_e/m_tau", "m_tau/m_b"]


# ---------------- ITEM 2 ----------------
def item2():
    print("=" * 78)
    print("ITEM 2: SEESAW-TYPE ROBUSTNESS OF THE ALIGNMENT OBSTRUCTION")
    print("=" * 78)
    worst = {"I": 0.0, "II": 0.0, "mixed": 0.0}
    for _ in range(300):
        p = rng.normal(size=30)
        f = sym(p[6:18])
        if abs(np.linalg.det(f)) < 1e-8:
            continue
        vL = p[18]+1j*p[19]
        vR = p[20]+1j*p[21]
        cD = p[22]+1j*p[23]
        MD = cD*vR*f
        MR = vR*f
        nuI = MD @ np.linalg.inv(MR) @ MD.T
        nuII = vL*f
        nuMix = nuII - nuI
        for key, Mn in (("I", nuI), ("II", nuII), ("mixed", nuMix)):
            # test proportionality to f via cross-product of vectorised matrices
            a = Mn.ravel(); b = f.ravel()
            num = np.linalg.norm(np.outer(a, b) - np.outer(b, a))
            den = np.linalg.norm(a)*np.linalg.norm(b)
            if den > 0:
                worst[key] = max(worst[key], num/den)
    for k in ("I", "II", "mixed"):
        print("  max relative deviation from M_nu proportional to f  [%-5s]: %.2e"
              % (k, worst[k]))
    print()
    print("  => In 120+126 ALL seesaw types give M_nu proportional to f, because")
    print("     M_D and M_R are both proportional to the single symmetric f.")
    print("     The obstruction is therefore INDEPENDENT of the seesaw mechanism.")
    print("     (This strengthens the paper: no type-I escape in the minimal content.)")


# ---------------- ITEM 3 ----------------
def build_with10(p):
    """10 + 120 + 126: h (sym, 10), A (anti, 120), S (sym, 126)."""
    h = sym(p[0:12])
    A = anti(p[12:18])
    S = sym(p[18:30])
    v = [p[30+2*t]+1j*p[31+2*t] for t in range(7)]
    v10u, v10d, v1u, v15u, v1d, v15d, su, sd = v[:8] if len(v) >= 8 else (v+[0]*(8-len(v)))
    Md = v10d*h + (v1d+v15d)*A + sd*S
    Me = v10d*h + (v1d-3*v15d)*A - 3*sd*S
    return Md, Me


def build_10_126(p):
    """10 + 126 only: h and S symmetric."""
    h = sym(p[0:12])
    S = sym(p[12:24])
    v = [p[24+2*t]+1j*p[25+2*t] for t in range(4)]
    v10d, v15d, su, sd = v
    Md = v10d*h + sd*S
    Me = v10d*h - 3*sd*S
    return Md, Me


def ratios_from(Md, Me):
    d = sv(Md); e = sv(Me)
    if min(d[0], e[0]) <= 1e-300:
        return None
    return np.array([d[1]/d[0], d[2]/d[0], e[1]/e[0], e[2]/e[0], e[0]/d[0]])


def fit(builder, npar, label, nstart=25, nit=200):
    def resid(p):
        Md, Me = builder(p)
        r = ratios_from(Md, Me)
        if r is None or np.any(r <= 0):
            return np.full(5, 50.0)
        return np.log(r/TGT)
    best = None
    for _ in range(nstart):
        p = rng.normal(size=npar)
        for it in range(nit):
            r = resid(p)
            J = np.zeros((5, npar)); h = 1e-7
            for k in range(npar):
                q = p.copy(); q[k] += h
                J[:, k] = (resid(q)-r)/h
            try:
                dp = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            n = np.linalg.norm(dp)
            if n > 1.0:
                dp = dp/n
            p = p + dp
            if np.linalg.norm(r) < 1e-9:
                break
        r = resid(p); c = np.linalg.norm(r)
        if best is None or c < best[0]:
            best = (c, ratios_from(*builder(p)))
    c, g = best
    print("  %-16s best |log resid| = %8.4f   m_tau/m_b = %7.3f" % (label, c, g[4]))
    return c, g


def item3():
    print()
    print("=" * 78)
    print("ITEM 3: DOES ADDING A 10 REMOVE THE m_tau/m_b OBSTRUCTION?")
    print("=" * 78)
    print("  targets:", ", ".join("%s=%.4g" % (n, t) for n, t in zip(NAMES, TGT)))
    print()
    c1, g1 = fit(build_10_126, 32, "10+126")
    c2, g2 = fit(build_with10, 44, "10+120+126")
    print()
    for label, c, g in (("10+126", c1, g1), ("10+120+126", c2, g2)):
        print("  %s fitted ratios:" % label)
        for n, t, v in zip(NAMES, TGT, g):
            print("    %-12s target %10.5g   fitted %10.5g" % (n, t, v))
        print()
    print("  Interpretation:")
    print("   - with a 10 present the diagonal identity is relaxed, so m_tau/m_b")
    print("     is no longer pinned near 3;")
    print("   - a small residual means the content CAN reproduce the observed")
    print("     charged ratios, i.e. it is phenomenologically viable, not merely")
    print("     structurally splitting.")


if __name__ == "__main__":
    item2()
    item3()
