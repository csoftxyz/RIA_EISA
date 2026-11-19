"""
ITEM 1, done properly: SCALE-FREE test of 120+126 viability.

The previous fit diverged because the overall scale is a free vev and the
optimiser ran along it.  Physical viability is about RATIOS, so we fit only
scale-invariant combinations:

    down:    m_s/m_b, m_d/m_b
    lepton:  m_mu/m_tau, m_e/m_tau
    cross:   m_tau/m_b

That is 5 numbers.  The model has 30 real parameters minus redundancies, so
plenty of freedom -- unless a structural obstruction exists.

The candidate obstruction: (M_e)_ii = -3 (M_d)_ii exactly (zero diagonal of A).
For strongly hierarchical matrices this pins m_tau/m_b near 3.  We test whether
the observed window 1.3-1.8 is reachable AT THE SAME TIME as the observed
hierarchies, by fitting the five ratios with a scale-free residual.
"""
import numpy as np

rng = np.random.default_rng(424242)


def anti(p):
    a, b, c = p[0]+1j*p[1], p[2]+1j*p[3], p[4]+1j*p[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(p):
    z = [p[2*t]+1j*p[2*t+1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)


def build(p):
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18+2*t]+1j*p[19+2*t] for t in range(6)]
    v1u, v15u, v1d, v15d, su, sd = v
    Mu = (v1u+v15u)*A + su*S
    Md = (v1d+v15d)*A + sd*S
    Me = (v1d-3*v15d)*A - 3*sd*S
    return Mu, Md, Me


sv = lambda M: np.linalg.svd(M, compute_uv=False)

# observed GUT-scale ratios (order of magnitude, from running literature)
TGT = np.array([
    0.022,    # m_s/m_b
    0.0011,   # m_d/m_b
    0.062,    # m_mu/m_tau
    0.00029,  # m_e/m_tau
    1.60,     # m_tau/m_b
])
NAMES = ["m_s/m_b", "m_d/m_b", "m_mu/m_tau", "m_e/m_tau", "m_tau/m_b"]


def ratios(p):
    Mu, Md, Me = build(p)
    d = sv(Md); e = sv(Me)
    if min(d[0], e[0]) <= 1e-300:
        return None
    return np.array([d[1]/d[0], d[2]/d[0], e[1]/e[0], e[2]/e[0], e[0]/d[0]])


def resid(p):
    r = ratios(p)
    if r is None or np.any(r <= 0):
        return np.full(5, 50.0)
    return np.log(r/TGT)


if __name__ == "__main__":
    print("=" * 78)
    print("SCALE-FREE FIT OF 120+126 TO THE OBSERVED CHARGED RATIOS")
    print("=" * 78)
    print("  targets:", ", ".join("%s=%.4g" % (n, t) for n, t in zip(NAMES, TGT)))
    print()
    best = None
    for trial in range(40):
        p = rng.normal(size=30)
        # keep vevs away from zero
        for k in range(18, 30, 2):
            p[k] = 0.8 + 0.3*abs(p[k])
        for it in range(250):
            r = resid(p)
            J = np.zeros((5, 30))
            h = 1e-7
            for k in range(30):
                q = p.copy(); q[k] += h
                J[:, k] = (resid(q)-r)/h
            try:
                dp = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            # trust-region-ish damping
            nrm = np.linalg.norm(dp)
            if nrm > 1.0:
                dp = dp/nrm
            p = p + dp
            if np.linalg.norm(r) < 1e-9:
                break
        r = resid(p)
        cost = np.linalg.norm(r)
        if best is None or cost < best[0]:
            best = (cost, p.copy(), ratios(p))
    cost, p, got = best
    print("  best |log residual| over 40 starts : %.4f" % cost)
    print()
    print("  %-12s %12s %12s %10s" % ("ratio", "target", "fitted", "log dev"))
    for n, t, g in zip(NAMES, TGT, got):
        print("  %-12s %12.5g %12.5g %10.3f" % (n, t, g, np.log(g/t)))
    print()
    mtb = got[4]
    print("  m_tau/m_b fitted = %.3f   observed window ~1.3-1.8" % mtb)
    if abs(np.log(mtb/1.6)) < 0.15 and cost < 0.5:
        print("  => the observed window IS reachable: content phenomenologically viable.")
        verdict = "VIABLE"
    else:
        print("  => the observed window is NOT reached together with the hierarchies.")
        verdict = "EXCLUDED"
    print()
    print("=" * 78)
    print("HOW CLOSE CAN m_tau/m_b GET, WITH HIERARCHIES FIXED?")
    print("=" * 78)
    # fit only the four hierarchy ratios, then report the resulting m_tau/m_b
    def resid4(p):
        r = ratios(p)
        if r is None or np.any(r <= 0):
            return np.full(4, 50.0)
        return np.log(r[:4]/TGT[:4])
    best4 = None
    for trial in range(40):
        p = rng.normal(size=30)
        for k in range(18, 30, 2):
            p[k] = 0.8 + 0.3*abs(p[k])
        for it in range(250):
            r = resid4(p)
            J = np.zeros((4, 30)); h = 1e-7
            for k in range(30):
                q = p.copy(); q[k] += h
                J[:, k] = (resid4(q)-r)/h
            try:
                dp = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            nrm = np.linalg.norm(dp)
            if nrm > 1.0:
                dp = dp/nrm
            p = p + dp
            if np.linalg.norm(r) < 1e-9:
                break
        r = resid4(p); c = np.linalg.norm(r)
        if best4 is None or c < best4[0]:
            best4 = (c, p.copy(), ratios(p))
    c4, p4, g4 = best4
    print("  best |log residual| on the FOUR hierarchies: %.4f" % c4)
    for n, t, g in zip(NAMES[:4], TGT[:4], g4[:4]):
        print("    %-12s target %.5g  fitted %.5g" % (n, t, g))
    print("  resulting m_tau/m_b = %.4f" % g4[4])
    print()
    print("  VERDICT: %s" % verdict)
    print()
    print("  Literature cross-check: Bajc-Melfo-Senjanovic-Vissani (hep-ph/0510139)")
    print("  and Babu-Bajc-Saad (1612.04329, Sec.2) state that 120+126 predicts")
    print("  m_tau/m_b ~ 3 with O(m_s/m_b) corrections, and m_t/m_b ~ 1.")
    print("  Our numerical finding should be compared with that statement.")
