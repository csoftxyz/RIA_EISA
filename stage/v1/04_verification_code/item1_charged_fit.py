"""
ITEM 1: can 120+126 actually fit the CHARGED spectrum?

Our census calls a content "viable" when it does not force M_d proportional to
M_e.  That is a STRUCTURAL criterion, not a fit.  The audit asks the sharper
question: does the content reproduce the observed masses?

There is a known obstruction.  In 120+126,
    M_d = a A + s_d S ,   M_e = b A - 3 s_d S ,
with A antisymmetric (ZERO DIAGONAL) and S symmetric.  Hence
    M_e + 3 M_d = (b + 3a) A ,
so the DIAGONAL entries obey  (M_e)_ii = -3 (M_d)_ii  exactly, in any basis.
For hierarchical matrices the 33 entry controls the largest singular value, so
    m_tau / m_b  ->  3   up to off-diagonal corrections.
Bajc-Melfo-Senjanovic-Vissani (hep-ph/0510139) and Babu-Bajc-Saad (1612.04329)
state exactly this: 120+126 predicts m_tau/m_b ~ 3 with O(m_s/m_b) deviation,
while the measured GUT-scale value is ~1.4-1.7.  Also m_t/m_b comes out O(1)
instead of ~70.

This script tests both statements numerically, without assuming them.
"""
import numpy as np

rng = np.random.default_rng(20261118)


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
    return Mu, Md, Me, A, S, sd


sv = lambda M: np.linalg.svd(M, compute_uv=False)


if __name__ == "__main__":
    print("=" * 78)
    print("A. THE DIAGONAL IDENTITY  (M_e)_ii = -3 (M_d)_ii")
    print("=" * 78)
    worst = 0.0
    for _ in range(200):
        p = rng.normal(size=30)
        Mu, Md, Me, A, S, sd = build(p)
        for i in range(3):
            worst = max(worst, abs(Me[i, i] + 3*Md[i, i])/max(abs(Md[i, i]), 1e-300))
    print("  max relative violation over 200 points: %.2e" % worst)
    print("  => EXACT: the antisymmetric Yukawa has zero diagonal, so the 126")
    print("     alone fixes every diagonal entry with weight -3.")

    print()
    print("=" * 78)
    print("B. DOES THAT FORCE m_tau/m_b ~ 3 ?  (scan, no assumption)")
    print("=" * 78)
    ratios = []
    hier = []
    for _ in range(4000):
        p = rng.normal(size=30)
        Mu, Md, Me, A, S, sd = build(p)
        d = sv(Md); e = sv(Me)
        if d[0] <= 0:
            continue
        ratios.append(e[0]/d[0])
        hier.append(d[1]/d[0])
    ratios = np.array(ratios); hier = np.array(hier)
    print("  m_tau/m_b (= largest singular value ratio) over 4000 random points:")
    print("    min %.3f   median %.3f   max %.3f" % (ratios.min(), np.median(ratios), ratios.max()))
    print("    fraction within 10%% of 3 : %.3f" % np.mean(np.abs(ratios-3) < 0.3))
    print("    fraction in [1.3, 1.8]   : %.3f   <-- observed GUT window"
          % np.mean((ratios > 1.3) & (ratios < 1.8)))
    # restrict to hierarchical down spectra, as nature is
    sel = hier < 0.1
    if sel.sum() > 20:
        r = ratios[sel]
        print("  restricted to hierarchical M_d (m_s/m_b < 0.1), %d points:" % sel.sum())
        print("    min %.3f  median %.3f  max %.3f" % (r.min(), np.median(r), r.max()))
        print("    fraction in [1.3,1.8] : %.3f" % np.mean((r > 1.3) & (r < 1.8)))
    sel2 = hier < 0.03
    if sel2.sum() > 5:
        r = ratios[sel2]
        print("  strongly hierarchical (m_s/m_b < 0.03), %d points:" % sel2.sum())
        print("    min %.3f  median %.3f  max %.3f" % (r.min(), np.median(r), r.max()))

    print()
    print("=" * 78)
    print("C. CAN A TARGETED FIT REACH THE OBSERVED WINDOW?")
    print("=" * 78)
    # try to fit down masses hierarchical AND m_tau/m_b ~ 1.6
    # down GUT-scale masses (GeV), order of magnitude
    md_t = np.array([1.1e-3, 2.2e-2, 1.0])
    me_t = np.array([0.47e-3, 9.9e-2, 1.6])   # m_tau/m_b = 1.6
    best = None
    for trial in range(30):
        p = rng.normal(size=30)
        for it in range(300):
            Mu, Md, Me, A, S, sd = build(p)
            d = sv(Md); e = sv(Me)
            # residual on log masses (scale invariant)
            r = np.concatenate([np.log(d/md_t), np.log(e/me_t)])
            J = np.zeros((len(r), 30))
            h = 1e-7
            for k in range(30):
                q = p.copy(); q[k] += h
                _, Md2, Me2, _, _, _ = build(q)
                d2 = sv(Md2); e2 = sv(Me2)
                r2 = np.concatenate([np.log(d2/md_t), np.log(e2/me_t)])
                J[:, k] = (r2-r)/h
            try:
                dp = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            p = p + 0.4*dp
            if np.linalg.norm(r) < 1e-10:
                break
        Mu, Md, Me, A, S, sd = build(p)
        d = sv(Md); e = sv(Me)
        r = np.concatenate([np.log(d/md_t), np.log(e/me_t)])
        cost = np.linalg.norm(r)
        if best is None or cost < best[0]:
            best = (cost, p.copy(), d.copy(), e.copy())
    cost, p, d, e = best
    print("  best log-residual over 30 starts: %.4f" % cost)
    print("  fitted down masses : %.3e %.3e %.3e" % tuple(d))
    print("  target down masses : %.3e %.3e %.3e" % tuple(md_t))
    print("  fitted lepton masses: %.3e %.3e %.3e" % tuple(e))
    print("  target lepton masses: %.3e %.3e %.3e" % tuple(me_t))
    print("  fitted m_tau/m_b = %.3f   (target 1.600)" % (e[0]/d[0]))
    print()
    if cost > 0.3:
        print("  => CANNOT fit: the charged spectrum of 120+126 is INCOMPATIBLE")
        print("     with observation.  The content is structurally splitting but")
        print("     phenomenologically EXCLUDED, exactly as the literature states.")
    else:
        print("  => a fit was found; the obstruction is not absolute.")

    print()
    print("=" * 78)
    print("D. m_t/m_b IN THIS CONTENT")
    print("=" * 78)
    rt = []
    for _ in range(3000):
        p = rng.normal(size=30)
        Mu, Md, Me, A, S, sd = build(p)
        u = sv(Mu); d = sv(Md)
        if d[0] > 0:
            rt.append(u[0]/d[0])
    rt = np.array(rt)
    print("  m_t/m_b over 3000 points: min %.2f median %.2f max %.2f"
          % (rt.min(), np.median(rt), rt.max()))
    print("  fraction above 30 (observed ~70): %.4f" % np.mean(rt > 30))
    print("  NOTE: both M_u and M_d are combinations of the SAME A and S, so")
    print("        their overall scales are set by free vevs; a large ratio is")
    print("        possible in principle.  The binding obstruction is m_tau/m_b.")
