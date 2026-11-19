"""
Decisive viability test demanded by audit 4, items 5 and 10.

The question: does 120+126 admit a point reproducing the observed charged
ratios, or is it excluded?  Earlier scripts used a weak damped Gauss-Newton
with few starts and returned large residuals (3.8-4.9 in log units).  A large
residual from a weak optimiser is NOT evidence of exclusion.  Here we use
Levenberg-Marquardt with batched finite differences and many starts.

Two target sets are fitted separately, because they test different claims:

  SET5 : m_s/m_b, m_d/m_b, m_mu/m_tau, m_e/m_tau, m_tau/m_b
  SET6 : SET5 plus m_t/m_b

Targets are INDICATIVE GUT-scale values (order of magnitude, as used in the
running literature), not precision inputs.  No RG evolution is performed.

Models (charged sector, two bidoublets for the 120 kept independent):
  120+126 : M_u=(v1u+v15u)A+s_u S, M_d=(v1d+v15d)A+s_d S,
            M_e=(v1d-3v15d)A-3 s_d S            30 real parameters
  10+126  : M_u=v_u10 h+s_u S, M_d=v_d10 h+s_d S, M_e=v_d10 h-3 s_d S
                                                32 real parameters

A fit reaching small residual is an EXISTENCE PROOF of a viable point for
those ratios.  Failure after many starts is evidence, not proof, of exclusion.
"""
import numpy as np

rng = np.random.default_rng(20261118)

TGT5 = np.array([0.022, 0.0011, 0.062, 0.00029, 1.60])
TGT6 = np.concatenate([TGT5, [70.0]])
NAMES = ["m_s/m_b", "m_d/m_b", "m_mu/m_tau", "m_e/m_tau", "m_tau/m_b", "m_t/m_b"]

N_STARTS = 120
N_ITER = 400


def sym_batch(P, off):
    n = P.shape[0]
    z = [P[:, off + 2 * t] + 1j * P[:, off + 2 * t + 1] for t in range(6)]
    M = np.zeros((n, 3, 3), complex)
    M[:, 0, 0] = z[0]
    M[:, 0, 1] = z[1]; M[:, 1, 0] = z[1]
    M[:, 0, 2] = z[2]; M[:, 2, 0] = z[2]
    M[:, 1, 1] = z[3]
    M[:, 1, 2] = z[4]; M[:, 2, 1] = z[4]
    M[:, 2, 2] = z[5]
    return M


def anti_batch(P, off):
    n = P.shape[0]
    a = P[:, off] + 1j * P[:, off + 1]
    b = P[:, off + 2] + 1j * P[:, off + 3]
    c = P[:, off + 4] + 1j * P[:, off + 5]
    M = np.zeros((n, 3, 3), complex)
    M[:, 0, 1] = a; M[:, 1, 0] = -a
    M[:, 0, 2] = b; M[:, 2, 0] = -b
    M[:, 1, 2] = c; M[:, 2, 1] = -c
    return M


def cvec(P, off):
    return (P[:, off] + 1j * P[:, off + 1])[:, None, None]


def build_120_126(P):
    A = anti_batch(P, 0)
    S = sym_batch(P, 6)
    v1u, v15u, v1d, v15d, su, sd = [cvec(P, 18 + 2 * t) for t in range(6)]
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    return Mu, Md, Me


def build_10_126(P):
    h = sym_batch(P, 0)
    S = sym_batch(P, 12)
    vu10, vd10, su, sd = [cvec(P, 24 + 2 * t) for t in range(4)]
    Mu = vu10 * h + su * S
    Md = vd10 * h + sd * S
    Me = vd10 * h - 3 * sd * S
    return Mu, Md, Me


MODELS = {"120+126": (build_120_126, 30), "10+126": (build_10_126, 32)}


def ratios_batch(build, P, ntarget):
    Mu, Md, Me = build(P)
    u = np.linalg.svd(Mu, compute_uv=False)
    d = np.linalg.svd(Md, compute_uv=False)
    e = np.linalg.svd(Me, compute_uv=False)
    tiny = 1e-300
    r = np.stack([d[:, 1] / np.maximum(d[:, 0], tiny),
                  d[:, 2] / np.maximum(d[:, 0], tiny),
                  e[:, 1] / np.maximum(e[:, 0], tiny),
                  e[:, 2] / np.maximum(e[:, 0], tiny),
                  e[:, 0] / np.maximum(d[:, 0], tiny)], axis=1)
    if ntarget == 6:
        r = np.concatenate([r, (u[:, 0] / np.maximum(d[:, 0], tiny))[:, None]], axis=1)
    return r


def resid_batch(build, P, tgt):
    r = ratios_batch(build, P, len(tgt))
    bad = ~np.isfinite(r) | (r <= 0)
    r = np.where(bad, 1.0, r)
    out = np.log(r / tgt)
    out[np.any(bad, axis=1)] = 50.0
    return out


def lm_fit(build, npar, tgt, p0):
    p = p0.copy()
    r = resid_batch(build, p[None, :], tgt)[0]
    cost = float(np.dot(r, r))
    lam = 1e-3
    h = 1e-7
    for _ in range(N_ITER):
        if cost < 1e-16:
            break
        P = np.repeat(p[None, :], npar + 1, axis=0)
        for k in range(npar):
            P[k + 1, k] += h
        R = resid_batch(build, P, tgt)
        J = ((R[1:] - R[0]) / h).T          # (ntarget, npar)
        g = J.T @ r
        H = J.T @ J
        diag = np.maximum(np.diag(H), 1e-12)
        ok = False
        for _try in range(12):
            try:
                dp = np.linalg.solve(H + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            pn = p + dp
            rn = resid_batch(build, pn[None, :], tgt)[0]
            cn = float(np.dot(rn, rn))
            if cn < cost:
                p, r, cost = pn, rn, cn
                lam = max(lam / 3.0, 1e-12)
                ok = True
                break
            lam *= 5.0
        if not ok:
            break
    return cost, p


def run(name, ntarget):
    build, npar = MODELS[name]
    tgt = TGT5 if ntarget == 5 else TGT6
    best = None
    costs = []
    for s in range(N_STARTS):
        p0 = rng.normal(size=npar) * (1.0 if s % 3 else 0.3)
        c, p = lm_fit(build, npar, tgt, p0)
        costs.append(c)
        if best is None or c < best[0]:
            best = (c, p.copy())
    costs = np.array(costs)
    c, p = best
    got = ratios_batch(build, p[None, :], ntarget)[0]
    return c, got, costs, p


if __name__ == "__main__":
    print("=" * 78)
    print("DECISIVE VIABILITY FIT  (Levenberg-Marquardt, %d starts, seed 20261118)"
          % N_STARTS)
    print("=" * 78)
    print("Targets are INDICATIVE GUT-scale ratios; no RG evolution is performed.")
    summary = {}
    for name in ["120+126", "10+126"]:
        for nt in [5, 6]:
            c, got, costs, p = run(name, nt)
            tgt = TGT5 if nt == 5 else TGT6
            print("\n" + "-" * 78)
            print("%s   fitting %d ratios" % (name, nt))
            print("-" * 78)
            print("  best squared residual : %.6f   (|log resid| = %.4f)"
                  % (c, np.sqrt(c)))
            print("  starts below 0.01     : %d / %d" % (int((costs < 0.01).sum()), N_STARTS))
            print("  starts below 0.10     : %d / %d" % (int((costs < 0.10).sum()), N_STARTS))
            print("  median best-of-start  : %.4f" % np.median(costs))
            print("  %-12s %12s %12s %10s" % ("ratio", "target", "fitted", "log dev"))
            for i in range(nt):
                print("  %-12s %12.5g %12.5g %10.3f"
                      % (NAMES[i], tgt[i], got[i], np.log(got[i] / tgt[i])))
            verdict = "VIABLE POINT FOUND" if c < 0.01 else (
                "close but not converged" if c < 0.5 else "NOT reached")
            print("  VERDICT: %s" % verdict)
            summary["%s_%d" % (name, nt)] = (c, verdict)

    print("\n" + "=" * 78)
    print("SENSITIVITY OF THE OBSTRUCTION TO THE ASSUMED m_tau/m_b WINDOW")
    print("=" * 78)
    print("GUT-scale m_tau/m_b is scheme dependent; the literature quotes values")
    print("from roughly 1.4 to 2.5.  We refit 120+126 for several targets, always")
    print("fitting the four hierarchies at the same time.")
    print("  %10s %14s %14s %12s" % ("target", "best sq.resid", "fitted", "log dev"))
    build, npar = MODELS["120+126"]
    for tb in [1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.76, 3.0]:
        tgt = TGT5.copy(); tgt[4] = tb
        best = None
        for s in range(40):
            p0 = rng.normal(size=npar) * (1.0 if s % 3 else 0.3)
            c, p = lm_fit(build, npar, tgt, p0)
            if best is None or c < best[0]:
                best = (c, p.copy())
        c, p = best
        got = ratios_batch(build, p[None, :], 5)[0]
        print("  %10.2f %14.6f %14.4f %12.3f"
              % (tb, c, got[4], np.log(got[4] / tb)))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, (c, v) in summary.items():
        print("  %-16s squared residual %10.6f   %s" % (k, c, v))
    print("\nInterpretation:")
    print("  A small residual is an existence proof of a point reproducing those")
    print("  ratios.  A large residual after many starts is evidence against, not")
    print("  a proof of exclusion.  Neither statement includes CKM, PMNS or RG.")
    print("=" * 78)
