"""Debug the charged H-word exact rank (14) vs the mathematical upper bound 13.

Genuine bi-unitary invariants are functions of the 13 charged observables, so
their Jacobian rank cannot exceed 13.  An exact value of 14 indicates an
implementation bug.  Here we compute the SAME word set numerically (float) and
compare, then isolate the offending direction.
"""
import numpy as np
import strict_obs as S


def H(M):
    return M @ M.conj().T


def charged_words(q):
    Mu, Md, Me = S.build_charged(q)
    Hs = [H(Mu), H(Md), H(Me)]
    out = []
    for Hx in Hs:
        H2 = Hx @ Hx; H3 = H2 @ Hx
        out += [np.trace(Hx), np.trace(H2), np.trace(H3)]
    for a in range(3):
        for b in range(a + 1, 3):
            X, Y = Hs[a], Hs[b]
            X2 = X @ X; Y2 = Y @ Y; XY = X @ Y
            out += [np.trace(XY), np.trace(X2 @ Y), np.trace(X @ Y2),
                    np.trace(X2 @ Y2), np.trace(XY @ XY), np.trace(XY @ X2 @ Y2)]
            Cm = XY - Y @ X
            out += [np.trace(Cm @ Cm @ Cm)]
    return np.array([complex(z) for z in out])


if __name__ == "__main__":
    for sd in (2, 5, 3):
        p = np.random.default_rng(sd).normal(size=30)
        f0 = charged_words(p)
        J = np.zeros((len(f0), 30), complex)
        h = 1e-6
        for k in range(30):
            q = p.copy(); q[k] += h
            J[:, k] = (charged_words(q) - f0) / h
        Jr = np.vstack([J.real, J.imag])
        s = np.linalg.svd(Jr, compute_uv=False)
        r = int((s > 1e-7 * s[0]).sum())
        print("seed %d  float charged H-word rank = %d  (bound 13)" % (sd, r))
        print("   s[12]/s0=%.2e s[13]/s0=%.2e" % (s[12] / s[0], s[13] / s[0]))
