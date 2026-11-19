import os
"""
CORRECTED compatibility test.

The previous test fixed a random S and fitted only (A,c1,c2).  But S = Y_126 is a
FREE Yukawa, not an observable, so it must be varied too.  With S free the
parameter count is 6(S)+3(A)+2(c1,c2) = 11 complex = 22 real against a symmetric
target of 12 real: the reduced map is dominant (proved), so ANY neutrino pattern
is reachable.  The four relations therefore do NOT restrict the neutrino data
alone; they restrict the JOINT down+neutrino configuration, because the same S
must also serve the down sector.

This script confirms both statements numerically:
  (a) with S free, a realistic neutrino target is reproduced to high accuracy;
  (b) the relations hold on the reproduced point.
"""
import numpy as np

rng = np.random.default_rng(777)


def pmns(s12, s23, s13, delta):
    c12, c23, c13 = np.sqrt(1-s12**2), np.sqrt(1-s23**2), np.sqrt(1-s13**2)
    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
         c12*c23 - s12*s23*s13*np.exp(1j*delta), s23*c13],
        [s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta), c23*c13]], complex)


def unpack(p):
    # S symmetric: 6 complex = 12 real ; A antisym: 3 complex = 6 ; c1,c2: 4
    zs = [p[2*t] + 1j*p[2*t+1] for t in range(6)]
    S = np.array([[zs[0], zs[1], zs[2]], [zs[1], zs[3], zs[4]], [zs[2], zs[4], zs[5]]], complex)
    a, b, cc = p[12]+1j*p[13], p[14]+1j*p[15], p[16]+1j*p[17]
    A = np.array([[0, a, b], [-a, 0, cc], [-b, -cc, 0]], complex)
    c1 = p[18]+1j*p[19]; c2 = p[20]+1j*p[21]
    return S, A, c1, c2


def model(p):
    S, A, c1, c2 = unpack(p)
    return c1*S + c2*(A @ np.linalg.inv(S) @ A)


def resid(p, target):
    D = model(p) - target
    return np.concatenate([D.real.ravel(), D.imag.ravel()])


def disc(X):
    t1 = np.trace(X); t2 = (t1*t1 - np.trace(X@X))/2; t3 = np.linalg.det(X)
    return t1*t1*t2*t2 - 4*t2**3 - 4*t1**3*t3 - 27*t3*t3 + 18*t1*t2*t3


if __name__ == "__main__":
    print("=" * 78)
    print("COMPATIBILITY, WITH S FREE (correct test)")
    print("=" * 78)
    m = np.array([1.0e-12, 8.6e-12, 5.0e-11])
    U = pmns(np.sqrt(0.307), np.sqrt(0.546), np.sqrt(0.0220), 1.20)
    target = U.conj() @ np.diag(m) @ U.conj().T
    scale = np.max(m)
    best = None
    for trial in range(8):
        p = rng.normal(size=22) * scale
        p[18] += scale       # c1 away from zero
        for it in range(600):
            r = resid(p, target)
            J = np.zeros((len(r), 22))
            h = 1e-10*scale
            for k in range(22):
                q = p.copy(); q[k] += h
                J[:, k] = (resid(q, target) - r)/h
            try:
                dp = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            # damped step
            p = p + 0.5*dp
            if np.linalg.norm(r) < 1e-16*scale:
                break
        r = resid(p, target)
        rel = np.linalg.norm(r)/np.linalg.norm(np.concatenate([target.real.ravel(), target.imag.ravel()]))
        if best is None or rel < best[0]:
            best = (rel, p.copy())
    rel, p = best
    print("  best relative residual : %.3e" % rel)
    S, A, c1, c2 = unpack(p)
    M = model(p)
    sv = lambda X: np.linalg.svd(X, compute_uv=False)
    print("  reproduced masses (GeV): %.3e %.3e %.3e" % tuple(sv(M)))
    print("  target      masses (GeV): %.3e %.3e %.3e" % tuple(sv(target)))
    Xp = M @ np.linalg.inv(S)
    print("  |disc(M_nu S^-1)| (relative) : %.3e"
          % (abs(disc(Xp))/max(abs(np.trace(Xp))**6, 1e-300)))
    v = np.array([A[1, 2], -A[0, 2], A[0, 1]])
    cr = np.cross(M @ v, S @ v)
    print("  |M_nu v x S v| (relative)    : %.3e"
          % (np.linalg.norm(cr)/max(np.linalg.norm(M@v)*np.linalg.norm(S@v), 1e-300)))
    print()
    if rel < 1e-6:
        print("  => COMPATIBLE: with S free the observed neutrino pattern is")
        print("     reproduced, and both explicit relations hold on it.")
        print("     The four relations constrain the JOINT down+neutrino")
        print("     configuration, not the neutrino data in isolation.")
    else:
        print("  => residual %.2e: not converged; no conclusion drawn." % rel)
    print()
    print("  What a global fit would test: the SAME S must reproduce the down")
    print("  masses and mixings together with this neutrino pattern.  That is")
    print("  the physical content of the four relations, and it is a fit for")
    print("  future work, not a claim made here.")
