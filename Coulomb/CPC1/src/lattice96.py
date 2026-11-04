"""Periodic-lattice Coulomb test on L^3, d = 3: point charge + neutralising background.

The Poisson problem  -Delta_lat V = delta_0 - L^-3  is solved exactly in Fourier
space with the *lattice* symbol of the Laplacian (not the continuum k^2).
Running this file reproduces Table 4 and Fig. 5 of the paper.

    python3 lattice96.py            # table for R = 8..24
"""
import numpy as np
import math
import sys
import time

try:                       # POSIX only; purely diagnostic
    import resource
except ImportError:        # pragma: no cover - Windows
    resource = None

__all__ = ["solve", "shell_stats", "table", "fit_ar_b_cr2", "OM3"]

OM3 = 4.0*math.pi


def solve(L, Q=1.0):
    """Return V on the periodic L^3 lattice with -Delta_lat V = Q*delta - Q/L^3."""
    k = 2.0*np.pi*np.fft.fftfreq(L)          # angular wavenumbers, spacing a = 1
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    # exact symbol of the lattice Laplacian:  -Delta -> 4*sum sin^2(k_i/2)
    sym = 4.0*(np.sin(KX/2)**2 + np.sin(KY/2)**2 + np.sin(KZ/2)**2)

    rho = np.zeros((L, L, L))
    rho[L//2, L//2, L//2] = Q
    rho -= Q/L**3

    rho_hat = np.fft.fftn(rho)
    V_hat = np.zeros_like(rho_hat)
    m = sym > 0.0                            # drop the zero mode
    V_hat[m] = rho_hat[m]/sym[m]
    return np.real(np.fft.ifftn(V_hat))


def shell_stats(V):
    """Per-shell mean potential and mean |x|^2 for shells S_R = {x : round(|x|) = R}."""
    L = V.shape[0]
    i = np.arange(L) - L//2
    X, Y, Z = np.meshgrid(i, i, i, indexing="ij")
    Rm = np.sqrt(X**2.0 + Y**2.0 + Z**2.0)
    n = np.rint(Rm).astype(int)
    out = {}
    for s in range(1, L//2):
        sel = n == s
        if sel.any():
            out[s] = dict(V=float(V[sel].mean()),
                          r2=float((Rm[sel]**2.0).mean()),
                          cnt=int(sel.sum()))
    return out


def table(st, L, Rs):
    """Recovered charge for the four estimators of Table 4.  Returns (R, 4 columns)."""
    rows = []
    for s in Rs:
        if s not in st or (s+1) not in st:
            continue
        dV = st[s]["V"] - st[s+1]["V"]
        rstar = math.sqrt(s*(s+1.0))
        bg = (st[s]["r2"] - st[s+1]["r2"])/(2.0*3.0*L**3)   # -Delta V_bg with V_bg = |x|^2/2dL^d
        rows.append((s,
                     OM3*float(s)**2*dV,
                     OM3*(s+0.5)**2*dV,
                     OM3*s*(s+1.0)*dV,
                     OM3*s*(s+1.0)*(dV - bg)))
    return rows


def fit_ar_b_cr2(st, L, Rs):
    """Least-squares fit of Vbar(R) = A/R + B + C R^2; returns (A, B, C, rms)."""
    R = np.array([s for s in Rs if s in st], dtype=float)
    Vb = np.array([st[int(s)]["V"] for s in R])
    M = np.vstack([1.0/R, np.ones_like(R), R**2.0]).T
    coef, *_ = np.linalg.lstsq(M, Vb, rcond=None)
    return tuple(coef) + (float((Vb - M @ coef).std()),)


if __name__ == "__main__":
    L = 96
    t0 = time.perf_counter()
    V = solve(L)
    t1 = time.perf_counter()
    st = shell_stats(V)
    t2 = time.perf_counter()

    rss = ("%.0f MB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0)
           if resource is not None else "n/a on this platform")
    print(f"radialflux lattice96: L={L}, solve {t1-t0:.2f} s, shells {t2-t1:.2f} s, "
          f"peak RSS {rss}")

    Rs = list(range(8, 25))
    rows = table(st, L, Rs)
    print("\n  R      left     midpoint    exact rho*   exact-bg")
    for s, a, b, c, e in rows:
        print(f"{s:3d}  {a:9.5f}  {b:9.5f}  {c:9.5f}  {e:9.5f}")
    arr = np.array([r[1:] for r in rows])
    print(f"mean  {arr.mean(0)[0]:9.5f}  {arr.mean(0)[1]:9.5f}  "
          f"{arr.mean(0)[2]:9.5f}  {arr.mean(0)[3]:9.5f}")

    A, B, C, rms = fit_ar_b_cr2(st, L, Rs)
    print(f"\nfit Vbar = A/R + B + C R^2 over R={Rs[0]}..{Rs[-1]}")
    print(f"   Q = 4 pi A = {4*math.pi*A:.4f}")
    print(f"   B = {B:.6f}   (constant offset: must not be dropped)")
    print(f"   C = {C:.3e}   (background; predicted 1/(6L^3) = {1/(6*L**3):.3e})")
    print(f"   rms residual of the fit = {rms:.2e}")
    sys.exit(0)
