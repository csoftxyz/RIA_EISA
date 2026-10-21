#!/usr/bin/env python3
"""Exhaustive scan over Z3-phase-locked textures with SMALL INTEGER entries.

Motivation (response to Reviewer 1): the five modulus ratios of the Z3LM
texture must not be tuned.  If the texture is instead written with integer
entries (the ratios are then automatic), the question becomes: how many
integer textures in a bounded box reproduce the oscillation data, and is the
Z3LM point the simplest one?

Texture family (phases locked to omega, (1,3) entry carries omega):

    M = m0 * [[ a,      3c,       c*omega ],
              [ 3c,     d,       -e       ],
              [ c*omega, -e,      f       ]]

Integer shape (a,c,d,e,f), continuous overall scale m0 (>0).
Shape observables (scale invariant): s12, s13, s23, delta, r = dm21/dm31.
m0 is then fixed by dm31.

Usage:  python3 integer_scan.py [NMAX] [--plot]
"""
import sys
import numpy as np

OMEGA = np.exp(2j * np.pi / 3)

TGT = dict(s12=0.305, s13=0.0223, s23=0.450, delta=-130.0, r=7.41 / 250.5)
ERR = dict(s12=0.010, s13=0.0006, s23=0.018, delta=27.5, r=0.00089)


def obs_batch(A, C, D, E, F):
    """Batched Takagi observables for integer shapes.

    A,C,D,E,F: 1-D arrays of ints (same length). Returns dict of arrays.
    """
    n = A.shape[0]
    M = np.zeros((n, 3, 3), dtype=complex)
    M[:, 0, 0] = A
    M[:, 0, 1] = M[:, 1, 0] = 3.0 * C
    M[:, 0, 2] = M[:, 2, 0] = C * OMEGA
    M[:, 1, 1] = D
    M[:, 1, 2] = M[:, 2, 1] = -E
    M[:, 2, 2] = F

    # Takagi via eigh of M M^dagger (batched)
    H = M @ np.conj(np.transpose(M, (0, 2, 1)))
    w2, V = np.linalg.eigh(H)          # ascending
    m = np.sqrt(np.maximum(w2, 0.0))
    U = np.conj(V)
    temp = np.einsum('kji,kjl,kli->ki', U, M, U)   # (U^T M U)_ii
    U = U * np.exp(-0.5j * np.angle(temp))[:, None, :]
    diag = np.einsum('kji,kjl,kli->ki', U, M, U)
    neg = np.real(diag) < 0
    U = U * np.where(neg, 1j, 1.0)[:, None, :]

    Ue = U
    s13 = np.clip(np.abs(Ue[:, 0, 2]) ** 2, 0, 1)
    # standard normalisation: s12 = |Ue2|^2/(1-s13), s23 = |Umu3|^2/(1-s13)
    s12 = np.abs(Ue[:, 0, 1]) ** 2 / np.maximum(1 - s13, 1e-12)
    s23 = np.abs(Ue[:, 1, 2]) ** 2 / np.maximum(1 - s13, 1e-12)
    J = np.imag(Ue[:, 0, 0] * Ue[:, 1, 1]
                * np.conj(Ue[:, 0, 1]) * np.conj(Ue[:, 1, 0]))
    t12 = np.arcsin(np.sqrt(np.clip(s12, 0, 1)))
    t13 = np.arcsin(np.sqrt(s13))
    t23 = np.arcsin(np.sqrt(np.clip(s23, 0, 1)))
    c12, c23, c13 = np.cos(t12), np.cos(t23), np.cos(t13)
    Dn = c12 * np.sqrt(s12) * c23 * np.sqrt(s23) * c13 ** 2 * np.sqrt(s13)
    with np.errstate(divide='ignore', invalid='ignore'):
        sind = np.where(Dn > 0, J / Dn, 0.0)
    sind = np.clip(sind, -1, 1)
    Um1 = np.abs(Ue[:, 1, 0]) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        cosd = (Um1 - s12 * c23 ** 2 - c12 ** 2 * s23 * s13) / \
               (2 * np.sqrt(s12) * c12 * np.sqrt(s23) * c23 * np.sqrt(s13) + 1e-300)
    cosd = np.clip(cosd, -1, 1)
    delta = np.degrees(np.arctan2(sind, cosd))
    dm21 = m[:, 1] ** 2 - m[:, 0] ** 2
    dm31 = m[:, 2] ** 2 - m[:, 0] ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(np.abs(dm31) > 0, dm21 / dm31, np.nan)
    return dict(s12=s12, s13=s13, s23=s23, delta=delta, r=r,
                dm21=dm21, dm31=dm31, m=m)


def chi2_shape(o):
    return ((o['s12'] - TGT['s12']) / ERR['s12']) ** 2 + \
           ((o['s13'] - TGT['s13']) / ERR['s13']) ** 2 + \
           ((o['s23'] - TGT['s23']) / ERR['s23']) ** 2 + \
           ((o['delta'] - TGT['delta']) / ERR['delta']) ** 2 + \
           ((o['r'] - TGT['r']) / ERR['r']) ** 2


def main():
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rng = np.arange(-NMAX, NMAX + 1)
    fpos = np.arange(1, NMAX + 1)
    A, C, D, E, F = np.meshgrid(rng, rng, rng, rng, fpos, indexing='ij')
    A, C, D, E, F = [x.ravel().astype(float) for x in (A, C, D, E, F)]
    global THRESH
    THRESH = 6.0
    total = A.size
    print(f"# box |a|,|c|,|d|,|e| <= {NMAX}, 1<=f<={NMAX}: {total:.3e} textures")

    hits = []
    B = 400000
    for i in range(0, total, B):
        sl = slice(i, min(i + B, total))
        a, c, d, e, f = A[sl], C[sl], D[sl], E[sl], F[sl]
        # cheap prefilter: avoid zero determinant / degenerate (all zero c)
        keep = (np.abs(c) > 0)
        if not keep.any():
            continue
        o = obs_batch(a[keep], c[keep], d[keep], e[keep], f[keep])
        x2 = chi2_shape(o)
        good = np.isfinite(x2) & (x2 < THRESH)
        if good.any():
            idx = np.where(good)[0]
            for j in idx:
                hits.append((x2[j], a[keep][j], c[keep][j], d[keep][j],
                             e[keep][j], f[keep][j],
                             o['s12'][j], o['s13'][j], o['s23'][j],
                             o['delta'][j], o['r'][j], o['dm31'][j]))
        if (i // B) % 10 == 0:
            print(f"  ... {i}/{total}", flush=True)

    hits.sort(key=lambda t: t[0])
    print(f"\n# {len(hits)} shapes with chi2_shape < {THRESH}")
    print(f"{'chi2':>8} {'a':>4} {'c':>4} {'d':>4} {'e':>4} {'f':>4} "
          f"{'s12':>7} {'s13':>7} {'s23':>7} {'delta':>7} {'r':>7}")
    for h in hits[:80]:
        print(f"{h[0]:8.2f} {int(h[1]):4d} {int(h[2]):4d} {int(h[3]):4d} "
              f"{int(h[4]):4d} {int(h[5]):4d} {h[6]:7.4f} {h[7]:7.5f} "
              f"{h[8]:7.4f} {h[9]:7.1f} {h[10]:7.4f}")
    np.save('integer_scan_hits.npy', np.array(hits))


if __name__ == '__main__':
    main()
