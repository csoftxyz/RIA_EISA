#!/usr/bin/env python3
"""Search for |G|-smooth rational Z3-locked textures.

Denominator theorem (App. E): moduli fixed by the CG structure of a finite
group G with denominator |G|-smooth.  Minimal flavour groups have
|G| in {12, 24, 27, 54}, i.e. only primes 2 and 3.

Question: is there a texture whose five moduli are p/q with
q in {1,2,3,4,6,8,9,12,16,18,24,27} that fits as well as Z3LM?
If yes, the No-Go obstruction disappears and a derivation becomes possible.

Stage 1: vectorised Gaussian shape chi2 over the whole grid.
Stage 2: refine the best candidates with the official NuFIT 5.2 block chi2.
"""
import numpy as np
import itertools
import z3lm
from nufit_parser import NuFIT52
from scipy.optimize import minimize_scalar

OMEGA = z3lm.OMEGA
NF = NuFIT52()

QS = [1,2,3,4,6,8,9,12,16,18,24,27,36,54]
R0 = np.array([0.5, 9/11, 45/22, 1/3, 6/19])
WIN = 0.10          # +-10% around the Z3LM values


def cand(r):
    lo, hi = r * (1 - WIN), r * (1 + WIN)
    out = set()
    for q in QS:
        for p in range(max(1, int(np.floor(lo * q))), int(np.ceil(hi * q)) + 1):
            v = p / q
            if lo <= v <= hi:
                out.add(round(v, 12))
    return sorted(out)


CAND = [cand(R0[i]) for i in (0, 1, 2, 4)]
print("candidates per ratio:", [len(c) for c in CAND])

# ---- stage 1: vectorised Gaussian metric --------------------------------
grid = np.array(list(itertools.product(*CAND)))
print("grid size:", grid.shape)

def shape_obs(R):
    """R: (n,4) = (r1,r2,r3,r5). Returns s12,s13,s23,delta,r."""
    n = R.shape[0]
    a12 = R[:, 3]
    d1 = -R[:, 0] * a12
    d2 = R[:, 1]
    d3 = np.ones(n)
    a23 = R[:, 2] * a12
    a13 = a12 / 3.0
    M = np.zeros((n, 3, 3), dtype=complex)
    M[:, 0, 0] = d1; M[:, 1, 1] = d2; M[:, 2, 2] = d3
    M[:, 0, 1] = M[:, 1, 0] = a12
    M[:, 1, 2] = M[:, 2, 1] = -a23
    M[:, 0, 2] = M[:, 2, 0] = a13 * OMEGA
    H = M @ np.conj(np.transpose(M, (0, 2, 1)))
    w2, V = np.linalg.eigh(H)
    m = np.sqrt(np.maximum(w2, 0.0))
    U = np.conj(V)
    tmp = np.einsum('kji,kjl,kli->ki', U, M, U)
    U = U * np.exp(-0.5j * np.angle(tmp))[:, None, :]
    dg = np.einsum('kji,kjl,kli->ki', U, M, U)
    U = U * np.where(np.real(dg) < 0, 1j, 1.0)[:, None, :]
    s13 = np.clip(np.abs(U[:, 0, 2]) ** 2, 0, 1)
    s12 = np.abs(U[:, 0, 1]) ** 2 / np.maximum(1 - s13, 1e-12)
    s23 = np.abs(U[:, 1, 2]) ** 2 / np.maximum(1 - s13, 1e-12)
    J = np.imag(U[:, 0, 0] * U[:, 1, 1] * np.conj(U[:, 0, 1]) * np.conj(U[:, 1, 0]))
    t12 = np.arcsin(np.sqrt(np.clip(s12, 0, 1)))
    t13 = np.arcsin(np.sqrt(s13))
    t23 = np.arcsin(np.sqrt(np.clip(s23, 0, 1)))
    c12, c13, c23 = np.cos(t12), np.cos(t13), np.cos(t23)
    Dn = c12 * np.sqrt(s12) * c23 * np.sqrt(s23) * c13 ** 2 * np.sqrt(s13)
    with np.errstate(divide='ignore', invalid='ignore'):
        sind = np.where(Dn > 0, J / Dn, 0.0)
    sind = np.clip(sind, -1, 1)
    Um1 = np.abs(U[:, 1, 0]) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        cosd = (Um1 - s12 * c23 ** 2 - c12 ** 2 * s23 * s13) / \
               (2 * np.sqrt(s12) * c12 * np.sqrt(s23) * c23 * np.sqrt(s13) + 1e-300)
    cosd = np.clip(cosd, -1, 1)
    delta = np.degrees(np.arctan2(sind, cosd))
    dm21 = m[:, 1] ** 2 - m[:, 0] ** 2
    dm31 = m[:, 2] ** 2 - m[:, 0] ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(np.abs(dm31) > 0, dm21 / dm31, np.nan)
    return s12, s13, s23, delta, rr


s12, s13, s23, delta, rr = shape_obs(grid)
chi = ((s12 - 0.305) / 0.010) ** 2 + ((s13 - 0.0223) / 0.0006) ** 2 \
    + ((s23 - 0.450) / 0.018) ** 2 + ((delta + 130) / 27.5) ** 2 \
    + ((rr - 7.41 / 250.5) / 0.00089) ** 2
chi = np.where(np.isfinite(chi), chi, np.inf)
order = np.argsort(chi)[:400]

print(f"best Gaussian chi2 in 2,3-smooth grid: {chi[order[0]]:.3f}")
print("Z3LM Gaussian chi2:", end=' ')
o = shape_obs(R0[[0, 1, 2, 4]][None, :]); g0 = ((o[0]-0.305)/0.010)**2 + ((o[1]-0.0223)/0.0006)**2 \
    + ((o[2]-0.450)/0.018)**2 + ((o[3]+130)/27.5)**2 + ((o[4]-7.41/250.5)/0.00089)**2
print(f"{float(g0[0]):.3f}")

# ---- stage 2: official block chi2 for the best candidates ---------------
def block_chi2(R):
    a12 = R[3]
    q = np.array([1.0, -R[0] * a12, R[1], 1.0, a12, R[2] * a12])
    def f(m0):
        M = z3lm.M6(q) * m0
        M[0, 2] = M[2, 0] = (a12 / 3.0) * m0 * OMEGA
        ob = z3lm.obs_vec(M)
        return NF.chi2_block(np.array([ob[0], ob[1], ob[2], ob[3], ob[4], ob[5]]))
    r = minimize_scalar(f, bounds=(0.025, 0.045), method='bounded',
                        options=dict(xatol=1e-7))
    return r.fun

ref = block_chi2(R0[[0, 1, 2, 4]])
print(f"\nofficial block chi2(Z3LM) = {ref:.3f}\n")
res = []
for i in order[:40]:
    R = grid[i]
    res.append((block_chi2(R), R))
res.sort(key=lambda t: t[0])
print("best 2,3-smooth rational textures (official block chi2):")
for c, R in res[:3]:
    print(f"  chi2={c:6.3f}   r=({R[0]:.5f}, {R[1]:.5f}, {R[2]:.5f}, 1/3, {R[3]:.5f})")
np.save('smooth_scan.npy', np.array([(c, *R) for c, R in res[:100]]))
