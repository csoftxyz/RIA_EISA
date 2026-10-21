#!/usr/bin/env python3
"""Look-elsewhere / uniqueness test for the Z3LM modulus ratios.

Question posed by Reviewer 1: are the five rational ratios a structural
signal or a post-hoc rationalisation of a numerical best fit?

Method
------
1. Propagate the NuFIT 5.2 measurement errors through the six-parameter
   texture (6 params, 6 observables -> dof = 0) to obtain the 1-sigma
   sensitivity of each modulus ratio.
2. Count how many textures of the SAME Z3-locked family, with ratios
   p/q (q <= Q, gcd=1), fall inside those 1-sigma windows.
   That number N(Q) is the look-elsewhere multiplicity: if N is large,
   the observed rational pattern is not a unique selection.
3. Report the density of small-denominator rationals in each window.

Run: python3 ratio_uniqueness.py
"""
import numpy as np
from math import gcd
import z3lm

RAD = np.pi / 180

# NuFIT 5.2 (NO, with SK) targets / errors, same as the paper
OBS = np.array([0.305, 0.0223, 0.450, -130.0, 7.41e-5, 2.505e-3])
ERR = np.array([0.010, 0.0006, 0.018, 27.5, 0.21e-5, 0.025e-3])

Q0 = z3lm.BEST_FIT.copy()          # (m0,d1,d2,d3,a12,a23)


def ratios(q):
    m0, d1, d2, d3, a12, a23 = q
    return np.array([abs(d1) / a12, d2 / d3, a23 / a12, (a12 / 3) / a12,
                     a12 / d3])


def obs6(q):
    """Shape observables only (scale invariant): s12,s13,s23,delta,r."""
    o = z3lm.obs_vec(z3lm.M6(q))
    return np.array([o[0], o[1], o[2], o[3], o[4] / o[5]])


SHAPE = (1, 2, 3, 4, 5)   # shape parameters (m0 excluded)


def jacobian(q, h=1e-5):
    o0 = obs6(q)
    J = np.zeros((5, 5))
    for k, i in enumerate(SHAPE):
        dq = q.copy()
        dq[i] *= (1 + h)
        J[:, k] = (obs6(dq) - o0) / (dq[i] - q[i])
    return J


J = jacobian(Q0)
print("Jacobian d(observable)/d(param):")
print(f"cond(J) = {np.linalg.cond(J):.3e}")
np.set_printoptions(precision=4, suppress=True)
print(J)

C_obs = np.diag(np.array([ERR[0], ERR[1], ERR[2], ERR[3], 8.9e-4]) ** 2)
# parameter covariance from linear propagation (dof = 0 fit)
Jinv = np.linalg.inv(J)
C_par = Jinv @ C_obs @ Jinv.T
sig = np.sqrt(np.diag(C_par))

# ratio covariance by linear propagation of the six parameters
def ratio_jac(q, h=1e-5):
    r0 = ratios(q)
    Jr = np.zeros((5, 5))
    for k, i in enumerate(SHAPE):
        dq = q.copy()
        dq[i] *= (1 + h)
        Jr[:, k] = (ratios(dq) - r0) / (dq[i] - q[i])
    return Jr

Jr = ratio_jac(Q0)
C_r = Jr @ C_par @ Jr.T
sig_r = np.sqrt(np.diag(C_r))

R = ratios(Q0)
print("\nratio        value      1sigma       rel.")
for name, v, s in zip(['|d1|/a12', 'd2/d3', 'a23/a12', 'a13/a12', 'a12/d3'],
                      R, sig_r):
    print(f"{name:>10}  {v:9.5f}  {s:9.5f}   {abs(s/v)*100:5.2f}%")


def count_rationals(center, halfwidth, Qmax, lo=0.0, hi=5.0):
    """Count reduced fractions p/q with 1<=q<=Qmax inside the window."""
    tot = 0
    hits = []
    for q in range(1, Qmax + 1):
        pmin = int(np.ceil((center - halfwidth) * q))
        pmax = int(np.floor((center + halfwidth) * q))
        for p in range(pmin, pmax + 1):
            if p < lo * q or p > hi * q:
                continue
            if gcd(abs(p), q) != 1:
                continue
            tot += 1
            hits.append((p, q))
    return tot, hits


for Qmax in (11, 22, 33, 50):
    total = 1
    details = []
    for v, s in zip(R, sig_r):
        # use the 1-sigma window; a13/a12 is fixed to 1/3 by construction,
        # so its own "window" is degenerate and is set to its nominal value
        hw = max(abs(s), 1e-9)
        n, hits = count_rationals(v, hw, Qmax)
        total *= max(n, 1)
        details.append((v, hw, n, hits[:6]))
    print(f"\nQmax = {Qmax}: N(candidates in 1sigma box) ~ {total:.3g}")
    for v, hw, n, hits in details:
        print(f"   r={v:8.5f} +-{hw:8.5f}  -> {n:4d} rationals  e.g. {hits}")
