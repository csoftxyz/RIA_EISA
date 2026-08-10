#!/usr/bin/env python3
"""
z3lm.py -- Core of the Z3LM six-parameter omega-locked Majorana texture.

Implements the texture of the paper:

    M_nu = m0 * [[ d1,      a12,      (a12/3)*omega ],
                 [ a12,     d2,       -a23          ],
                 [ (a12/3)*omega, -a23, d3          ]]

with omega = exp(2*pi*i/3), plus the Takagi decomposition of the
complex-symmetric mass matrix and extraction of all low-energy
observables (mixing angles, Dirac phase, Jarlskog invariant, masses,
sum of masses, m_beta_beta).

Reference numbers (paper Sec. 4):
    m0=0.02243, d1=-0.2226, d2=1.1539, d3=1.4095,
    a12=0.4456, a23=0.9119, a13=a12/3=0.1485
    -> sin^2 t12=0.3051, sin^2 t23=0.4500, sin^2 t13=0.02230,
       delta=-83.4 deg (276.6 deg; standard PDG/Jarlskog
       convention, rephasing-invariant extraction), dm21=7.413e-5,
       dm31=2.505e-3, J=-0.0332, m=(0.0076,0.0115,0.0506),
       Sum m=0.0697, mbb=0.0050

v1.2 (2026-08-09): the Dirac phase is now extracted from
rephasing-invariant quantities (sin delta = J/D, cos delta from
|U_mu1|^2), which guarantees J = D sin delta.  Earlier versions
fixed U_t3 real and extracted delta = -arg(U_e3), which is not the
PDG convention and gave delta = 237.5 deg inconsistent with J.
The correct standard-PDG value is delta = 276.6 deg (-83.4 deg).

Author: Yuxuan Zhang  (paper code, v1.2)
"""
import numpy as np
from scipy.linalg import eigh

OMEGA = np.exp(2j * np.pi / 3)          # cubic root of unity
RAD = np.pi / 180.0


# ----------------------------------------------------------------------
# Texture
# ----------------------------------------------------------------------

def M6(q):
    """Build the Z3LM mass matrix from the six real parameters
    q = (m0, d1, d2, d3, a12, a23)."""
    m0, d1, d2, d3, a12, a23 = q
    return m0 * np.array([
        [d1,        a12,      (a12 / 3.0) * OMEGA],
        [a12,       d2,       -a23],
        [(a12 / 3.0) * OMEGA, -a23, d3],
    ], dtype=complex)


# ----------------------------------------------------------------------
# Takagi decomposition
# ----------------------------------------------------------------------

def takagi_all(M):
    """Takagi (Autonne-Takagi) decomposition of a complex-symmetric
    matrix M = U diag(m) U^T with U unitary, m_i >= 0.

    Algorithm (paper Appendix A):
      1. eigendecompose H = M M^dagger  ->  V, sigma^2
      2. use the CONJUGATED basis  U0 = V*
      3. fix column phases so that U^T M U is diagonal real >= 0
    Returns (U, m) with m sorted ascending.
    """
    w2, V = eigh(M @ M.conj().T)
    idx = np.argsort(w2)
    m = np.sqrt(np.maximum(w2[idx], 0.0))
    U = V[:, idx].conj()
    temp = U.T @ M @ U
    U *= np.exp(-0.5j * np.angle(np.diag(temp)))[None, :]
    d = np.diag(U.T @ M @ U)
    for i in range(3):
        if np.real(d[i]) < 0:
            U[:, i] *= 1j
    return U, m


def takagi_residual(M, U=None, m=None):
    """Max |(U^T M U)_ij - delta_ij m_i|; diagnostic for correctness."""
    if U is None:
        U, m = takagi_all(M)
    return np.max(np.abs(U.T @ M @ U - np.diag(m)))


# ----------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------

def obs_vec(M):
    """Extract the six observables used in the fit:
       (sin2_12, sin2_13, sin2_23, delta_deg, dm21, dm31)
    plus derived quantities (J, masses, mbb) available via obs_full.

    The Dirac phase is extracted by REPHASING-INVARIANT quantities
    (standard PDG/Jarlskog convention):
        sin(delta) = J / D,  D = c12 s12 c23 s23 c13^2 s13
        cos(delta) from |U_mu1|^2 = s12^2 c23^2 + c12^2 s23^2 s13^2
                                    + 2 s12 c12 s23 c23 s13 cos(delta)
    This guarantees J = D sin(delta) exactly, independent of the
    phase convention used for the Takagi matrix.  (The earlier
    versions extracted delta = -arg(U_e3) after fixing U_t3 real,
    which is NOT the PDG convention and is inconsistent with J;
    fixed in v1.2.)"""
    U, m = takagi_all(M)
    t12 = np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
    t23 = np.arctan2(abs(U[1, 2]), abs(U[2, 2]))
    t13 = np.arcsin(min(max(abs(U[0, 2]), 0.0), 1.0))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))

    s12, c12 = np.sin(t12), np.cos(t12)
    s23, c23 = np.sin(t23), np.cos(t23)
    s13, c13 = np.sin(t13), np.cos(t13)
    D = c12 * s12 * c23 * s23 * c13 ** 2 * s13

    # cos(delta) from |U_mu1|^2 (PDG: U_mu1 = -s12 c23 - c12 s23 s13 e^{i delta})
    Um1 = abs(U[1, 0]) ** 2
    cosd = (Um1 - s12 ** 2 * c23 ** 2 - c12 ** 2 * s23 ** 2 * s13 ** 2) \
        / (2 * s12 * c12 * s23 * c23 * s13)
    cosd = float(np.clip(cosd, -1.0, 1.0))
    sind = float(np.clip(J / D, -1.0, 1.0))
    delta = np.arctan2(sind, cosd)   # in (-pi, pi]
    d_deg = np.degrees(delta)
    # report in [-180, 180) to match the NuFIT table convention
    if d_deg >= 180.0:
        d_deg -= 360.0

    return np.array([np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2,
                     d_deg, m[1] ** 2 - m[0] ** 2, m[2] ** 2 - m[0] ** 2])


def obs_full(M):
    """All low-energy outputs as a dict (for tables in the paper)."""
    U, m = takagi_all(M)
    o = obs_vec(M)
    ms = np.sort(m)
    Ue = U[:, np.argsort(m)]
    mbb = abs(Ue[0, 0] ** 2 * ms[0] + Ue[0, 1] ** 2 * ms[1]
              + Ue[0, 2] ** 2 * ms[2])
    t12 = np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
    t23 = np.arctan2(abs(U[1, 2]), abs(U[2, 2]))
    t13 = np.arcsin(min(max(abs(U[0, 2]), 0.0), 1.0))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    return dict(
        s12=o[0], s13=o[1], s23=o[2], delta=o[3],
        dm21=o[4], dm31=o[5],
        t12=np.degrees(t12), t23=np.degrees(t23), t13=np.degrees(t13),
        J=J, m1=ms[0], m2=ms[1], m3=ms[2], sum_m=ms.sum(), mbb=mbb,
        ordering="NO" if ms[2] > ms[1] else "IO",
    )


BEST_FIT = np.array([0.02243, -0.2226, 1.1539, 1.4095, 0.4456, 0.9119])
"""Six-parameter REFERENCE point (structural branch, d2,d3>0) from
which the ratios were read off; chi2 = 1.33 (dof = 0).  This is NOT
the final model.  The final one-parameter model (all ratios fixed,
chi2 = 1.41, dof = 5) is implemented in z3lm_1param.py."""


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    q = BEST_FIT
    M = M6(q)
    o = obs_full(M)
    print("Best-fit Z3LM texture (paper Sec. 4):")
    print("  params :", q)
    print("  matrix :")
    print(np.round(M / q[0], 4))
    print(f"  Takagi residual : {takagi_residual(M):.2e}")
    print(f"  sin2 t12={o['s12']:.4f}  sin2 t23={o['s23']:.4f}  sin2 t13={o['s13']:.5f}")
    print(f"  delta={o['delta']:.1f} deg  J={o['J']:+.4f}")
    print(f"  dm21={o['dm21']:.4e}  dm31={o['dm31']:.4e}")
    print(f"  m1={o['m1']:.4f}  m2={o['m2']:.4f}  m3={o['m3']:.4f}")
    print(f"  Sum m={o['sum_m']:.4f}  mbb={o['mbb']:.4f}  ordering={o['ordering']}")
