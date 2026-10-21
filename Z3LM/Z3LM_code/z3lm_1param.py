#!/usr/bin/env python3
"""
z3lm_1param.py -- The one-parameter Z3LM texture (final version).

All five modulus ratios are FIXED (phenomenological inputs, not derived;
see paper Sec. 4.4 and App. C No-Go result):

    |d1|/a12 = 1/2          (from reconnaissance, 0.10% off)
    d2/d3    = 9/11         (from reconnaissance, 0.05% off)
    a23/a12  = 45/22        (from reconnaissance, 0.05% off)
    a13/a12  = 1/3          (structural constraint of the texture)
    a12/d3   = 6/19         (structural selection; 1/sqrt(10) within 1sigma)

Phases locked to the cubic root of unity (Z3 CP seed):
    M12 in R+,  M13 = (a12/3) omega,  M23 in R-.

With these relations, d3 and a12 enter only through the common scale
m0*d3, so the texture has a SINGLE effective free parameter: the
absolute mass scale.  We fix the gauge d3 = 1 and scan m0.

Outputs (official NuFIT 5.2 release tables, NO with SK):
    chi2_min = 1.41 (dof = 5: six observables - one parameter)
    sin2t12, sin2t23, sin2t13, delta, dm21, dm31, J,
    m1, m2, m3, sum m, m_bb, ordering.

NOTE (v1.2): the Dirac phase is extracted from rephasing-invariant
quantities (sin delta = J/D, cos delta from |U_mu1|^2), so that
J = D sin(delta) holds exactly.  The correct standard-PDG value is
delta = -83.4 deg (276.6 deg); earlier versions reported 237.6 deg
from a non-standard phase fixing that is inconsistent with J.
"""
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, '/root/.openclaw/workspace/paper/code')
from nufit_parser import NuFIT52, table_available  # noqa: E402

OMEGA = np.exp(2j * np.pi / 3)

# --- fixed ratios (phenomenological inputs) --------------------------
R_D1 = 1.0 / 2.0        # |d1| / a12
R_D2 = 9.0 / 11.0       # d2 / d3
R_A23 = 45.0 / 22.0     # a23 / a12
R_A13 = 1.0 / 3.0       # a13 / a12
R_A12 = 6.0 / 19.0      # a12 / d3   (structural selection)


def takagi_all(M):
    """Autonne-Takagi decomposition: M = U diag(m) U^T, U unitary, m>=0."""
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


def M_texture(m0, d3=1.0):
    """One-parameter texture matrix (d3 gauge fixed; scale = m0*d3)."""
    a12 = R_A12 * d3
    d1 = -R_D1 * a12
    d2 = R_D2 * d3
    a23 = R_A23 * a12
    a13 = R_A13 * a12
    return m0 * np.array([[d1, a12, a13 * OMEGA],
                          [a12, d2, -a23],
                          [a13 * OMEGA, -a23, d3]], dtype=complex)


def obs_full(M):
    """All low-energy outputs; delta extracted rephasing-invariantly
    (standard PDG/Jarlskog: sin = J/D, cos from |U_mu1|^2)."""
    U, m = takagi_all(M)
    t12 = np.arctan2(abs(U[0, 1]), abs(U[0, 0]))
    t23 = np.arctan2(abs(U[1, 2]), abs(U[2, 2]))
    t13 = np.arcsin(min(max(abs(U[0, 2]), 0), 1))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    s12, c12 = np.sin(t12), np.cos(t12)
    s23, c23 = np.sin(t23), np.cos(t23)
    s13, c13 = np.sin(t13), np.cos(t13)
    D = c12 * s12 * c23 * s23 * c13 ** 2 * s13
    Um1 = abs(U[1, 0]) ** 2
    cosd = float(np.clip((Um1 - s12 ** 2 * c23 ** 2
                          - c12 ** 2 * s23 ** 2 * s13 ** 2)
                         / (2 * s12 * c12 * s23 * c23 * s13), -1, 1))
    sind = float(np.clip(J / D, -1, 1))
    d = np.degrees(np.arctan2(sind, cosd))
    if d >= 180:
        d -= 360
    ms = np.sort(m)
    Ue = U[:, np.argsort(m)]
    mbb = abs(Ue[0, 0] ** 2 * ms[0] + Ue[0, 1] ** 2 * ms[1]
              + Ue[0, 2] ** 2 * ms[2])
    return dict(s12=np.sin(t12) ** 2, s13=np.sin(t13) ** 2,
                s23=np.sin(t23) ** 2, delta=d,
                dm21=m[1] ** 2 - m[0] ** 2, dm31=m[2] ** 2 - m[0] ** 2,
                J=J, m1=ms[0], m2=ms[1], m3=ms[2], sum_m=ms.sum(),
                mbb=mbb, ordering="NO" if ms[2] > ms[1] else "IO")


def chi2_at(m0, nf):
    o = obs_full(M_texture(m0))
    return nf.chi2_1d_sum(np.array([o["s12"], o["s13"], o["s23"],
                                    o["delta"], o["dm21"], o["dm31"]]))


def main():
    nf = NuFIT52() if table_available() else None
    if nf is None:
        print("Official NuFIT 5.2 tables not found; cannot evaluate chi2.")
        return 1

    # one-dimensional scan over the single free parameter m0
    grid = np.linspace(0.005, 0.10, 30000)
    chis = np.array([chi2_at(m, nf) for m in grid])
    i = int(np.argmin(chis))
    m0_best = grid[i]

    print("=== ONE-PARAMETER Z3LM TEXTURE (final version) ===")
    print("fixed ratios: |d1|/a12 = 1/2, d2/d3 = 9/11, a23/a12 = 45/22,")
    print("              a13/a12 = 1/3, a12/d3 = 6/19  (all phenomenological)")
    print(f"chi2_min = {chis[i]:.4f}  (dof = 5: six observables, one parameter)")
    print(f"m0 (d3 gauge) = {m0_best:.6f} eV   scale m0*d3 = {m0_best:.6f} eV")

    o = obs_full(M_texture(m0_best))
    for k in ["s12", "s23", "s13", "delta", "dm21", "dm31", "J",
              "m1", "m2", "m3", "sum_m", "mbb", "ordering"]:
        print(f"  {k:8s} = {o[k]:.6g}" if isinstance(o[k], float)
              else f"  {k:8s} = {o[k]}")

    # 1-sigma interval on the absolute scale (Delta chi2 = 1)
    lo = grid[chis <= chis[i] + 1.0].min()
    hi = grid[chis <= chis[i] + 1.0].max()
    olo, ohi = obs_full(M_texture(lo)), obs_full(M_texture(hi))
    print(f"\n1-sigma interval on m0: [{lo:.5f}, {hi:.5f}] eV")
    print(f"  sum_m: {olo['sum_m']:.4f} -- {ohi['sum_m']:.4f} eV")
    print(f"  m_bb : {olo['mbb']:.4f} -- {ohi['mbb']:.4f} eV")
    print(f"  delta: {olo['delta']:.1f} -- {ohi['delta']:.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
