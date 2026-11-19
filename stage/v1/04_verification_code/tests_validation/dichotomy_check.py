"""
Rank-alignment dichotomy for one antisymmetric plus one symmetric Yukawa.

Referenced by Proposition 1 and Appendix A of the paper.

    M_d = a A + s_d S,   M_e = b A - 3 s_d S,   M_nu = v_L S,
    A antisymmetric, S symmetric.

Dichotomy:
  * a = b = 0  =>  M_d proportional to M_e (rank one);
  * a or b nonzero  =>  M_e has a nonzero antisymmetric part, so M_e is not
    normal and cannot be simultaneously diagonalised with M_nu proportional
    to S for generic S.

Diagnostic used: the relative Frobenius norm of the commutator
    C = [M_e M_e^dagger, S S^dagger],
normalised by ||M_e M_e^dagger|| * ||S S^dagger||.  Simultaneous
diagonalisability of two Hermitian matrices requires C = 0.

This is a GENERICITY test over a fixed-seed sample, not a measure-zero theorem.
"""
import numpy as np

rng = np.random.default_rng(602719)

N_POINTS = 50000
TOL = 1e-9


def anti(p):
    a, b, c = p[0] + 1j * p[1], p[2] + 1j * p[3], p[4] + 1j * p[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(p):
    z = [p[2 * t] + 1j * p[2 * t + 1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]],
                     [z[1], z[3], z[4]],
                     [z[2], z[4], z[5]]], complex)


def herm(M):
    return M @ M.conj().T


def rel_commutator(Me, S):
    H1, H2 = herm(Me), herm(S)
    C = H1 @ H2 - H2 @ H1
    denom = np.linalg.norm(H1) * np.linalg.norm(H2)
    if denom <= 1e-300:
        return 0.0
    return float(np.linalg.norm(C) / denom)


def rel_proportional(Md, Me):
    """Distance from proportionality: min over k of ||Me - k Md|| / ||Me||."""
    nMd = np.linalg.norm(Md)
    nMe = np.linalg.norm(Me)
    if nMd <= 1e-300 or nMe <= 1e-300:
        return 0.0
    k = np.vdot(Md.ravel(), Me.ravel()) / (nMd ** 2)
    return float(np.linalg.norm(Me - k * Md) / nMe)


def rel_antisym_part(Me):
    n = np.linalg.norm(Me)
    if n <= 1e-300:
        return 0.0
    return float(np.linalg.norm(Me - Me.T) / (2.0 * n))


if __name__ == "__main__":
    print("=" * 78)
    print("RANK-ALIGNMENT DICHOTOMY  (seed 602719, %d points)" % N_POINTS)
    print("=" * 78)

    # ---- branch 1: a = b = 0  => proportional (rank one) ----
    worst_prop = 0.0
    exc_prop = 0
    for _ in range(N_POINTS // 5):
        S = sym(rng.normal(size=12))
        s_d = rng.normal() + 1j * rng.normal()
        Md = s_d * S
        Me = -3.0 * s_d * S
        r = rel_proportional(Md, Me)
        worst_prop = max(worst_prop, r)
        if r > TOL:
            exc_prop += 1

    print("\nBRANCH 1  (a = b = 0): expect M_d proportional to M_e")
    print("  points                                : %d" % (N_POINTS // 5))
    print("  exceptions (deviation > %g)           : %d" % (TOL, exc_prop))
    print("  worst relative deviation              : %.3e" % worst_prop)

    # ---- branch 2: a or b nonzero => not simultaneously diagonalisable ----
    comms = []
    exc_split = 0
    min_comm = np.inf
    min_anti = np.inf
    for _ in range(N_POINTS):
        A = anti(rng.normal(size=6))
        S = sym(rng.normal(size=12))
        a = rng.normal() + 1j * rng.normal()
        b = rng.normal() + 1j * rng.normal()
        s_d = rng.normal() + 1j * rng.normal()
        Md = a * A + s_d * S
        Me = b * A - 3.0 * s_d * S
        c = rel_commutator(Me, S)
        comms.append(c)
        min_comm = min(min_comm, c)
        min_anti = min(min_anti, rel_antisym_part(Me))
        # the dichotomy would fail if the splitting branch still commuted
        if c <= TOL:
            exc_split += 1

    comms = np.array(comms)
    print("\nBRANCH 2  (a or b nonzero): expect a NONZERO commutator")
    print("  points                                : %d" % N_POINTS)
    print("  exceptions (commutator <= %g)         : %d" % (TOL, exc_split))
    print("  relative commutator  min              : %.4f" % comms.min())
    print("  relative commutator  median           : %.4f" % np.median(comms))
    print("  relative commutator  max              : %.4f" % comms.max())
    print("  smallest antisymmetric fraction of Me : %.4f" % min_anti)

    total_exc = exc_prop + exc_split
    print("\n" + "=" * 78)
    print("TOTAL EXCEPTIONS TO THE DICHOTOMY: %d / %d"
          % (total_exc, N_POINTS + N_POINTS // 5))
    print("VERDICT: %s" % ("PASS" if total_exc == 0 else "FAIL"))
    print("NOTE: genericity evidence over a fixed-seed sample, not a proof")
    print("      that no special texture can evade the obstruction.")
    print("=" * 78)
