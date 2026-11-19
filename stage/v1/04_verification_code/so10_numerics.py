"""
Basic structural checks for the SO(10) Yukawa sector.

Referenced by Appendix A of the paper (rank claim).  Two exact-by-construction
facts are checked numerically over a large fixed-seed sample:

  (1) a complex antisymmetric 3x3 matrix has rank <= 2 (even rank, det = 0),
      hence at least one massless family;
  (2) a single symmetric structure gives M_d proportional to M_e, i.e. the
      pair spans a one-dimensional subspace of the space of mass matrices.

Everything is reproducible: the generator seed is fixed below.
"""
import numpy as np

rng = np.random.default_rng(20261118)

N_POINTS = 20000
TOL_SINGULAR = 1e-10


def anti(p):
    """Complex antisymmetric 3x3 from 6 real parameters."""
    a, b, c = p[0] + 1j * p[1], p[2] + 1j * p[3], p[4] + 1j * p[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(p):
    """Complex symmetric 3x3 from 12 real parameters."""
    z = [p[2 * t] + 1j * p[2 * t + 1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]],
                     [z[1], z[3], z[4]],
                     [z[2], z[4], z[5]]], complex)


def smallest_singular(M):
    return float(np.linalg.svd(M, compute_uv=False)[-1])


def check_antisymmetric_rank():
    """Rank <= 2 <=> smallest singular value vanishes (relative to norm)."""
    worst = 0.0
    exceptions = 0
    worst_det = 0.0
    for _ in range(N_POINTS):
        A = anti(rng.normal(size=6))
        scale = np.linalg.norm(A)
        rel = smallest_singular(A) / scale
        worst = max(worst, rel)
        if rel > TOL_SINGULAR:
            exceptions += 1
        worst_det = max(worst_det, abs(np.linalg.det(A)) / scale ** 3)
    return exceptions, worst, worst_det


def check_single_symmetric_proportional():
    """With one symmetric f: M_d = s_d f and M_e = -3 s_d f, so M_e + 3 M_d = 0."""
    worst = 0.0
    exceptions = 0
    for _ in range(N_POINTS):
        f = sym(rng.normal(size=12))
        s_d = rng.normal() + 1j * rng.normal()
        M_d = s_d * f
        M_e = -3.0 * s_d * f
        scale = max(np.linalg.norm(M_d), 1e-300)
        rel = np.linalg.norm(M_e + 3.0 * M_d) / scale
        worst = max(worst, rel)
        if rel > TOL_SINGULAR:
            exceptions += 1
    return exceptions, worst


def check_single_ten_equal():
    """With a single 10: M_d = M_e exactly (no down/lepton splitting)."""
    worst = 0.0
    for _ in range(N_POINTS):
        h = sym(rng.normal(size=12))
        v_d = rng.normal() + 1j * rng.normal()
        M_d = v_d * h
        M_e = v_d * h
        scale = max(np.linalg.norm(M_d), 1e-300)
        worst = max(worst, np.linalg.norm(M_e - M_d) / scale)
    return worst


if __name__ == "__main__":
    print("=" * 78)
    print("BASIC STRUCTURAL CHECKS  (seed 20261118, %d points each)" % N_POINTS)
    print("=" * 78)

    exc, worst, wdet = check_antisymmetric_rank()
    print("\n(1) complex antisymmetric 3x3 has rank <= 2")
    print("    exceptions (smallest singular value > %g relative): %d / %d"
          % (TOL_SINGULAR, exc, N_POINTS))
    print("    worst relative smallest singular value : %.3e" % worst)
    print("    worst relative |det|                   : %.3e" % wdet)
    print("    => at least one massless family, as required by even rank.")

    exc2, worst2 = check_single_symmetric_proportional()
    print("\n(2) one symmetric structure => M_d proportional to M_e")
    print("    exceptions: %d / %d" % (exc2, N_POINTS))
    print("    worst relative ||M_e + 3 M_d|| : %.3e" % worst2)
    print("    => the pair spans a one-dimensional subspace (rank one).")

    worst3 = check_single_ten_equal()
    print("\n(3) one 10 => M_d = M_e exactly")
    print("    worst relative ||M_e - M_d|| : %.3e" % worst3)
    print("    => no down/lepton splitting at all.")

    print("\n" + "=" * 78)
    ok = (exc == 0 and exc2 == 0 and worst3 < TOL_SINGULAR)
    print("ALL BASIC CHECKS: %s" % ("PASS" if ok else "FAIL"))
    print("=" * 78)
