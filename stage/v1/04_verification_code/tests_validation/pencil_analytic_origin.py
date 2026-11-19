"""Analytic origin of the pencil identity (breakthrough).

The pencil identity disc(Mnu S^-1)=0 was previously verified numerically.  Here
we give and verify the full algebraic derivation chain, valid for EVERY point:

  M_D = a A + b S   (A antisymmetric 3x3, S symmetric; a=v1u-3v15u, b=-3su)
  Mnu = vL S - (1/vR) M_D S^-1 M_D^T
      = vL S - (1/vR)(a A S^-1 + b I)(-a A + b S)        [A^T = -A]
      = vL S - (1/vR)(-a^2 B A + ab(B S - A) + b^2 S)     [B = A S^-1]
  But B S = A S^-1 S = A, so the cross term ab(B S - A) = 0 EXACTLY.
      = (vL - b^2/vR) S + (a^2/vR) A S^-1 A
      = c1 S + c2 A S^-1 A,   c1 = vL - b^2/vR,  c2 = a^2/vR.

  Now Mnu S^-1 = c1 I + c2 (A S^-1)(A S^-1) = c1 I + c2 B^2.
  A is 3x3 antisymmetric => det A = 0 => B = A S^-1 has rank <= 2 and tr B = 0
  (tr(A S^-1) = tr(S^-1 A) = 0 since S^-1 symmetric, A antisymmetric).
  A traceless rank-2 3x3 matrix has eigenvalues {lambda, -lambda, 0}.
  Hence B^2 has eigenvalues {lambda^2, lambda^2, 0}: a REPEATED eigenvalue.
  Therefore Mnu S^-1 = c1 I + c2 B^2 has a repeated eigenvalue => disc = 0.  QED.

The repeated root is forced by the ANTISYMMETRY of the 120 Yukawa (Pfaffian
structure det A = 0), not by any parameter choice.  This is the group-theoretic
origin of the pencil identity.

This script verifies each link of the chain numerically to machine precision.
"""
import numpy as np
import strict_obs as S


def verify_chain(n=200, seed=17):
    rng = np.random.default_rng(seed)
    max_cross = 0.0      # cross term ab(BS - A) should vanish
    max_reduction = 0.0  # Mnu == c1 S + c2 A S^-1 A
    max_B2 = 0.0         # A S^-1 A S^-1 == B^2
    max_trB = 0.0        # tr B == 0
    max_rankB = 0        # rank B <= 2
    max_repeat = 0.0     # B^2 repeated eigenvalue (two equal)
    max_disc = 0.0       # disc(Mnu S^-1) == 0
    for _ in range(n):
        A = S.anti(rng.normal(size=6)); Sy = S.sym(rng.normal(size=12))
        if abs(np.linalg.det(Sy)) < 1e-8:
            continue
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
        if abs(vR) < 1e-8:
            continue
        vL = kappa * su * sd / vR
        a = v1u - 3 * v15u; b = -3 * su
        Si = np.linalg.inv(Sy)
        B = A @ Si
        MD = a * A + b * Sy
        Mnu = vL * Sy - MD @ np.linalg.inv(vR * Sy) @ MD.T
        c1 = vL - b * b / vR; c2 = a * a / vR
        # link 1: cross term vanishes
        max_cross = max(max_cross, np.max(np.abs(B @ Sy - A)))
        # link 2: reduction
        max_reduction = max(max_reduction, np.max(np.abs(Mnu - (c1 * Sy + c2 * A @ Si @ A))))
        # link 3: B^2 identity
        max_B2 = max(max_B2, np.max(np.abs(A @ Si @ A @ Si - B @ B)))
        # link 4: tr B = 0
        max_trB = max(max_trB, abs(np.trace(B)))
        # link 5: rank B <= 2
        max_rankB = max(max_rankB, np.linalg.matrix_rank(B, tol=1e-7))
        # link 6: B^2 repeated eigenvalue (min pairwise gap ~0)
        eB2 = np.linalg.eigvals(B @ B)
        gaps = [abs(eB2[i] - eB2[j]) for i in range(3) for j in range(i + 1, 3)]
        max_repeat = max(max_repeat, min(gaps))  # closest pair = the repeated root
        # link 7: disc(Mnu S^-1) = 0
        M = Mnu @ Si
        t1 = np.trace(M); t2 = (t1 * t1 - np.trace(M @ M)) / 2; t3 = np.linalg.det(M)
        disc = t1 * t1 * t2 * t2 - 4 * t2**3 - 4 * t1**3 * t3 - 27 * t3**2 + 18 * t1 * t2 * t3
        sc = max(np.max(np.abs(np.linalg.eigvals(M))), 1e-12)
        max_disc = max(max_disc, abs(disc) / sc**6)
    return {"n": n,
            "cross_term_max": float(max_cross),
            "reduction_max": float(max_reduction),
            "B2_identity_max": float(max_B2),
            "trB_max": float(max_trB),
            "rankB_max": int(max_rankB),
            "B2_repeat_gap_max": float(max_repeat),
            "disc_rel_max": float(max_disc),
            "chain_verified": bool(max_cross < 1e-9 and max_reduction < 1e-7
                                   and max_B2 < 1e-9 and max_trB < 1e-9
                                   and max_rankB <= 2 and max_disc < 1e-8),
            "chain": "disc(Mnu S^-1)=0 follows from antisymmetry of A: B=A S^-1 "
                     "traceless rank-2 => B^2 has a repeated eigenvalue => pencil. "
                     "The repeated root is forced by det A=0 (Pfaffian), not by parameters. "
                     "Analytic chain, grade (b): derived algebraically and confirmed numerically."}


if __name__ == "__main__":
    r = verify_chain()
    print("PENCIL ANALYTIC ORIGIN -- derivation chain verification (n=%d)" % r["n"])
    print("  cross term ab(BS-A) max      : %.2e (should ~0)" % r["cross_term_max"])
    print("  reduction Mnu=c1S+c2AS^-1A   : %.2e" % r["reduction_max"])
    print("  B^2 = AS^-1AS^-1 max         : %.2e" % r["B2_identity_max"])
    print("  tr B max                     : %.2e (should ~0)" % r["trB_max"])
    print("  rank B max                   : %d (should <=2)" % r["rankB_max"])
    print("  B^2 repeated-root gap max    : %.2e (should ~0)" % r["B2_repeat_gap_max"])
    print("  disc(Mnu S^-1) rel max       : %.2e (should ~0)" % r["disc_rel_max"])
    print("  CHAIN VERIFIED:", r["chain_verified"])
    print("  ANALYTIC CHAIN (grade b, numerically confirmed):", r["chain"])
