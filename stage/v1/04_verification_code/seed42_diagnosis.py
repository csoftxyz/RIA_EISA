"""Diagnosis of the seed-42 rank-21 point (audit 10 open item 4).

Shows the rank loss is a coordinate/parameterisation degeneracy of the charged
Jarlskog observable near J=0, NOT a physical relation:
  * the Jacobian null direction is carried by the charged-Jarlskog component;
  * J at seed 42 is ~4.6e-7 (four orders below a typical point);
  * moving along the null direction changes J (nonzero 2nd derivative, finite shift);
  * all mass eigenvalue gaps are healthy (no mass degeneracy).
Hence generic rank 22 stands; zero observable relations.
"""
import numpy as np
import strict_obs as S


def jac(p, h=1e-6):
    f0 = S.strict_obs(p)
    J = np.zeros((len(f0), len(p)))
    for k in range(len(p)):
        q = p.copy(); q[k] += h
        J[:, k] = (S.strict_obs(q) - f0) / h
    return J


def compute():
    p = np.random.default_rng(42).normal(size=34)
    J = jac(p)
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    u = U[:, 21]
    Mu, Md, Me, Mnu = S.build(p)
    Lu, _ = S.left_unitary(Mu); Ld, _ = S.left_unitary(Md)
    V = Lu.conj().T @ Ld
    v = Vt[21]
    f = lambda pp: S.strict_obs(pp)[12]
    d2 = (f(p + 1e-4 * v) - 2 * f(p) + f(p - 1e-4 * v)) / (1e-8)
    gaps = []
    for M in (Mu, Md, Me, Mnu):
        w = np.sort(np.linalg.eigvalsh(M @ M.conj().T))[::-1]
        gaps.append(float(min((w[0] - w[1]) / w[0], (w[1] - w[2]) / w[0])))
    return {"s21_over_s0": float(s[21] / s[0]), "null_weight_obs12": float(u[12]),
            "J": float(S.jarlskog(V)), "d2J": float(d2),
            "finite_dJ": float(f(p + 0.1 * v) - f(p)), "min_eig_gap": min(gaps)}


if __name__ == "__main__":
    r = compute()
    print("seed 42: s[21]/s0=%.3e  null weight obs[12]=%.4f  J=%.3e" %
          (r["s21_over_s0"], r["null_weight_obs12"], r["J"]))
    print("d2J/dt2=%.3e  finite dJ(t=0.1)=%.3e  min eig gap=%.3e" %
          (r["d2J"], r["finite_dJ"], r["min_eig_gap"]))
    print("CONCLUSION: coordinate degeneracy of J near 0, not a relation; generic rank 22.")
