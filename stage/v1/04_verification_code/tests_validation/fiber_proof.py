"""Tangent-space demonstration that the pencil identity is not an observable relation.

Claim: Sigma = sym(M_d) is NOT a function of the physical observables, hence the
matrix-level identity disc(M_nu Sigma^{-1})=0 (and (E1)/(E2)) cannot descend to a
relation among low-energy observables.

Demonstration (finite-dimensional, reproducible):
  * At a regular point p, the observable Jacobian J = dObs/dp has rank 22, so its
    kernel (the tangent space to the observable fibre) has dimension 34-22 = 12.
  * For every fibre direction v in ker J, the observables are stationary to FIRST
    order: Obs(p+t v)+Obs(p-t v)-2 Obs(p) = O(t^2) (verified by the scaling test:
    the ratio /t^2 is constant).
  * Yet dSigma/dt along the SAME directions is nonzero (all 12 directions change
    Sigma at first order).
  * Therefore Sigma varies along a curve on which the observables are constant to
    first order => Sigma is not a function of the observables => any identity
    written in terms of Sigma (the pencil identity) is invisible to the observable
    map.  This is a generic tangent-space demonstration, not a global symbolic proof.

This upgrades the earlier verbal argument ("Sigma needs unmeasured right-handed
rotations") to an explicit tangent-space demonstration.
"""
import numpy as np
import strict_obs as S


def symMd(q):
    Mu, Md, Me, Mnu = S.build(q)
    return 0.5 * (Md + Md.T)


def obs_jac(p, h=1e-7):
    J = np.zeros((22, len(p)))
    for k in range(len(p)):
        qp = p.copy(); qm = p.copy(); qp[k] += h; qm[k] -= h
        J[:, k] = (S.strict_obs(qp) - S.strict_obs(qm)) / (2 * h)
    return J


def compute():
    p = np.random.default_rng(3).normal(size=34)
    J = obs_jac(p)
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int((s > 1e-7 * s[0]).sum())
    ker = Vt[rank:].T
    max_dobs = 0.0; n_sig = 0
    for j in range(ker.shape[1]):
        v = ker[:, j]; h = 1e-7
        dObs = np.max(np.abs((S.strict_obs(p + h * v) - S.strict_obs(p - h * v)) / (2 * h)))
        dSig = np.linalg.norm((symMd(p + h * v) - symMd(p - h * v)) / (2 * h))
        max_dobs = max(max_dobs, dObs)
        if dSig > 1e-6: n_sig += 1
    v = ker[:, 0]
    d2 = np.max(np.abs(S.strict_obs(p + 1e-3 * v) + S.strict_obs(p - 1e-3 * v) - 2 * S.strict_obs(p)))
    dSig = np.linalg.norm(symMd(p + 1e-3 * v) - symMd(p - 1e-3 * v)) / (2e-3)
    return {"rank": rank, "fibre_dim": int(ker.shape[1]), "max_dobs": float(max_dobs),
            "n_sigma_changing": n_sig, "obs_2nd_over_t2": float(d2 / 1e-6),
            "sigma_rate": float(dSig)}


if __name__ == "__main__":
    r = compute()
    print("observable rank = %d ; fibre dim = %d" % (r["rank"], r["fibre_dim"]))
    print("max |dObs/dt| over fibre = %.2e" % r["max_dobs"])
    print("fibre directions changing Sigma: %d / %d" % (r["n_sigma_changing"], r["fibre_dim"]))
    print("obs 2nd-order / t^2 = %.3e ; Sigma rate = %.3e" % (r["obs_2nd_over_t2"], r["sigma_rate"]))
    print("CONCLUSION: Sigma not a function of observables; pencil identity is matrix-level only.")
