"""DEPRECATED OBSERVABLE CONSTRUCTIONS -- DO NOT USE FOR DIMENSION VERDICTS.

The ckm_obs / charged_obs / full_obs functions below build CKM/PMNS from
np.linalg.eigh(M M^dag) eigenvectors and Majorana phases from
angle[(Un^T Mnu Un)_ii].  These are NOT basis-independent (audit 9): they were
the source of the withdrawn 19/three count.  They are kept ONLY because
audit5_tests imports the builders; the observable functions themselves must not
be used to judge the physical dimension.

The CORRECT, basis-independent observables and the single source of truth for
dimensions live in strict_obs.py (Autonne-Takagi).  Use strict_obs.run().
"""
import numpy as np


def anti(q):
    a, b, c = q[0] + 1j * q[1], q[2] + 1j * q[3], q[4] + 1j * q[5]
    return np.array([[0, a, b], [-a, 0, c], [-b, -c, 0]], complex)


def sym(q):
    z = [q[2 * t] + 1j * q[2 * t + 1] for t in range(6)]
    return np.array([[z[0], z[1], z[2]], [z[1], z[3], z[4]], [z[2], z[4], z[5]]], complex)


def masses(M):
    return np.sort(np.linalg.svd(M, compute_uv=False))[::-1]


def ckm_obs(Mu, Md):
    Hu = Mu @ Mu.conj().T; Hd = Md @ Md.conj().T
    wu, Uu = np.linalg.eigh(Hu); wd, Ud = np.linalg.eigh(Hd)
    Uu = Uu[:, np.argsort(wu)[::-1]]; Ud = Ud[:, np.argsort(wd)[::-1]]
    V = Uu.conj().T @ Ud
    return [abs(V[0, 1]) ** 2, abs(V[0, 2]) ** 2, abs(V[1, 2]) ** 2,
            np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2] * V[1, 1]))]


def charged_obs(Mu, Md, Me):
    out = []
    for M in (Mu, Md, Me):
        out += list(masses(M) ** 2)          # 9 squared masses
    out += ckm_obs(Mu, Md)                    # 4 CKM
    return np.array(out)                      # 13


def build_120_126_charged(p):
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(6)]
    v1u, v15u, v1d, v15d, su, sd = v
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    return Mu, Md, Me


def build_10_120_126(p):
    h = sym(p[0:12]); A = anti(p[12:18]); S = sym(p[18:30])
    v = [p[30 + 2 * t] + 1j * p[31 + 2 * t] for t in range(10)]
    au, ad_, v1u, v15u, v1d, v15d, su, sd, vR, vL = v
    Mu = au * h + (v1u + v15u) * A + su * S
    Md = ad_ * h + (v1d + v15d) * A + sd * S
    Me = ad_ * h + (v1d - 3 * v15d) * A - 3 * sd * S
    MD = au * h + (v1u - 3 * v15u) * A - 3 * su * S
    Mnu = vL * S - MD @ np.linalg.inv(vR * S) @ MD.T
    return Mu, Md, Me, Mnu


def full_obs(Mu, Md, Me, Mnu):
    out = list(charged_obs(Mu, Md, Me))
    Hn = Mnu @ Mnu.conj().T
    out += list(masses(Mnu) ** 2)             # 3 nu masses
    He = Me @ Me.conj().T
    we, Ue = np.linalg.eigh(He); wn, Un = np.linalg.eigh(Hn)
    Ue = Ue[:, np.argsort(we)[::-1]]; Un = Un[:, np.argsort(wn)[::-1]]
    P = Ue.conj().T @ Un
    out += [abs(P[0, 1]) ** 2, abs(P[0, 2]) ** 2, abs(P[1, 2]) ** 2,
            np.imag(P[0, 1] * P[1, 2] * np.conj(P[0, 2] * P[1, 1]))]
    Dg = Un.T @ Mnu @ Un
    out += [np.angle(Dg[0, 0]), np.angle(Dg[1, 1])]
    return np.array(out)                      # 22


def rank_of(obs_fn, build_fn, npar, seeds=(1, 2, 3, 42, 99)):
    ranks = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        p = rng.normal(size=npar)
        f0 = obs_fn(*build_fn(p))
        J = np.zeros((len(f0), npar)); h = 1e-7
        for k in range(npar):
            q = p.copy(); q[k] += h
            J[:, k] = (obs_fn(*build_fn(q)) - f0) / h
        s = np.linalg.svd(J, compute_uv=False)
        ranks.append(int((s > 1e-8 * s[0]).sum()))
    return ranks


def build_120_126_vLconstrained(p):
    """120+126 with v_L tied to the seesaw scale: v_L = kappa * (su*sd)/vR
    (a D-flatness-like relation), so v_L is NOT a free parameter.
    Parameters: A(6) S(12) v1u v15u v1d v15d su sd vR kappa (16) = 34.
    We replace the free vL by kappa*su*sd/vR."""
    A = anti(p[0:6]); S = sym(p[6:18])
    v = [p[18 + 2 * t] + 1j * p[19 + 2 * t] for t in range(8)]
    v1u, v15u, v1d, v15d, su, sd, vR, kappa = v
    vL = kappa * su * sd / vR
    Mu = (v1u + v15u) * A + su * S
    Md = (v1d + v15d) * A + sd * S
    Me = (v1d - 3 * v15d) * A - 3 * sd * S
    MD = (v1u - 3 * v15u) * A - 3 * su * S
    Mnu = vL * S - MD @ np.linalg.inv(vR * S) @ MD.T
    return Mu, Md, Me, Mnu


if __name__ == "__main__":
    print("charged-only 120+126 (13 obs, 30 par):",
          rank_of(charged_obs, build_120_126_charged, 30))
    print("full 10+120+126 (22 obs, 50 par):",
          rank_of(full_obs, build_10_120_126, 50))
    print("120+126 v_L constrained (22 obs, 34 par):",
          rank_of(full_obs, build_120_126_vLconstrained, 34))
