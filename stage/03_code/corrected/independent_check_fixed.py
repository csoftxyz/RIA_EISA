# -*- coding: utf-8 -*-
"""独立核验脚本（不复用论文代码）。

本文件只做两件事（其余检查在 independent_check2 / check3 中）：
1) 用独立参数化 M_d = P + Q, M_e = -3P - Q, M_u = aP + bQ（P 对称、Q 反对称）
   重算 SO(10) 120+126 的 rank Dpi，300 个随机点，种子与论文不同。
2) 提取 Jacobian 左零空间方向 n(p)，统计其逐分量漂移，
   用以判定这条关系是线性（n 恒定）还是非线性（n 随点变化）。
"""
import numpy as np

rng = np.random.default_rng(999983)   # 与论文种子 20261113 不同


# ---------- 可观测量（独立实现） ----------
def obs(Mu, Md, Me):
    su = np.linalg.svd(Mu, compute_uv=False)
    sd = np.linalg.svd(Md, compute_uv=False)
    se = np.linalg.svd(Me, compute_uv=False)

    def VL(M):
        _, U = np.linalg.eigh(M @ M.conj().T)
        return U[:, ::-1]

    V = VL(Mu).conj().T @ VL(Md)
    J = (V[0, 0] * V[1, 1] * np.conj(V[0, 1]) * np.conj(V[1, 0])).imag
    return np.concatenate([su, sd, se,
                           [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J]])


NAMES = ["mu1", "mu2", "mu3", "md1", "md2", "md3", "me1", "me2", "me3",
         "|Vus|", "|Vub|", "|Vcb|", "J"]


def jac(f, p, h=1e-6):
    n = len(p)
    Jm = np.zeros((13, n))
    for j in range(n):
        a = p.copy(); a[j] += h
        b = p.copy(); b[j] -= h
        Jm[:, j] = (f(a) - f(b)) / (2 * h)
    return Jm


def nrank(Jm, tol=1e-7):
    s = np.linalg.svd(Jm, compute_uv=False)
    return int((s > s[0] * tol).sum()), s


def rand_sym(rng):
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return (Z + Z.T) / 2


def rand_anti(rng):
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return (Z - Z.T) / 2


# =====================================================================
# 第一部分：SO(10) 120+126，独立参数化
#   参数 22 实：P(对称,6复=12) Q(反对称,3复=6) a(2) b(2)
# =====================================================================
def unpack120_126(p):
    """p: 22 实向量 -> (P, Q, a, b)"""
    P = np.zeros((3, 3), complex)
    k = 0
    for i in range(3):
        for j in range(i, 3):
            P[i, j] = p[k] + 1j * p[k + 1]
            P[j, i] = P[i, j]
            k += 2
    Q = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(i + 1, 3):
            Q[i, j] = p[k] + 1j * p[k + 1]
            Q[j, i] = -Q[i, j]
            k += 2
    a = p[18] + 1j * p[19]
    b = p[20] + 1j * p[21]
    return P, Q, a, b


def f_120_126(p):
    P, Q, a, b = unpack120_126(p)
    Md = P + Q
    Me = -3 * P - Q
    Mu = a * P + b * Q
    return obs(Mu, Md, Me)


def good_point(p, f, tries=200):
    for _ in range(tries):
        pp = rng.normal(size=len(p)) * 0.6
        pp[18] = 1.1 + 0.2 * rng.normal(); pp[19] = 0.3 * rng.normal()
        pp[20] = 0.7 + 0.2 * rng.normal(); pp[21] = 0.4 * rng.normal()
        P, Q, a, b = unpack120_126(pp)
        Ms = (a * P + b * Q, P + Q, -3 * P - Q)
        ok = True
        for M in Ms:
            s = np.linalg.svd(M, compute_uv=False)
            if np.min(-np.diff(s)) < 1e-3:
                ok = False
        if ok:
            return pp
    return None


print("=" * 74)
print("PART 1  SO(10) 120+126 -- 独立参数化，秩普查")
print("=" * 74)
p0 = np.zeros(22)
ranks = []
null_vecs = []
pts = []
while len(ranks) < 300:
    pp = good_point(p0, f_120_126)
    Jm = jac(f_120_126, pp)
    if not np.all(np.isfinite(Jm)):
        continue
    r, s = nrank(Jm)
    ranks.append(r)
    if r == 12:
        U = np.linalg.svd(Jm)[0]
        n = U[:, -1]
        if n[NAMES.index("md1")] < 0:
            n = -n
        null_vecs.append(n)
        pts.append(pp)
from collections import Counter
print("300 个随机点的秩分布 :", dict(Counter(ranks)))
print("论文声称 rank=12 (P_add=1) ->", "一致" if set(ranks) == {12} else "不一致")

NV = np.array(null_vecs)
print("\n左零空间方向 n(p) 的逐分量统计（13 个可观测量）：")
print("%-8s %12s %12s %12s" % ("obs", "mean", "std", "std/|mean|"))
for i, nm in enumerate(NAMES):
    mu_ = NV[:, i].mean(); sd_ = NV[:, i].std()
    flag = "  <== 恒定" if sd_ < 1e-6 * max(abs(mu_), 1e-12) else ""
    print("%-8s %12.4e %12.4e %12.3f%s" % (nm, mu_, sd_, sd_ / max(abs(mu_), 1e-30), flag))

print("\n若 n(p) 恒定 => 关系是线性的 sum_i n_i O_i = const")
n_mean = NV.mean(axis=0)
vals = []
for pp in pts[:60]:
    vals.append(float(n_mean @ f_120_126(pp)))
vals = np.array(vals)
print("  用平均法向量算 sum n_i O_i：mean=%.6e  std=%.6e" % (vals.mean(), vals.std()))
print("  相对波动 = %.3e" % (vals.std() / max(abs(vals.mean()), 1e-30)))
