# -*- coding: utf-8 -*-
"""第二部分：独立复现其余 6 个 splitting 内容 + 实 vev 降秩检验。
参数量按 CG 表从头独立推算，与论文 n_param 逐项对表。
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(777761)   # 又一个不同种子


def obs(Mu, Md, Me):
    su = np.linalg.svd(Mu, compute_uv=False)
    sd = np.linalg.svd(Md, compute_uv=False)
    se = np.linalg.svd(Me, compute_uv=False)

    def VL(M):
        _, U = np.linalg.eigh(M @ M.conj().T)
        return U[:, ::-1]

    V = VL(Mu).conj().T @ VL(Md)
    J = (V[0, 0] * V[1, 1] * np.conj(V[0, 1]) * np.conj(V[1, 0])).imag
    return np.concatenate([su, sd, se, [abs(V[0, 1]), abs(V[0, 2]), abs(V[1, 2]), J]])


def jac(f, p, h=1e-6):
    Jm = np.zeros((13, len(p)))
    for j in range(len(p)):
        a = p.copy(); a[j] += h
        b = p.copy(); b[j] -= h
        Jm[:, j] = (f(a) - f(b)) / (2 * h)
    return Jm


def nrank(Jm, tol=1e-7):
    s = np.linalg.svd(Jm, compute_uv=False)
    return int((s > s[0] * tol).sum()), s


def build_sym(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(a, 3):
            M[a, b] = p[i] + 1j * p[i + 1]; M[b, a] = M[a, b]; i += 2
    return M, i


def build_anti(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(a + 1, 3):
            M[a, b] = p[i] + 1j * p[i + 1]; M[b, a] = -M[a, b]; i += 2
    return M, i


def build_gen(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(3):
            M[a, b] = p[i] + 1j * p[i + 1]; i += 2
    return M, i


def build_sym_real(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(a, 3):
            M[a, b] = p[i]; M[b, a] = M[a, b]; i += 1
    return M, i


def build_anti_real(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(a + 1, 3):
            M[a, b] = p[i]; M[b, a] = -M[a, b]; i += 1
    return M, i


def build_gen_real(p):
    M = np.zeros((3, 3), complex); i = 0
    for a in range(3):
        for b in range(3):
            M[a, b] = p[i]; i += 1
    return M, i


MODELS = {}


def mk_su5_5_45(real=False):
    bs = build_sym_real if real else build_sym
    ba = build_anti_real if real else build_anti
    bg = build_gen_real if real else build_gen
    st = 1 if real else 2

    def f(p):
        i = 0
        S5, k = bs(p[i:]); i += 6 * st
        A45, k = ba(p[i:]); i += 3 * st
        Y1, k = bg(p[i:]); i += 9 * st
        Y2, k = bg(p[i:]); i += 9 * st
        v = []
        for _ in range(4):
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * S5 + v[2] * A45
        Md = v[1] * Y1 + v[3] * Y2
        Me = v[1] * Y1.T - 3 * v[3] * Y2.T
        return obs(Mu, Md, Me)
    return f, (6 + 3 + 9 + 9) * st + 4 * st


def mk_su5_5_45_50(real=False):
    bs = build_sym_real if real else build_sym
    ba = build_anti_real if real else build_anti
    bg = build_gen_real if real else build_gen
    st = 1 if real else 2

    def f(p):
        i = 0
        S5, k = bs(p[i:]); i += 6 * st
        A45, k = ba(p[i:]); i += 3 * st
        S50, k = bs(p[i:]); i += 6 * st
        Y1, k = bg(p[i:]); i += 9 * st
        Y2, k = bg(p[i:]); i += 9 * st
        v = []
        for _ in range(5):   # 50_H 只喂 up，故只有 v50u
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * S5 + v[2] * A45 + v[4] * S50
        Md = v[1] * Y1 + v[3] * Y2
        Me = v[1] * Y1.T - 3 * v[3] * Y2.T
        return obs(Mu, Md, Me)
    return f, (6 + 3 + 6 + 9 + 9) * st + 5 * st


def mk_so10_10_120(real=False):
    bs = build_sym_real if real else build_sym
    ba = build_anti_real if real else build_anti
    st = 1 if real else 2

    def f(p):
        i = 0
        S, k = bs(p[i:]); i += 6 * st
        A, k = ba(p[i:]); i += 3 * st
        v = []
        for _ in range(4):
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * S + v[2] * A
        Md = v[1] * S - v[3] * A
        Me = v[1] * S + v[3] * A
        return obs(Mu, Md, Me)
    return f, (6 + 3) * st + 4 * st


def mk_so10_10_126(real=False):
    bs = build_sym_real if real else build_sym
    st = 1 if real else 2

    def f(p):
        i = 0
        S1, k = bs(p[i:]); i += 6 * st
        S2, k = bs(p[i:]); i += 6 * st
        v = []
        for _ in range(4):
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * S1 + v[2] * S2
        Md = v[1] * S1 + v[3] * S2
        Me = v[1] * S1 - 3 * v[3] * S2
        return obs(Mu, Md, Me)
    return f, 12 * st + 4 * st


def mk_so10_10_120_126(real=False):
    bs = build_sym_real if real else build_sym
    ba = build_anti_real if real else build_anti
    st = 1 if real else 2

    def f(p):
        i = 0
        S10, k = bs(p[i:]); i += 6 * st
        A120, k = ba(p[i:]); i += 3 * st
        S126, k = bs(p[i:]); i += 6 * st
        v = []
        for _ in range(6):
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * S10 + v[2] * A120 + v[4] * S126
        Md = v[1] * S10 - v[3] * A120 + v[5] * S126
        Me = v[1] * S10 + v[3] * A120 - 3 * v[5] * S126
        return obs(Mu, Md, Me)
    return f, (6 + 3 + 6) * st + 6 * st


def mk_ps_1_15(real=False):
    bg = build_gen_real if real else build_gen
    st = 1 if real else 2

    def f(p):
        i = 0
        Y1, k = bg(p[i:]); i += 9 * st
        Y2, k = bg(p[i:]); i += 9 * st
        v = []
        for _ in range(4):
            v.append(p[i] + (1j * p[i + 1] if not real else 0)); i += st
        Mu = v[0] * Y1 + v[2] * Y2
        Md = v[1] * Y1 + v[3] * Y2
        Me = v[1] * Y1 - 3 * v[3] * Y2
        return obs(Mu, Md, Me)
    return f, 18 * st + 4 * st


MODELS = {
    "SU(5) 5+45":          (mk_su5_5_45,        62),
    "SU(5) 5+45+50":       (mk_su5_5_45_50,     76),
    "SO(10) 10+120":       (mk_so10_10_120,     26),
    "SO(10) 10+126":       (mk_so10_10_126,     32),
    "SO(10) 10+120+126":   (mk_so10_10_120_126, 42),
    "PS (1,2,2)+(15,2,2)": (mk_ps_1_15,         44),
}
EXPECT = {"SU(5) 5+45": 13, "SU(5) 5+45+50": 13, "SO(10) 10+120": 10,
          "SO(10) 10+126": 13, "SO(10) 10+120+126": 13, "PS (1,2,2)+(15,2,2)": 13}


def sample_ranks(f, n, npts):
    ranks = []
    tried = 0
    while len(ranks) < npts and tried < npts * 6:
        tried += 1
        p = rng.normal(size=n) * 0.6
        Jm = jac(f, p)
        if not np.all(np.isfinite(Jm)):
            continue
        r, s = nrank(Jm)
        ranks.append(r)
    return Counter(ranks), len(ranks)


print("=" * 82)
print("PART 2  其余 6 个 splitting 内容：独立实现的秩普查（每模型 60 个随机点）")
print("=" * 82)
print("%-24s %7s %8s %6s %12s %s" % ("model", "n(我)", "n(论文)", "点数", "秩分布", "判定"))
for name, (mk, npaper) in MODELS.items():
    f, n = mk(real=False)
    c, got = sample_ranks(f, n, 60)
    ok = (set(c.keys()) == {EXPECT[name]}) and (n == npaper)
    print("%-24s %7d %8d %6d %12s %s"
          % (name, n, npaper, got, dict(c), "一致" if ok else "** 需复核 **"))

print()
print()
print("=" * 82)
print("PART 3  实 vev 检验：复 Yukawa + 实 vev（22 参数）应降到 11")
print("=" * 82)
print("注：早期版本把 Yukawa 也取成实数（11 参数），那是另一个模型，秩为 8。")
print("    这里按 D5 §2 的冗余计数，只把 vev 取实，Yukawa 保持复。")
print()


def f_120_126_realvev(p):
    """P 复对称(12) + Q 复反对称(6) + 4 个实 vev = 22 参数。"""
    i = 0
    P, k = build_sym(p[i:]); i += 12
    Q, k = build_anti(p[i:]); i += 6
    va, vs, ba, bs = p[i], p[i + 1], p[i + 2], p[i + 3]
    return obs(ba * Q + bs * P, -va * Q + vs * P, va * Q - 3 * vs * P)


def f_120_126_cplxvev(p):
    """同一模型，vev 取复：26 参数。"""
    i = 0
    P, k = build_sym(p[i:]); i += 12
    Q, k = build_anti(p[i:]); i += 6
    va = p[i] + 1j * p[i + 1]; vs = p[i + 2] + 1j * p[i + 3]
    ba = p[i + 4] + 1j * p[i + 5]; bs = p[i + 6] + 1j * p[i + 7]
    return obs(ba * Q + bs * P, -va * Q + vs * P, va * Q - 3 * vs * P)


def f_120_126_realall(p):
    """对照：Yukawa 与 vev 皆实（11 参数）——不是本文的模型。"""
    i = 0
    P, k = build_sym_real(p[i:]); i += 6
    Q, k = build_anti_real(p[i:]); i += 3
    a, b = p[i], p[i + 1]
    return obs(a * P + b * Q, P + Q, -3 * P - Q)


for label, f, n, want in [("复 Yukawa + 复 vev (26)", f_120_126_cplxvev, 26, 12),
                          ("复 Yukawa + 实 vev (22)", f_120_126_realvev, 22, 11),
                          ("实 Yukawa + 实 vev (11) 对照", f_120_126_realall, 11, 8)]:
    c, got = sample_ranks(f, n, 80)
    ok = set(c.keys()) == {want}
    print("%-32s 秩分布 %-12s 有效点 %3d  预期 %2d  %s"
          % (label, dict(c), got, want, "一致" if ok else "** 需复核 **"))

print()
print("=> 22 参数版本给出 rank 11，与 D5 的冗余计数 22-2-9-0=11 一致。")
print("   26 参数版本给出 12，与论文一致；11 参数版本给出 8，属另一个模型。")
