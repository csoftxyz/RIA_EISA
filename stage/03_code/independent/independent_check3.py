# -*- coding: utf-8 -*-
"""第三部分（改正）：按论文原意，只把 vev 限制为实数，Yukawa 矩阵保持复数。
论文补充材料 §1 声称：SO(10) 120+126 在实 vev 下 rank 从 12 降到 11，
"会凭空多出一条伪关系"。此处核验该说法。
同时对其余 splitting 内容做同样的实 vev 对照，看降秩是否是普遍现象。
"""
import numpy as np
from collections import Counter

rng = np.random.default_rng(515011)


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
    return int((s > s[0] * tol).sum())


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


# ------- 120+126：复 Yukawa（12+6=18 实参数）+ vev 复(4)/实(2) -------
def f_126_cplx_vev(p):      # 22 参数
    P, k = build_sym(p[0:12])
    Q, _ = build_anti(p[12:18])
    a = p[18] + 1j * p[19]
    b = p[20] + 1j * p[21]
    return obs(a * P + b * Q, P + Q, -3 * P - Q)


def f_126_real_vev(p):      # 20 参数：Yukawa 复，vev 实
    P, k = build_sym(p[0:12])
    Q, _ = build_anti(p[12:18])
    a = p[18]; b = p[19]
    return obs(a * P + b * Q, P + Q, -3 * P - Q)


# ------- 10+120 同样对照 -------
def f_10120_cplx(p):        # 26 参数
    S, k = build_sym(p[0:12])
    A, _ = build_anti(p[12:18])
    v = [p[18] + 1j * p[19], p[20] + 1j * p[21], p[22] + 1j * p[23], p[24] + 1j * p[25]]
    return obs(v[0] * S + v[2] * A, v[1] * S - v[3] * A, v[1] * S + v[3] * A)


def f_10120_real(p):        # 22 参数：Yukawa 复，vev 实
    S, k = build_sym(p[0:12])
    A, _ = build_anti(p[12:18])
    v = [p[18], p[19], p[20], p[21]]
    return obs(v[0] * S + v[2] * A, v[1] * S - v[3] * A, v[1] * S + v[3] * A)


# ------- 10+126（满秩组）对照 -------
def f_10126_cplx(p):        # 32
    S1, k = build_sym(p[0:12]); S2, _ = build_sym(p[12:24])
    v = [p[24] + 1j * p[25], p[26] + 1j * p[27], p[28] + 1j * p[29], p[30] + 1j * p[31]]
    return obs(v[0] * S1 + v[2] * S2, v[1] * S1 + v[3] * S2, v[1] * S1 - 3 * v[3] * S2)


def f_10126_real(p):        # 28
    S1, k = build_sym(p[0:12]); S2, _ = build_sym(p[12:24])
    v = [p[24], p[25], p[26], p[27]]
    return obs(v[0] * S1 + v[2] * S2, v[1] * S1 + v[3] * S2, v[1] * S1 - 3 * v[3] * S2)


# ------- SU(5) 5+45（满秩组）对照 -------
def f_su5_cplx(p):          # 62
    S5, k = build_sym(p[0:12]); A45, _ = build_anti(p[12:18])
    Y1, k2 = build_gen(p[18:36]); Y2, _ = build_gen(p[36:54])
    v = [p[54 + 2 * i] + 1j * p[55 + 2 * i] for i in range(4)]
    return obs(v[0] * S5 + v[2] * A45, v[1] * Y1 + v[3] * Y2,
               v[1] * Y1.T - 3 * v[3] * Y2.T)


def f_su5_real(p):          # 58
    S5, k = build_sym(p[0:12]); A45, _ = build_anti(p[12:18])
    Y1, k2 = build_gen(p[18:36]); Y2, _ = build_gen(p[36:54])
    v = [p[54], p[55], p[56], p[57]]
    return obs(v[0] * S5 + v[2] * A45, v[1] * Y1 + v[3] * Y2,
               v[1] * Y1.T - 3 * v[3] * Y2.T)


CASES = [
    ("SO(10) 120+126", f_126_cplx_vev, 22, f_126_real_vev, 20, 12),
    ("SO(10) 10+120",  f_10120_cplx,   26, f_10120_real,   22, 10),
    ("SO(10) 10+126",  f_10126_cplx,   32, f_10126_real,   28, 13),
    ("SU(5) 5+45",     f_su5_cplx,     62, f_su5_real,     58, 13),
]


def scan(f, n, npts=70):
    c = Counter(); got = 0
    while got < npts and got < npts * 5:
        p = rng.normal(size=n) * 0.6
        Jm = jac(f, p)
        if not np.all(np.isfinite(Jm)):
            continue
        c[nrank(Jm)] += 1; got += 1
    return c


print("=" * 88)
print("PART 3(改正)  仅把 vev 限制为实数（Yukawa 保持复数）：论文补充材料 §1 的声称")
print("=" * 88)
print("%-18s %16s %16s %10s %s"
      % ("model", "复 vev 秩", "实 vev 秩", "论文复vev", "是否降秩"))
for name, fc, nc, fr, nr, expect in CASES:
    cc = scan(fc, nc)
    cr = scan(fr, nr)
    rc = list(cc.keys())[0] if len(cc) == 1 else dict(cc)
    rr = list(cr.keys())[0] if len(cr) == 1 else dict(cr)
    drop = (isinstance(rr, int) and isinstance(rc, int) and rr < rc)
    print("%-18s %16s %16s %10d %s"
          % (name, str(cc), str(cr), expect, "降秩 (-%d)" % (rc - rr) if drop else "不降"))

print()
print("论文原话：实 vev 使 120+126 的秩 12 -> 11，'会凭空多出一条伪关系'")
print("上表第一行即为该声称的独立检验结果。")
