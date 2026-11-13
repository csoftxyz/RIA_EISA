# -*- coding: utf-8 -*-
"""D5 最终验证：解析迹恒等式 -> 可观测量的精确关系与可检验预言。

推导（见 accompanying 文档）：
  取 SO(10) flavor 规范，使对称部分 S 经 Takagi 分解为 S = U D U^T，
  用 U(3) 合同冗余固定 S -> D（实对角正）。则
      M_d = D + A,   M_e = -3D - A,   M_u = beta (D + rho A)
  其中 A 反对称，rho = alpha/beta 复，beta > 0。
  令 G := A A^dagger (半正定), Z := -D A^* + A D (Hermitian 无迹)，则
      H_d := M_d M_d^dagger = D^2 + G + Z
      H_e := M_e M_e^dagger = 9 D^2 + G + 3 Z
  取迹（tr Z = 0, tr(D^2 Z) = 0）：
      tr H_d = Delta + Gamma ,  tr H_e = 9 Delta + Gamma
      Delta := tr(D^2) = tr(S S^dagger) > 0   （合同不变量！）
      Gamma := tr(G)   = tr(A A^dagger) >= 0  （合同不变量！）
  =>  可观测量精确关系：
      sum_i m_{e,i}^2 - sum_i m_{d,i}^2 = 8 tr(S S^dagger) > 0
      9 sum_i m_{d,i}^2 - sum_i m_{e,i}^2 = 8 tr(A A^dagger) >= 0
  =>  可检验双重不等式： sum m_d^2 < sum m_e^2 <= 9 sum m_d^2
"""
import numpy as np

rng = np.random.default_rng(271828)


def sv(M):
    return np.linalg.svd(M, compute_uv=False)


def rand_sym():
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return (Z + Z.T) / 2


def rand_anti():
    Z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return (Z - Z.T) / 2


print("=" * 82)
print("STEP A  精确迹恒等式（原始 vev 定义，不做任何规范固定）")
print("=" * 82)
print("S = -(M_d+M_e)/2 ,  A = (3M_d+M_e)/2 ,  Delta=tr(SS^dag), Gamma=tr(AA^dag)")
print()
w1 = w2 = w3 = 0.0
for _ in range(3000):
    S = rand_sym(); A = rand_anti()
    v = {k: (0.8 + 0.4 * rng.normal()) + 1j * 0.5 * rng.normal()
         for k in ["Au", "Ad", "Su", "Sd"]}
    # 原始定义（CG: 120 -> (1,-1,1), 126 -> (1,1,-3)）
    Md = -v["Ad"] * A + v["Sd"] * S
    Me = v["Ad"] * A - 3 * v["Sd"] * S
    # 从 M_d, M_e 反解 S, A
    Sb = -(Md + Me) / 2
    Ab = (3 * Md + Me) / 2
    w1 = max(w1, np.max(np.abs(Sb - v["Sd"] * S)) / np.max(np.abs(v["Sd"] * S)))
    w2 = max(w2, np.max(np.abs(Ab - (-v["Ad"] * A))) / np.max(np.abs(v["Ad"] * A)))
    Dl = np.trace(Sb @ Sb.conj().T).real
    Gm = np.trace(Ab @ Ab.conj().T).real
    Sd2 = np.sum(sv(Md) ** 2)          # = tr H_d = sum m_d^2
    Se2 = np.sum(sv(Me) ** 2)          # = tr H_e = sum m_e^2
    w3 = max(w3, max(abs((Se2 - Sd2) - 8 * Dl) / Se2,
                     abs((9 * Sd2 - Se2) - 8 * Gm) / Se2))
print("max 相对误差 |S-(M_d+M_e)/2 - v_Sd*S|      = %.3e" % w1)
print("max 相对误差 |(3M_d+M_e)/2 - (-v_Ad*A)|    = %.3e" % w2)
print("max 相对误差 (两条迹恒等式)                 = %.3e" % w3)
print("=> 两条恒等式在原始 vev 参数化下精确成立：")
print("     sum m_e^2 - sum m_d^2 = 8 tr(S S^dag) = 8 Delta > 0")
print("     9 sum m_d^2 - sum m_e^2 = 8 tr(A A^dag) = 8 Gamma >= 0")
print()

print("=" * 82)
print("STEP B  由此得到的纯可观测量双重不等式，在模型上大规模检验")
print("=" * 82)
print("    sum m_d^2 < sum m_e^2 <= 9 sum m_d^2")
print()


def so10_120_126(scale=1.0):
    S = rand_sym() * scale; A = rand_anti() * scale
    return S + A, -3 * S - A


def so10_10_126():
    """对照：两个对称矩阵"""
    S1 = rand_sym(); S2 = rand_sym()
    v = [(0.9 + 0.3 * rng.normal()) + 1j * 0.5 * rng.normal() for _ in range(4)]
    return v[1] * S1 + v[3] * S2, v[1] * S1 - 3 * v[3] * S2


def so10_10_120():
    """对照：M_e = M_d^T"""
    S = rand_sym(); A = rand_anti()
    v = [(0.9 + 0.3 * rng.normal()) + 1j * 0.5 * rng.normal() for _ in range(4)]
    return v[1] * S - v[3] * A, v[1] * S + v[3] * A


def su5_5_45():
    """对照：SU(5)，down/lepton 用不同 flavor 矩阵"""
    Y1 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Y2 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    v = [(0.9 + 0.3 * rng.normal()) + 1j * 0.5 * rng.normal() for _ in range(4)]
    return v[1] * Y1 + v[3] * Y2, v[1] * Y1.T - 3 * v[3] * Y2.T


print("%-24s %10s %10s %10s %s" % ("model", "违反下界", "违反上界", "样本", "判定"))
for name, gen in [("SO(10) 120+126", so10_120_126),
                  ("SO(10) 10+126", so10_10_126),
                  ("SO(10) 10+120", so10_10_120),
                  ("SU(5) 5+45", su5_5_45)]:
    N = 6000; v_lo = v_hi = 0
    margins_lo = []; margins_hi = []
    for _ in range(N):
        Md, Me = gen()
        sd2 = np.sum(sv(Md) ** 2); se2 = np.sum(sv(Me) ** 2)
        if not (se2 > sd2):
            v_lo += 1
        if not (se2 <= 9 * sd2 * (1 + 1e-12)):
            v_hi += 1
        margins_lo.append(se2 / sd2)
        margins_hi.append(9 * sd2 / se2)
    print("%-24s %10d %10d %10d %s"
          % (name, v_lo, v_hi, N,
             "恒成立" if v_lo == 0 and v_hi == 0 else "被违反 -> 该不等式是 120+126 特有"))
    if name == "SO(10) 120+126":
        ml = np.array(margins_lo)
        print("      sum m_e^2 / sum m_d^2 的分布: min=%.4f  中位=%.4f  max=%.4f"
              % (ml.min(), np.median(ml), ml.max()))
print()

print("=" * 82)
print("STEP C  共线定理给出的第二条可观测不等式")
print("=" * 82)
print("  存在单位向量 x（A 的零向量）使 M_e x = -3 M_d x, M_u x = beta M_d x")
print("  => 3 m_{d,min} <= m_{e,max}  且  m_{e,min} <= 3 m_{d,max}")
print()
print("%-24s %12s %12s %s" % ("model", "违反 3md<=Me", "违反 me<=3Md", "判定"))
for name, gen in [("SO(10) 120+126", so10_120_126),
                  ("SO(10) 10+126", so10_10_126),
                  ("SU(5) 5+45", su5_5_45)]:
    N = 6000; v1 = v2 = 0
    for _ in range(N):
        Md, Me = gen()
        sd = sv(Md); se = sv(Me)
        if not (3 * sd[-1] <= se[0] * (1 + 1e-9)):
            v1 += 1
        if not (se[-1] <= 3 * sd[0] * (1 + 1e-9)):
            v2 += 1
    print("%-24s %12d %12d %s" % (name, v1, v2,
                                  "恒成立" if v1 == 0 and v2 == 0 else "被违反"))
print()

print("=" * 82)
print("STEP D  用真实 GUT 尺度谱检验这两条预言")
print("=" * 82)
print("数据来源: S. Antusch, M. Spinrath, Phys. Rev. D 78 (2008) 075020 [arXiv:0804.0717]")
print("  Table 3 = case 0（无 SUSY 阈值，即 SM 式跑动），Table 5 = 三种阈值修正情形。")
print("  表中给的是 GUT 尺度的 Yukawa；对 d 型夸克与带电轻子 m = y*v*cos(beta)，")
print("  故 sum m^2 之比等于 sum y^2 之比。")
print()
SCEN = {
    "case 0, tan beta = 30": dict(e=(0.62e-4, 1.31e-2, 0.23), d=(1.57e-4, 0.30e-2, 0.18)),
    "case 0, tan beta = 40": dict(e=(0.88e-4, 1.85e-2, 0.34), d=(2.22e-4, 0.43e-2, 0.26)),
    "case 0, tan beta = 50": dict(e=(1.21e-4, 2.55e-2, 0.51), d=(3.06e-4, 0.59e-2, 0.39)),
    "MSSM g+ (tb=40, mid)":  dict(e=(0.92e-4, 1.945e-2, 0.365), d=(1.88e-4, 0.36e-2, 0.23)),
    "MSSM g- (tb=40, mid)":  dict(e=(0.945e-4, 1.995e-2, 0.375), d=(3.10e-4, 0.595e-2, 0.23)),
    "MSSM a  (tb=40, mid)":  dict(e=(1.18e-4, 2.50e-2, 0.48), d=(4.20e-4, 0.81e-2, 0.29)),
}
print("%-26s %10s %10s %8s %s" % ("scenario", "sum y_e^2", "sum y_d^2", "ratio", "verdict"))
ratios = []
for name, s_ in SCEN.items():
    se2 = sum(x ** 2 for x in s_["e"]); sd2 = sum(x ** 2 for x in s_["d"])
    r = se2 / sd2; ratios.append(r)
    inside = 1.0 < r <= 9.0
    print("%-26s %10.5f %10.5f %8.3f %s"
          % (name, se2, sd2, r, "满足" if inside else "** 违反 **"))
print()
print("  所有情形都在 (1,9] 内，且都更靠近下界而非中点。")
print("  下界 1 才是约束性的：它等价于 GUT 尺度上 y_tau/y_b > 1，")
print("  而精确的 b-tau 统一恰好把这个比值压到 1。")
assert all(1.0 < r <= 9.0 for r in ratios), "bound violated by a tabulated spectrum"
print("  断言通过。")
print()
