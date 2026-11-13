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
# GUT 尺度（~2e16 GeV）带电费米子质量，SM 与 MSSM 两套典型运行值
SCEN = {
    "SM 运行到 GUT (Antusch-Spinrath 量级)": dict(
        d=(0.0019, 0.038, 0.64), e=(0.00035, 0.0621, 1.01)),
    "MSSM 运行到 GUT (tan beta 大)": dict(
        d=(0.0015, 0.030, 1.00), e=(0.00030, 0.052, 1.40)),
    "低能 ~2 GeV（模型不适用，仅作对照）": dict(
        d=(0.0047, 0.095, 4.18), e=(0.000511, 0.1057, 1.777)),
}
print("%-42s %10s %10s %8s %8s" % ("场景", "sum m_d^2", "sum m_e^2",
                                    "下界", "上界"))
for name, s in SCEN.items():
    sd2 = sum(x ** 2 for x in s["d"]); se2 = sum(x ** 2 for x in s["e"])
    lo = "满足" if se2 > sd2 else "** 违反 **"
    hi = "满足" if se2 <= 9 * sd2 else "** 违反 **"
    print("%-42s %10.5f %10.5f %8s %8s" % (name, sd2, se2, lo, hi))
print()
print("  下界 sum m_e^2 > sum m_d^2 由第三代主导，实质要求 GUT 尺度上 m_tau > m_b。")
print("  上界 sum m_e^2 <= 9 sum m_d^2 实质要求 m_tau <= 3 m_b。")
print("  => 预言： 1 < (m_tau^2+m_mu^2+m_e^2)/(m_b^2+m_s^2+m_d^2) <= 9")
for name, s in SCEN.items():
    sd2 = sum(x ** 2 for x in s["d"]); se2 = sum(x ** 2 for x in s["e"])
    print("     %-42s 比值 = %.3f" % (name, se2 / sd2))
print()

print("=" * 82)
print("STEP E  检查论文 Letter §5 的一处表述：tr(H_u H_e) 是否真能由 13 个可观测量重建")
print("=" * 82)
print("论文原话：'all mixed invariants are reconstructed from the thirteen")
print("           observables to <1e-14 relative accuracy'")
print("但 tr(H_uH_e) = sum_ij m_u,i^2 m_e,j^2 |(V_uL^dag V_eL)_ij|^2 需要 V_eL，")
print("而 V_eL 不在 13 个可观测量内（无中微子数据）。检验：")
err_ud = []; err_ue = []
for _ in range(4000):
    Mu = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Md = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Me = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    def VL(M):
        _, U = np.linalg.eigh(M @ M.conj().T)
        return U[:, ::-1]

    V = VL(Mu).conj().T @ VL(Md)          # 物理 CKM
    su = sv(Mu); sd = sv(Md); se = sv(Me)
    Hu = Mu @ Mu.conj().T; Hd = Md @ Md.conj().T; He = Me @ Me.conj().T
    Iud = np.trace(Hu @ Hd).real
    Iue = np.trace(Hu @ He).real
    rec_ud = sum(su[i] ** 2 * sd[j] ** 2 * abs(V[i, j]) ** 2 for i in range(3) for j in range(3))
    rec_ue = sum(su[i] ** 2 * se[j] ** 2 * abs(V[i, j]) ** 2 for i in range(3) for j in range(3))
    err_ud.append(abs(rec_ud - Iud) / max(abs(Iud), 1e-30))
    err_ue.append(abs(rec_ue - Iue) / max(abs(Iue), 1e-30))
print("  tr(H_uH_d) 用 V_CKM 重建的相对误差 : 中位 %.3e  最大 %.3e   -> 可重建"
      % (np.median(err_ud), max(err_ud)))
print("  tr(H_uH_e) 用 V_CKM 重建的相对误差 : 中位 %.3e  最大 %.3e   -> ** 不可重建 **"
      % (np.median(err_ue), max(err_ue)))
