#!/usr/bin/env python3
"""
z44_2loop_v2.py  (四审 ④-1 的数值估计)
目标: 估计两圈 pole<->MSbar 转换的**味依赖**位移 δ2 ln(m_mu/m_e), 与所需 -9.755e-6 比.
诚实分层: (A) RGE 跑动严格为零 (可证); (B) 味依赖在两圈**转换**(fermion-loop 真空极化);
          (C) 阈值解耦幂次压低. (B) 的量级可估, 符号需显式两圈计算.
"""
import numpy as np
from mpmath import mp, mpf, log, pi, quad, sqrt
mp.dps=40

alpha=mpf(1)/mpf('137.035999177'); a=alpha/pi   # alpha/pi = 0.002323
R=mpf('206.7682830'); L=log(R)
need=mpf('9.755e-6')  # |所需相对位移| (to erase M0 deviation in some scheme)
print("a=alpha/pi =",mp.nstr(a,6),"  ln(m_mu/m_e) =",mp.nstr(L,6))
print("所需相对位移 = -",mp.nstr(need,5),"\n")

print("(A) RGE 跑动: γ_m 无质量定义 => 味盲 => d ln(m_i/m_j)/dlnmu = 0  [严格, 任意圈]")
print("(B) 两圈味依赖只在转换系数: δ2 ln(m_mu/m_e) = a^2 * C2 * (fermion-loop 系数)")
print("    结构: δ2 = a^2 * [k2 * ln(m_mu/m_e) + const],  k2 由 2-loop self-energy 定\n")

# 对 k2 做扫描 (k2 是未知的两圈系数, O(1))
print("    k2 扫描:")
for k2 in [0.25,0.5,1.0,1.5,2.0,3.0]:
    d2=a**2*k2*L
    print(f"      k2={k2:<4}: |δ2| = {mp.nstr(abs(d2),3)}  = {mp.nstr(abs(d2)/need,3)} x 所需位移")
print("\n(C) 阈值解耦 (轻→重): 贡献 ~ a^2*(m/M)^2 ~ 1e-8..1e-10, 可忽略")

print("\n================ 判决 ================")
print("1) 跑动 = 0 (严格) -> 无连续 dial.")
print("2) 两圈转换的味依赖 |δ2| = a^2*|k2|*ln(R) ~ (0.5-3)*|k2| x 10^-5；")
print("   与所需 9.8e-6 同量级 => 若 k2>~0.8 且符号为负, 则**可能**实现 2/9.")
print("3) k2 的符号/值需显式两圈 QED self-energy 计算 (或查 Koide-RG 先行文献).")
print("=> 结论: 两圈**量级上可覆盖**所需位移, 但符号未知 => 该问题**开放**(既非 excluded 也非 rescued).")
print("   与四审一致; 且严格纠正: 味依赖来自'转换/匹配', 不是 RGE 跑动.")
