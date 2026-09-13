#!/usr/bin/env python3
"""
z44_eps_scheme.py  (v3: 修正 四审指出的三处 bug)
- mpmath 高精度, 不用 1e-6 网格
- 直接解 (f_mu/f_e)^2 = 206.7682830 (r=sqrt2) 得 phi* 到 10+ 位
- 加**符号分析**: R_pole = R_MSbar(1+eps) => 方案移动的方向
"""
from mpmath import mp, mpf, cos, sqrt, pi, findroot, log
mp.dps = 60

me,mmu,mtau = mpf('0.510998950'), mpf('105.6583755'), mpf('1776.86')
R_pole = mmu/me
print("R_pole = m_mu/m_e =", mp.nstr(R_pole,12))
print("CODATA m_mu/m_e = 206.7682830")

# ---- 1) phi* 精确解 (r=sqrt2, 由 m_mu/m_e 定) ----
# f_i = 1+sqrt2 cos(phi+2pi i/3); 代: f0=tau(最大), f1=e(最小), f2=mu
def ratio_eq(phi):
    f1 = 1+sqrt(2)*cos(phi+2*pi/3)
    f2 = 1+sqrt(2)*cos(phi+4*pi/3)
    return (f2/f1)**2 - mpf('206.7682830')
phi_star = findroot(ratio_eq, mpf('0.2222'))
print("\n[1] phi* (M1, r=sqrt2, 由 m_mu/m_e 精解) =", mp.nstr(phi_star, 12))
two9 = mpf(2)/9
print("    2/9 =", mp.nstr(two9, 12), "  |phi*-2/9| =", mp.nstr(abs(phi_star-two9), 6), "rad")

# ---- 2) eps 与相位/各量模糊度 ----
A_ME = mpf(137.035999177)
def inv_alpha(mu):
    inv=A_ME
    for mf in [mpf('0.510999'),mpf('105.658'),mpf('1776.86')]:
        if mu>mf: inv -= (2/(3*pi))*log(mu/mf)
    return inv
a_e=1/inv_alpha(me); a_mu=1/inv_alpha(mmu); a_mZ=1/inv_alpha(mpf('91187.6'))
print("\n[2] Delta_alpha_lep/alpha =", mp.nstr((a_mu-a_e)/a_e,6))
S=mpf('56.15')
eps = (a_mu-a_e)/pi
print("    eps = c(alpha(m_mu)-alpha(m_e))/pi =", mp.nstr(eps,6), " (c=1)")
print("    delta_phi_th = eps/S =", mp.nstr(eps/S,6), "rad")
print("    delta_Q_th   = 0.0145 eps =", mp.nstr(mpf('0.0145')*eps,6))
print("    |phi*-2/9| / delta_phi_th =", mp.nstr(abs(phi_star-two9)/(eps/S),4))

# ---- 3) 符号分析 (四审: 方案移动的方向) ----
R_MSbar = R_pole*(1-eps)
R_M0 = mpf('206.7703')
print("\n[3] 符号分析 (一圈): R_pole = R_MSbar(1+eps), eps>0 => R_MSbar < R_pole")
print("    R_MSbar =", mp.nstr(R_MSbar,10))
print("    R_pole  =", mp.nstr(R_pole,10))
print("    R_M0    =", mp.nstr(R_M0,10))
print(f"    M0 偏差: pole {mp.nstr(R_M0-R_pole,4)} ; MSbar {mp.nstr(R_M0-R_MSbar,4)}  => MSbar 更大!")
sig_pole = abs(R_M0-R_pole)/(mpf('4.4e-6')); sig_ms = abs(R_M0-R_MSbar)/(mpf('4.4e-6'))
print(f"    显著性: pole {mp.nstr(sig_pole,5)} sigma ; MSbar {mp.nstr(sig_ms,5)} sigma")
need = -(R_M0-R_pole)/R_pole
print(f"    要抹平需 eps = {mp.nstr(need,4)} (即 alpha(mu)<alpha(m_e)) -> 不可能 (alpha 在 Thomson 取极小)")
print("\n结论: 一圈下 M0 在 pole 与 MSbar 两个标准方案中**都被排除** (455σ / 1370σ);")
print("      方案移动方向相反, 'case (b)' 为假. 两圈味依赖跑动 (~1e-4, 符号未定) 才能救 -> 开放问题.")
