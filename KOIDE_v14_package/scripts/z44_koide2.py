#!/usr/bin/env python3
"""
z44_koide2.py  (方向2 续: theta 格点化)
发现假设: Koide 相位 theta ≡ 2/9 rad (mod 2pi/3), 且锥角=45度(格点原生).
检验数据 + 重建质量比.
"""
import numpy as np
np.set_printoptions(suppress=True, precision=8)
me,mmu,mtau=0.51099895,105.6583755,1776.86
m=np.array([me,mmu,mtau]); sq=np.sqrt(m)
print(f"观测 sqrt(m) = {sq}")

# 由数据精确解 theta
mu=sq.mean(); d=sq-mu
c1=(2/3)*np.sum(d*np.exp(-2j*np.pi*np.arange(3)/3))
theta=np.angle(c1); r=abs(c1)/mu
print(f"\n[1] 精确解: mu={mu:.6f}  r={r:.8f} (sqrt2={np.sqrt(2):.8f})  theta={theta:.8f} rad = {np.degrees(theta):.6f} deg")

two3=2*np.pi/3
print(f"\n[2] theta mod (2pi/3):  theta-2pi/3 = {theta-two3:.8f} rad")
print(f"    2/9 = {2/9:.8f} rad   偏差 = {abs((theta-two3)-2/9):.2e} rad = {np.degrees(abs((theta-two3)-2/9)):.3e} deg")
print(f"    也试: (theta) mod (2pi/3) = {theta%two3:.8f} ; (theta+2pi/3)%(2pi/3)={(theta+two3)%two3:.8f}")

print(f"\n[3] 锥角: tan(alpha)=|w|/|(1,1,1)|, |w|=r*sqrt(3/2), |(1,1,1)|=sqrt3")
alpha=np.degrees(np.arctan(r*np.sqrt(3/2)/np.sqrt(3)))
print(f"    alpha = arctan(r/sqrt2) = {alpha:.6f} deg   (45 度 = 格点角谱成员)")

print(f"\n[4] 重建质量 (用 r=sqrt2, theta=2/9 (mod 2pi/3)):")
for th,lab in [(2/9,"theta=2/9"),(0,"theta=0"),(two3,"theta=2pi/3")]:
    v=1+np.sqrt(2)*np.cos(th+2*np.pi*np.arange(3)/3)
    mm=(mu**2)*v**2
    ratio=mm/mm[0]
    print(f"    {lab:<12}: sqrt(m)={np.round(mu*v,4)}  比值 m/m_e = {np.round(ratio,3)}")
mo=np.array([me,mmu,mtau]); mo=mo/mo[0]
v=1+np.sqrt(2)*np.cos(2/9+2*np.pi*np.arange(3)/3); mm=v**2; mm=mm/mm[0]
print(f"    观测比值 m/m_e = {np.round(mo,3)}")
print(f"    重建 vs 观测 相对偏差: {np.round((mm-mo)/mo*100,4)} %")

print(f"\n[5] 格点原生性核对:")
print(f"    振幅 r=sqrt2 = root壳(L^2=2)模长 ✓")
print(f"    相位 2/9 = 2/3^2 (3-adic) ; 2pi/3 = Z3 triality")
print(f"    r 与 2/9 是否有关? sqrt2 与 2/9 ...")
print(f"    锥角 45 deg in {{0,30,35.26,45,54.74,60,90}} ✓")
print("\n结论见对话.")
