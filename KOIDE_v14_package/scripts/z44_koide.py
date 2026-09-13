#!/usr/bin/env python3
"""
z44_koide.py (方向2: 不动点/民主链 <-> 质量层级)
关键线索: 带电轻子 Koide 关系 Q=(sum m)/(sum sqrt(m))^2 = 2/3 (观测到极高精度).
已知极坐标形式: sqrt(m_i) = mu*(1 + r*cos(theta + 2pi i/3)), 且 Q=2/3 <=> r=sqrt2.
检验: r=sqrt2 是否 = 44 晶格八面体(root)壳的向量模 |v|=sqrt(L^2=2) ? 以及 theta 是否格点原生.
"""
import numpy as np
np.set_printoptions(suppress=True, precision=6)

# 观测带电轻子质量 (MeV)
me,mmu,mtau=0.51099895,105.6583755,1776.86
m=np.array([me,mmu,mtau]); sq=np.sqrt(m)
Q=(m.sum())/(sq.sum()**2)
print("观测带电轻子:")
print(f"  m = {m}")
print(f"  Koide Q = (sum m)/(sum sqrt m)^2 = {Q:.8f}   (2/3 = {2/3:.8f})  偏差 {abs(Q-2/3)/ (2/3)*100:.4f}%")

# 极坐标: sqrt(m_i)=mu*(1+r cos(theta+2pi i/3))
print("\n[1] 极坐标形式 sqrt(m_i)=mu(1+r cos(theta+2pi i/3)):")
# 由数据解 (mu, r, theta)
x=sq
mu=x.mean()
# 三个 x_i - mu 应 = mu*r*cos(theta+2pi i/3)
d=x-mu
# 用 DFT: 分量 c1 = (2/3) sum d_i e^{-2pi i i/3} -> 幅值 = mu*r
c1=(2/3)*np.sum(d*np.exp(-2j*np.pi*np.arange(3)/3))
r=abs(c1)/mu; theta=np.angle(c1)
print(f"  mu={mu:.5f}  r={r:.6f}  (sqrt2={np.sqrt(2):.6f})  theta={np.degrees(theta):.4f} deg")
print(f"  重算 Q (用 r): Q=(1+r^2/2)/3 = {(1+r*r/2)/3:.8f}")

print("\n[2] 代数: Q=2/3 <=> r=sqrt2")
for rr in [1.0,np.sqrt(2),1.5]:
    print(f"   r={rr:.4f} -> Q=(1+r^2/2)/3={(1+rr*rr/2)/3:.6f}")

print("\n[3] 格点原生量 r=sqrt2:")
print("   44 晶格八面体(root)壳: L^2=2 -> |v|=sqrt(2) ✓  (=r)")
print("   民主方向 [1,1,1]/sqrt3: |v|=1 ; [1,1,1]: |v|=sqrt3")
print("   => Koide 振幅 r=sqrt2 = root 壳向量模 (L^2=2), 格点原生!")

print("\n[4] theta 是否格点原生? 与格点角谱比对:")
th=np.degrees(theta)%360
grid=[0,30,35.264,45,54.736,60,90]
print(f"   theta={th:.4f} deg; 格点角谱 {grid}")
d2=min(abs(th-g) for g in grid)
print(f"   最近格点角距离 = {d2:.3f} deg")
print(f"   theta/2 = {th/2:.4f} deg ; (90-theta)={90-th:.4f} deg")

print("\n[5] 同一检验用于夸克 (上/下型, 各自三代):")
for nm,ms in [("up (u,c,t) MeV",[2.16,1270,172760]),("down (d,s,b) MeV",[4.67,93.4,4180])]:
    mm=np.array(ms); ss=np.sqrt(mm); Qq=mm.sum()/ss.sum()**2
    print(f"   {nm}: Q={Qq:.5f} (偏离 2/3 = {abs(Qq-2/3)/(2/3)*100:.1f}%)")

print("\n结论见对话.")
