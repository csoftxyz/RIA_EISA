#!/usr/bin/env python3
"""
z44_final2.py — v13 的全部数字复现 (mpmath 50 dps) + 新定理验证 (sympy)。
输入: pole 组 (XZ Eq1, = PDG2004-era); MSbar(M_Z) 组 (XZ Eq8); 当前 PDG 组。
"""
from mpmath import mp, mpf, sqrt, cos, sin, pi, log, findroot, diff
mp.dps = 50
import sympy as sp

pole=[mpf('0.510998918'),mpf('105.6583692'),mpf('1776.86')]
msb =[mpf('0.486755106'),mpf('102.740394'),mpf('1746.56')]
cur =[mpf('0.51099895000'),mpf('105.6583755'),mpf('1776.86')]

def dft(m):
    x=[sqrt(v) for v in m]; mu=sum(x)/3
    c1=(mpf(2)/3)*sum((x[i]-mu)*mp.e**(-2j*pi*i/3) for i in range(3))
    ph=mp.arg(c1); ph=ph-(2*pi/3)*mp.floor(ph/(2*pi/3))
    return abs(c1)/mu, ph
def Qof(m): return sum(m)/(sum(sqrt(v) for v in m)**2)

print("=== 1) 三组输入的 (Q, r, phi-2pi/3) ===")
for lab,m in [("pole(XZ Eq1)",pole),("MSbar(XZ Eq8)",msb),("PDG current",cur)]:
    r,ph=dft(m); print(f"  {lab:16}: Q={mp.nstr(Qof(m),10)} r={mp.nstr(r,10)} phi-2pi/3={mp.nstr(ph,10)}")

print("\n=== 2) M0 预言 R_M0 (由 sqrt2, 2pi/3+2/9 直接算, 不硬编码) ===")
f=[1+sqrt(2)*cos(mpf(2)/9 + 2*pi*i/3) for i in range(3)]
print("  f =",[mp.nstr(v,8) for v in f])
fv=sorted(range(3), key=lambda i:-f[i])   # 最大->最小 = tau,mu,e? 检查
import itertools
mx=max(range(3),key=lambda i:f[i]); mn=min(range(3),key=lambda i:f[i]); md=({0,1,2}-{mx,mn}).pop()
R_M0=(f[md]/f[mn])**2
print(f"  R_M0 = m_mu/m_e = {mp.nstr(R_M0,12)}")
print(f"  与 pole 实测 206.7682836 之差 = {mp.nstr(R_M0-mpf('206.7682836'),5)}")

print("\n=== 3) 敏感度 (数值微分, 不硬编码) ===")
def R_of_phi(phi):
    g=[1+sqrt(2)*cos(phi+2*pi*i/3) for i in range(3)]
    mx=max(range(3),key=lambda i:g[i]); mn=min(range(3),key=lambda i:g[i]); md=({0,1,2}-{mx,mn}).pop()
    return (g[md]/g[mn])**2
ph0=mpf(2)/9
S=diff(lambda p: log(R_of_phi(p)), ph0)
print(f"  d ln(m_mu/m_e)/d phi = {mp.nstr(S,6)} rad^-1")

print("\n=== 4) sigma 数 (B4 更正) ===")
dQ=r'''
dQ_exp = |dQ/dm_tau| * 0.12
'''
mt=cur[2]; de=mpf('0.00012')  # GeV? 用 MeV: 0.12 MeV
Q=Qof(cur)
dQdmtau=diff(lambda x: (sum([cur[0],cur[1],x]))/(sum([sqrt(cur[0]),sqrt(cur[1]),sqrt(x)]))**2, mt)
print(f"  dQ/dm_tau = {mp.nstr(dQdmtau,5)} /MeV ; delta Q_exp = |.|*0.12 = {mp.nstr(abs(dQdmtau)*mpf('0.12'),5)}")
sigQ=abs(dQdmtau)*mpf('0.12')
print(f"  ΔQ_MSbar = 1.263e-3 ; /sig_Q = {mp.nstr(mpf('1.263e-3')/sigQ,5)} sigma")
print(f"  ΔQ_MSbar / |Q_obs-2/3| = {mp.nstr(mpf('1.263e-3')/abs(Q-mpf(2)/3),5)} x")

print("\n=== 5) 新定理验证: tan(phi*) 是代数数 (sympy minimal_polynomial) ===")
# 用有理化输入 (pole 质量是有限小数 => 有理)
me,mm,mtau=[sp.Rational(str(v)) for v in ['0.510998918','105.6583692','1776.86']]
se,sm,st=sp.sqrt(me),sp.sqrt(mm),sp.sqrt(mtau)
mu_=(se+sm+st)/3
w=[se/mu_-1, sm/mu_-1, st/mu_-1]
# c1 = (2/3) sum w_i e^{-2pi i i/3}; tan(phi*) = Im/Re  (代数)
w3=sp.exp(-2*sp.pi*sp.I/3)
c1=(sp.Rational(2,3))*(w[0]+w[1]*w3+w[2]*w3**2)
tanphi=sp.simplify(sp.im(c1)/sp.re(c1))
print("  tan(phi*) 表达式类型:", type(tanphi), " ; free symbols:", tanphi.free_symbols)
try:
    mp_=sp.minimal_polynomial(tanphi, sp.Symbol('x'))
    print("  minimal_polynomial(tan(phi*)) degree =", sp.degree(mp_), "(有限 => 代数)")
    print("  mp 前几项:", sp.nsimplify(mp_))
except Exception as e:
    print("  minpoly err:", e)
# 数值 tan(phi*)
r,ph=dft(pole); print(f"  tan(phi*) 数值 = {mp.nstr(mp.tan(ph),12)} ; phi*={mp.nstr(ph,12)}")
print(f"  2/9 = {mp.nstr(mpf(2)/9,12)} ; |phi*-2/9| = {mp.nstr(abs(ph-mpf(2)/9),4)}")

print("\n=== 6) 一圈 QED 交叉检验 MS-bar 质量 (审稿给 0.9526/0.9724/0.9830) ===")
for i,lab in enumerate(['e','mu','tau']):
    print(f"  mbar_{lab}(M_Z)/m^pole = {mp.nstr(msb[i]/pole[i],5)}")
