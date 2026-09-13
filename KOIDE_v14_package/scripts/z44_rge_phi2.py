#!/usr/bin/env python3
"""z44_rge_phi2.py — 用 XZ 的 pole 与 M_Z 质量，插值跑动，求 Q(mu), phi(mu)，找交叉点。"""
import numpy as np
from scipy.optimize import brentq
w=np.exp(2j*np.pi/3)
pole=np.array([0.510998918e-3,105.6583692e-3,1776.86e-3])
mZ  =np.array([0.486755106e-3,102.740394e-3,1746.56e-3])
lnMZ=np.log(91.1876)
D=np.log(pole/mZ)                      # Δ_i = ∫γ dlnμ
lnmi=np.log(pole)

def masses(mu):
    out=[]
    for i in range(3):
        if mu<=pole[i]: out.append(pole[i])
        else:
            frac=np.log(mu/pole[i])/np.log(91.1876/pole[i])
            out.append(pole[i]*np.exp(-D[i]*frac))
    return np.array(out)

def Qphi(mu):
    m=masses(mu); Q=m.sum()/(np.sqrt(m).sum()**2)
    x=np.sqrt(m); mu_=x.mean(); d=x-mu_
    c1=(2/3)*np.sum(d*np.exp(-2j*np.pi*np.arange(3)/3)); r=abs(c1)/mu_; phi=np.angle(c1)%(2*np.pi/3)
    return Q,r,phi

print("mu(GeV)   Q          r         phi-2pi/3     |phi-2/9|")
for mu in [1.777,1.80,1.9,2.2,3,10,91.1876]:
    Q,r,phi=Qphi(mu); print(f"{mu:7.3f}  {Q:.8f}  {r:.6f}  {phi:.8f}  {abs(phi-2/9):.3e}")

f=lambda mu: Qphi(mu)[2]-2/9
mu_cross=brentq(f,1.777,91.1876,xtol=1e-10)
Qc,rc,pc=Qphi(mu_cross)
print(f"\n*** phi(mu)=2/9 的标度: mu = {mu_cross:.5f} GeV  (= {mu_cross/1.77686:.4f} x m_tau) ***")
print(f"    该标度上: Q={Qc:.8f}  (|Q-2/3|={abs(Qc-2/3):.3e})  r={rc:.6f} (sqrt2={np.sqrt(2):.6f})")
print(f"    => 在 mu≈{mu_cross:.2f} GeV, Q=2/3 与 phi=2/9 同时成立 (Q 到 {abs(Qc-2/3):.1e})")
print(f"    running 到 M_Z 会使 |phi-2/9| 涨到 {abs(Qphi(91.19)[2]-2/9):.2e} (放大 {abs(Qphi(91.19)[2]-2/9)/abs(pole[0]*0+7.4e-6):.0f}x)")
