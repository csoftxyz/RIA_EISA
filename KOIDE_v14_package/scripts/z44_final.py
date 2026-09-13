#!/usr/bin/env python3
"""
z44_final.py  — 高精度 (mpmath, 50 dps) 复算 v12 的全部数字.
输入 (MeV): pole = XZ Eq(1); MSbar(M_Z) = XZ Eq(8).
"""
from mpmath import mp, mpf, sqrt, cos, pi, log, findroot, quad
mp.dps = 50

pole = [mpf('0.510998918'), mpf('105.6583692'), mpf('1776.86')]
msb  = [mpf('0.486755106'), mpf('102.740394'), mpf('1746.56')]

def Q_of(m): return sum(m)/(sum(sqrt(x) for x in m)**2)
def phi_of(m):
    x=[sqrt(v) for v in m]; mu=sum(x)/3
    w=mp.e**(2j*pi/3)
    c1=(mpf(2)/3)*sum((x[i]-mu)*mp.e**(-2j*pi*i/3) for i in range(3))
    r=abs(c1)/mu; ph=mp.arg(c1)
    ph = ph - (2*pi/3)*mp.floor(ph/(2*pi/3))
    return r, ph

print("=== v12 高精度数字 (mpmath, 50 dps) ===")
for lab,m in [("pole",pole),("MSbar(M_Z)",msb)]:
    Q=Q_of(m); r,ph=phi_of(m)
    print(f"{lab:11}: Q={mp.nstr(Q,10)}  r={mp.nstr(r,10)}  phi-2pi/3={mp.nstr(ph,10)}  |phi-2/9|={mp.nstr(abs(ph-mpf(2)/9),4)}")
Qp=Q_of(pole); Qm=Q_of(msb); _,php=phi_of(pole); _,phm=phi_of(msb)
print(f"\nQ 变化: {mp.nstr(Qm-Qp,4)}  ({mp.nstr((Qm/Qp-1)*100,4)}%)  = {mp.nstr((Qm-Qp)/mpf('6.1e-6'),4)} sigma (sig=6.1e-6)")
print(f"phi 变化: {mp.nstr(phm-php,4)} rad = {mp.nstr(abs(phm-php)/mpf('1.75e-7'),4)} x |phi*-2/9|")
print(f"(m_mu/m_e): pole {mp.nstr(pole[1]/pole[0],9)}  MSbar {mp.nstr(msb[1]/msb[0],9)}  比 {mp.nstr((msb[1]/msb[0])/(pole[1]/pole[0]),9)}")
print(f"(m_tau/m_e): pole {mp.nstr(pole[2]/pole[0],9)}  MSbar {mp.nstr(msb[2]/msb[0],9)}  比 {mp.nstr((msb[2]/msb[0])/(pole[2]/pole[0]),9)}")
# 闭式解
def closed(m): a=sqrt(m[0]); b=sqrt(m[1]); return (2*(a+b)+sqrt(3)*sqrt(m[0]+4*a*b+m[1]))**2
print(f"\nm_tau 闭式解(pole mass inputs): {mp.nstr(closed(pole),17)} MeV")
print(f"m_tau 闭式解(MSbar inputs):     {mp.nstr(closed(msb),17)} MeV  (实际 {msb[2]}, 偏 {mp.nstr(closed(msb)-msb[2],6)})")
# 敏感度
S=mpf('56.15'); print(f"\n敏感度 d ln(m_mu/m_e)/d phi = {S} rad^-1 (input)")
print(f"455 sigma 检验: |m_mu/m_e(pole)-R_M0|/sig ; R_M0=206.7703, sig=4.4e-6")
print(f"  = {mp.nstr(abs(mpf('206.7703')-(pole[1]/pole[0]))/mpf('4.4e-6'),5)} sigma")
