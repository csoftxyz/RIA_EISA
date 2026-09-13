#!/usr/bin/env python3
"""
z44_derive_2over9.py  (方向2(b): 2/9 的格点来源)
搜寻: 2/9 rad (=12.7324 deg) 是否等于某个格点角度/比值/精确组合.
"""
import numpy as np, math
from itertools import combinations
np.set_printoptions(suppress=True, precision=6)
T=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
def gen44():
    seed=np.vstack([np.eye(3),[np.array([1,1,1])/np.sqrt(3),-np.array([1,1,1])/np.sqrt(3)]])
    uniq=set(tuple(np.round(v,8)) for v in seed); cur=seed.tolist()
    for _ in range(15):
        new=[]
        for v in cur:
            v1=T@v;v2=T@v1; new+=[v1,v2,v1-v,v2-v]
            cr=np.cross(v,v1)
            if np.linalg.norm(cr)>1e-6: new.extend([cr,cr/np.linalg.norm(cr)])
        for nv in new:
            if np.linalg.norm(nv)>1e-6: uniq.add(tuple(np.round(nv,8)))
        allv=[np.array(u) for u in uniq]; cur=[v.tolist() for v in allv[:100]]
    vecs=[np.array(t) for t in uniq if np.linalg.norm(np.array(t))>1e-6]
    vecs.sort(key=lambda x:(round(np.linalg.norm(x),4),np.sum(np.abs(x)))); return vecs[:44]
V=gen44()
target=2/9; tgt_deg=math.degrees(target)
print(f"目标: 2/9 rad = {target:.6f} rad = {tgt_deg:.4f} deg")

print("\n[1] 格点矢量间角度 (找 {:.4f} deg):".format(tgt_deg))
found=[]
for i in range(len(V)):
    for j in range(i+1,len(V)):
        c=V[i]@V[j]/(np.linalg.norm(V[i])*np.linalg.norm(V[j]))
        a=math.degrees(math.acos(max(-1,min(1,c))))
        if abs(a-tgt_deg)<1.0: found.append((round(a,4),i,j))
print("   命中(<1deg):",found[:10] if found else "无")

print("\n[2] 特殊方向的角度:")
ref=[np.array([1,1,1])/np.sqrt(3),np.array([1,1,1]),np.array([1,0,0]),np.array([1,1,0]),np.array([1,-1,0]),np.array([1,1,-2])]
for r in ref:
    for i in range(len(V)):
        c=r@V[i]/(np.linalg.norm(r)*np.linalg.norm(V[i]))
        a=math.degrees(math.acos(max(-1,min(1,c))))
        if abs(a-tgt_deg)<2.0: print(f"   |ref {np.round(r,3)}| vs v#{i} {np.round(V[i],3)}: {a:.4f} deg")

print("\n[3] 精确分数/根式组合是否 = 2/9:")
cands={
 "根壳 L^2 / 3^2 = 2/9":2/9,
 "(2/3)(1/3)":(2/3)*(1/3),
 "L^2_root/(N/ ... )":"-",
 "1/3 - 1/9":1/3-1/9,
 "(sqrt2)^2/9":(np.sqrt(2))**2/9,
 "2*L^2_root/ (2*9)":2*2/18,
}
for k,v in cands.items():
    print(f"   {k:<24} = {v}")

print("\n[4] tan(2/9)=? 是否为格点比值:")
print(f"   tan(2/9) = {math.tan(2/9):.6f}")
for a in [1,2,3,4,1/3,2/3,np.sqrt(2),np.sqrt(3),1/np.sqrt(2),1/np.sqrt(3),1/(3*np.sqrt(3))]:
    if abs(math.tan(2/9)-a)<0.03: print(f"     ~ {a:.4f}")

print("\n[5] arctan(格点比值) 是否 = 2/9 rad:")
import itertools
grid_vals=[1,2,3,4,6,9,13,27,44,1/np.sqrt(2),1/np.sqrt(3),np.sqrt(2),np.sqrt(3),1/3,2/3,1/9,2/9,3/9]
for a in grid_vals:
    ang=math.atan(a)
    if abs(ang-target)<0.01: print(f"   arctan({a:.4f})={ang:.6f} (target {target:.6f})")

print("\n[6] 3-adic: 2/9 = 2 * 3^-2 ; 相关量:")
for n in [1,2,3]:
    print(f"   2/3^{n} = {2/3**n:.6f}   (2/9 是 n=2)")

print("\n[7] 关键重写: 2/9 = L^2_root * (1/3)^2  (root壳模方 × |Z3|^-2)")
print(f"   L^2_root = 2 (root壳);  |Z3|=3;  2 * (1/3)^2 = {2*(1/3)**2:.6f}")
print("\n结论见对话.")
