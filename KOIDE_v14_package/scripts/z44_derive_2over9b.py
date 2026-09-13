#!/usr/bin/env python3
"""
z44_derive_2over9b.py (方向1: 2/9 的动力学/本征来源)
试: 2/9 rad 是否 = (a) 格点算子的本征相位; (b) 质量矢量在民主正交平面内与格点方向的夹角;
    (c) 某个自然 3x3 算子的本征矢角度; (d) 44-晶格 Cayley Laplacian 谱里.
"""
import numpy as np, math
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
V=gen44(); tgt=2/9
print(f"目标 2/9 rad = {tgt:.6f} = {math.degrees(tgt):.4f} deg")

# 民主正交平面 基
n=np.array([1,1,1])/np.sqrt(3)
u1=np.array([2,-1,-1])/np.sqrt(6); u2=np.array([0,1,-1])/np.sqrt(2)
def planangle(v):
    a=v@u1; b=v@u2
    return math.atan2(b,a)

print("\n[a] 质量矢量 sqrt(m) 的民主正交方向 (由观测质量):")
me,mmu,mtau=0.51099895,105.6583755,1776.86
sq=np.sqrt([me,mmu,mtau]); sqbar=sq.mean(); d=sq-sqbar
c1=(2/3)*np.sum(d*np.exp(-2j*np.pi*np.arange(3)/3))
theta=data_ang=np.angle(c1)%(2*np.pi)
print(f"   theta = {theta:.6f} rad ; theta-2pi/3 = {theta-2*np.pi/3:.6f}")
v_mass=np.array([math.cos(theta),math.cos(theta+2*np.pi/3),math.cos(theta+4*np.pi/3)])
fa=planangle(v_mass)
print(f"   在民主正交平面内的方向角 = {fa:.6f} rad = {math.degrees(fa):.3f} deg")

print("\n[b] 各格点矢量投影到民主正交平面的方向角 (找与质量矢量差 = 2/9):")
diffs=[]
for i,v in enumerate(V):
    vp=v-(v@n)*n
    if np.linalg.norm(vp)<1e-6: continue
    a=planangle(vp)
    dd=(a-fa+math.pi)%(2*math.pi)-math.pi
    diffs.append((abs(dd),math.degrees(dd),i,round(np.linalg.norm(v),4)))
diffs.sort()
for ad,deg,i,L in diffs[:6]:
    print(f"    v#{i:>2} (|v|={L}): 夹角={deg:+8.3f} deg   |.|-12.732|={abs(abs(deg)-12.732):.3f}")
print(f"    最小 |夹角| = {diffs[0][0]:.5f} rad ({diffs[0][1]:.3f} deg)")

print("\n[c] 自然 3x3 算子的本征矢角度:")
J=np.ones((3,3))/3; F=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)
C=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)
for nm,M in [("circulant(0,1,0)",C),("C^2",C@C),("J+C",J+C)]:
    w_,ev=np.linalg.eig(M); 
    for k in range(3):
        vv=np.real(ev[:,k]); vv=vv/np.linalg.norm(vv)
        a=math.degrees(planangle(vv))%360
        print(f"    {nm:<16} eig#{k}: 平面角={a:8.3f} deg")

print("\n[d] 44-晶格 Cayley Laplacian (w=1/d^2) 谱里找 2/9:")
N=len(V); idx={tuple(np.round(v,6)):i for i,v in enumerate(V)}
A=np.zeros((N,N))
for i,v in enumerate(V):
    v1=T@v; cr=np.cross(v,v1)
    for cv in [v1,cr]:
        if np.linalg.norm(cv)<1e-6: continue
        k=tuple(np.round(cv,6))
        if k in idx and idx[k]!=i:
            j=idx[k]; d2=np.sum((v-V[j])**2)
            if d2>1e-12: A[i,j]=A[j,i]=1.0/d2
L=np.diag(A.sum(1))-A
ev=np.linalg.eigvalsh(L)
print("   Laplacian 最小非零本征值:",np.round(ev[:6],6))
print("   是否有比例 = 2/9 ?")
for a in range(len(ev)):
    for b in range(len(ev)):
        if abs(ev[b])>1e-9 and abs(ev[a]/ev[b]-tgt)<1e-4: print(f"     ev[{a}]/ev[{b}] = {ev[a]/ev[b]:.6f}")

print("\n[e] 简单函数里是否有格点比: asin/acos/atan(2/9) 等")
print("   sin(2/9)=",round(math.sin(tgt),6)," cos(2/9)=",round(math.cos(tgt),6))
for name,val in [("1/(2sqrt2)",1/(2*np.sqrt(2))),("1/3-1/9",1/3-1/9),("(2/3)^2/2",(2/3)**2/2),("sqrt2/6.363",np.sqrt(2)/6.363)]:
    print(f"   {name} = {val:.6f}")
print("\n结论见对话.")
