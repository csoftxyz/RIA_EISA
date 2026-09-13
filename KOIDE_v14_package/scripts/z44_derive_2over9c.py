#!/usr/bin/env python3
"""
z44_derive_2over9c.py (方向1 续: 2/9 = 六角格点离散化残差?)
检验: 质量矢量相对晶格 30° 六角网格的偏离 (=2/9 rad) 是否是"离散化残差",
      即把质量矢量投到格点方向后能否得到更干净的格点值.
"""
import numpy as np, math
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
n=np.array([1,1,1])/np.sqrt(3); u1=np.array([2,-1,-1])/np.sqrt(6); u2=np.array([0,1,-1])/np.sqrt(2)
def pa(v):
    vp=v-(v@n)*n
    return None if np.linalg.norm(vp)<1e-6 else math.degrees(math.atan2(vp@u2,vp@u1))%360

me,mmu,mtau=0.51099895,105.6583755,1776.86
sq=np.sqrt([me,mmu,mtau]); d=sq-sq.mean()
c1=(2/3)*np.sum(d*np.exp(-2j*np.pi*np.arange(3)/3)); th=np.angle(c1)
vm=np.array([math.cos(th),math.cos(th+2*np.pi/3),math.cos(th+4*np.pi/3)])
fam=pa(vm)%360
print(f"质量矢量平面角 = {fam:.6f} deg ; theta={th:.6f} ; -theta%360={(-th)%360:.6f}")
print(f"最近 30 度格点: {round(fam/30)*30} ; 偏离 = {abs(fam-round(fam/30)*30):.6f} deg ; 2/9 rad = {math.degrees(2/9):.6f}")

print("\n[1] 把质量矢量投到最近格点方向(240 deg) -> 理想质量:")
th_ideal=2*np.pi/3     # 使 fam=240
v=1+np.sqrt(2)*np.cos(th_ideal+2*np.pi*np.arange(3)/3)
m_ideal=v**2; m_ideal=m_ideal/m_ideal.min()
print(f"   th=2pi/3: sqrt(m)={np.round(v,6)}  m/m_e={np.round(m_ideal,4)}  -> 退化(e=mu), 不符")

print("\n[2] 质量矢量平面角 vs 各格点方向, 偏移列表:")
grid=sorted(set(round(pa(v),6) for v in V if pa(v) is not None))
offs=sorted((abs(((g-fam+180)%360)-180),g) for g in grid)
for off,g in offs[:6]:
    print(f"   到 {g:7.3f} deg: 偏离 {off:.6f} deg")

print("\n[3] 2/9 是否 = 六角网格的某个自然量?")
for nm,val in [("30/ (2/9 rad)", math.radians(30)/(2/9)),("2/9 rad / (pi/6)", (2/9)/(math.pi/6)),
               ("tan(2/9)",math.tan(2/9)),("偏离/15",math.degrees(2/9)/15),
               ("(240-180)/... ","-")]:
    print(f"   {nm} = {val}")

print("\n[4] 6 个六角方向中, 质量矢量到最近的两条:")
print("   fam=%.4f ; 到210=%.4f ; 到240=%.4f ; 17.267?=%.4f"%(fam,abs(fam-210),abs(240-fam),abs(fam-210)))
print("   17.267 deg 是否干净? 17.267/30=%.4f ; 17.267 deg=%.5f rad"%(17.267/30,math.radians(17.267)))

print("\n[5] 另一可能: 质量矢量在 3D 里与某格点矢量的夹角:")
best=[]
for i,v in enumerate(V):
    c=(vm@v)/(np.linalg.norm(vm)*np.linalg.norm(v))
    a=math.degrees(math.acos(max(-1,min(1,c))))
    best.append((a,i,round(np.linalg.norm(v),4)))
best.sort()
for a,i,L in best[:5]: print(f"   v#{i:>2}(|v|={L}): {a:.4f} deg")

print("\n结论见对话.")
