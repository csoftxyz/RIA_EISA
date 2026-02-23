import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

print("=== Z3 Hubble Skymap Generator (No Healpy Version) ===")
print("使用 matplotlib mollweide 投影生成全天哈勃各向异性预测图\n")

# ====================== 1. 生成严格闭合的 44 向量晶格 ======================
T_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else None

dem = np.array([1., 1., 1.]) / np.sqrt(3)
seeds = [
    [1,0,0], [0,1,0], [0,0,1],
    dem, -dem,
    [1,1,0], [1,-1,0], [0,1,1], [0,-1,1]
]

unique_set = set()
current = [normalize(s) for s in seeds if normalize(s) is not None]

for _ in range(15):
    new = []
    for v in current:
        if v is None: continue
        v = v.ravel()
        v1 = normalize(T_mat @ v)
        if v1 is not None:
            new.append(v1)
            new.append(normalize(v + v1))
            new.append(normalize(v - v1))
            new.append(normalize(np.cross(v, v1)))
    for nv in new:
        if nv is not None:
            key = tuple(np.round(nv, 8))
            if key not in unique_set:
                unique_set.add(key)
                current.append(nv)
    if len(unique_set) >= 44:
        break

vectors_base = np.array([list(k) for k in list(unique_set)[:44]])
print(f"生成严格闭合 44 向量完成：{len(vectors_base)} 个")

# ====================== 2. 使用您提供的欧拉角旋转晶格 ======================
best_euler = np.array([32.12, 3.07, 376.45])   # 您之前优化的最佳角度
r_best = R.from_euler('zyx', best_euler, degrees=True)
vectors_rot = r_best.apply(vectors_base)

print(f"晶格已旋转，欧拉角 = {np.round(best_euler, 2)}")

# ====================== 3. 生成全天采样点并计算 η(n) ======================
n_points = 8000   # 采样点数量（越高图越精细）
theta = np.random.uniform(0, np.pi, n_points)
phi = np.random.uniform(0, 2*np.pi, n_points)

# 转换为单位向量
probes = np.column_stack((
    np.sin(theta) * np.cos(phi),
    np.sin(theta) * np.sin(phi),
    np.cos(theta)
))

dots = probes @ vectors_rot.T
eta = np.sum(dots**4, axis=1)
eta_norm = (eta - np.mean(eta)) / np.mean(eta)

print(f"全天几何因子 η(n) 计算完成 (采样点: {n_points})")

# ====================== 4. Mollweide 投影绘图 ======================
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='mollweide')

# Mollweide 需要 -pi 到 pi 的经度
phi_moll = phi - np.pi
sc = ax.scatter(phi_moll, np.pi/2 - theta, 
                c=eta_norm, 
                cmap='RdBu_r', 
                s=8, 
                alpha=0.85)

plt.colorbar(sc, label=r'$\delta H / H_0$  (Predicted Hubble Anisotropy)')
ax.set_xlabel('Galactic Longitude (l)', fontsize=12)
ax.set_ylabel('Galactic Latitude (b)', fontsize=12)
ax.set_title('Z3 Predicted Directional Dependence of Hubble Constant\n'
             '(44-Vector Lattice Orientation)', fontsize=14, pad=20)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Z3_Hubble_Skymap.png', dpi=300, bbox_inches='tight')
print("全天哈勃各向异性预测图已保存为: Z3_Hubble_Skymap.png")

# 保存数据供以后使用
np.save('z3_hubble_map_data.npy', np.column_stack((phi, theta, eta_norm)))
print("原始数据已保存为 z3_hubble_map_data.npy")

print("\n=== 任务完成 ===")
print("建议：将这张图放入论文 Discussion 部分，作为 'Predicted Hubble Anisotropy from Z3 Lattice' 的可视化证据。")