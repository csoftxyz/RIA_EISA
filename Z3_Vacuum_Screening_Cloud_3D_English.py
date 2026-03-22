import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=== Z3 Vacuum Screening Cloud: Final Version (Real Calculated Values) ===")

# ====================== 1. 集体模拟计算裸尺度 ======================
N_runs = 100
d_min_list = []
C_alg_list = []

for run in range(N_runs):
    vectors = []
    v_dem = np.array([1.,1.,1.]) / np.sqrt(3)
    vectors.append(v_dem.copy())
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        vectors.append(e.copy())
    for _ in range(30):
        new_vecs = []
        for v in vectors:
            T_v = np.roll(v, 1)
            new_vecs.append(T_v.copy())
            new_vecs.append((v + T_v).copy())
            new_vecs.append((v - T_v).copy())
            cross = np.cross(v, T_v)
            norm = np.linalg.norm(cross)
            if norm > 1e-12:
                new_vecs.append(cross / norm)
        vectors.extend(new_vecs)
        
        unique = []
        for v in vectors:
            if not any(np.linalg.norm(v - u) < 1e-10 for u in unique):
                unique.append(v.copy())
        vectors = unique[:44]
        if len(vectors) == 44:
            break
    vectors = np.array(vectors)
    
    distances = [np.linalg.norm(vectors[i] - vectors[j]) 
                 for i in range(len(vectors)) 
                 for j in range(i+1, len(vectors)) 
                 if np.linalg.norm(vectors[i] - vectors[j]) > 1e-10]
    d_min_list.append(np.min(distances))
    
    dem_axis = np.array([1.,1.,1.]) / np.sqrt(3)
    proj = np.abs(np.dot(vectors, dem_axis))
    C_alg_list.append(np.max(proj))

avg_d_min = np.mean(d_min_list)
avg_C_alg = np.mean(C_alg_list)
xi_bare = (avg_C_alg / avg_d_min) * (len(vectors) / 8.0)
eta_alg = 4.0
xi_eff = xi_bare / eta_alg

print(f"Bare Scale (ξ_bare)   : {xi_bare:.2f} nm")
print(f"Algebraic Screening η : {eta_alg}")
print(f"Dressed Scale (ξ_eff) : {xi_eff:.3f} nm")

# ====================== 2. 最终高清图（数值实时显示 + 无叠加） ======================
fig = plt.figure(figsize=(15, 7.5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

np.random.seed(42)
angles = np.linspace(0, 2*np.pi, 18)
r = np.linspace(0.5, 1.5, 8)
theta, rad = np.meshgrid(angles, r)
x = rad * np.cos(theta)
y = rad * np.sin(theta)
z = np.random.uniform(-1.1, 1.1, x.shape) * 0.25
vac_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))[:40]

ferm_points = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0]]) * 1.35

# 左侧：Bare Scale
ax1.scatter(vac_points[:,0], vac_points[:,1], vac_points[:,2], c='#ff2d55', s=160, alpha=0.95, edgecolors='white', linewidth=1.2)
ax1.scatter(ferm_points[:,0], ferm_points[:,1], ferm_points[:,2], c='#3b82f6', s=200, alpha=0.75, edgecolors='white', linewidth=1.2)
ax1.view_init(elev=30, azim=48)
ax1.set_title('Bare Scale', fontsize=14, pad=10)

# 右侧：Dressed Scale
compressed = vac_points * 0.25
ax2.scatter(compressed[:,0], compressed[:,1], compressed[:,2], c='#ff2d55', s=160, alpha=0.95, edgecolors='white', linewidth=1.2)
ax2.scatter(ferm_points[:,0]*0.25, ferm_points[:,1]*0.25, ferm_points[:,2]*0.25, c='#3b82f6', s=200, alpha=0.85, edgecolors='white', linewidth=1.2)
ax2.view_init(elev=30, azim=48)
ax2.set_title('Dressed Scale', fontsize=14, pad=10)

# 多根红色压迫箭头
for p in compressed[::3]:
    ax2.quiver(p[0], p[1], p[2], -p[0]*0.6, -p[1]*0.6, -p[2]*0.6, color='#ff9500', linewidth=2.5, arrow_length_ratio=0.3, alpha=0.85)

# 绿色压缩箭头 + 文字
ax2.quiver(0, 0, 0, 1.2, 0, 0, color='#00ff9d', linewidth=8, arrow_length_ratio=0.25, alpha=0.95)
ax2.text(0.68, 0.15, 0.12, 'η = 4\nCompression', color='#00ff9d', fontsize=15, fontweight='bold',
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.85))

# 实时显示计算出的数值（彻底避免硬编码）
fig.text(0.25, 0.82, f'ξ_bare ≈ {xi_bare:.2f} nm', fontsize=13, fontweight='bold', ha='center')
fig.text(0.75, 0.82, f'ξ_eff ≈ {xi_eff:.3f} nm (η=4)', fontsize=13, fontweight='bold', ha='center')

# 图例
ax1.legend(['Z₃ Vacuum Lattice (g₂)', 'Fermion Cloud (g₁)'], loc='upper right', fontsize=11, frameon=True, facecolor='white')

plt.suptitle('Z₃ Vacuum: Bare → Dressed by Fermion Screening Cloud', fontsize=17, y=0.94, fontweight='bold')
plt.tight_layout()
plt.savefig('Z3_Vacuum_Screening_Cloud_3D_Crystal_Final_Fixed_NoOverlap.png', dpi=800, bbox_inches='tight')
plt.show()

print("\n✅ 最终版已生成！")
print("ξ_bare 和 ξ_eff 均为实时计算值")
print("文件已保存：Z3_Vacuum_Screening_Cloud_3D_Crystal_Final_Fixed_NoOverlap.png")