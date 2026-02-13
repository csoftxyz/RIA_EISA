import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== 1. 民主向量和CKM混合向量 ===============
v_dem = np.array([1.0, 1.0, 1.0])
norm_dem = np.linalg.norm(v_dem)
v_dem_norm = v_dem / norm_dem

# 来自论文的最佳向量（示例，实际运行搜索可得到更精确的）
ckm_vectors = {
    'V_us (Cabibbo ≈0.2245)': np.array([-19, -12, -12]),
    'V_cb (≈0.0412)': np.array([-24, -22, -22]),
    'V_ub (≈0.0038)': np.array([-99, -97, -97])  # 示例大向量，实际搜索可得更好
}

# 计算每个向量的sin θ
sines = {}
for name, u in ckm_vectors.items():
    u_norm = np.linalg.norm(u)
    cos_theta = abs(np.dot(u, v_dem)) / (u_norm * norm_dem)
    cos_theta = min(cos_theta, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)
    sines[name] = sin_theta

# 观测值（PDG 2024近似）
observed = {
    'V_us (Cabibbo ≈0.2245)': 0.2245,
    'V_cb (≈0.0412)': 0.0412,
    'V_ub (≈0.0038)': 0.0038
}

# =============== 2. 3D可视化 ===============
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('white')

# 单位球面网格
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax1.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴
ax1.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6)
ax1.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6)
ax1.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6)
ax1.text(1.3,0,0, "e1", color='black')
ax1.text(0,1.3,0, "e2", color='black')
ax1.text(0,0,1.3, "e3", color='black')

# 民主向量（金色大星）
ax1.scatter(v_dem_norm[0], v_dem_norm[1], v_dem_norm[2], c='gold', s=300, marker='*', edgecolors='black', linewidth=1.5, label='Democratic [1,1,1]/√3')
ax1.plot([0, v_dem_norm[0]], [0, v_dem_norm[1]], [0, v_dem_norm[2]], c='gold', linewidth=3)

# CKM向量（归一化后，彩色）
colors = ['cyan', 'magenta', 'lime']
for i, (name, u) in enumerate(ckm_vectors.items()):
    u_normed = u / np.linalg.norm(u)
    color = colors[i]
    ax1.scatter(u_normed[0], u_normed[1], u_normed[2], c=color, s=150, marker='o', edgecolors='black', linewidth=1)
    ax1.plot([0, u_normed[0]], [0, u_normed[1]], [0, u_normed[2]], c=color, linewidth=2)
    ax1.text(u_normed[0]*1.1, u_normed[1]*1.1, u_normed[2]*1.1, f"{name}\n sinθ ≈ {sines[name]:.4f}", color=color, fontsize=9)

ax1.set_axis_off()
ax1.view_init(elev=20, azim=45)
ax1.set_title("Z₃ Lattice: Misalignment Angles to Democratic Direction", fontsize=14)

# 图例和说明
ax1.text2D(0.02, 0.90, "Gold ★: Democratic reference", transform=ax1.transAxes, color='gold', fontsize=12)
ax1.text2D(0.02, 0.85, "Colored lines: CKM-like hybrid vectors", transform=ax1.transAxes, color='black', fontsize=12)
ax1.text2D(0.02, 0.80, "sinθ ≈ |V_ij| (numerical coincidence)", transform=ax1.transAxes, color='black', fontsize=12)

# =============== 3. 柱状图：预测 vs 观测 ===============
ax2 = fig.add_subplot(122)
names = list(sines.keys())
pred_vals = [sines[n] for n in names]
obs_vals = [observed[n] for n in names]

x = np.arange(len(names))
width = 0.35

ax2.bar(x - width/2, pred_vals, width, label='Predicted sinθ (lattice)', color='lightblue')
ax2.bar(x + width/2, obs_vals, width, label='Observed |V_ij| (PDG)', color='orange')

ax2.set_ylabel('|V_ij| or sinθ')
ax2.set_title('CKM Magnitude Coincidences')
ax2.set_xticks(x)
ax2.set_xticklabels([n.split(' ')[0] for n in names], rotation=45)
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()