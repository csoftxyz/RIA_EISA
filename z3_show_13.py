import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== 1. 向量定义 ===============
# 基础向量 (flavor-like)
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# 对称双分向量 (大气混合-like)
bisector_23 = np.array([0, 1, 1]) / np.sqrt(2)

# 民主向量 (太阳混合-like)
dem = np.array([1, 1, 1]) / np.sqrt(3)

# 示例hybrid向量 (夸克-like小角度)
hybrid_example = np.array([-2, 1, 1]) / np.linalg.norm(np.array([-2, 1, 1]))

# =============== 2. 计算角度 ===============
# 大气混合: e2/e3 到 bisector_23 的角度 (45°)
cos_23 = np.dot(e2, bisector_23)
theta_23_deg = np.degrees(np.arccos(cos_23))
sin2_23 = np.sin(np.radians(theta_23_deg))**2

# 太阳混合: e1 到 democratic 的角度 (magic ≈54.74°)
cos_sol = np.dot(e1, dem)
theta_sol_deg = np.degrees(np.arccos(cos_sol))
cos2_sol = cos_sol**2

# 示例hybrid小角度
cos_hybrid = np.dot(e1, hybrid_example)
theta_hybrid_deg = np.degrees(np.arccos(abs(cos_hybrid)))  # 取绝对值避免符号
sin_hybrid = np.sin(np.radians(theta_hybrid_deg))

print(f"Atmospheric-like (to [0,1,1]/√2): {theta_23_deg:.2f}° → sin²θ = {sin2_23:.4f} (exact 0.5)")
print(f"Solar-like (to [1,1,1]/√3): {theta_sol_deg:.2f}° → cos²θ = {cos2_sol:.4f} (exact 1/3)")
print(f"Hybrid example [-2,1,1]: misalignment ≈ {theta_hybrid_deg:.2f}° (small, quark-like)")

# =============== 3. 3D可视化 ===============
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('white')

# 左图: Neutrino-like 大角度 (TBM)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('white')

# 单位球面网格
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax1.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴
ax1.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6, label='e1 (flavor basis)')
ax1.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6)
ax1.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6)
ax1.text(1.3,0,0, "e1", color='black', fontsize=12)
ax1.text(0,1.3,0, "e2", color='black', fontsize=12)
ax1.text(0,0,1.3, "e3", color='black', fontsize=12)

# Bisector [0,1,1]/√2 (大气混合)
ax1.quiver(0,0,0, *bisector_23, length=1.0, color='green', linewidth=4, label='Bisector [0,1,1]/√2')
ax1.text(bisector_23[0]*1.1, bisector_23[1]*1.1, bisector_23[2]*1.1, 
         f"45° → sin²θ₂₃ = 0.5\n(maximal)", color='green', fontsize=11, ha='center')

# Democratic [1,1,1]/√3 (太阳混合)
ax1.quiver(0,0,0, *dem, length=1.0, color='gold', linewidth=4, label='Democratic [1,1,1]/√3')
ax1.text(dem[0]*1.1, dem[1]*1.1, dem[2]*1.1, 
         f"54.74° → cos²θ₁₂ = 1/3\n(TBM solar)", color='gold', fontsize=11, ha='center')

ax1.set_axis_off()
ax1.view_init(elev=20, azim=45)
ax1.set_title("Neutrino-like Large Mixing (Tri-Bimaximal)\nExact Geometric Reproduction", fontsize=14)

# 右图: Quark-like 小角度 (hybrid示例)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_facecolor('white')

ax2.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴 (同左图)
ax2.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6)
ax2.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6)
ax2.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6)
ax2.text(1.3,0,0, "e1", color='black', fontsize=12)
ax2.text(0,1.3,0, "e2", color='black', fontsize=12)
ax2.text(0,0,1.3, "e3", color='black', fontsize=12)

# Democratic作为参考
ax2.quiver(0,0,0, *dem, length=1.0, color='gold', linewidth=3, alpha=0.7)

# Hybrid示例
ax2.quiver(0,0,0, *hybrid_example, length=1.0, color='magenta', linewidth=4, label='Hybrid [-2,1,1] (normalized)')
ax2.text(hybrid_example[0]*1.1, hybrid_example[1]*1.1, hybrid_example[2]*1.1, 
         f"Small misalignment\nquark-like hierarchy", color='magenta', fontsize=11, ha='center')

ax2.set_axis_off()
ax2.view_init(elev=20, azim=45)
ax2.set_title("Quark-like Small Mixing\nAnisotropic Hybrid Perturbation", fontsize=14)

ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()