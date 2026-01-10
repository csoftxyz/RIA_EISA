import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ====================== 数据定义 ======================
# 1. 核心晶格：44个向量（这里用简化示例，实际运行z3_mass_6.py可得到完整44个）
#    为演示，我们取部分典型向量（包括单位基、民主方向、根等）
core_vectors = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],                     # 基向量
    [0.57735, 0.57735, 0.57735], [-0.57735, -0.57735, -0.57735],  # 民主方向
    [0.7071, -0.7071, 0], [0, 0.7071, -0.7071],           # 典型根向量（约值）
    # ... 实际44个向量太长，这里省略，实际可用代码生成后复制
    # 补充一些示例向量让点更多
    [1, -1, 0], [0, 1, -1], [1, 0, -1],
    [-1, 1, 0], [0, -1, 1], [-1, 0, 1],
])

# 2. 粒子谱数据（来自文章实现）
particles = {
    "Top":     {"vector": [0, 0, 1],      "L2": 1.0,     "mass_pred": 172760, "mass_exp": 172760},
    "Bottom":  {"vector": [1, 2, 7],      "L2": 54.0,    "mass_pred": 3199,   "mass_exp": 4180},
    "Tau":     {"vector": [0, 9, 9],      "L2": 162.0,   "mass_pred": 1066,   "mass_exp": 1776},
    "Muon":    {"vector": [0, 27, 27],    "L2": 1458.0,  "mass_pred": 118.5,  "mass_exp": 105.7},
    "Down":    {"vector": [1, 46, 193],   "L2": 39366.0, "mass_pred": 4.39,   "mass_exp": 4.70},
    "Electron":{"vector": [3, 138, 579],  "L2": 354294.0,"mass_pred": 0.488,  "mass_exp": 0.511},
}

# ====================== 可视化 ======================
fig = plt.figure(figsize=(16, 8))

# 子图1：3D 核心晶格（44向量散点图）
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Core Lattice: 44 Vectors (Ground State)\nGenerates sin²θ_W = 0.25", fontsize=12)

# 画向量：从原点出发的箭头
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

for vec in core_vectors:
    a = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                mutation_scale=10, lw=1, arrowstyle="-|>", color="blue", alpha=0.7)
    ax1.add_artist(a)

ax1.scatter(core_vectors[:,0], core_vectors[:,1], core_vectors[:,2], c='red', s=30)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_xlim(-1.5,1.5); ax1.set_ylim(-1.5,1.5); ax1.set_zlim(-1.5,1.5)
ax1.text2D(0.05, 0.95, "Finite 44-Vector Lattice\n(Z₃ Triality Closed Orbit)", transform=ax1.transAxes, fontsize=10)

# 子图2：粒子质量谱（对数刻度条形图 + 向量标注）
ax2 = fig.add_subplot(122)
ax2.set_title("Fermion Mass Spectrum (Geometric Seesaw)\nm ∝ 1/L² from Extended Lattice", fontsize=12)

names = list(particles.keys())[::-1]  # 从轻到重
pred_masses = [particles[p]["mass_pred"] for p in names]
exp_masses = [particles[p]["mass_exp"] for p in names]
vectors_str = [str(particles[p]["vector"]) for p in names]
l2_values = [particles[p]["L2"] for p in names]

y_pos = np.arange(len(names))
ax2.barh(y_pos, pred_masses, color='skyblue', label='Predicted (Geometric)')
ax2.barh(y_pos, exp_masses, color='salmon', alpha=0.6, label='Experimental')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{n}\nL²={l2_values[i]:,.0f}\nVec={vectors_str[i]}" for i,n in enumerate(names)])
ax2.set_xlabel('Mass (MeV)')
ax2.set_xscale('log')
ax2.set_xlim(0.1, 200000)
ax2.legend()
ax2.grid(True, axis='x', linestyle='--')

plt.suptitle("Z₃ Discrete Vacuum Geometry: Core Lattice (44 Vectors) + Mass Spectrum", fontsize=14)
plt.tight_layout()
plt.show()