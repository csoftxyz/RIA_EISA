import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ====================== 数据定义 ======================
# 1. 核心晶格：44个向量（这里仍使用简化示例集以保持可读性）
#    实际完整44向量可通过单独生成脚本得到，这里选取代表性向量（基向量、民主方向、典型根向量等）
core_vectors = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],                     # 单位基向量
    [1, 1, 1], [-1, -1, -1],                              # 民主方向（整数形式）
    [1, -1, 0], [0, 1, -1], [1, 0, -1],                  # 典型根向量
    [-1, 1, 0], [0, -1, 1], [-1, 0, 1],
    [2, -1, -1], [-1, 2, -1], [-1, -1, 2],               # 常见混合向量示例
], dtype=float)

# 为美观，对长度大于1的向量进行归一化（仅用于绘图箭头）
core_normalized = np.array([v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in core_vectors])

# 2. 更新后的粒子谱数据（基于最新mod-9约束结果，包含Strange）
particles = {
    "Top (anchor)":     {"vector": [0, 0, 1],      "L2": 1,       "mass_pred": 173000, "mass_exp": 173000},
    "Bottom":           {"vector": [1, 2, 7],      "L2": 54,      "mass_pred": 3207,   "mass_exp": 4180},
    "Charm":            {"vector": [0, 9, 9],      "L2": 162,     "mass_pred": 1068,   "mass_exp": 1275},
    "Strange (new)":    {"vector": [0, 27, 33],    "L2": 1818,    "mass_pred": 95.2,   "mass_exp": 95},
    "Muon":             {"vector": [0, 27, 27],    "L2": 1458,    "mass_pred": 118.7,  "mass_exp": 105.7},
    "Down":             {"vector": [1, 46, 193],   "L2": 39366,   "mass_pred": 4.40,   "mass_exp": 4.7},
    "Electron":         {"vector": [3, 138, 579],  "L2": 354294,  "mass_pred": 0.488,  "mass_exp": 0.511},
}

# ====================== 可视化 ======================
fig = plt.figure(figsize=(18, 9))

# 子图1：3D 核心晶格散点 + 箭头
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Core Lattice: Representative Vectors\n(Finite 44-Vector Closed Set → sin²θ_W = 0.25)", fontsize=13)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# 绘制归一化箭头（更美观）
for vec in core_normalized:
    a = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                mutation_scale=12, lw=1.2, arrowstyle="-|>", color="steelblue", alpha=0.8)
    ax1.add_artist(a)

ax1.scatter(core_normalized[:,0], core_normalized[:,1], core_normalized[:,2], c='darkred', s=50, depthshade=True)
ax1.set_xlabel('X component')
ax1.set_ylabel('Y component')
ax1.set_zlabel('Z component')
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_zlim(-1.2, 1.2)
ax1.text2D(0.02, 0.95, "Z₃ Triality-Saturated Core Lattice\n(44 vectors total, representative shown)", 
           transform=ax1.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# 子图2：费米子质量谱（对数条形图 + 向量与L²标注）
ax2 = fig.add_subplot(122)
ax2.set_title("Charged Fermion Mass Hierarchy\nm ∝ 1/L² (Extended ℤ³ Lattice, mod-9 resonance)", fontsize=13)

# 按质量从轻到重排序
order = ["Electron", "Down", "Strange (new)", "Muon", "Charm", "Bottom", "Top (anchor)"]
names = order
pred_masses = [particles[p]["mass_pred"] for p in names]
exp_masses = [particles[p]["mass_exp"] for p in names]
vectors_str = [str(particles[p]["vector"]) for p in names]
l2_values = [particles[p]["L2"] for p in names]

y_pos = np.arange(len(names))
height = 0.35  # 条形高度

pred_y = y_pos - height / 2
exp_y = y_pos + height / 2

bars1 = ax2.barh(pred_y, pred_masses, height=height, color='cornflowerblue', label='Geometric Prediction')
bars2 = ax2.barh(exp_y, exp_masses, height=height, color='lightcoral', label='Experimental')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{n}\nVec = {vectors_str[i]}\nL² = {l2_values[i]:,}" for i, n in enumerate(names)])
ax2.set_xlabel('Mass (MeV)', fontsize=12)
ax2.set_xscale('log')
ax2.set_xlim(0.3, 300000)
ax2.legend(fontsize=11)
ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

# 在条形末端标注数值（错开y位置避免重叠）
for i in range(len(names)):
    pred = pred_masses[i]
    exp = exp_masses[i]
    pred_text = f"{pred:.0f}" if pred >= 100 else f"{pred:.2f}"
    exp_text = f"{exp:.0f}" if exp >= 100 else f"{exp:.2f}"
    
    # 预测值标签（略微向下偏移）
    ax2.text(pred * 1.05 if pred > 0 else 0.5, pred_y[i], pred_text, 
             va='center', ha='left', fontsize=10, color='blue')
    
    # 实验值标签（略微向上偏移）
    ax2.text(exp * 1.05 if exp > 0 else 0.5, exp_y[i], exp_text, 
             va='center', ha='left', fontsize=10, color='darkred')

plt.suptitle("Z₃-Graded Discrete Vacuum Geometry\nCore Lattice + Fermion Mass Spectrum (Updated with Strange Quark & mod-9 Constraint)", 
             fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()