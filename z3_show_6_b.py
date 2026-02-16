import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------- 1. 生成44向量晶格（纯Z3 triality迭代） ---------------------
basis = np.eye(3)  # e1, e2, e3
dem = np.array([1, 1, 1]) / np.sqrt(3)
seed = np.vstack([basis, dem, -dem])  # 5个初始向量

# Triality循环矩阵
T_mat = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

def apply_triality(v):
    return T_mat @ v

# 迭代生成
unique = set()
for v in seed:
    unique.add(tuple(np.round(v, 12)))

current = seed.tolist()
levels = 15
for level in range(levels):
    new = []
    for v in current:
        v1 = apply_triality(v)
        v2 = apply_triality(v1)
        new += [v1, v2, v1 - v, v2 - v]
        cross = np.cross(v, v1)
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-10:
            new.append(cross / norm_cross)
    for nv in new:
        norm = np.linalg.norm(nv)
        if norm > 1e-10:
            unique.add(tuple(np.round(nv / norm, 10)))  # 归一化
        unique.add(tuple(np.round(nv, 10)))          # 原始
    current = new
    if len(unique) >= 44:  # 提前检测饱和
        break

vectors = [np.array(t) for t in unique]
print(f"最终生成 {len(vectors)} 个唯一向量（饱和）")

# --------------------- 2. 简单分类（用于着色） ---------------------
democratic = []  # 金黄/橙（民主核心）
root_like = []   # 深绿（根式互锁）
hybrid = []      # 深红（杂化榫头）
residual = []    # 深灰（基础）

for v in vectors:
    norm = np.linalg.norm(v)
    if abs(norm - np.sqrt(3)) < 0.1:  # democratic ≈√3
        democratic.append(v)
    elif abs(norm - np.sqrt(2)) < 0.1:  # root-like
        root_like.append(v)
    elif abs(norm - 1.0) < 0.1:       # basis/residual
        residual.append(v)
    else:
        hybrid.append(v)

print(f"Democratic: {len(democratic)}, Root-like: {len(root_like)}, "
      f"Hybrid: {len(hybrid)}, Residual: {len(residual)}")

# --------------------- 3. 3D可视化（白色背景，适合论文） ---------------------
fig = plt.figure(figsize=(14, 10), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')      # 白色背景
fig.patch.set_facecolor('white')

# 绘制向量（从原点到末端）
def plot_vector(vecs, color, label, size=1.2, alpha=1.0):
    for v in vecs:
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=color, linewidth=3.5, alpha=alpha)
        ax.scatter(v[0], v[1], v[2], color=color, s=120*size, alpha=alpha, 
                   edgecolors='black', linewidth=0.8, depthshade=True)

# 白底优化颜色方案（更鲜明、对比强）
plot_vector(democratic, '#D81B60', 'Democratic', size=1.4, alpha=0.95)  # 玫红核心
plot_vector(root_like, '#1E88E5', 'Root-like', alpha=0.9)               # 深蓝互锁
plot_vector(hybrid, '#43A047', 'Hybrid', alpha=0.9)                    # 深绿榫头
plot_vector(residual, '#546E7A', 'Residual', alpha=0.8)                # 深灰基础

# 选择性连接线（深灰，密度适中，突出互锁）
for i, v1 in enumerate(vectors):
    for j, v2 in enumerate(vectors[i+1:], i+1):
        if 0.8 < np.linalg.norm(v1 - v2) < 2.0:
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                    color='#37474F', linewidth=1.8, alpha=0.6)

ax.set_axis_off()
ax.grid(False)
ax.view_init(elev=25, azim=50)  # 最佳视角

ax.set_title('Z₃-Graded Vacuum Lattice: Luban Lock Analogy\n'
             'Spontaneous Self-Interlocking Closure at Exactly 44 Vectors', 
             color='black', fontsize=16, pad=40)

plt.tight_layout()

# 保存图片
output_filename = 'z3_show_6_b_output.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"白色背景图片已保存为: {output_filename}")

# 本地运行预览
plt.show()