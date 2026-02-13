import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== 1. 生成44向量晶格 ====================
basis = np.eye(3)  # Basis axes for quarks (root-like)
dem = np.array([1, 1, 1]) / np.sqrt(3)  # Democratic axis for neutrinos
seed = np.vstack([basis, [dem, -dem]])  # Seed vectors

T_mat = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])  # Triality rotation matrix

def apply_triality(v):
    return T_mat @ v

unique = set()
for v in seed:
    unique.add(tuple(np.round(v, 12)))

current = seed.tolist()
levels = 15
max_per_level = 200

for level in range(levels):
    new = []
    for v in current:
        v1 = apply_triality(v)
        v2 = apply_triality(v1)
        new += [v1, v2]
        new.append(v1 - v)
        new.append(v2 - v)
        cross = np.cross(v, v1)
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-10:
            new.append(cross / norm_cross)
    
    for nv in new:
        norm = np.linalg.norm(nv)
        if norm > 1e-10:
            unique.add(tuple(np.round(nv / norm, 10)))  # normalized
        unique.add(tuple(np.round(nv, 10)))  # raw
    
    current = new[:max_per_level]

vectors = np.array([np.array(t) for t in unique])
print(f"Generated vectors: {len(vectors)}")  # Should be around 44

# ==================== 2. 分类向量（突出物理意义） ====================
democratic = []  # Neutrino-related: flatter hierarchies, blue
root_like = []   # Quark-related: strong hierarchies, green
hybrid = []      # Mixed axes, red
l1_vectors = []  # L=1: basic units, purple spheres (low mass level)
l6_vectors = []  # L=6: extended, orange stars (higher mass level)
others = []

for v in vectors:
    norm_sq = np.round(np.linalg.norm(v)**2, 5)  # L^2 for levels
    if np.isclose(norm_sq, 1.0, atol=1e-5):
        l1_vectors.append(v)
    elif np.isclose(norm_sq, 6.0, atol=1e-5):  # L^2=6 examples like [2,1,1]
        l6_vectors.append(v)
    
    # Classify axes
    if np.allclose(np.abs(v), np.abs(dem), atol=1e-5):  # Democratic for neutrinos
        democratic.append(v)
    elif np.allclose(np.sort(np.abs(v)), np.sort([1,1,0]/np.sqrt(2)), atol=1e-5):  # Root-like for quarks
        root_like.append(v)
    elif np.allclose(np.sort(np.abs(v)), np.sort([2,1,1]/np.sqrt(6)), atol=1e-5):  # Hybrid
        hybrid.append(v)
    else:
        others.append(v)

print(f"Democratic (Neutrino axes): {len(democratic)}")
print(f"Root-like (Quark axes): {len(root_like)}")
print(f"Hybrid: {len(hybrid)}")
print(f"L=1 (Low mass level): {len(l1_vectors)}")
print(f"L=6 (Higher mass level): {len(l6_vectors)}")
print(f"Others: {len(others)}")

# ==================== 3. 漂亮3D可视化 ====================
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 画其他点（灰色小点，背景）
if others:
    others_arr = np.array(others)
    ax.scatter(others_arr[:,0], others_arr[:,1], others_arr[:,2], c='lightgray', s=15, alpha=0.4, label='Other Vectors')

# 画Root-like主轴（绿色，夸克强层级，稍大点）
if root_like:
    rl_arr = np.array(root_like)
    ax.scatter(rl_arr[:,0], rl_arr[:,1], rl_arr[:,2], c='green', s=60, marker='o', label='Root-like Axes (Quark Hierarchies)')

# 画Democratic主轴（蓝色，中微子平坦层级，大点）
if democratic:
    dem_arr = np.array(democratic)
    ax.scatter(dem_arr[:,0], dem_arr[:,1], dem_arr[:,2], c='blue', s=80, marker='o', label='Democratic Axes (Neutrino Hierarchies)')

# 画Hybrid（红色，混合轴，中等点）
if hybrid:
    hy_arr = np.array(hybrid)
    ax.scatter(hy_arr[:,0], hy_arr[:,1], hy_arr[:,2], c='red', s=50, marker='^', label='Hybrid Axes (Axis Differences)')

# 突出L=1（紫色球，低质量层级）
if l1_vectors:
    l1_arr = np.array(l1_vectors)
    ax.scatter(l1_arr[:,0], l1_arr[:,1], l1_arr[:,2], c='purple', s=120, marker='o', edgecolor='black', label='L=1 Vectors (Low Mass Level)')

# 突出L=6（橙色星，更高质量层级）
if l6_vectors:
    l6_arr = np.array(l6_vectors)
    ax.scatter(l6_arr[:,0], l6_arr[:,1], l6_arr[:,2], c='orange', s=120, marker='*', edgecolor='black', label='L=6 Vectors (Higher Mass Level)')

# 连接最近邻（银色线，形成晶格结构，阈值调小以清晰）
all_vec = vectors  # All for connections
for i in range(len(all_vec)):
    for j in range(i+1, len(all_vec)):
        dist = np.linalg.norm(all_vec[i] - all_vec[j])
        if dist < 1.5:  # Smaller threshold for cleaner crystal view
            ax.plot([all_vec[i][0], all_vec[j][0]],
                    [all_vec[i][1], all_vec[j][1]],
                    [all_vec[i][2], all_vec[j][2]],
                    color='silver', linewidth=0.8, alpha=0.5)

# 美化轴和标签
ax.set_xlabel('X Component', fontsize=12, labelpad=10)
ax.set_ylabel('Y Component', fontsize=12, labelpad=10)
ax.set_zlabel('Z Component', fontsize=12, labelpad=10)
ax.set_title('Z₃ 44-Vector Crystal Lattice: Neutrino vs. Quark Mass Hierarchies\n(L=1 Low Level Purple, L=6 Higher Level Orange)', fontsize=14, pad=20)

# 图例（右上角，美观布局）
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10, frameon=True)

# 视角优化（可旋转查看差异）
ax.view_init(elev=25, azim=135)  # Good angle to see axis differences

# 紧凑布局
plt.tight_layout()

# 保存高清图（论文用）
plt.savefig('z3_crystal_44_schematic.png', dpi=300, bbox_inches='tight')

plt.show()