import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== 1. 生成44向量晶格 ====================
basis = np.eye(3)
dem = np.array([1, 1, 1]) / np.sqrt(3)
seed = np.vstack([basis, [dem, -dem]])

T_mat = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])

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
print(f"生成向量数: {len(vectors)}")  # 应为44

# ==================== 2. 分类向量（突出物理意义） ====================
# 民主型: 接近[±0.577, ±0.577, ±0.577]
democratic = []
# Hybrid不对称: 如[-2,1,1]类 (raw整数不对称)
hybrid = []
# Root-like: 如[1,-1,0]类
root_like = []
others = []

for v in vectors:
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-5:
        continue  # 跳过零向量
    v_norm = v / norm_v if norm_v > 1e-10 else v
    
    # 民主判断
    if np.allclose(abs(v_norm), 0.57735, atol=0.01):
        democratic.append(v)
    # Hybrid: raw整数向量，不对称（如一个分量|2|倍）
    elif norm_v > 1 and np.all(np.abs(v - np.round(v)) < 1e-8):
        if len(set(np.abs(v))) > 1:  # 不全等
            hybrid.append(v)
    # Root-like: 归一化后像(1,-1,0)/sqrt(2)
    elif np.sum(np.abs(v_norm) > 0.1) == 2 and np.any(np.abs(v_norm) > 0.7):
        root_like.append(v)
    else:
        others.append(v)

print(f"民主型: {len(democratic)}, Hybrid: {len(hybrid)}, Root-like: {len(root_like)}, 其他: {len(others)}")

# ==================== 3. 3D可视化 ====================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 画其他点（灰色小）
if others:
    others_arr = np.array(others)
    ax.scatter(others_arr[:,0], others_arr[:,1], others_arr[:,2], c='gray', s=20, alpha=0.5, label='Other')

# 画Root-like（绿色）
if root_like:
    rl_arr = np.array(root_like)
    ax.scatter(rl_arr[:,0], rl_arr[:,1], rl_arr[:,2], c='green', s=50, label='Root-like (e.g., [1,-1,0])')

# 画Hybrid（红色大）
if hybrid:
    hy_arr = np.array(hybrid)
    ax.scatter(hy_arr[:,0], hy_arr[:,1], hy_arr[:,2], c='red', s=80, label='Hybrid (e.g., [-2,1,1])')

# 画民主（蓝色最大）
if democratic:
    dem_arr = np.array(democratic)
    ax.scatter(dem_arr[:,0], dem_arr[:,1], dem_arr[:,2], c='blue', s=100, label='Democratic ([1,1,1])')

# 连接最近邻（形成晶格边，阈值调小看清晰结构）
all_vec = np.vstack([democratic, hybrid, root_like, others]) if others else np.vstack([democratic, hybrid, root_like])
for i in range(len(all_vec)):
    for j in range(i+1, len(all_vec)):
        dist = np.linalg.norm(all_vec[i] - all_vec[j])
        if dist < 2.5:  # 调阈值控制连线密度（2.5看清晰晶体感）
            ax.plot(*zip(all_vec[i], all_vec[j]), color='silver', linewidth=1, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Z₃ Vacuum Lattice: 44 Vectors Emergent Crystal (Interactive 3D)')
ax.legend()
ax.view_init(elev=20, azim=45)  # 初始视角，可鼠标旋转

plt.show()