import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# =============== 1. 参数和目标 ===============
TARGET_SIN2_THETA13 = 0.0224
TOLERANCE = 0.001  # 宽松容差用于展示候选
MAX_L2 = 100000    # 搜索上限 (L² ≤ 100000, 足够覆盖输出中的候选)

print(f"Searching for basis projection candidates with sin²θ₁₃ ≈ {TARGET_SIN2_THETA13}")
print(f"Target 1/sin² ≈ {1/TARGET_SIN2_THETA13:.2f} (looking for near-integer ~44–46)\n")

# =============== 2. 生成整数向量并搜索θ13候选 (basis轴投影) ===============
candidates = []
limit = int(np.sqrt(MAX_L2)) + 10

for x in range(1, limit):
    for y in range(0, x + 1):
        for z in range(0, y + 1):
            l2 = x*x + y*y + z*z
            if l2 > MAX_L2 or l2 <= x*x:  # 跳过纯basis (sin²=0)
                continue
                
            sin2 = 1.0 - (x*x) / l2
            denom = 1.0 / sin2
            error = abs(sin2 - TARGET_SIN2_THETA13)
            
            if error < 0.005:  # 宽松筛选展示用
                candidates.append({
                    'vec': (x, y, z),
                    'L²': l2,
                    'sin²': sin2,
                    '1/sin²': denom,
                    'error': error,
                    'int_score': abs(denom - round(denom))
                })

# 转为DataFrame并排序 (先整数分数，再误差)
import pandas as pd
df = pd.DataFrame(candidates)
df = df.sort_values(['int_score', 'error']).head(20)  # Top 20

print("Top Candidates (basis projection):")
print(df[['vec', 'L²', 'sin²', '1/sin²', 'int_score']].to_string(index=False))

# =============== 3. 3D可视化：高亮Top候选向量 ===============
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# 单位球面网格
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x_sphere = np.cos(u) * np.sin(v)
y_sphere = np.sin(u) * np.sin(v)
z_sphere = np.cos(v)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴
ax.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6, label='e1 (flavor basis)')
ax.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6)
ax.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6)
ax.text(1.3,0,0, "e1", color='black')
ax.text(0,1.3,0, "e2", color='black')
ax.text(0,0,1.3, "e3", color='black')

# 所有候选向量 (归一化，半透明)
for _, row in df.iterrows():
    vec = np.array(row['vec'])
    normed = vec / np.linalg.norm(vec)
    score = row['int_score']
    color = 'gold' if score < 0.01 else 'cyan' if score < 0.1 else 'lightblue'
    size = 200 if score < 0.01 else 100
    ax.scatter(normed[0], normed[1], normed[2], c=color, s=size, edgecolors='black', alpha=0.8)
    ax.plot([0, normed[0]], [0, normed[1]], [0, normed[2]], c=color, linewidth=2, alpha=0.7)
    
    # 标注最佳几个
    if score < 0.1:
        ax.text(normed[0]*1.1, normed[1]*1.1, normed[2]*1.1,
                f"{row['vec']}\n1/sin²≈{row['1/sin²']:.2f}", 
                color='darkred' if score < 0.01 else 'black', fontsize=9)

ax.set_axis_off()
ax.view_init(elev=20, azim=45)
ax.set_title(f"Z₃ Lattice: Candidates for sin²θ₁₃ ≈ {TARGET_SIN2_THETA13}\n"
             f"Basis Projection (1/sin² near integer, top {len(df)} shown)", fontsize=14)

# 图例
ax.text2D(0.02, 0.92, "Gold: Near-perfect integer (dev <0.01)", transform=ax.transAxes, color='gold', fontsize=12)
ax.text2D(0.02, 0.88, "Cyan: Good integer (dev <0.1)", transform=ax.transAxes, color='cyan', fontsize=12)
ax.text2D(0.02, 0.84, "Target 1/sin² ≈44.64 (experimental)", transform=ax.transAxes, color='red', fontsize=12)

plt.tight_layout()
plt.show()