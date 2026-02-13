import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== 1. 生成并锁定44向量晶格 ===============
def generate_lattice_vectors():
    basis = np.eye(3)
    dem = np.array([1, 1, 1]) / np.sqrt(3)
    seed = np.vstack([basis, dem.reshape(1,3), -dem.reshape(1,3)])
    
    T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    
    def apply(v):
        return T @ v
    
    unique_core = set()
    for v in seed:
        unique_core.add(tuple(np.round(v, 8)))
    
    current = seed.tolist()
    
    for level in range(15):
        new = []
        for v in current:
            v = np.array(v)
            v1 = apply(v)
            v2 = apply(v1)
            new += [v1, v2, v1-v, v2-v]
            
            cross = np.cross(v, v1)
            if np.linalg.norm(cross) > 1e-6:
                new.append(cross)
                new.append(cross / np.linalg.norm(cross))
        
        for nv in new:
            nv = np.array(nv)
            if np.linalg.norm(nv) > 1e-6:
                unique_core.add(tuple(np.round(nv, 8)))
        
        all_vecs = [np.array(u) for u in unique_core]
        all_vecs.sort(key=lambda x: (np.round(np.linalg.norm(x), 4), np.sum(np.abs(x))))
        
        if len(all_vecs) >= 44:
            ground_state = all_vecs[:44]
            break
        current = all_vecs[:100]
    
    print(f"生成向量总数 (锁定): {len(ground_state)}")
    return ground_state

vectors = generate_lattice_vectors()
vectors_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-8 else v for v in vectors]

# =============== 2. 组合因子计算 ===============
N = len(vectors)  # 44
combinatorial_factor = N**4
print(f"4-point combinatorial factor N^4 = {N}^4 = {combinatorial_factor:.2e}")

# Notional values for illustration
planck_scale = 0          # log10(Λ / M_Pl^4) = 0
qft_estimate = 0          # naive QFT ~ M_Pl^4
notional_suppression = -128
observed_cc = -122
compensation_needed = 6   # ~10^6 to bridge -128 → -122

# =============== 3. 可视化：左3D晶格 + 右尺度图 ===============
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('white')

# 左图：44向量晶格
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('white')

u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax1.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

ax1.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6)
ax1.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6)
ax1.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6)

for vec in vectors_normed:
    ax1.scatter(vec[0], vec[1], vec[2], c='cyan', s=40, edgecolors='black', linewidth=0.3)

dem = np.array([1,1,1])/np.sqrt(3)
ax1.scatter(dem[0], dem[1], dem[2], c='gold', s=200, marker='*', edgecolors='black')

ax1.set_axis_off()
ax1.view_init(elev=20, azim=45)
ax1.set_title(f"Z₃ Lattice (N = {N} vectors)\nCombinatorial Factor N⁴ ≈ {combinatorial_factor:.2e}", fontsize=14)

# 右图：宇宙常数尺度层次（对数图）
ax2 = fig.add_subplot(122)
ax2.set_facecolor('white')

scales = {
    'Planck / QFT estimate': 0,
    'Notional suppression': notional_suppression,
    'Lattice compensation (+log₁₀ N⁴ ≈ +6.57)': notional_suppression + np.log10(combinatorial_factor),
    'Observed Λ': observed_cc
}

logs = list(scales.values())
names = list(scales.keys())

y_pos = np.arange(len(names))
colors = ['red', 'orange', 'green', 'blue']
bars = ax2.barh(y_pos, logs, color=colors, edgecolor='black')

# 标注
for i, (name, logval) in enumerate(scales.items()):
    ax2.text(logval + 2 if logval >= 0 else logval - 8, y_pos[i], f'{10**logval:.0e} M_Pl⁴', 
             va='center', fontsize=11, fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(names)
ax2.set_xlabel('log₁₀(Λ / M_Pl⁴)')
ax2.set_title('Cosmological Constant Hierarchy\nLattice Combinatorial "Compensation"', fontsize=14)
ax2.grid(True, axis='x', alpha=0.3)
ax2.axvline(observed_cc, color='blue', linestyle='--', linewidth=2, label='Observed ~10^{-122}')
ax2.legend()

# 箭头展示补偿
ax2.annotate('', xy=(notional_suppression + np.log10(combinatorial_factor), 2.5),
             xytext=(notional_suppression, 2.5),
             arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax2.text(notional_suppression + 2, 2.7, f'+ log₁₀(N⁴) ≈ +6.57\n→ ~10^{{-122}}', color='green', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()