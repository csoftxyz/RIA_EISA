import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============== 1. 生成44向量晶格 (背景参考) ===============
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
    
    print(f"背景晶格生成向量数: {len(ground_state)}")
    return ground_state

background_vectors = generate_lattice_vectors()
background_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-8 else v for v in background_vectors]

# =============== 2. 物理费米子向量数据集 ===============
fermions = {
    "Top (Anchor)": np.array([0,0,1]),
    "Bottom": np.array([1,2,7]),
    "Charm": np.array([0,9,9]),
    "Strange": np.array([0,27,33]),
    "Muon": np.array([0,27,27]),
    "Down": np.array([1,46,193]),
    "Electron": np.array([3,138,579])
}

# Triality permutation τ(v) = (v3, v1, v2)
def triality_perm(v):
    return np.array([v[2], v[0], v[1]])

# 计算指标
results = []
for name, vec in fermions.items():
    l2 = np.dot(vec, vec)
    mod9 = "YES" if (l2 % 9 == 0) else "NO" if name != "Top (Anchor)" else "Exempt"
    
    tau_vec = triality_perm(vec)
    cross = np.cross(vec, tau_vec)
    delta = np.dot(cross, cross) / (l2 ** 2) if l2 > 0 else 0
    
    results.append({
        "Particle": name,
        "Vector": str(list(vec)),
        "L²": int(l2),
        "Mod 9": mod9,
        "Δ (Stability)": f"{delta:.6f}"
    })

# 打印表格
print(f"{'Particle':<15} {'Vector':<25} {'L²':<10} {'Mod 9':<8} {'Δ (Stability)':<15}")
print("-" * 75)
for res in results:
    print(f"{res['Particle']:<15} {res['Vector']:<25} {res['L²']:<10} {res['Mod 9']:<8} {res['Δ (Stability)']:<15}")
print("-" * 75)

# =============== 3. 3D可视化：高亮物理向量 ===============
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# 背景晶格 (淡灰)
bg_arr = np.array(background_normed)
ax.scatter(bg_arr[:,0], bg_arr[:,1], bg_arr[:,2], c='lightgray', s=30, alpha=0.4, label='44-Vector Background Lattice')

# 单位球面
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴
ax.quiver(0,0,0,1.2,0,0, length=1.2, color='black', alpha=0.7)
ax.quiver(0,0,0,0,1.2,0, length=1.2, color='black', alpha=0.7)
ax.quiver(0,0,0,0,0,1.2, length=1.2, color='black', alpha=0.7)

# 物理费米子向量 (归一化，大点彩色)
colors = ['gold', 'red', 'orange', 'magenta', 'cyan', 'purple', 'lime']
for i, (name, vec) in enumerate(fermions.items()):
    normed = vec / np.linalg.norm(vec)
    color = colors[i]
    ax.scatter(normed[0], normed[1], normed[2], c=color, s=300, marker='*', edgecolors='black', linewidth=2)
    ax.plot([0, normed[0]], [0, normed[1]], [0, normed[2]], c=color, linewidth=4)
    ax.text(normed[0]*1.1, normed[1]*1.1, normed[2]*1.1, 
            f"{name}\nL²={int(np.dot(vec,vec))}\nΔ={results[i]['Δ (Stability)']}", 
            color=color, fontsize=10, fontweight='bold')

ax.set_axis_off()
ax.view_init(elev=20, azim=45)
ax.set_title('Z₃ Vacuum Lattice: Physical Fermion Vectors\n'
             '(Modulo-9 Resonance + Triality Stability Δ Highlighted)', fontsize=16)

plt.tight_layout()
plt.show()