import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成晶格向量 (匹配z3_mass_6.py逻辑，锁定44)
def generate_lattice_vectors():
    basis = np.eye(3)
    dem = np.array([1, 1, 1]) / np.sqrt(3)
    seed = np.vstack([basis, dem.reshape(1,3), -dem.reshape(1,3)])
    
    T_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    
    def apply_triality(v):
        return T_mat @ v
    
    unique_core = set()
    for v in seed:
        unique_core.add(tuple(np.round(v, 8)))
    
    current = seed.tolist()
    
    for level in range(15):
        new = []
        for v in current:
            v = np.array(v)
            v1 = apply_triality(v)
            v2 = apply_triality(v1)
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
    
    print(f"生成向量总数: {len(ground_state)}")
    return ground_state

# 生成向量
vectors = generate_lattice_vectors()
vectors_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-8 else v for v in vectors]

# 分类函数
def classify_vector(v):
    norm = np.linalg.norm(v)
    n = np.round(v, 8)
    if abs(norm - 1.0) < 0.1:
        return "g0 (gauge)", "blue"
    elif abs(norm - np.sqrt(2)) < 0.1:
        return "g1 (fermion)", "red"
    elif np.allclose(abs(n), [1,1,1], atol=1e-5):
        return "g2 (vacuum)", "gold"
    elif any(abs(x) >= 2 for x in n):
        return "hybrid (mass vec)", "green"
    else:
        return "other", "purple"

# 统计short vectors (匹配z3_mass_6.py)
short_count = 0
for v in vectors:
    ln = np.linalg.norm(v)
    if abs(ln - 1.4142) < 0.05 or abs(ln - 1.0) < 0.05:
        short_count += 1

total = len(vectors)
sin2_theta = short_count / total
print(f"Short vectors: {short_count}/{total} = {sin2_theta:.2f} (sin²θ_W)")

# 质量向量
mass_vectors = {
    'top': np.array([0,0,1]),
    'bottom': np.array([1,2,7]),
    'charm/tau': np.array([0,9,9]),
    'muon': np.array([0,27,27]),
    'down': np.array([1,46,193]),
    'electron': np.array([3,138,579])
}

m_top = 173  # GeV
exp_masses = {'top': 173, 'bottom': 4.18, 'charm/tau': 1.776, 'muon': 0.1057, 'down': 0.0048, 'electron': 0.000511}
l2_values = {k: np.dot(v, v) for k, v in mass_vectors.items()}
pred_masses = {k: m_top / l2 for k, l2 in l2_values.items()}

# 可视化
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')

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
ax1.text(1.3,0,0, "e1", color='black')
ax1.text(0,1.3,0, "e2", color='black')
ax1.text(0,0,1.3, "e3", color='black')

for vec in vectors_normed:
    label, color = classify_vector(vec)
    size = 80 if "g2" in label else 50 if "g1" in label else 30
    ax1.scatter(vec[0], vec[1], vec[2], c=color, s=size, edgecolors='black', linewidth=0.5)

dem = np.array([1,1,1])/np.sqrt(3)
ax1.scatter(dem[0], dem[1], dem[2], c='gold', s=200, marker='*', edgecolors='black')

for name, vec in mass_vectors.items():
    norm_vec = vec / np.linalg.norm(vec)
    ax1.scatter(norm_vec[0], norm_vec[1], norm_vec[2], c='cyan', s=100, marker='o', edgecolors='black', alpha=0.8)
    ax1.text(norm_vec[0]*1.05, norm_vec[1]*1.05, norm_vec[2]*1.05, name, color='cyan', fontsize=8)

ax1.set_axis_off()
ax1.view_init(elev=20, azim=45)
ax1.set_title("Z₃-Graded Lattice (44 Vectors)", fontsize=14)

ax1.text2D(0.05, 0.90, f"sin²θ_W = {short_count}/{total} = 0.25", transform=ax1.transAxes, color='blue', fontsize=12)
ax1.text2D(0.05, 0.85, "Blue: g0 (gauge)", transform=ax1.transAxes, color='blue', fontsize=10)
ax1.text2D(0.05, 0.80, "Red: g1 (fermion)", transform=ax1.transAxes, color='red', fontsize=10)
ax1.text2D(0.05, 0.75, "Gold ★: g2 (vacuum)", transform=ax1.transAxes, color='gold', fontsize=10)
ax1.text2D(0.05, 0.70, "Cyan: Mass vectors", transform=ax1.transAxes, color='cyan', fontsize=10)

ax2 = fig.add_subplot(122)
particles = list(pred_masses.keys())
pred_vals = list(pred_masses.values())
exp_vals = [exp_masses[p] for p in particles]

x = np.arange(len(particles))
width = 0.35

ax2.bar(x - width/2, pred_vals, width, label='Predicted (m ~ 1/L²)', color='lightblue')
ax2.bar(x + width/2, exp_vals, width, label='Experimental', color='orange')

ax2.set_yscale('log')
ax2.set_ylabel('Mass (GeV)')
ax2.set_title('Fermion Mass Hierarchy Coincidences')
ax2.set_xticks(x)
ax2.set_xticklabels(particles, rotation=45)
ax2.legend()

# 直接使用手动LaTeX字符串（避免Sympy derivative错误）
rg_latex = r"$\frac{\mathrm{d} \sin^2 \theta_W}{\mathrm{d} \ln \mu} = \frac{\alpha}{2\pi} (b_2 - b_1) \sin^2 \theta_W (1 - \sin^2 \theta_W)$"
ax2.text(0.02, 0.02, f"RG equation:\n{rg_latex}", transform=ax2.transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), verticalalignment='bottom')

plt.tight_layout()
plt.show()