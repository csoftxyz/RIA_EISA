import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== 1. Generate the 44-vector lattice ====================
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
        v = np.array(v)
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
print(f"Generated vectors: {len(vectors)} (should be ~44)")

# ==================== 2. Classify vectors ====================
democratic = []
root_like = []
hybrid = []
l1_vectors = []  # L² ≈ 1
l6_vectors = []  # L² ≈ 6
others = []

for v in vectors:
    norm_sq = np.round(np.linalg.norm(v)**2, 5)
    
    if np.isclose(norm_sq, 1.0, atol=0.1):
        l1_vectors.append(v)
    elif np.isclose(norm_sq, 6.0, atol=0.1):
        l6_vectors.append(v)
    
    v_abs = np.abs(v)
    v_norm = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else v
    
    if np.allclose(v_abs, np.abs(dem), atol=0.05):
        democratic.append(v_norm)
    elif np.any(np.abs(v_norm - np.array([0, 1, 1])/np.sqrt(2)) < 0.1) or \
         np.any(np.abs(v_norm - np.array([1, 0, 1])/np.sqrt(2)) < 0.1) or \
         np.any(np.abs(v_norm - np.array([1, 1, 0])/np.sqrt(2)) < 0.1):
        root_like.append(v_norm)
    elif np.any(np.abs(v_norm - np.array([-2, 1, 1])/np.sqrt(6)) < 0.1) or \
         np.any(np.abs(v_norm - np.array([2, -1, 1])/np.sqrt(6)) < 0.1) or \
         np.any(np.abs(v_norm - np.array([1, -2, 1])/np.sqrt(6)) < 0.1):
        hybrid.append(v_norm)
    else:
        others.append(v_norm)

# ==================== 3. Enhanced 3D Visualization ====================
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# Unit sphere background
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.08, linewidth=0.5)

# Coordinate axes
ax.quiver(0,0,0,1.2,0,0, length=1.2, color='black', alpha=0.7, linewidth=1.5)
ax.quiver(0,0,0,0,1.2,0, length=1.2, color='black', alpha=0.7, linewidth=1.5)
ax.quiver(0,0,0,0,0,1.2, length=1.2, color='black', alpha=0.7, linewidth=1.5)
ax.text(1.35, 0, 0, "e1", color='black', fontsize=12)
ax.text(0, 1.35, 0, "e2", color='black', fontsize=12)
ax.text(0, 0, 1.35, "e3", color='black', fontsize=12)

# Plot vectors by class
if others:
    others_arr = np.array(others)
    ax.scatter(others_arr[:,0], others_arr[:,1], others_arr[:,2], c='lightgray', s=30, alpha=0.6, label='Other vectors')

if root_like:
    rl_arr = np.array(root_like)
    ax.scatter(rl_arr[:,0], rl_arr[:,1], rl_arr[:,2], c='green', s=100, marker='o', edgecolors='darkgreen', linewidth=1, label='Root-like [1,1,0]/√2 type')

if democratic:
    dem_arr = np.array(democratic)
    ax.scatter(dem_arr[:,0], dem_arr[:,1], dem_arr[:,2], c='blue', s=150, marker='D', edgecolors='navy', linewidth=1.5, label='Democratic [1,1,1]/√3')

if hybrid:
    hy_arr = np.array(hybrid)
    ax.scatter(hy_arr[:,0], hy_arr[:,1], hy_arr[:,2], c='red', s=120, marker='^', edgecolors='darkred', linewidth=1, label='Hybrid [-2,1,1]/√6 type')

if l1_vectors:
    l1_arr = np.array([v / np.linalg.norm(v) for v in l1_vectors])
    ax.scatter(l1_arr[:,0], l1_arr[:,1], l1_arr[:,2], c='purple', s=200, marker='o', edgecolors='black', linewidth=2, label='L² ≈ 1 (low mass level)')

if l6_vectors:
    l6_arr = np.array([v / np.linalg.norm(v) for v in l6_vectors])
    ax.scatter(l6_arr[:,0], l6_arr[:,1], l6_arr[:,2], c='orange', s=200, marker='*', edgecolors='black', linewidth=2, label='L² ≈ 6 (higher mass level)')

# Connect nearest neighbors for crystal structure
all_normed = np.array([v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else v for v in vectors])
for i in range(len(all_normed)):
    for j in range(i+1, len(all_normed)):
        dist = np.linalg.norm(all_normed[i] - all_normed[j])
        if dist < 0.8:  # Tighter threshold for clearer lattice
            ax.plot([all_normed[i][0], all_normed[j][0]],
                    [all_normed[i][1], all_normed[j][1]],
                    [all_normed[i][2], all_normed[j][2]],
                    color='silver', linewidth=0.8, alpha=0.7)

ax.set_title('Emergent 44-Vector Crystal Lattice from Z₃ Triality\n'
             '(Purely Mathematical Structure – Curious Numerical Coincidences Only)', 
             fontsize=16, pad=30)

ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=11, frameon=True, fancybox=True, shadow=True)

ax.view_init(elev=25, azim=135)
ax.set_axis_off()

plt.tight_layout()
plt.show()