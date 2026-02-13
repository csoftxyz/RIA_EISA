import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== 1. Generate 44-vector lattice ====================
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
print(f"Generated vectors: {len(vectors)}")  # Should be around 44

# ==================== 2. Classification ====================
democratic = []  # Neutrino-related: flatter hierarchies, blue
root_like = []   # Quark-related: strong hierarchies, green
hybrid = []      # Mixed axes, red
l1_vectors = []  # L=1: basic units, purple spheres (low mass level)
l6_vectors = []  # L=6: extended, orange stars (higher mass level)
gauge_basis = [] # Gauge basis vectors (e.g., permutations of [1,0,0])
others = []

# Helper: check if vector is permutation of [1,0,0] (up to sign and normalization)
def is_gauge_basis(v, atol=0.1):
    abs_v = np.abs(v)
    sorted_abs = np.sort(abs_v)
    return np.allclose(sorted_abs, [0,0,1], atol=atol)

for v in vectors:
    norm_sq = np.round(np.linalg.norm(v)**2, 5)  # L^2 for levels
    if np.isclose(norm_sq, 1.0, atol=1e-5):
        l1_vectors.append(v)
    elif np.isclose(norm_sq, 6.0, atol=1e-5):  # L^2=6 examples like [2,1,1]
        l6_vectors.append(v)
    
    # Classify axes
    if np.allclose(np.abs(v), np.abs(dem), atol=1e-5):  # Democratic for neutrinos
        democratic.append(v)
    elif is_gauge_basis(v):  # Gauge basis permutations
        gauge_basis.append(v)
    elif np.allclose(np.sort(np.abs(v)), np.sort([1,1,0]/np.sqrt(2)), atol=1e-5):  # Root-like for quarks
        root_like.append(v)
    elif np.allclose(np.sort(np.abs(v)), np.sort([2,1,1]/np.sqrt(6)), atol=1e-5):  # Hybrid
        hybrid.append(v)
    else:
        others.append(v)

print(f"Democratic (Neutrino axes): {len(democratic)}")
print(f"Gauge Basis: {len(gauge_basis)}")
print(f"Root-like (Quark axes): {len(root_like)}")
print(f"Hybrid: {len(hybrid)}")
print(f"L=1 (Low mass level): {len(l1_vectors)}")
print(f"L=6 (Higher mass level): {len(l6_vectors)}")
print(f"Others: {len(others)}")

# ==================== 3. Beautiful 3D Visualization ====================
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# To avoid array ambiguity, convert to tuples for set operations
def array_to_tuple(arr):
    return tuple(np.round(arr, 10))

classified_set = set(array_to_tuple(v) for v in democratic + gauge_basis + root_like + hybrid + l1_vectors + l6_vectors)

others = [v for v in vectors if array_to_tuple(v) not in classified_set]

# Plot others (light gray background)
if others:
    others_arr = np.array(others)
    ax.scatter(others_arr[:,0], others_arr[:,1], others_arr[:,2], c='lightgray', s=15, alpha=0.4, label='Other Vectors')

# Plot Gauge Basis (purple, large - low mass)
if gauge_basis:
    gb_arr = np.array(gauge_basis)
    ax.scatter(gb_arr[:,0], gb_arr[:,1], gb_arr[:,2], c='purple', s=120, marker='o', edgecolor='black', label='Gauge Basis (Low Mass Level)')

# Plot Democratic (blue, large)
if democratic:
    dem_arr = np.array(democratic)
    ax.scatter(dem_arr[:,0], dem_arr[:,1], dem_arr[:,2], c='blue', s=80, marker='o', label='Democratic Axes (Neutrino Hierarchies)')

# Plot Root-like (green)
if root_like:
    rl_arr = np.array(root_like)
    ax.scatter(rl_arr[:,0], rl_arr[:,1], rl_arr[:,2], c='green', s=60, marker='o', label='Root-like Axes (Quark Hierarchies)')

# Plot Hybrid (red)
if hybrid:
    hy_arr = np.array(hybrid)
    ax.scatter(hy_arr[:,0], hy_arr[:,1], hy_arr[:,2], c='red', s=50, marker='^', label='Hybrid Axes')

# Highlight L=1 (purple large spheres)
if l1_vectors:
    l1_arr = np.array(l1_vectors)
    ax.scatter(l1_arr[:,0], l1_arr[:,1], l1_arr[:,2], c='purple', s=150, marker='o', edgecolor='white', linewidth=2, label='L=1 Vectors (Low Mass)')

# Highlight L=6 (orange stars)
if l6_vectors:
    l6_arr = np.array(l6_vectors)
    ax.scatter(l6_arr[:,0], l6_arr[:,1], l6_arr[:,2], c='orange', s=120, marker='*', edgecolor='black', label='L=6 Vectors (Higher Mass)')

# Dense crystal connections
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        dist = np.linalg.norm(vectors[i] - vectors[j])
        if dist < 1.5:
            ax.plot([vectors[i][0], vectors[j][0]],
                    [vectors[i][1], vectors[j][1]],
                    [vectors[i][2], vectors[j][2]],
                    color='silver', linewidth=0.8, alpha=0.5)

ax.set_xlabel('X Component', fontsize=12, labelpad=10)
ax.set_ylabel('Y Component', fontsize=12, labelpad=10)
ax.set_zlabel('Z Component', fontsize=12, labelpad=10)
ax.set_title('Z₃ 44-Vector Crystal Lattice: Neutrino vs. Quark Mass Hierarchies\n(L=1 Low Level Purple, L=6 Higher Level Orange)', fontsize=14, pad=20)

ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10, frameon=True)

ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.savefig('z3_crystal_44_schematic.png', dpi=300, bbox_inches='tight')
plt.show()