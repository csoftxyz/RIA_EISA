import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

# 生成44向量（从之前代码复用）
def generate_lattice_vectors():
    basis = np.eye(3)
    dem = np.array([1., 1., 1.]) / np.sqrt(3)
    seed = np.vstack([basis, dem, -dem])
    
    T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    
    def apply_triality(v):
        return T @ v
    
    vectors = set()
    current = [v for v in seed]
    
    for _ in range(10):
        new_vectors = []
        for v in current:
            v1 = apply_triality(v)
            v2 = apply_triality(v1)
            
            candidates = [v1, v2, v1 - v, v2 - v, np.cross(v, v1)]
            
            for cand in candidates:
                norm = np.linalg.norm(cand)
                if norm > 1e-8:
                    normalized = cand / norm
                    key_norm = tuple(np.round(normalized, 10))
                    key_raw = tuple(np.round(cand, 10))
                    if key_norm not in vectors:
                        vectors.add(key_norm)
                        new_vectors.append(normalized)
                    if key_raw not in vectors:
                        vectors.add(key_raw)
                        new_vectors.append(cand)
        current = new_vectors
    
    vec_list = [np.array(t) for t in vectors if np.linalg.norm(np.array(t)) > 1e-8]
    print(f"生成向量总数: {len(vec_list)}")
    return vec_list

vectors = generate_lattice_vectors()
vectors_normed = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-8 else v for v in vectors]

# 分类函数（复用）
def classify_vector(v):
    norm = np.linalg.norm(v)
    n = np.round(v, 8)
    if abs(norm - 1.0) < 0.1:
        return "g0 (bosonic)", "blue"  # 对应degree 0
    elif abs(norm - np.sqrt(2)) < 0.1:
        return "g1 (fermionic)", "red"  # degree 1
    elif np.allclose(abs(n), [1,1,1], atol=1e-5):
        return "g2 (vacuum)", "gold"  # degree 2
    elif any(abs(x) >= 2 for x in n):
        return "hybrid", "green"
    else:
        return "other", "purple"

# 构建图：节点为向量，边基于triality或cross product连接（简化：如果内积>0.5或叉积norm>0.5则连边）
G = nx.Graph()
for i, v in enumerate(vectors_normed):
    G.add_node(i, pos=v, label=classify_vector(v)[0])

for i in range(len(vectors_normed)):
    for j in range(i+1, len(vectors_normed)):
        v1 = vectors_normed[i]
        v2 = vectors_normed[j]
        dot = np.abs(np.dot(v1, v2))
        cross_norm = np.linalg.norm(np.cross(v1, v2))
        if dot > 0.5 or cross_norm > 0.5:  # 模拟"操作连接"
            G.add_edge(i, j)

# 计算邻接矩阵A，Tr(A^4)
A = nx.to_numpy_array(G)
trace_A4 = np.trace(np.linalg.matrix_power(A, 4))
print(f"Tr(A^4) ≈ {trace_A4:.2e} (combinatorial factor estimate)")

# 可视化：3D图 + 图论网络
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 单位球面网格
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.1, linewidth=0.5)

# 坐标轴
ax.quiver(0,0,0,1,0,0, length=1.2, color='black', alpha=0.6, linewidth=1.5)
ax.quiver(0,0,0,0,1,0, length=1.2, color='black', alpha=0.6, linewidth=1.5)
ax.quiver(0,0,0,0,0,1, length=1.2, color='black', alpha=0.6, linewidth=1.5)
ax.text(1.3, 0, 0, "e1", color='black', fontsize=12)
ax.text(0, 1.3, 0, "e2", color='black', fontsize=12)
ax.text(0, 0, 1.3, "e3", color='black', fontsize=12)

# 绘制节点和边
pos = nx.get_node_attributes(G, 'pos')
for node, p in pos.items():
    label, color = classify_vector(p)
    size = 120 if "g2" in label else 60 if "g1" in label else 30
    ax.scatter(p[0], p[1], p[2], c=color, s=size, depthshade=True, edgecolors='black', linewidth=0.5)

for edge in G.edges():
    p1 = pos[edge[0]]
    p2 = pos[edge[1]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.3, linewidth=0.5)

# 高亮g2真空
dem = np.array([1,1,1])/np.sqrt(3)
ax.scatter(dem[0], dem[1], dem[2], c='gold', s=300, marker='*', edgecolors='black', linewidth=1)

# 文字说明
ax.text2D(0.02, 0.92, "Z₃-Graded Lattice Structure (44 nodes)", transform=ax.transAxes, color='black', fontsize=16, weight='bold')
ax.text2D(0.02, 0.87, "Blue: g0 (bosonic)", transform=ax.transAxes, color='blue', fontsize=14)
ax.text2D(0.02, 0.83, "Red: g1 (fermionic)", transform=ax.transAxes, color='red', fontsize=14)
ax.text2D(0.02, 0.79, "Gold ★: g2 (vacuum)", transform=ax.transAxes, color='gold', fontsize=14)
ax.text2D(0.02, 0.75, f"Graph Tr(A^4) ≈ {trace_A4:.2e} (~10^6-10^7)", transform=ax.transAxes, color='purple', fontsize=14)
ax.text2D(0.02, 0.71, "Gray lines: Triality/cross connections", transform=ax.transAxes, color='gray', fontsize=12)

ax.set_axis_off()
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()