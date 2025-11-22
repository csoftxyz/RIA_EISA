# show_z3_algebra_final.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.DiGraph()

# 等边三角形顶点（完美对称）
radius = 4.8
pos = {
    "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$": ( radius * np.cos(np.radians(90)),  radius * np.sin(np.radians(90))),
    "Grade 1\nFermions (4 dim)":                     ( radius * np.cos(np.radians(210)), radius * np.sin(np.radians(210))),
    "Grade 2\nVacuum (3 dim)":                        ( radius * np.cos(np.radians(330)), radius * np.sin(np.radians(330))),
}
node_colors = ["#5DADE2", "#58D68D", "#EC7063"]   # 天蓝、薄荷绿、珊瑚红

for node, p in pos.items():
    G.add_node(node, pos=p)

# ====================== 边 ======================
edges = [
    # 主要括号（实线）
    ("Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$",
     "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "[B,B] → B", "solid"),
    ("Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "Grade 1\nFermions (4 dim)", "[B,F] → F", "solid"),
    ("Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "Grade 2\nVacuum (3 dim)", "[B,ζ] → ζ", "solid"),
    ("Grade 1\nFermions (4 dim)", "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "[F,F] (d=0)", "solid"),
    ("Grade 1\nFermions (4 dim)", "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "[F,ζ] → B", "solid"),
    ("Grade 2\nVacuum (3 dim)", "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "[ζ,F] skew", "solid"),
    ("Grade 2\nVacuum (3 dim)", "Grade 1\nFermions (4 dim)", "[ζ,ζ] → F", "solid"),
    ("Grade 1\nFermions (4 dim)", "Grade 2\nVacuum (3 dim)", "{F,F,F} → ζ\n(optional)", "dashed"),
]

for u, v, label, style in edges:
    G.add_edge(u, v, label=label, style=style)

# ====================== 绘制 ======================
plt.figure(figsize=(14, 12), facecolor='white')
ax = plt.gca()
ax.set_aspect('equal')

# 节点（大圆 + 清晰黑边）
nx.draw_networkx_nodes(G, pos, node_size=11000, node_color=node_colors,
                       edgecolors='black', linewidths=3.5, alpha=0.97)

# 节点文字（LaTeX）
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 实线边
solid = [(u,v) for u,v,d in G.edges(data=True) if d['style']=='solid']
nx.draw_networkx_edges(G, pos, edgelist=solid, edge_color='#2E4053',
                       arrows=True, arrowsize=28, arrowstyle='->', width=3,
                       connectionstyle='arc3,rad=0.18')

# 虚线边（三次括号）
dashed = [(u,v) for u,v,d in G.edges(data=True) if d['style']=='dashed']
nx.draw_networkx_edges(G, pos, edgelist=dashed, edge_color='#9B59B6',
                       style='dashed', arrows=True, arrowsize=28, arrowstyle='->', width=3.5,
                       connectionstyle='arc3,rad=0.25', alpha=0.9)

# 边标签（白色背景不遮挡）
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10.5,
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.4))

# Z₃ triality 循环：用三条柔和圆弧虚线（不遮挡文字）
triality_arcs = [
    ("Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", "Grade 1\nFermions (4 dim)", 0.4),
    ("Grade 1\nFermions (4 dim)", "Grade 2\nVacuum (3 dim)", 0.4),
    ("Grade 2\nVacuum (3 dim)", "Grade 0\nBosons (12 dim)\n$\\mathfrak{su}(3)\\oplus\\mathfrak{su}(2)\\oplus\\mathfrak{u}(1)$", 0.4),
]
for u, v, rad in triality_arcs:
    arc = nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                 edge_color='#3498DB', style='--', alpha=0.6,
                                 arrows=False, width=5,
                                 connectionstyle=f'arc3,rad={rad}')

# 中心文字（小一点 + 透明背景）
plt.text(0, 0, "Z₃ Cyclic\nGrading &\nEmergent\nTriality",
         ha='center', va='center', fontsize=15, fontweight='bold', color='#2E4053',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='#3498DB', boxstyle='round,pad=0.8', linewidth=1.5))

plt.title("Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality\n"
          "Minimal 19-dimensional example (dim = 12 ⊕ 4 ⊕ 3)",
          fontsize=19, fontweight='bold', pad=40, color='#2E4053')

plt.axis('off')
plt.tight_layout()
plt.savefig("z3_algebra_perfect.pdf", dpi=400, bbox_inches='tight')
plt.savefig("z3_algebra_perfect.png", dpi=400, bbox_inches='tight')
plt.show()