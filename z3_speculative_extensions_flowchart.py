import matplotlib.pyplot as plt
import networkx as nx

# 创建有向图
G = nx.DiGraph()

# 节点（分层）
# 左：基础
G.add_node("Z3 Algebra", pos=(0, 4), color='lightblue')
G.add_node("Triality τ³=id", pos=(0, 2), color='lightblue')
G.add_node("Cubic Invariant", pos=(0, 0), color='lightblue')

# 中：核心
G.add_node("Core Lattice\n44 Vectors", pos=(3, 2), color='lightgreen')

# 右：扩展
G.add_node("Effective Action\nSTr(F∧F∧φ)", pos=(6, 4), color='wheat')
G.add_node("Ternary Interference\nCP Asymmetry", pos=(6, 2), color='wheat')
G.add_node("Dim-5 Interactions\nζ E·B", pos=(6, 0), color='wheat')
G.add_node("E8 Embedding\n248 → 19 Projection", pos=(6, -2), color='wheat')

# 边（带公式标签）
edges = [
    ("Z3 Algebra", "Core Lattice\n44 Vectors", "Triality Operations"),
    ("Triality τ³=id", "Core Lattice\n44 Vectors", "Saturation"),
    ("Cubic Invariant", "Core Lattice\n44 Vectors", "Vacuum Alignment"),
    ("Core Lattice\n44 Vectors", "Effective Action\nSTr(F∧F∧φ)", "Formal Curvature"),
    ("Core Lattice\n44 Vectors", "Ternary Interference\nCP Asymmetry", "{F,F,ζ} Brackets"),
    ("Core Lattice\n44 Vectors", "Dim-5 Interactions\nζ E·B", "[F,ζ] Couplings"),
    ("Core Lattice\n44 Vectors", "E8 Embedding\n248 → 19 Projection", "Iterated Mapping")
]

G.add_edges_from([(a, b, {'label': l}) for a, b, l in edges])

# 绘图
pos = nx.get_node_attributes(G, 'pos')
colors = nx.get_node_attributes(G, 'color').values()

plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=4000, font_size=10,
        font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20,
        edge_color='gray', width=2, alpha=0.9)

# 边标签
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# 标题和说明
plt.title("Speculative Mathematical Extensions in Z₃ Vacuum Geometry\n"
          "(Formal Analogies – Purely Algebraic, No Physical Claim)", fontsize=16, pad=20)
plt.text(3, -3, "All patterns are mathematical curiosities from triality closure.\n"
                "No mechanism or physical relevance is proposed.", ha='center', fontsize=12, style='italic')

plt.axis('off')
plt.tight_layout()
plt.savefig('z3_speculative_extensions_flowchart.png', dpi=300, bbox_inches='tight')
plt.show()