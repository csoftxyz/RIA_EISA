"""
z3_44_lattice_3D_figure.py
Publication-quality 3D visualization of the 44-vector Z3 lattice
for the PRL paper figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# ============================================================
# Generate 44-vector lattice
# ============================================================
basis = np.eye(3)
dem_vec = np.array([1, 1, 1]) / np.sqrt(3)
seed = np.vstack([basis, [dem_vec, -dem_vec]])
T_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

def apply_triality(v):
    return T_mat @ v

uniq = set()
for v in seed:
    uniq.add(tuple(np.round(v, 10)))
current = seed.tolist()

for level in range(15):
    new = []
    for v in current:
        v1 = apply_triality(v); v2 = apply_triality(v1)
        new += [v1, v2, v1 - v, v2 - v]
        cr = np.cross(v, v1)
        if np.linalg.norm(cr) > 1e-6:
            new.extend([cr, cr / np.linalg.norm(cr)])
    for nv in new:
        if np.linalg.norm(nv) > 1e-6:
            uniq.add(tuple(np.round(nv, 10)))
    all_v = [np.array(u) for u in uniq]
    current = [v.tolist() for v in all_v[:100]]
    if not new: break

vectors_all = [np.array(t) for t in uniq]
vectors_all = [v for v in vectors_all if np.linalg.norm(v) > 1e-6]
vectors_all.sort(key=lambda x: (round(np.linalg.norm(x), 4), np.sum(np.abs(x))))
vectors_44 = np.array(vectors_all[:44])

# Shell classification
shells = {}
for i, v in enumerate(vectors_44):
    L2 = round(np.sum(v**2), 4)
    shells.setdefault(L2, []).append(i)

# ============================================================
# Colors by shell type
# ============================================================
# Cool colormap: blue (inner) -> red (outer)
shell_colors = {
    1.0: '#1f77b4',      # blue - basis
    2.0: '#2ca02c',      # green - root shell 1
    3.0: '#d62728',      # red - democratic 1
    6.0: '#ff7f0e',      # orange - root shell 2
    18.0: '#9467bd',     # purple - root shell 3
    27.0: '#d62728',     # red - democratic 2
    54.0: '#8c564b',     # brown - root shell 4
    162.0: '#e377c2',    # pink - root shell 5
    243.0: '#d62728',    # red - democratic 3
    486.0: '#7f7f7f',    # grey - root shell 6
}

# ============================================================
# Figure setup
# ============================================================
fig = plt.figure(figsize=(16, 7))

# --- Panel (a): Full 44-vector lattice ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('(a) 44-Vector Z$_3$ Vacuum Lattice', fontsize=13, fontweight='bold', pad=15)

# Plot all vectors by shell
for L2 in sorted(shells.keys()):
    idxs = shells[L2]
    pts = vectors_44[idxs]
    color = shell_colors.get(L2, '#333333')
    size = 80 if len(idxs) == 1 else 50
    marker = 'D' if len(idxs) == 1 else 'o'
    alpha = 1.0 if len(idxs) == 1 else 0.75
    zorder = 10 if len(idxs) == 1 else 5
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                c=color, s=size, marker=marker, alpha=alpha,
                edgecolors='k', linewidth=0.3, zorder=zorder)

# Draw octahedron edges for L^2=2 shell (first root shell)
def draw_octahedron(ax, pts, color, alpha=0.3, lw=0.8):
    """Draw K_{2,2,2} octahedron edges: each vertex connects to 4 non-antipodal others."""
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] + pts[j]) > 0.01:  # not antipodal
                ax.plot3D([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], [pts[i,2], pts[j,2]],
                         color=color, alpha=alpha, linewidth=lw)

# Draw octahedra for selected root shells
for L2 in [2.0, 6.0, 18.0]:
    pts = vectors_44[shells[L2]]
    draw_octahedron(ax1, pts, shell_colors[L2], alpha=0.25, lw=0.6)

# Democratic axis
dem_pts = np.array([[0,0,0], [9,9,9]])
ax1.plot3D(dem_pts[:,0], dem_pts[:,1], dem_pts[:,2], 
           '--', color='red', alpha=0.5, linewidth=1.2, label='Democratic axis [111]')

# Highlight democratic nodes
dem_nodes_idx = [11, 24, 37]
for idx in dem_nodes_idx:
    v = vectors_44[idx]
    ax1.scatter([v[0]], [v[1]], [v[2]], c='red', s=150, marker='D',
                edgecolors='darkred', linewidth=1.5, zorder=20)

# Labels
ax1.set_xlabel('$v_x$', labelpad=8)
ax1.set_ylabel('$v_y$', labelpad=8)
ax1.set_zlabel('$v_z$', labelpad=8)
ax1.view_init(elev=22, azim=-42)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_zlim(-10, 10)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ca02c', alpha=0.7, label='Root shells ($K_{2,2,2}$)'),
    mpatches.Patch(facecolor='red', alpha=0.7, label='Democratic nodes (Higgs)'),
    plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Democratic axis [111]'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

# --- Panel (b): Shell structure diagram ---
ax2 = fig.add_subplot(122)

# Radial projection plot
L2_vals = sorted(shells.keys())
for L2 in L2_vals:
    idxs = shells[L2]
    n = len(idxs)
    r = np.sqrt(L2)
    color = shell_colors.get(L2, '#333333')
    
    if n == 1:
        # Democratic - red diamond
        ax2.scatter(r, 0, c=color, s=200, marker='D', edgecolors='darkred', linewidth=1.5, zorder=10)
        if L2 < 100:
            ax2.annotate(f'$L^2 = {L2:.0f}$\n$[{int(np.sqrt(L2/3))},{int(np.sqrt(L2/3))},{int(np.sqrt(L2/3))}]$', 
                        xy=(r, 0.05), fontsize=7, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    elif n == 6:
        # Root shell - scatter at slightly different y for visibility
        jitter = np.linspace(-0.08, 0.08, 6)
        ax2.scatter([r]*6, jitter, c=color, s=40, alpha=0.7, zorder=5)
        ax2.annotate(f'$L^2 = {L2:.0f}$\n(${n}$ vectors)', 
                    xy=(r, -0.15), fontsize=7, ha='center', va='top', color=color)
    else:
        jitter = np.linspace(-0.06, 0.06, n)
        ax2.scatter([r]*n, jitter, c=color, s=30, alpha=0.6, zorder=5)

# Fermion shell assignments
up_shells = [2.0, 162.0, 486.0]
down_shells = [6.0, 162.0, 486.0]
lep_shells = [1.0, 18.0, 486.0]

for shells_list, y_pos, label, color in [
    (up_shells, 0.35, 'Up-type', '#e41a1c'),
    (down_shells, 0.25, 'Down-type', '#377eb8'),
    (lep_shells, 0.15, 'Leptons', '#4daf4a')]:
    for L2 in shells_list:
        r = np.sqrt(L2)
        ax2.axvline(x=r, ymin=0.42, ymax=0.58, color=color, alpha=0.4, linewidth=2, linestyle='-')
    # Label
    r_mid = np.sqrt(shells_list[1])
    ax2.text(r_mid, y_pos, label, fontsize=9, fontweight='bold', color=color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color))

ax2.set_title('(b) Shell Structure & Fermion Assignments', fontsize=13, fontweight='bold')
ax2.set_xlabel('Radial distance $|\\mathbf{v}| = \\sqrt{L^2}$', fontsize=11)
ax2.set_ylabel('Shell multiplicity', fontsize=11)
ax2.set_xlim(0, 23)
ax2.set_ylim(-0.25, 0.5)
ax2.set_yticks([])
ax2.axhline(y=0, color='grey', alpha=0.3, linewidth=0.5)

# Geometric progression annotation
ax2.annotate('$L_k = \\sqrt{2}\\cdot(\\sqrt{3})^{k-1}$\nGeometric progression\n(ratio $\\sqrt{3}$ per shell)',
            xy=(14, 0.42), fontsize=8, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

plt.tight_layout(pad=2)
plt.savefig('./Z3_44_Lattice_3D.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('./Z3_44_Lattice_3D.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved: ./Z3_44_Lattice_3D.pdf, ./Z3_44_Lattice_3D.png")
plt.close()
