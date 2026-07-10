#!/usr/bin/env python3
"""
Figure 1: The 44-Vector Z₃ Quaternion Orbit Closure
====================================================
Publication-quality multi-panel figure for
"Quaternions as Quantization" (Zhang, Hu & Zhang 2026)

Panels:
  (a) Full 44-vector lattice — 3D view with color-coded shells
  (b) V₀⊕V₁ decomposition — democratic axis + perpendicular plane
  (c) A₂ hexagonal root lattice — the D₃-symmetric V₁ sector
  (d) Shell structure — L² distribution with cumulative count
"""

import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Global settings ───────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'figure.facecolor': '#0a0a14',
    'axes.facecolor': '#0d0d1a',
    'text.color': '#c8d6e5',
    'axes.edgecolor': '#334466',
    'axes.labelcolor': '#8899aa',
    'xtick.color': '#556677',
    'ytick.color': '#556677',
    'grid.alpha': 0.15,
})

# ── Constants ─────────────────────────────────────────────
d = np.array([1., 1., 1.]) / sqrt(3)  # democratic axis
e1, e2, e3 = np.eye(3)
e1p = np.array([2./3., -1./3., -1./3.])  # e₁⊥
T = lambda v: np.array([v[2], v[0], v[1]])
Delta = lambda v: T(v) - v

# ── Generate 44-vector lattice ────────────────────────────
def generate_lattice():
    seeds = [np.array(v) for v in [[1,0,0],[0,1,0],[0,0,1], d, -d]]
    uniq = {tuple(np.round(s, 12)) for s in seeds}

    # V₁ root shells at L² = 2, 6, 18, 54, 162, 486
    v1_vecs = {}
    v = e1p.copy()
    for k in range(6):
        L2 = round(np.dot(v, v), 4)
        # D₃ orbit: ±v, ±T(v), ±T²(v)
        orbit = []
        for sgn in [1, -1]:
            w = sgn * v.copy()
            for _ in range(3):
                orbit.append(tuple(np.round(w, 12)))
                w = T(w)
        v1_vecs[L2] = [np.array(o) for o in orbit]
        uniq.update(orbit)
        v = Delta(v)

    # V₀ democratic shells at L² = 3, 27, 243
    v0_vecs = {}
    for L2 in [3.0, 27.0, 243.0]:
        s = sqrt(L2)
        pos = d * s
        neg = -d * s
        v0_vecs[L2] = [pos, neg]
        uniq.update([tuple(np.round(pos,12)), tuple(np.round(neg,12))])

    # Sort all by norm (smallest first, ground state)
    all_v = sorted([np.array(u) for u in uniq],
                   key=lambda x: (round(np.linalg.norm(x), 8), np.sum(np.abs(x))))
    return all_v[:44], v1_vecs, v0_vecs

all_44, v1_vecs, v0_vecs = generate_lattice()

# Classify by type
def classify(v):
    L2 = round(np.dot(v, v), 8)
    if L2 == 1.0: return 'seed'
    if abs(np.dot(v, d)) > 0.99 * np.linalg.norm(v): return 'V0'
    return 'V1'

# ── Colors ────────────────────────────────────────────────
seed_color   = '#ff6644'
v1_palette   = ['#44ccff', '#33aadd', '#2288bb', '#116699', '#005588', '#004477']
v0_color     = '#ffcc33'

# ── Shell colors for V₁ ───────────────────────────────────
v1_L2_list = [2, 6, 18, 54, 162, 486]
v1_color_map = dict(zip(v1_L2_list, v1_palette))

def get_color(v):
    tp = classify(v)
    if tp == 'seed': return seed_color
    if tp == 'V0': return v0_color
    L2 = round(np.dot(v, v), 4)
    return v1_color_map.get(L2, '#888888')

# ── Create figure ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
fig.patch.set_facecolor('#0a0a14')

# ── Panel (a): Full 44-vector lattice ─────────────────────
ax_a = fig.add_subplot(2, 2, 1, projection='3d')
ax_a.set_facecolor('#0d0d1a')
ax_a.set_title('(a) Full 44-Vector Lattice\n$|L_{44}| = 4 \\times 11 = 44$',
               color='#c8d6e5', pad=-2)

# Plot all 44 vectors as points
for v in all_44:
    c = get_color(v)
    s = 25 if classify(v) == 'seed' else (20 if classify(v) == 'V0' else 10)
    ax_a.scatter(*v, c=c, s=s, alpha=0.9, edgecolors='none', depthshade=True)

# Democratic axis line
ax_a.plot([-d[0]*20, d[0]*20], [-d[1]*20, d[1]*20], [-d[2]*20, d[2]*20],
          color='#556688', alpha=0.4, linewidth=0.8, linestyle='--')

# V₁ plane hint: a few ring circles
for r in [6, 13]:
    theta = np.linspace(0, 2*pi, 100)
    # Circle in V₁ plane (perpendicular to d)
    u = np.array([1., -1., 0.]) / sqrt(2)  # orthonormal basis for V₁
    v_dir = np.cross(d, u)
    circle = np.array([r*(np.cos(t)*u + np.sin(t)*v_dir) for t in theta])
    ax_a.plot(circle[:,0], circle[:,1], circle[:,2],
              color='#334466', alpha=0.25, linewidth=0.5)

ax_a.set_xlim(-14, 14); ax_a.set_ylim(-14, 14); ax_a.set_zlim(-14, 14)
ax_a.set_xlabel('x', color='#556688'); ax_a.set_ylabel('y', color='#556688')
ax_a.set_zlabel('z', color='#556688')
ax_a.view_init(elev=22, azim=-48)
for spine in ax_a.spines.values(): spine.set_visible(False)

# Legend
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=seed_color, markersize=7, label='Seeds (L²=1, 5)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#44ccff', markersize=5, label='V₁ Root shells (6×6=36)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=v0_color, markersize=6, label='V₀ Democratic (3×1=3)'),
    Line2D([0],[0], linestyle='--', color='#556688', linewidth=1, label='d = (1,1,1)/√3'),
]
leg = ax_a.legend(handles=leg_elements, loc='upper left', fontsize=7,
                  facecolor='#0d0d1a88', edgecolor='#334466',
                  labelcolor='#8899aa', framealpha=0.7)

# ── Panel (b): V₀⊕V₁ Decomposition ────────────────────────
ax_b = fig.add_subplot(2, 2, 2, projection='3d')
ax_b.set_facecolor('#0d0d1a')
ax_b.set_title('(b) $\\mathbb{R}^3 = V_0 \\oplus V_1$ Decomposition\n$V_0 = \\mathrm{span}\\{\\mathbf{d}\\}$ (trivial), $V_1 \\perp \\mathbf{d}$ (120° rotation)',
               color='#c8d6e5', pad=-2)

# V₁ plane (semi-transparent disc)
theta = np.linspace(0, 2*pi, 60)
u_v1 = np.array([2., -1., -1.]) / sqrt(6)
v_v1 = np.cross(d, u_v1)
r_max = 14
ring_pts = []
for r in np.linspace(0, r_max, 20):
    for t in theta[::4]:
        ring_pts.append(r*(np.cos(t)*u_v1 + np.sin(t)*v_v1))
ring_pts = np.array(ring_pts)
ax_b.scatter(ring_pts[:,0], ring_pts[:,1], ring_pts[:,2],
             c='#223355', s=0.5, alpha=0.3, depthshade=False)

# V₁ plane boundary ring
circle_pts = np.array([r_max*(np.cos(t)*u_v1 + np.sin(t)*v_v1) for t in theta])
ax_b.plot(circle_pts[:,0], circle_pts[:,1], circle_pts[:,2],
          color='#4466aa', alpha=0.5, linewidth=1.2)

# Democratic axis (thick, prominent)
ax_b.plot([-d[0]*15, d[0]*15], [-d[1]*15, d[1]*15], [-d[2]*15, d[2]*15],
          color='#aaccee', alpha=0.7, linewidth=2)
ax_b.scatter([0],[0],[0], c='#ffffff', s=30, alpha=0.8)

# Arrow tip
tip = d * 15.5
ax_b.scatter(*tip, c='#aaccee', s=40, marker='^', alpha=0.8)
ax_b.text(tip[0]+0.8, tip[1]+0.8, tip[2]+0.8, '$V_0$ (d-axis)',
          color='#aaccee', fontsize=8, fontweight='bold')

# Show a few V₁ root vectors in the plane
for L2 in [2, 6, 18]:
    if L2 in v1_vecs:
        for v in v1_vecs[L2][:3]:
            ax_b.quiver(0, 0, 0, v[0], v[1], v[2],
                        color='#44ccff', alpha=0.5, linewidth=0.8,
                        arrow_length_ratio=0.08)

# Show seeds (decomposed)
for s in [e1, e2, e3]:
    par = np.dot(s, d) * d
    perp = s - par
    ax_b.quiver(0, 0, 0, par[0], par[1], par[2], color='#ff6644', alpha=0.6, linewidth=1.5, arrow_length_ratio=0.06)
    ax_b.quiver(par[0], par[1], par[2], perp[0], perp[1], perp[2], color='#44ccff', alpha=0.5, linewidth=1, arrow_length_ratio=0.08)

ax_b.set_xlim(-15, 15); ax_b.set_ylim(-15, 15); ax_b.set_zlim(-15, 15)
ax_b.set_xlabel('x', color='#556688'); ax_b.set_ylabel('y', color='#556688')
ax_b.set_zlabel('z', color='#556688')
ax_b.view_init(elev=18, azim=-55)
ax_b.text2D(0.05, 0.95, '$V_0$: trivial (1D)\n$V_1$: 120° rotation (2D)',
            transform=ax_b.transAxes, color='#8899aa', fontsize=7, va='top')

# ── Panel (c): A₂ Hexagonal Root Lattice ──────────────────
ax_c = fig.add_subplot(2, 2, 3, projection='3d')
ax_c.set_facecolor('#0d0d1a')
ax_c.set_title('(c) $A_2$ Hexagonal Root Lattice (V₁ sector)\n$\\mathbf{r}_1=(-1,1,0),\\; \\mathbf{r}_2=(0,-1,1),\\; \\|\\mathbf{r}_k\\|^2=2$',
               color='#c8d6e5', pad=-2)

# Project all V₁ vectors into the V₁ plane
# Use coordinates in the (u_v1, v_v1) basis
def proj_V1(v):
    return np.array([np.dot(v, u_v1), np.dot(v, v_v1)])

# Show hexagonal lattice in 2D within 3D
for L2 in v1_L2_list:
    if L2 in v1_vecs:
        pts = v1_vecs[L2]
        for pt in pts:
            ax_c.scatter(*pt, c=v1_color_map[L2], s=40, alpha=0.85, edgecolors='none', depthshade=True)

# Connect hexagon vertices
for L2 in v1_L2_list[:3]:  # first 3 shells for clarity
    if L2 in v1_vecs:
        pts = v1_vecs[L2]
        # Sort by angle in V₁ plane
        angles = [np.arctan2(np.dot(p, v_v1), np.dot(p, u_v1)) for p in pts]
        sorted_pts = [p for _, p in sorted(zip(angles, pts))]
        # Close the loop
        sorted_pts.append(sorted_pts[0])
        xs = [p[0] for p in sorted_pts]; ys = [p[1] for p in sorted_pts]; zs = [p[2] for p in sorted_pts]
        ax_c.plot(xs, ys, zs, color=v1_color_map[L2], alpha=0.5, linewidth=1.2)

# Fundamental roots
r1 = Delta(e1p)  # (-1, 1, 0)
r2 = Delta(T(e1p))  # (0, -1, 1)
ax_c.quiver(0, 0, 0, r1[0], r1[1], r1[2], color='#ff8866', linewidth=2.5, arrow_length_ratio=0.1)
ax_c.quiver(0, 0, 0, r2[0], r2[1], r2[2], color='#ffaa44', linewidth=2.5, arrow_length_ratio=0.1)
ax_c.text(r1[0]*1.15, r1[1]*1.15, r1[2]*1.15, '$\\mathbf{r}_1^0$', color='#ff8866', fontsize=9, fontweight='bold')
ax_c.text(r2[0]*1.15, r2[1]*1.15, r2[2]*1.15, '$\\mathbf{r}_2^0$', color='#ffaa44', fontsize=9, fontweight='bold')

# V₁ plane disc
ax_c.plot(circle_pts[:,0], circle_pts[:,1], circle_pts[:,2],
          color='#4466aa', alpha=0.3, linewidth=0.8, linestyle='--')

ax_c.set_xlim(-13, 13); ax_c.set_ylim(-13, 13); ax_c.set_zlim(-13, 13)
ax_c.set_xlabel('x', color='#556688'); ax_c.set_ylabel('y', color='#556688')
ax_c.set_zlabel('z', color='#556688')
ax_c.view_init(elev=70, azim=-45)
ax_c.text2D(0.05, 0.95, '$\\|\\Delta(\\mathbf{v})\\| = \\sqrt{3}\\,\\|\\mathbf{v}\\|$\n6 shells × 6 vectors = 36',
            transform=ax_c.transAxes, color='#8899aa', fontsize=7, va='top')

# ── Panel (d): Shell Structure ────────────────────────────
ax_d = fig.add_subplot(2, 2, 4)
ax_d.set_facecolor('#0d0d1a')
ax_d.set_title('(d) Shell Structure & Cumulative Count\nTotal: 5 + 36 + 3 = 44',
               color='#c8d6e5')

# Shell data
shell_data = [
    (1, 5, 'seed', 'Basis'),
    (2, 6, 'V1', 'V₁ Root'),
    (3, 1, 'V0', 'V₀ Dem.'),
    (6, 6, 'V1', 'V₁ Root'),
    (18, 6, 'V1', 'V₁ Root'),
    (27, 1, 'V0', 'V₀ Dem.'),
    (54, 6, 'V1', 'V₁ Root'),
    (162, 6, 'V1', 'V₁ Root'),
    (243, 1, 'V0', 'V₀ Dem.'),
    (486, 6, 'V1', 'V₁ Root'),
]

L2_vals = [s[0] for s in shell_data]
counts = [s[1] for s in shell_data]
types = [s[2] for s in shell_data]
labels = [s[3] for s in shell_data]
cumulative = np.cumsum(counts)

# Bar chart with color coding
colors_d = [seed_color if t == 'seed' else (v0_color if t == 'V0' else '#44ccff') for t in types]
bars = ax_d.bar(range(len(shell_data)), counts, color=colors_d, alpha=0.8, edgecolor='#1a1a2e', linewidth=0.5)
ax_d.set_xticks(range(len(shell_data)))
ax_d.set_xticklabels([f'$L^2$={l}' for l in L2_vals], rotation=45, ha='right', fontsize=7, color='#8899aa')
ax_d.set_ylabel('Vectors per shell', color='#8899aa')
ax_d.set_ylim(0, 8)

# Cumulative line on twin axis
ax_d2 = ax_d.twinx()
ax_d2.plot(range(len(shell_data)), cumulative, 'o-', color='#ffcc33', linewidth=2, markersize=6)
ax_d2.set_ylabel('Cumulative count', color='#ffcc33')
ax_d2.set_ylim(0, 48)
ax_d2.axhline(y=44, color='#ffcc33', linestyle='--', alpha=0.4, linewidth=1)
ax_d2.text(len(shell_data)-0.5, 44.5, '44', color='#ffcc33', fontsize=9, fontweight='bold', ha='right')
ax_d2.tick_params(colors='#ffcc33')

# Annotate bars
for bar, count, lbl in zip(bars, counts, labels):
    if count > 0:
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                  f'{count}', ha='center', va='bottom', fontsize=7, color='#c8d6e5')

ax_d.grid(axis='y', alpha=0.1)
ax_d.tick_params(colors='#556677')
for spine in ax_d.spines.values(): spine.set_color('#334466')

# ── Supertitle ────────────────────────────────────────────
fig.suptitle('The 44-Vector $\\mathbb{Z}_3$ Quaternion Orbit Closure\n'
             '$\\mathbf{Quaternions\\ as\\ Quantization}$',
             color='#c8d6e5', fontsize=15, fontweight='bold', y=0.98)

# Attribution line
fig.text(0.5, 0.01, 'Zhang, Hu & Zhang (2026)  ·  arXiv:XXXX.XXXXX',
         ha='center', color='#556688', fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# ── Save ──────────────────────────────────────────────────
outpath = 'fig_quaternion_quantization.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#0a0a14')
print(f'✅ Saved → {outpath}')
print(f'   {len(all_44)} vectors: 5 seeds + 36 V₁ + 3 V₀ = 44')
plt.close()
