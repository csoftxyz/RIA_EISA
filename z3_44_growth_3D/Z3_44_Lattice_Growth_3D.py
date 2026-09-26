#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3 44-Vector Vacuum Lattice  —  3D growth animation (point by point)
===================================================================
Five seeds  ->  Z3 triality closure  ->  44-vector, 10-shell lattice.

Growth story (每一点逐个出现，每一步空间可视化):
  Gen 0 :  5 seeds            L^2 = 1        (3 basis  e1,e2,e3  +  2 democratic +-(1,1,1)/v3)
  Gen 1 :  6 root vectors     L^2 = 2        K2,2,2 octahedron        (from  Delta: v -> Tv - v)
  Gen 2 :  1 + 6              L^2 = 3, 6     [111] democratic  +  v3-scaled octahedron
  Gen 3 :  6 + 1              L^2 = 18, 27   octahedron       +  [333]
  Gen 4 :  6 + 1              L^2 = 54, 243  octahedron       +  [999]
  Gen 5 :  6                  L^2 = 162      octahedron
  Gen 6 :  6                  L^2 = 486      octahedron
  ---------------------------------------------------------------------
  total  =  5 + 6 + 7 + 7 + 7 + 6 + 6  =  44

Left panel  : true scale (真空晶格真实尺度, |v| up to sqrt(486) ~ 22)
Right panel : normalized directions on the unit sphere (角度结构, octahedron visible)

Output: Z3_44_Lattice_Growth.gif
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ----------------------------------------------------------------------
# 0.  CJK font
# ----------------------------------------------------------------------
for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 1.  Z3 triality closure  ->  generation level of every vector
# ----------------------------------------------------------------------
T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)     # cyclic permutation  (x,y,z)->(z,x,y)


def kk(v, d=6):
    return tuple(np.round(v, d))


basis = np.eye(3)
dem = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
seed_vecs = [basis[0], basis[1], basis[2], dem, -dem]
seed_names = ["e1=(1,0,0)", "e2=(0,1,0)", "e3=(0,0,1)",
              "d+=(1,1,1)/√3", "d-=-(1,1,1)/√3"]

level_of = {}
for s in seed_vecs:
    level_of[kk(s)] = 0

frontier = [s.copy() for s in seed_vecs]
for lev in range(1, 16):
    new = []
    for v in frontier:
        v1 = T @ v
        v2 = T @ v1
        cands = [v1, v2, v1 - v, v2 - v]          # T / T^2 / Δ / Δ∘T
        cr = np.cross(v, v1)                       # × : v × Tv
        if np.linalg.norm(cr) > 1e-9:
            cands += [cr, cr / np.linalg.norm(cr)]
        for c in cands:
            if np.linalg.norm(c) < 1e-9:
                continue
            k = kk(c)
            if k not in level_of:
                level_of[k] = lev
                new.append(c)
    frontier = new

# ---- keep the 44 shortest / simplest vectors -------------------------
allv = [(np.array(k), l) for k, l in level_of.items()]
allv.sort(key=lambda x: (round(float(np.linalg.norm(x[0])), 4),
                         float(np.sum(np.abs(x[0])))))
top44 = allv[:44]

# ---- group by generation level, then by shell L^2 --------------------
gens = {}
for v, l in top44:
    gens.setdefault(l, []).append(v)

GEN_INFO = {
    0: ("5 个种子 — 原始输入",          "5 seeds: 3 Cartesian basis + 2 democratic ±"),
    1: ("第 1 代 — Δ 差分产生八面体",    "Δ: v→Tv−v  →  K₂,₂,₂ octahedron  (L²=2)"),
    2: ("第 2 代 — × 叉乘产生民主方向",  "× : v×Tv  →  [1,1,1]  +  √3-scaled octahedron"),
    3: ("第 3 代 — 尺度跳一级",          "L²=18 octahedron  +  [3,3,3]"),
    4: ("第 4 代 — 尺度再跳一级",        "L²=54 octahedron  +  [9,9,9]"),
    5: ("第 5 代 — 外层根壳",            "L²=162 octahedron"),
    6: ("第 6 代 — 外壳收束 (饱和)",      "L²=486 octahedron  →  total = 44 vectors"),
}

# ---- build the ordered growth list ----------------------------------
growth = []            # each item: (vector, level, is_democratic, shell_L2)
for lev in sorted(gens):
    vs = gens[lev]
    # democratic (all-equal components) first inside a mixed generation
    vs = sorted(vs, key=lambda v: (0 if (abs(v[0]-v[1]) < 1e-6 and abs(v[1]-v[2]) < 1e-6 and abs(v[0]) > 1e-6) else 1,
                                   float(np.sum(v**2))))
    for v in vs:
        L2 = float(np.dot(v, v))
        is_dem = (abs(v[0] - v[1]) < 1e-6 and abs(v[1] - v[2]) < 1e-6 and abs(v[0]) > 1e-6)
        growth.append((v, lev, is_dem, round(L2, 4)))

assert len(growth) == 44, len(growth)

# ----------------------------------------------------------------------
# 2.  styling helpers
# ----------------------------------------------------------------------
SEED_BASIS = "#4dc9f6"      # cyan
SEED_DEM = "#ffd700"        # gold
ROOT_COLORS = ["#ff4466", "#ff7744", "#ffaa44", "#ffcc66", "#e0b060", "#c09040"]
DEM_COLORS = ["#ffd700", "#ffe066", "#fff2b0"]


def root_color(L2):
    root_L2 = [2.0, 6.0, 18.0, 54.0, 162.0, 486.0]
    try:
        i = root_L2.index(round(L2, 4))
    except ValueError:
        i = 0
    return ROOT_COLORS[i % len(ROOT_COLORS)]


def dem_color(L2):
    dem_L2 = [3.0, 27.0, 243.0]
    try:
        i = dem_L2.index(round(L2, 4))
    except ValueError:
        i = 0
    return DEM_COLORS[i % len(DEM_COLORS)]


def point_color(v, lev, is_dem):
    L2 = round(float(np.dot(v, v)), 4)
    if lev == 0:
        return SEED_DEM if is_dem else SEED_BASIS
    return dem_color(L2) if is_dem else root_color(L2)


# ---- octahedron (K2,2,2) edges for a set of 6 vectors ---------------
def octa_edges(vecs):
    e = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if np.linalg.norm(vecs[i] + vecs[j]) < 1e-4:   # antipodal -> no edge
                continue
            e.append((vecs[i], vecs[j]))
    return e


ROOT_L2 = [2.0, 6.0, 18.0, 54.0, 162.0, 486.0]

# ----------------------------------------------------------------------
# 3.  figure
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(13.2, 6.4), dpi=100)
fig.patch.set_facecolor("#07070f")

axL = fig.add_axes([0.01, 0.06, 0.47, 0.86], projection="3d")   # true scale
axR = fig.add_axes([0.50, 0.06, 0.47, 0.86], projection="3d")   # unit sphere

for ax, title in ((axL, "真实尺度  True scale"), (axR, "单位方向球  Unit direction sphere")):
    ax.set_facecolor("#07070f")
    ax.set_title(title, color="#9ab", fontsize=12, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.03, 0.03, 0.08, 1.0))
        pane.pane.set_edgecolor((0.15, 0.15, 0.3, 0.4))
        pane.line.set_color((0.2, 0.2, 0.35, 0.4))
    ax.grid(False)

# dynamic limit for the true-scale panel (zooms out as the lattice grows)
L = 24.0

# faint wire sphere on the right panel
u = np.linspace(0, 2 * np.pi, 24)
w = np.linspace(0, np.pi, 12)
for th in w[::2]:
    axR.plot(np.sin(th) * np.cos(u), np.sin(th) * np.sin(u), np.cos(th) * np.ones_like(u),
             color="#1a1a33", lw=0.4, alpha=0.5)
for ph in u[::3]:
    axR.plot(np.sin(w) * np.cos(ph), np.sin(w) * np.sin(ph), np.cos(w),
             color="#1a1a33", lw=0.4, alpha=0.5)
axR.set_xlim(-1.15, 1.15); axR.set_ylim(-1.15, 1.15); axR.set_zlim(-1.15, 1.15)

# header / footer text
txt_gen = fig.text(0.5, 0.965, "", ha="center", va="top", color="#ffffff",
                   fontsize=15, fontweight="bold")
txt_sub = fig.text(0.5, 0.925, "", ha="center", va="top", color="#8ad", fontsize=10.5)
txt_stat = fig.text(0.5, 0.022, "", ha="center", va="bottom", color="#cde", fontsize=11)
txt_vec = fig.text(0.5, 0.055, "", ha="center", va="bottom", color="#ffe9a0", fontsize=10.5)

fig.text(0.012, 0.012, "Z₃  44-晶格 生长过程", color="#fff", fontsize=13, fontweight="bold")

# legend
legend_items = [("种子·基矢", SEED_BASIS), ("种子·民主 ±", SEED_DEM),
                ("根壳 L²=2 (八面体)", ROOT_COLORS[0]), ("根壳 L²=162/486", ROOT_COLORS[4]),
                ("民主链 [111]", DEM_COLORS[0])]
for i, (lab, col) in enumerate(legend_items):
    fig.text(0.795, 0.855 - i * 0.028, "● " + lab, color=col, fontsize=8.5)

# ----------------------------------------------------------------------
# 4.  frame state  and  render
# ----------------------------------------------------------------------
N = len(growth)


def build_frames():
    """return list of (n_visible, highlight_index, hold_tag)"""
    frames = []
    for i in range(N):
        # pulse: two frames for the newly added point
        frames.append((i + 1, i, None))
        frames.append((i + 1, i, None))
        # hold a bit when a generation (level) is completed
        if i + 1 < N and growth[i][1] != growth[i + 1][1]:
            for _ in range(3):
                frames.append((i + 1, -1, "hold"))
    frames.append((N, -1, "end"))
    for _ in range(14):
        frames.append((N, -1, "end"))
    return frames


FRAMES = build_frames()


def draw(fidx):
    n_vis, hi, tag = FRAMES[fidx]
    for ax in (axL, axR):
        ax.clear()
        ax.set_facecolor("#07070f")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor((0.03, 0.03, 0.08, 1.0))
            pane.pane.set_edgecolor((0.15, 0.15, 0.3, 0.4))
            pane.line.set_color((0.2, 0.2, 0.35, 0.4))
        ax.grid(False)

    # ---- camera (slow 3D rotation) ----------------------------------
    az = -62 + fidx * 0.75
    el = 20 + 9 * np.sin(fidx * 0.045)
    for ax in (axL, axR):
        ax.view_init(elev=el, azim=az)

    # ---- draw on both panels ----------------------------------------
    vis = growth[:n_vis]

    # true-scale panel geometry: dynamic zoom so every new point is legible
    maxn = max((float(np.linalg.norm(g[0])) for g in vis), default=1.0)
    Ld = max(2.6, 1.18 * maxn)
    axL.set_xlim(-Ld, Ld); axL.set_ylim(-Ld, Ld); axL.set_zlim(-Ld, Ld)

    # unit-sphere wireframe
    for th in w[::2]:
        axR.plot(np.sin(th) * np.cos(u), np.sin(th) * np.sin(u), np.cos(th) * np.ones_like(u),
                 color="#1a1a33", lw=0.4, alpha=0.5)
    for ph in u[::3]:
        axR.plot(np.sin(w) * np.cos(ph), np.sin(w) * np.sin(ph), np.cos(w),
                 color="#1a1a33", lw=0.4, alpha=0.5)
    axR.set_xlim(-1.15, 1.15); axR.set_ylim(-1.15, 1.15); axR.set_zlim(-1.15, 1.15)

    # ---- root-shell octahedra (edges) -------------------------------
    for L2 in ROOT_L2:
        vs = [v for (v, lev, d, l2) in vis if l2 == L2]
        if len(vs) < 6:
            continue
        col = root_color(L2)
        segs_true = [(a, b) for (a, b) in octa_edges(vs)]
        segs_unit = [(a / np.linalg.norm(a), b / np.linalg.norm(b)) for (a, b) in segs_true]
        lc1 = Line3DCollection(segs_true, colors=col, linewidths=0.9, alpha=0.35)
        lc2 = Line3DCollection(segs_unit, colors=col, linewidths=0.9, alpha=0.35)
        axL.add_collection3d(lc1)
        axR.add_collection3d(lc2)

    # ---- democratic chain line --------------------------------------
    dem_pts = [v for (v, lev, d, l2) in vis if d and lev > 0]
    if len(dem_pts) >= 1:
        chain = sorted(dem_pts, key=lambda v: np.linalg.norm(v))
        end = 10.5 * np.array([1, 1, 1]) / np.sqrt(3)
        pts = [np.zeros(3)] + chain + [end]
        P = np.array(pts)
        axL.plot(P[:, 0], P[:, 1], P[:, 2], color="#ffd700", lw=0.9, alpha=0.35, ls="--")
        Q = P / np.linalg.norm(P, axis=1, keepdims=True)
        axR.plot(Q[:, 0], Q[:, 1], Q[:, 2], color="#ffd700", lw=0.9, alpha=0.35, ls="--")

    # ---- points ------------------------------------------------------
    for idx, (v, lev, is_dem, L2) in enumerate(vis):
        col = point_color(v, lev, is_dem)
        nv = np.linalg.norm(v)
        base = 34 if lev == 0 else max(14, 34 - 2.2 * lev)
        size = base
        alpha = 0.95
        edge = "none"
        if idx == hi:
            size = base * 3.4
            alpha = 1.0
            edge = "#ffffff"
        axL.scatter([v[0]], [v[1]], [v[2]], s=size, c=col, alpha=alpha,
                    edgecolors=edge, linewidths=1.1, depthshade=True)
        u_ = v / nv
        axR.scatter([u_[0]], [u_[1]], [u_[2]], s=size, c=col, alpha=alpha,
                    edgecolors=edge, linewidths=1.1, depthshade=True)

    # ---- text --------------------------------------------------------
    lev = growth[min(n_vis - 1, N - 1)][1] if n_vis <= N else 6
    if n_vis >= N:
        lev = 6
    gname, gsub = GEN_INFO.get(lev, ("", ""))
    if n_vis < N and hi >= 0:
        gname, gsub = GEN_INFO.get(growth[hi][1], (gname, gsub))
    txt_gen.set_text(gname)
    txt_sub.set_text(gsub)

    n_root = sum(1 for (v, l, d, l2) in vis if (not d) and l > 0 and l2 in ROOT_L2)
    n_dem = sum(1 for (v, l, d, l2) in vis if d and l > 0)
    n_seed = sum(1 for (v, l, d, l2) in vis if l == 0)
    txt_stat.set_text(
        f"已生成 {n_vis} / 44 个点    |    种子 {n_seed}   根壳点 {n_root}   民主点 {n_dem}    "
        f"|    生成代数: T: v→(z,x,y)   Δ: v→Tv−v   × : v×Tv"
    )

    if hi >= 0:
        v, lev, is_dem, L2 = growth[hi]
        vec = "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"
        kind = "民主 [111] 链" if is_dem else ("种子" if lev == 0 else f"根壳 L²={L2:g}")
        txt_vec.set_text(f"新增 #{hi+1}:  {vec}    →  {kind}")
    else:
        txt_vec.set_text("")

    return []


# ----------------------------------------------------------------------
# 5.  render GIF
# ----------------------------------------------------------------------
out = "Z3_44_Lattice_Growth.gif"
writer = PillowWriter(fps=16)
with writer.saving(fig, out, dpi=100):
    for f in range(len(FRAMES)):
        draw(f)
        writer.grab_frame(facecolor="#07070f")
        if f % 20 == 0:
            print(f"  frame {f+1}/{len(FRAMES)}")

print(f"\n✅ saved {out}  ({len(FRAMES)} frames, "
      f"{os.path.getsize(out)/1e6:.2f} MB)")
