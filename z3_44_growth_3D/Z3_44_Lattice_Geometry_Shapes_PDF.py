#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3 44-Vector Vacuum Lattice — GEOMETRY SHAPES CATALOG (PDF)
===========================================================
Take the final 44-vector lattice and connect the points under every natural
algebraic / metric rule, then catalogue all resulting spatial geometric shapes.

Connection rules
  R1  T-orbit        v -> Tv                (Z3 cyclic)          -> equilateral triangle
  R2  antipodal      v <-> -v                                    -> centrosymmetric pair
  R3  same-shell NN  equal radius, nearest                       -> regular hexagon C6
  R4  two triangles in one shell                                 -> hexagram / Star of David
  R5  triality rule  all pairs except antipodal                  -> K2,2,2 (octahedron graph)
  R6  cross product  a x b = c  (c in lattice)                   -> right / equilateral triangle
  R7  democratic     collinear [111],[333],[999]                 -> line / chain
  R8  orthogonality  v . w = 0                                   -> orthogonal web
  R9  translation    v - w  in  lattice                          -> lattice bond graph
  R10 concentric     shells at |v| = 3^(k/2)*sqrt(2) ...         -> nested shells

Output: Z3_44_Lattice_Geometry_Shapes.pdf
"""

import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from collections import defaultdict, Counter

# ---------------- font ----------------
for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. lattice
# ============================================================
T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
kk = lambda v, d=6: tuple(np.round(v, d))

basis = np.eye(3)
dem = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
seeds = [basis[0], basis[1], basis[2], dem, -dem]
level_of = {kk(s): 0 for s in seeds}
frontier = [s.copy() for s in seeds]
for lev in range(1, 16):
    new = []
    for v in frontier:
        v1, v2 = T @ v, T @ (T @ v)
        cands = [v1, v2, v1 - v, v2 - v]
        cr = np.cross(v, v1)
        if np.linalg.norm(cr) > 1e-9:
            cands += [cr, cr / np.linalg.norm(cr)]
        for c in cands:
            if np.linalg.norm(c) < 1e-9:
                continue
            if kk(c) not in level_of:
                level_of[kk(c)] = lev
                new.append(c)
    frontier = new

allv = [(np.array(k), l) for k, l in level_of.items()]
allv.sort(key=lambda x: (round(float(np.linalg.norm(x[0])), 4), float(np.sum(np.abs(x[0])))))
V = np.array([v for v, l in allv[:44]])
LEV = np.array([l for v, l in allv[:44]])
N = 44
KV = {kk(v): i for i, v in enumerate(V)}
D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)

shells = defaultdict(list)
for i in range(N):
    shells[round(float(V[i] @ V[i]), 4)].append(i)
ROOT_L2 = [2.0, 6.0, 18.0, 54.0, 162.0, 486.0]
DEM_L2 = [3.0, 27.0, 243.0]

# T-orbits
def orbit_of(i):
    o = [i]
    w = T @ V[i]
    while kk(w) != kk(V[i]):
        o.append(KV[kk(w)])
        w = T @ w
    return o

orbits, seen_o = [], set()
for i in range(N):
    if i in seen_o:
        continue
    o = orbit_of(i)
    seen_o.update(o)
    orbits.append(o)
tri_orbits = [o for o in orbits if len(o) == 3]
sing_orbits = [o for o in orbits if len(o) == 1]

# rules
anti = [(i, j) for i in range(N) for j in range(i + 1, N) if np.linalg.norm(V[i] + V[j]) < 1e-6]
orth = [(i, j) for i in range(N) for j in range(i + 1, N) if abs(V[i] @ V[j]) < 1e-6]

cross_tri = set()
for i in range(N):
    for j in range(i + 1, N):
        c = np.cross(V[i], V[j])
        if np.linalg.norm(c) < 1e-9:
            continue
        for s in (1, -1):
            k = kk(s * c)
            if k in KV:
                u, w_, x = sorted((i, j, KV[k]))
                cross_tri.add((u, w_, x))

delta_edges = set()
for i in range(N):
    for j in range(N):
        if i != j and kk(V[i] - V[j]) in KV:
            delta_edges.add(tuple(sorted((i, j))))
delta_edges = sorted(delta_edges)

hex_edges = []
for L2 in ROOT_L2:
    idx = shells[L2]
    for a, b in itertools.combinations(idx, 2):
        if abs(D[a, b] - np.sqrt(L2)) < 1e-6:
            hex_edges.append((a, b))

k22_edges = []
for L2 in ROOT_L2:
    idx = shells[L2]
    for a, b in itertools.combinations(idx, 2):
        if np.linalg.norm(V[a] + V[b]) < 1e-6:
            continue
        k22_edges.append((a, b))

# ============================================================
# 2. style
# ============================================================
SEED_BASIS = "#0e8fd6"
SEED_DEM = "#e0a800"
ROOT_COLORS = ["#e03050", "#e86030", "#e89030", "#c8a030", "#a08028", "#806020"]
DEM_COLORS = ["#e0a800", "#d8b830", "#c8c060"]


def pcolor(i):
    v, l = V[i], LEV[i]
    L2 = round(float(v @ v), 4)
    isdem = (abs(v[0] - v[1]) < 1e-6 and abs(v[1] - v[2]) < 1e-6 and abs(v[0]) > 1e-6)
    if l == 0:
        return SEED_DEM if isdem else SEED_BASIS
    if isdem:
        return DEM_COLORS[min(DEM_L2.index(L2), 2)] if L2 in DEM_L2 else DEM_COLORS[0]
    return ROOT_COLORS[ROOT_L2.index(L2)] if L2 in ROOT_L2 else "#888"


def shell_color(L2):
    if L2 in ROOT_L2:
        return ROOT_COLORS[ROOT_L2.index(L2)]
    if L2 in DEM_L2:
        return DEM_COLORS[min(DEM_L2.index(L2), 2)]
    return "#333"


# ============================================================
# 3. page machinery
# ============================================================
PAGES = []


def page(fn):
    PAGES.append(fn)
    return fn


def new_fig(title_cn, title_en, note=""):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.962, title_cn, ha="center", va="top", fontsize=16, fontweight="bold", color="#111")
    if title_en:
        fig.text(0.5, 0.925, title_en, ha="center", va="top", fontsize=10.5, color="#456")
    if note:
        fig.text(0.5, 0.895, note, ha="center", va="top", fontsize=9.5, color="#666")
    return fig


def mk_ax(fig, rect, view=(-58, 22), limits=None, title=""):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_facecolor("#fbfbfd")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.pane.set_facecolor((0.97, 0.97, 0.99, 1.0))
        p.pane.set_edgecolor((0.78, 0.78, 0.84, 0.8))
        p.line.set_color((0.72, 0.72, 0.8, 0.9))
    ax.grid(False)
    ax.view_init(elev=view[1], azim=view[0])
    if limits:
        ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_zlim(*limits)
    if title:
        ax.set_title(title, fontsize=10.5, color="#223", pad=2)
    return ax


def add_edges(ax, edges, color, lw=1.2, alpha=0.6, ls="-"):
    if not edges:
        return
    segs = [(V[i], V[j]) for i, j in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha,
                                         linestyles=ls))


def add_points(ax, idxs=None, sizes=None, highlight=None):
    if idxs is None:
        idxs = range(N)
    for i in idxs:
        s = 40 if LEV[i] == 0 else max(16, 40 - 2.6 * LEV[i])
        if sizes is not None:
            s = sizes
        edge, lw = ("none", 0)
        if highlight is not None and i == highlight:
            s, edge, lw = s * 2.6, "#111", 1.2
        ax.scatter([V[i][0]], [V[i][1]], [V[i][2]], s=s, c=pcolor(i),
                   alpha=0.95, edgecolors=edge, linewidths=lw, depthshade=True)


def footer(fig, txt):
    fig.text(0.5, 0.018, txt, ha="center", va="bottom", fontsize=9, color="#777")
    fig.text(0.02, 0.018, "Z3 44-vector lattice", fontsize=8, color="#aaa")
    fig.text(0.98, 0.018, f"{len(PAGES)+1}", fontsize=8, color="#aaa", ha="right")


# ============================================================
# 4. pages
# ============================================================
@page
def p_cover(pdf):
    fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.93, "Z₃ 44-晶格：空间几何形状图鉴", ha="center", fontsize=23, fontweight="bold", color="#111")
    fig.text(0.5, 0.885, "The 44-Vector Vacuum Lattice — Catalogue of All Spatial Geometries",
             ha="center", fontsize=12, color="#456")
    fig.text(0.5, 0.845, "5 seeds  →  Z₃ triality closure  →  44 vectors / 10 shells",
             ha="center", fontsize=11, color="#777")

    rows = [
        ("R1  T-轨道  v→Tv (Z₃)", "等边三角形", f"{len(tri_orbits)} 个三角形 + {len(sing_orbits)} 个民主不动点"),
        ("R2  对跖配对  v↔−v", "中心对称对", f"{len(anti)} 对"),
        ("R3  同壳最近邻 (等半径)", "正六边形 C₆", f"{len(shells)} 壳；边长=半径"),
        ("R4  同壳两个三角叠加", "六角星 ✡ (Star of David)", f"{len(ROOT_L2)} 个根壳"),
        ("R5  三性规则 除对跖全连", "K₂,₂,₂ 八面体图", f"{len(k22_edges)} 条边（抽象八面体）"),
        ("R6  叉乘闭合  a×b=c", "直角 / 等边三角形", f"{len(cross_tri)} 个闭合三角"),
        ("R7  民主链  [111]→[333]→[999]", "共线直线", "3 点共线（+种子 ±）"),
        ("R8  正交  v·w=0", "正交网 / 三脚架", f"{len(orth)} 对正交"),
        ("R9  平移闭合  v−w ∈ L", "晶格键图", f"{len(delta_edges)} 条键"),
        ("R10 同心嵌套", "嵌套壳层 (比例 √3)", f"{len(shells)} 层"),
    ]
    y = 0.775
    fig.text(0.10, y, "规则", fontsize=11, fontweight="bold", color="#111")
    fig.text(0.34, y, "几何形状", fontsize=11, fontweight="bold", color="#111")
    fig.text(0.62, y, "数量 / 说明", fontsize=11, fontweight="bold", color="#111")
    fig.lines.append(plt.Line2D([0.10, 0.90], [y - 0.008, y - 0.008], color="#999", lw=0.8))
    y -= 0.045
    for r, s, c in rows:
        fig.text(0.10, y, r, fontsize=9.8, color="#223")
        fig.text(0.34, y, s, fontsize=9.8, color="#a11", fontweight="bold")
        fig.text(0.62, y, c, fontsize=9.8, color="#555")
        y -= 0.043
    fig.text(0.5, 0.132, "壳层结构：  基矢壳 L²=1 (5)   根壳 L²=2·3ᵏ, k=0..5 (6×6=36)   民主壳 L²=3ᵏ, k=1..3 (3)",
             ha="center", fontsize=10, color="#333")
    fig.text(0.5, 0.098, "44 = 5 (种子) + 36 (6 个根壳×6) + 3 (3 个民主壳×1)",
             ha="center", fontsize=10, color="#333")
    fig.text(0.5, 0.055, "注：根壳 6 矢量共面（全部 ⊥ (1,1,1)）→ 实际嵌入是平面正六边形；"
                         "论文中的「八面体」是三性规则下的抽象图 K₂,₂,₂（理想实现为 ±e₁,±e₂,±e₃）。",
             ha="center", fontsize=9, color="#a11")
    footer(fig, "封面 / 规则总览")
    pdf.savefig(fig); plt.close(fig)


@page
def p_overview(pdf):
    fig = new_fig("全 44 点 + 全部规则并集",
                  "All 44 vectors with the union of every connection rule",
                  "左侧：点 + 全部连线并集（R1–R9）。右侧：按壳层着色的点云 + 同心壳层示意。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], limits=(-25, 25), title="全部规则并集 (union of R1–R9)")
    allE = set(hex_edges) | set(k22_edges) | set(delta_edges) | set(anti) | set(orth)
    for a, b, c in cross_tri:
        allE |= {(a, b), (b, c), (a, c)}
    for o in tri_orbits:
        allE |= {(o[0], o[1]), (o[1], o[2]), (o[2], o[0])}
    allE = {tuple(sorted(e)) for e in allE}
    add_edges(ax, list(allE), "#5566aa", lw=0.7, alpha=0.35)
    add_points(ax)

    ax2 = mk_ax(fig, [0.53, 0.08, 0.45, 0.79], limits=(-25, 25), title="壳层着色 (10 shells)")
    for L2 in sorted(shells):
        idx = shells[L2]
        r = np.sqrt(L2)
        th = np.linspace(0, 2 * np.pi, 60)
        # circle drawn in the plane perpendicular to (1,1,1) for root/dem shells, else xy
        n = np.array([1, 1, 1]) / np.sqrt(3) if L2 != 1.0 else np.array([0, 0, 1.0])
        a1 = np.cross(n, [1, 0, 0]);  a1 /= np.linalg.norm(a1)
        a2 = np.cross(n, a1)
        pts = np.outer(np.cos(th) * r, a1) + np.outer(np.sin(th) * r, a2)
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=shell_color(L2), lw=0.9, alpha=0.5)
        add_edges(ax2, [(idx[0], idx[k]) for k in range(1, len(idx))], shell_color(L2), lw=1.4, alpha=0.8)
    add_points(ax2)
    footer(fig, f"并集连线 {len(allE)} 条   |   正交 {len(orth)}   平移键 {len(delta_edges)}   对跖 {len(anti)}   三性边 {len(k22_edges)}   六边形边 {len(hex_edges)}")
    pdf.savefig(fig); plt.close(fig)


@page
def p_hexagon_nested(pdf):
    fig = new_fig("R3/R10  根壳正六边形 & 同心嵌套",
                  "Root shells: regular hexagons  (C6)  nested with ratio √3",
                  "每个根壳的 6 个矢量等长、共面、且全部 ⊥ (1,1,1) → 平面正六边形。相邻壳半径比 = √3。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title="全部 6 个根壳 (nested hexagons)")
    for L2 in ROOT_L2:
        idx = shells[L2]
        add_edges(ax, [(idx[a], idx[b]) for a, b in itertools.combinations(range(6), 2)
                       if (idx[a], idx[b]) in hex_edges or (idx[b], idx[a]) in hex_edges],
                  shell_color(L2), lw=1.6, alpha=0.85)
        add_edges(ax, [(idx[0], idx[k]) for k in range(1, 6)], shell_color(L2), lw=0.5, alpha=0.25, ls=":")
    add_points(ax)

    ax2 = mk_ax(fig, [0.51, 0.08, 0.24, 0.79], view=(0, 90), limits=(-4.5, 4.5), title="L²=2 壳 俯视")
    idx = shells[2.0]
    add_edges(ax2, [(idx[a], idx[b]) for a, b in itertools.combinations(range(6), 2)
                    if (idx[a], idx[b]) in hex_edges or (idx[b], idx[a]) in hex_edges],
              shell_color(2.0), lw=2.0, alpha=0.9)
    add_points(ax2, idx)
    ax3 = mk_ax(fig, [0.76, 0.08, 0.23, 0.79], view=(0, 90), limits=(-24, 24), title="全部 6 壳 俯视")
    for L2 in ROOT_L2:
        idx = shells[L2]
        add_edges(ax3, [(idx[a], idx[b]) for a, b in itertools.combinations(range(6), 2)
                        if (idx[a], idx[b]) in hex_edges or (idx[b], idx[a]) in hex_edges],
                  shell_color(L2), lw=1.3, alpha=0.85)
    add_points(ax3)
    footer(fig, "半径: " + "  <  ".join(f"{np.sqrt(L2):.2f}" for L2 in ROOT_L2) + "   (每级 ×√3)")
    pdf.savefig(fig); plt.close(fig)


@page
def p_hexagram(pdf):
    fig = new_fig("R4  六角星 ✡  (同壳两个 Z₃ 三角形叠加)",
                  "Hexagram (Star of David): the two T-orbits inside one root shell",
                  "每个根壳的 6 点 = 两个 T-轨道（各 3 点，等边三角形）。两三角错开 60° 叠加 → 六角星。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(0, 90), limits=(-4.5, 4.5), title="L²=2 壳：two triangles → ✡")
    idx = shells[2.0]
    orbs = [o for o in tri_orbits if set(o) <= set(idx)]
    cols = ["#e03050", "#0e8fd6"]
    for o, c in zip(orbs, cols):
        tri = [(o[0], o[1]), (o[1], o[2]), (o[2], o[0])]
        add_edges(ax, tri, c, lw=2.2, alpha=0.9)
    add_points(ax, idx)

    ax2 = mk_ax(fig, [0.51, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title="全部根壳的 12 个三角形")
    pal = ["#e03050", "#0e8fd6", "#e89030", "#7a4fd0", "#2aa060", "#c8a030",
           "#d04080", "#3a80c0", "#a0603a", "#508060", "#8060a0", "#c06060"]
    for n, o in enumerate([o for o in tri_orbits if o[0] in sum([shells[L2] for L2 in ROOT_L2], [])]):
        c = pal[n % len(pal)]
        add_edges(ax2, [(o[0], o[1]), (o[1], o[2]), (o[2], o[0])], c, lw=1.6, alpha=0.8)
    add_points(ax2)
    footer(fig, f"T-轨道三角形共 {len([o for o in tri_orbits if o[0] in sum([shells[L2] for L2 in ROOT_L2], [])])} 个（= 6 壳 × 2）")
    pdf.savefig(fig); plt.close(fig)


@page
def p_seed_shapes(pdf):
    fig = new_fig("种子壳的三种形状：正交三脚架 / 对跖二重态 / 民主链",
                  "Seed-shell geometries: orthonormal triad, antipodal pair, democratic line",
                  "R8 三脚架 (e₁,e₂,e₃ 两两正交)   R2 对跖 (±(1,1,1)/√3)   R7 共线民主链 [111]→[333]→[999]")
    ax = mk_ax(fig, [0.03, 0.08, 0.30, 0.79], limits=(-2.2, 2.2), title="① 正交三脚架")
    b = [KV[kk(basis[k])] for k in range(3)]
    add_edges(ax, [(b[0], b[1]), (b[1], b[2]), (b[2], b[0])], "#0e8fd6", lw=2.0, alpha=0.9)
    add_edges(ax, [(i, KV[kk(np.zeros(3))]) for i in b if kk(np.zeros(3)) in KV], "#0e8fd6", lw=0.6, alpha=0.3, ls=":")
    add_points(ax, b)
    ax.scatter([0], [0], [0], s=20, c="#aaa")

    ax2 = mk_ax(fig, [0.37, 0.08, 0.28, 0.79], limits=(-1.2, 1.2), title="② 对跖二重态 ±")
    dp, dm = KV[kk(dem)], KV[kk(-dem)]
    add_edges(ax2, [(dp, dm)], "#e0a800", lw=2.2, alpha=0.9)
    add_points(ax2, [dp, dm])

    ax3 = mk_ax(fig, [0.68, 0.08, 0.29, 0.79], limits=(-1, 10), title="③ 共线民主链")
    chain = [KV[kk(np.array([n, n, n], float))] for n in (1, 3, 9)]
    add_edges(ax3, [(chain[0], chain[1]), (chain[1], chain[2])], "#e0a800", lw=2.0, alpha=0.9)
    add_edges(ax3, [(KV[kk(dem)], KV[kk(-dem)])], "#e0a800", lw=0.8, alpha=0.4, ls=":")
    add_points(ax3, chain + [KV[kk(dem)], KV[kk(-dem)]])
    footer(fig, "三种种子形状：正交三脚架 (3 点)  |  对跖二重态 (2 点)  |  共线民主链 (3 点)")
    pdf.savefig(fig); plt.close(fig)


@page
def p_octahedron(pdf):
    fig = new_fig("R5  三性规则 K₂,₂,₂：抽象八面体  vs  实际平面六边形",
                  "Triality rule K2,2,2: abstract octahedron vs the actual coplanar realisation",
                  "除对跖对外全连 → 完全二分图 K₂,₂,₂ = 八面体图。理想实现 ±e₁,±e₂,±e₃ (左)；"
                  "实际根壳 6 矢量共面 → 退化为平面六边形+对角线 (右)。")
    OV = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.]])
    OE = [(i, j) for i in range(6) for j in range(i + 1, 6) if i // 2 != j // 2]
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], limits=(-1.3, 1.3), title="理想八面体 ±e₁,±e₂,±e₃  (K₂,₂,₂, 12 边 8 面)")
    ax.add_collection3d(Line3DCollection([(OV[i], OV[j]) for i, j in OE], colors="#c02040", linewidths=1.8, alpha=0.8))
    for p in OV:
        ax.scatter([p[0]], [p[1]], [p[2]], s=60, c="#c02040", alpha=0.95)
    fax = [OV[t] for t in ([0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5])]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(fax, facecolor="#c02040", alpha=0.08, edgecolor="#c02040", linewidths=0.4))

    ax2 = mk_ax(fig, [0.51, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-2.2, 2.2), title="实际 L²=2 根壳：同一 K₂,₂,₂ 规则")
    idx = shells[2.0]
    add_edges(ax2, [(idx[a], idx[b]) for a, b in itertools.combinations(range(6), 2) if (idx[a], idx[b]) in k22_edges or (idx[b], idx[a]) in k22_edges], "#c02040", lw=1.6, alpha=0.8)
    add_points(ax2, idx)
    footer(fig, "K₂,₂,₂ 图: 6 顶点 / 12 边 / 8 面；拉普拉斯谱 {0,4,4,4,6,6} = 1⊕3⊕2 = U(1)⊕SU(2)⊕SU(3)")
    pdf.savefig(fig); plt.close(fig)


@page
def p_orbits(pdf):
    fig = new_fig("R1  T-轨道：等边三角形（Z₃ 三阶循环）",
                  "T-orbits: equilateral triangles from the Z3 cyclic symmetry v→Tv",
                  "每个 size-3 轨道在 ⊥(1,1,1) 平面内构成等边三角形；民主点是不动点 (size-1)。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title="全部 14 个 T-轨道三角形")
    pal = ["#e03050", "#0e8fd6", "#e89030", "#7a4fd0", "#2aa060", "#c8a030",
           "#d04080", "#3a80c0", "#a0603a", "#508060", "#8060a0", "#c06060", "#4090a0", "#b07040"]
    for n, o in enumerate(tri_orbits):
        c = pal[n % len(pal)]
        add_edges(ax, [(o[0], o[1]), (o[1], o[2]), (o[2], o[0])], c, lw=1.8, alpha=0.85)
    for o in sing_orbits:
        ax.scatter([V[o[0]][0]], [V[o[0]][1]], [V[o[0]][2]], s=70, c="#e0a800", marker="*", alpha=1.0)
    add_points(ax)

    ax2 = mk_ax(fig, [0.51, 0.08, 0.45, 0.79], view=(0, 90), limits=(-4.5, 4.5), title="L²=2 壳中的 2 个三角形（俯视）")
    for o, c in zip([o for o in tri_orbits if set(o) <= set(shells[2.0])], ["#e03050", "#0e8fd6"]):
        add_edges(ax2, [(o[0], o[1]), (o[1], o[2]), (o[2], o[0])], c, lw=2.2, alpha=0.9)
    add_points(ax2, shells[2.0])
    footer(fig, f"T-轨道: {len(tri_orbits)} 个等边三角形 + {len(sing_orbits)} 个民主不动点 (★)")
    pdf.savefig(fig); plt.close(fig)


@page
def p_cross(pdf):
    fig = new_fig("R6  叉乘闭合三角形  a × b = c",
                  "Cross-product closed triangles: a × b = c with c also in the lattice",
                  f"共 {len(cross_tri)} 个三元组满足叉乘闭合。例：(-1,1,0)×(0,-1,1)=(1,1,1)（种子民主矢量的来源）。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title=f"全部 {len(cross_tri)} 个叉乘闭合三角")
    for a, b, c in cross_tri:
        add_edges(ax, [(a, b), (b, c), (c, a)], "#7040c0", lw=0.9, alpha=0.35)
    add_points(ax)

    ax2 = mk_ax(fig, [0.51, 0.08, 0.45, 0.79], limits=(-1.6, 1.6), title="典型例：种子民主矢量的诞生")
    i1, i2 = KV[kk(np.array([-1., 1, 0]))], KV[kk(np.array([0., -1, 1]))]
    i3 = KV[kk(np.array([1., 1, 1]))]
    add_edges(ax2, [(i1, i2)], "#0e8fd6", lw=2.2, alpha=0.9)
    add_edges(ax2, [(i1, i3), (i2, i3)], "#7040c0", lw=1.6, alpha=0.7)
    add_points(ax2, [i1, i2, i3], highlight=i3)
    footer(fig, "叉乘闭合：a×b 仍落在 44 晶格内 → 代数在晶格上封闭")
    pdf.savefig(fig); plt.close(fig)


@page
def p_orth(pdf):
    fig = new_fig("R8  正交图  v · w = 0",
                  "Orthogonality graph",
                  f"共 {len(orth)} 对正交矢量。种子壳内 (e₁,e₂,e₃) 是唯一「正交且等长」的三脚架（3 对）。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title=f"正交图 ({len(orth)} 条边)")
    add_edges(ax, orth, "#2aa060", lw=0.7, alpha=0.3)
    add_points(ax)
    ax2 = mk_ax(fig, [0.51, 0.08, 0.45, 0.79], limits=(-1.4, 1.4), title="正交三脚架 (e₁,e₂,e₃)")
    b = [KV[kk(basis[k])] for k in range(3)]
    add_edges(ax2, [(b[0], b[1]), (b[1], b[2]), (b[2], b[0])], "#2aa060", lw=2.0, alpha=0.9)
    add_points(ax2, b)
    footer(fig, f"正交对 {len(orth)}   其中「正交且等长」3 对（即 e₁⊥e₂⊥e₃）")
    pdf.savefig(fig); plt.close(fig)


@page
def p_anti(pdf):
    fig = new_fig("R2  对跖配对（中心对称）",
                  "Antipodal pairs — the ± degeneracy carried by the lattice",
                  f"共 {len(anti)} 对 v ↔ −v（19 对）。它们是 CP 破缺 / 真空二重性的几何载体。")
    ax = mk_ax(fig, [0.03, 0.08, 0.45, 0.79], view=(-58, 24), limits=(-25, 25), title=f"{len(anti)} 条对跖轴")
    add_edges(ax, anti, "#d04080", lw=1.2, alpha=0.5)
    add_points(ax)
    # histogram of pair lengths
    ax2 = fig.add_axes([0.55, 0.16, 0.38, 0.62])
    x = sorted(np.linalg.norm(V[i]) for i, j in anti)
    ax2.hist(x, bins=np.arange(0, 24, 1.0), color="#d04080", alpha=0.75, edgecolor="white")
    ax2.set_xlabel("|v|  (对跖对的半径)", fontsize=10)
    ax2.set_ylabel("对数", fontsize=10)
    ax2.set_title("对跖对的半径分布", fontsize=11)
    ax2.grid(alpha=0.25)
    footer(fig, "19 对 = 1 (民主种子 ±) + 18 (6 个根壳 × 3 对)")
    pdf.savefig(fig); plt.close(fig)


@page
def p_delta(pdf):
    fig = new_fig("R9  平移闭合键图  v − w ∈ 晶格",
                  "Translation-closure bond graph: an edge whenever v − w is again a lattice vector",
                  f"共 {len(delta_edges)} 条键。这是 44 点在「差矢量仍闭合」规则下最完整的骨架。")
    ax = mk_ax(fig, [0.03, 0.08, 0.47, 0.79], view=(-58, 24), limits=(-25, 25), title=f"平移闭合键 ({len(delta_edges)} 边)")
    add_edges(ax, delta_edges, "#3a80c0", lw=0.8, alpha=0.3)
    add_points(ax)
    ax2 = mk_ax(fig, [0.53, 0.08, 0.44, 0.79], view=(0, 90), limits=(-4.5, 4.5), title="内层：L²≤6 的键 (俯视)")
    inner = [e for e in delta_edges if max(np.linalg.norm(V[e[0]]), np.linalg.norm(V[e[1]])) <= 2.6]
    add_edges(ax2, inner, "#3a80c0", lw=1.6, alpha=0.7)
    add_points(ax2, [i for i in range(N) if np.linalg.norm(V[i]) <= 2.6])
    footer(fig, "键规则：只要 v−w 仍是 44 晶格中的矢量就连一条边 → 晶格的平移闭合结构")
    pdf.savefig(fig); plt.close(fig)


@page
def p_summary(pdf):
    fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, "汇总：44 晶格上「所有」可连出的空间几何形状", ha="center",
             fontsize=18, fontweight="bold", color="#111")
    fig.text(0.5, 0.90, "Summary of every spatial figure formed by connecting the 44 lattice points",
             ha="center", fontsize=11, color="#456")
    rows = [
        ("正交三脚架", "3 点 / 3 边", "e₁,e₂,e₃ 两两正交等长", "R8"),
        ("等边三角形", f"{len(tri_orbits)} 个", "T-轨道 (Z₃ 循环)", "R1"),
        ("正六边形 C₆", "6 个 (每根壳 1 个)", "同壳等半径最近邻", "R3"),
        ("六角星 ✡", "6 个", "同壳两三角叠加", "R4"),
        ("八面体 K₂,₂,₂", "6 个 (抽象)", "除对跖全连；实际退化平面", "R5"),
        ("直角/叉乘三角形", f"{len(cross_tri)} 个", "a×b=c 闭合", "R6"),
        ("对跖二重态", f"{len(anti)} 对", "v↔−v 中心对称", "R2"),
        ("共线民主链", "1 条 (3 点)", "[111]→[333]→[999]", "R7"),
        ("正交网", f"{len(orth)} 对", "v·w=0", "R8"),
        ("平移闭合键图", f"{len(delta_edges)} 边", "v−w ∈ L", "R9"),
        ("同心嵌套壳层", "10 层", "半径比 √3 (根壳)", "R10"),
    ]
    y = 0.845
    cols = [0.07, 0.30, 0.48, 0.72, 0.86]
    for t, x in zip(["形状", "规模", "生成规则", "规则编号"], cols):
        fig.text(x, y, t, fontsize=11.5, fontweight="bold", color="#111")
    fig.lines.append(plt.Line2D([0.07, 0.93], [y - 0.008, y - 0.008], color="#999", lw=0.8))
    y -= 0.048
    for name, size, rule, tag in rows:
        fig.text(cols[0], y, name, fontsize=10.5, color="#a11", fontweight="bold")
        fig.text(cols[1], y, size, fontsize=10, color="#333")
        fig.text(cols[2], y, rule, fontsize=10, color="#333")
        fig.text(cols[3], y, tag, fontsize=10, color="#777")
        y -= 0.045
    fig.text(0.5, 0.14, "核心结论：", ha="center", fontsize=12, fontweight="bold", color="#111")
    for i, t in enumerate([
        "• 5 个种子生成 44 个点；所有「连线」结果只落到有限几种形状上（三角形 / 六边形 / 六角星 / 八面体图 / 对跖对 / 共线链 / 正交网 / 嵌套壳）。",
        "• 三性规则 K₂,₂,₂ 给出抽象八面体（谱 {0,4,4,4,6,6}=1⊕3⊕2）；但根壳矢量共面，实际嵌入是平面正六边形。",
        "• 因此「八面体」是图的对称性，不是嵌入的凸包——凸包是平面六边形。这一点在正文中需明确区分。",
    ]):
        fig.text(0.09, 0.108 - i * 0.032, t, fontsize=9.8, color="#333")
    footer(fig, "汇总表")
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# 5. render
# ============================================================
out = "Z3_44_Lattice_Geometry_Shapes.pdf"
with PdfPages(out) as pdf:
    for i, fn in enumerate(PAGES, 1):
        fn(pdf)
        print(f"  page {i}/{len(PAGES)}  {fn.__name__}")

print(f"\n✅ saved {out}  ({len(PAGES)} pages, {os.path.getsize(out)/1e6:.2f} MB)")
