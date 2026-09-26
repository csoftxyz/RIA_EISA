#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3 44-Vector Vacuum Lattice — frame-by-frame PDF (44 pages)
===========================================================
One page per growth step: page k shows the lattice after the k-th vector appears,
with that vector highlighted. Left = true scale, Right = unit direction sphere.

Output: Z3_44_Lattice_Growth_FrameByFrame.pdf  (44 pages, A4 landscape)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------- font ----------
for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- Z3 closure -> generation level ----------
T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def kk(v, d=6):
    return tuple(np.round(v, d))


basis = np.eye(3)
dem = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
seed_vecs = [basis[0], basis[1], basis[2], dem, -dem]

level_of = {}
for s in seed_vecs:
    level_of[kk(s)] = 0
frontier = [s.copy() for s in seed_vecs]
for lev in range(1, 16):
    new = []
    for v in frontier:
        v1 = T @ v
        v2 = T @ v1
        cands = [v1, v2, v1 - v, v2 - v]
        cr = np.cross(v, v1)
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

allv = [(np.array(k), l) for k, l in level_of.items()]
allv.sort(key=lambda x: (round(float(np.linalg.norm(x[0])), 4), float(np.sum(np.abs(x[0])))))
top44 = allv[:44]

gens = {}
for v, l in top44:
    gens.setdefault(l, []).append(v)

GEN_INFO = {
    0: ("5 个种子 — 原始输入", "3 Cartesian basis e1,e2,e3  +  2 democratic ±(1,1,1)/√3"),
    1: ("第 1 代 — Δ 差分产生八面体", "Δ: v→Tv−v   →   K₂,₂,₂ octahedron (L²=2)"),
    2: ("第 2 代 — × 叉乘产生民主方向", "× : v×Tv   →   [1,1,1] + √3-scaled octahedron (L²=3, 6)"),
    3: ("第 3 代 — 尺度跳一级", "octahedron L²=18   +   [3,3,3] democratic"),
    4: ("第 4 代 — 尺度再跳一级", "octahedron L²=54   +   [9,9,9] democratic"),
    5: ("第 5 代 — 外层根壳", "octahedron L²=162"),
    6: ("第 6 代 — 外壳收束（饱和）", "octahedron L²=486   →   total 44 vectors"),
}

growth = []
for lev in sorted(gens):
    vs = gens[lev]
    vs = sorted(vs, key=lambda v: (0 if (abs(v[0]-v[1]) < 1e-6 and abs(v[1]-v[2]) < 1e-6 and abs(v[0]) > 1e-6) else 1,
                                   float(np.sum(v**2))))
    for v in vs:
        is_dem = (abs(v[0]-v[1]) < 1e-6 and abs(v[1]-v[2]) < 1e-6 and abs(v[0]) > 1e-6)
        growth.append((v, lev, is_dem, round(float(np.dot(v, v)), 4)))
assert len(growth) == 44, len(growth)

# ---------- style ----------
SEED_BASIS = "#22b8f0"
SEED_DEM = "#ffd700"
ROOT_COLORS = ["#ff4466", "#ff7744", "#ffaa44", "#ffcc66", "#e0b060", "#c09040"]
DEM_COLORS = ["#ffd700", "#ffe066", "#fff2b0"]
ROOT_L2 = [2.0, 6.0, 18.0, 54.0, 162.0, 486.0]
DEM_L2 = [3.0, 27.0, 243.0]


def root_color(L2):
    i = ROOT_L2.index(round(L2, 4)) if round(L2, 4) in ROOT_L2 else 0
    return ROOT_COLORS[i % len(ROOT_COLORS)]


def dem_color(L2):
    i = DEM_L2.index(round(L2, 4)) if round(L2, 4) in DEM_L2 else 0
    return DEM_COLORS[i % len(DEM_COLORS)]


def point_color(v, lev, is_dem):
    L2 = round(float(np.dot(v, v)), 4)
    if lev == 0:
        return SEED_DEM if is_dem else SEED_BASIS
    return dem_color(L2) if is_dem else root_color(L2)


def octa_edges(vecs):
    e = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if np.linalg.norm(vecs[i] + vecs[j]) < 1e-4:
                continue
            e.append((vecs[i], vecs[j]))
    return e


# ---------- page renderer ----------
def draw_page(pdf, k):
    n_vis = k
    hi = k - 1
    vis = growth[:n_vis]

    fig = plt.figure(figsize=(11.69, 8.27))          # A4 landscape
    fig.patch.set_facecolor("#ffffff")

    axL = fig.add_axes([0.03, 0.10, 0.44, 0.78], projection="3d")
    axR = fig.add_axes([0.53, 0.10, 0.44, 0.78], projection="3d")

    for ax, title in ((axL, "真实尺度  True scale  (|v|≤√486≈22)"),
                      (axR, "单位方向球  Unit direction sphere")):
        ax.set_facecolor("#fbfbfd")
        ax.set_title(title, color="#334", fontsize=11, pad=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor((0.97, 0.97, 0.99, 1.0))
            pane.pane.set_edgecolor((0.75, 0.75, 0.82, 0.8))
            pane.line.set_color((0.7, 0.7, 0.78, 0.9))
        ax.grid(False)
        ax.view_init(elev=22, azim=-58)

    maxn = max((float(np.linalg.norm(g[0])) for g in vis), default=1.0)
    Ld = max(2.6, 1.18 * maxn)
    axL.set_xlim(-Ld, Ld); axL.set_ylim(-Ld, Ld); axL.set_zlim(-Ld, Ld)

    # unit-sphere wireframe
    u = np.linspace(0, 2 * np.pi, 24)
    w = np.linspace(0, np.pi, 12)
    for th in w[::2]:
        axR.plot(np.sin(th)*np.cos(u), np.sin(th)*np.sin(u), np.cos(th)*np.ones_like(u),
                 color="#c8c8d8", lw=0.5)
    for ph in u[::3]:
        axR.plot(np.sin(w)*np.cos(ph), np.sin(w)*np.sin(ph), np.cos(w), color="#c8c8d8", lw=0.5)
    axR.set_xlim(-1.15, 1.15); axR.set_ylim(-1.15, 1.15); axR.set_zlim(-1.15, 1.15)

    # octahedron edges
    for L2 in ROOT_L2:
        vs = [v for (v, lev, d, l2) in vis if l2 == L2]
        if len(vs) < 6:
            continue
        col = root_color(L2)
        segs = octa_edges(vs)
        axL.add_collection3d(Line3DCollection(segs, colors=col, linewidths=1.1, alpha=0.5))
        axR.add_collection3d(Line3DCollection(
            [(a/np.linalg.norm(a), b/np.linalg.norm(b)) for a, b in segs],
            colors=col, linewidths=1.1, alpha=0.5))

    # democratic chain
    dem_pts = [v for (v, lev, d, l2) in vis if d and lev > 0]
    if dem_pts:
        chain = sorted(dem_pts, key=lambda v: np.linalg.norm(v))
        end = 10.5 * np.array([1, 1, 1]) / np.sqrt(3)
        P = np.array([np.zeros(3)] + chain + [end])
        axL.plot(P[:, 0], P[:, 1], P[:, 2], color="#e6b800", lw=1.0, alpha=0.6, ls="--")
        Q = P / np.linalg.norm(P, axis=1, keepdims=True)
        axR.plot(Q[:, 0], Q[:, 1], Q[:, 2], color="#e6b800", lw=1.0, alpha=0.6, ls="--")

    # points
    for idx, (v, lev, is_dem, L2) in enumerate(vis):
        col = point_color(v, lev, is_dem)
        nv = np.linalg.norm(v)
        base = 40 if lev == 0 else max(16, 40 - 2.6 * lev)
        size, alpha, edge = base, 0.92, "none"
        if idx == hi:
            size, alpha, edge = base * 3.2, 1.0, "#111111"
        axL.scatter([v[0]], [v[1]], [v[2]], s=size, c=col, alpha=alpha, edgecolors=edge, linewidths=1.0)
        uu = v / nv
        axR.scatter([uu[0]], [uu[1]], [uu[2]], s=size, c=col, alpha=alpha, edgecolors=edge, linewidths=1.0)

    # labels
    lev = growth[hi][1]
    gname, gsub = GEN_INFO[lev]
    fig.text(0.5, 0.955, f"Z₃ 44-晶格 逐帧生长过程    —    第 {k} / 44 步",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#111")
    fig.text(0.5, 0.905, f"{gname}      {gsub}", ha="center", va="top", fontsize=10.5, color="#335")

    v, lev, is_dem, L2 = growth[hi]
    vec = "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"
    kind = "民主 [111] 链" if is_dem else ("种子" if lev == 0 else f"根壳 L²={L2:g}")
    fig.text(0.5, 0.055, f"新增第 {k} 点:  {vec}    →   {kind}", ha="center", va="bottom",
             fontsize=10.5, color="#8a6d00")

    n_root = sum(1 for (v_, l, d, l2) in vis if (not d) and l > 0 and l2 in ROOT_L2)
    n_dem = sum(1 for (v_, l, d, l2) in vis if d and l > 0)
    n_seed = sum(1 for (v_, l, d, l2) in vis if l == 0)
    fig.text(0.5, 0.018,
             f"已生成 {n_vis} / 44    |    种子 {n_seed}    根壳点 {n_root}    民主点 {n_dem}    "
             f"|    算子: T: v→(z,x,y)   Δ: v→Tv−v   × : v×Tv",
             ha="center", va="bottom", fontsize=9.5, color="#555")

    fig.text(0.02, 0.018, "Z₃ 44-vector vacuum lattice", fontsize=8, color="#999")
    fig.text(0.98, 0.018, f"5 seeds → Z₃ closure → 44 vectors   |   {k}/44",
             fontsize=8, color="#999", ha="right")

    pdf.savefig(fig)
    plt.close(fig)


out = "Z3_44_Lattice_Growth_FrameByFrame.pdf"
with PdfPages(out) as pdf:
    for k in range(1, 45):
        draw_page(pdf, k)
        if k % 10 == 0:
            print(f"  page {k}/44")

print(f"\n✅ saved {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
