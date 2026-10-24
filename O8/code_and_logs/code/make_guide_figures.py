#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_guide_figures.py -- high-quality figures for the illustrated guide.

All figures are saved as vector PDF and 300-dpi PNG in ../guide_figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpmath as mp

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "guide_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "mathtext.fontset": "dejavuserif",
})

BLUE = "#1f4e79"
ORANGE = "#c55a11"
TEAL = "#2e7d6f"
RED = "#a63d2f"
GRAY = "#8c8c8c"
LIGHT = "#dbe5f1"

# geometry ---------------------------------------------------------------
AX = {"x": 0, "y": 1, "z": 2}
VERTS = [(a, s) for a in range(3) for s in (1, -1)]
COORD = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
FACE_SIGNS = [(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
FACES = [np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], float) for sx, sy, sz in FACE_SIGNS]
COL = {0: BLUE, 1: TEAL, 2: ORANGE}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_3d(ax):
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)


def octahedron(ax, highlight_faces=(), highlight_edges=(), labels=True,
               alpha=0.22, lw=1.0):
    for k, (tri, sg) in enumerate(zip(FACES, FACE_SIGNS)):
        fc = "#f6c99f" if k in highlight_faces else LIGHT
        al = 0.75 if k in highlight_faces else alpha
        pc = Poly3DCollection([tri], facecolor=fc, edgecolor="0.55",
                              linewidth=0.6, alpha=al)
        ax.add_collection3d(pc)
    # edges
    for i in range(6):
        for j in range(i + 1, 6):
            if VERTS[i][0] != VERTS[j][0]:
                x = [COORD[i][0], COORD[j][0]]
                y = [COORD[i][1], COORD[j][1]]
                z = [COORD[i][2], COORD[j][2]]
                ax.plot(x, y, z, color="0.65", lw=0.6, zorder=1)
    for (i, j) in highlight_edges:
        ax.plot([COORD[i][0], COORD[j][0]], [COORD[i][1], COORD[j][1]],
                [COORD[i][2], COORD[j][2]], color=RED, lw=3.0, zorder=5)
    for i, (a, s) in enumerate(VERTS):
        ax.scatter(*COORD[i], s=45, color=COL[a], edgecolor="white",
                   linewidth=0.6, depthshade=False, zorder=10)
        if labels:
            lab = ("+" if s > 0 else "-") + "xyz"[a]
            ax.text(COORD[i][0] * 1.18, COORD[i][1] * 1.18,
                    COORD[i][2] * 1.18, lab, fontsize=8, ha="center",
                    va="center", color="0.2")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-1.25, 1.25)
    set_3d(ax)


# ---------------------------------------------------------------- figure 1
def fig_geometry():
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    octahedron(ax, highlight_faces=(7,), alpha=0.18)
    ax.set_title("octahedron  $K_{2,2,2}$", fontsize=9, pad=0)
    # dual cube
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    cv = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                   for sz in (-1, 1)], float) / np.sqrt(3)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(cv[i] != cv[j]) == 1:
                ax2.plot([cv[i][0], cv[j][0]], [cv[i][1], cv[j][1]],
                         [cv[i][2], cv[j][2]], color=TEAL, lw=1.4)
    ax2.scatter(cv[:, 0], cv[:, 1], cv[:, 2], s=40, color=TEAL,
                edgecolor="white", linewidth=0.6, depthshade=False)
    for i, s in enumerate(FACE_SIGNS):
        lab = "".join("+" if x > 0 else "-" for x in s)
        ax2.text(cv[i][0] * 1.22, cv[i][1] * 1.22, cv[i][2] * 1.22,
                 lab, fontsize=6.5, ha="center", va="center", color="0.25")
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4); ax2.set_zlim(-1.4, 1.4)
    set_3d(ax2)
    ax2.set_title("dual graph = cube  (one integer $n$)", fontsize=9, pad=0)
    fig.subplots_adjust(wspace=0.02)
    save(fig, "g1_geometry")


# ---------------------------------------------------------------- figure 2
def fig_wilson():
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # A = 2 loop: x+ y+ x- z+
    idx = [0, 2, 1, 4]
    edges = [(idx[k], idx[(k + 1) % 4]) for k in range(4)]
    faces_hl = (7, 5)  # (+++) and (-++)
    octahedron(ax, highlight_faces=faces_hl, highlight_edges=edges)
    ax.set_title("loop with $A=2$ enclosed faces", fontsize=9, pad=0)
    ax.view_init(elev=26, azim=28)

    # exact values
    def I(n):
        return mp.besseli(int(n), 1)

    Z = sum(I(n) ** 8 for n in range(-30, 31))
    W = [float(sum(I(n) ** A * I(n + 1) ** (8 - A) for n in range(-30, 31)) / Z)
         for A in (1, 2, 3, 4)]
    r = float(I(1) / I(0))

    ax2 = fig.add_subplot(1, 2, 2)
    A = np.arange(1, 5)
    ax2.semilogy(A, W, "o", ms=6, color=BLUE, label="exact")
    ax2.semilogy(A, r ** A, "--", color=GRAY, lw=1.0, label=r"$r^{A}$")
    for a, w in zip(A, W):
        ax2.annotate("%.4f" % w, (a, w), textcoords="offset points",
                     xytext=(5, 7), fontsize=7.2, color=BLUE)
    ax2.set_xticks(A)
    ax2.set_xlabel("enclosed area $A$")
    ax2.set_ylabel(r"$\langle W(C)\rangle$")
    ax2.set_ylim(0.06, 0.75)
    ax2.legend(loc="lower left")
    ax2.set_title("exact area law at $\\beta=1$", fontsize=9)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "g2_wilson")


# ---------------------------------------------------------------- figure 3
def fig_finite_size():
    def I(n):
        return mp.besseli(int(n), 1)

    def face(F):
        Z = sum(I(n) ** F for n in range(-30, 31))
        return sum(I(n) * I(n + 1) ** (F - 1) for n in range(-30, 31)) / Z

    r = I(1) / I(0)
    sig = -mp.log(r)
    bracket = 1 + (I(2) / I(1)) * r - 2 * r ** 2
    Fs = np.array([4, 6, 8, 10, 12, 16, 20])
    ex = np.array([float(-mp.log(face(int(F)))) for F in Fs])
    asym = np.array([float(sig - r ** (F - 2) * bracket) for F in Fs])
    rel = np.abs(asym - ex) / ex

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    a = ax[0]
    Ff = np.linspace(3.5, 21, 200)
    a.plot(Ff, [float(sig)] * len(Ff), "--", color=GRAY, lw=1)
    a.plot(Fs, ex, "o", ms=4.5, color=BLUE)
    a.set_xlabel("$F$"); a.set_ylabel(r"$\sigma_F$")
    a.set_title("(a) finite-volume tension", fontsize=8.5)

    b = ax[1]
    b.semilogy(Fs, ex - np.array([float(sig)] * len(Fs)), "o", ms=4.5,
               color=BLUE, label="exact")
    b.semilogy(Ff, [float(r ** (F - 2) * bracket) for F in Ff], "-",
               color=ORANGE, lw=1.1, label="analytic")
    b.set_xlabel("$F$"); b.set_ylabel(r"$\sigma_\infty-\sigma_F$")
    b.legend(fontsize=7.5, loc="upper right")
    b.set_title("(b) exponential approach", fontsize=8.5)

    c = ax[2]
    c.semilogy(Fs, rel, "s-", ms=4, lw=1, color=TEAL)
    c.axhline(1e-4, color=GRAY, ls=":", lw=0.8)
    c.set_xlabel("$F$"); c.set_ylabel("relative error")
    c.set_title("(c) asymptotic error", fontsize=8.5)
    fig.tight_layout()
    save(fig, "g3_finite_size")


# ---------------------------------------------------------------- figure 4
def fig_thermo():
    from scipy.special import iv
    n = np.arange(0, 40)

    def cbeta(b, F=8):
        def lZ(x):
            w = iv(n, x) ** F
            w = np.concatenate([w[:1], 2 * w[1:]])
            return np.log(w.sum())
        h = 1e-4 * max(b, 0.1)
        return b * b * (lZ(b + h) - 2 * lZ(b) + lZ(b - h)) / h ** 2

    bs = np.linspace(0.05, 6, 160)
    cs = [cbeta(b) for b in bs]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a = ax[0]
    a.plot(bs, cs, color=BLUE, lw=1.5)
    a.axhline(3.5, color=GRAY, ls="--", lw=1)
    a.plot([2.05], [5.745], "o", color=RED, ms=5)
    a.annotate("crossover\n$\\beta\\simeq2.05$", (2.05, 5.745),
               xytext=(18, -8), textcoords="offset points", fontsize=7.5,
               color=RED)
    a.text(4.6, 3.62, "$(F-1)/2$", color=GRAY, fontsize=8)
    a.set_xlabel(r"$\beta$"); a.set_ylabel(r"$c(\beta)$")
    a.set_title("(a) specific heat: crossover, no singularity", fontsize=8.5)

    b = ax[1]
    ws = [float(iv(k, 1) ** 8 / sum(iv(m, 1) ** 8 for m in range(-30, 31)))
          for k in range(0, 5)]
    b.bar(range(5), ws, color=[BLUE, TEAL, TEAL, ORANGE, ORANGE],
          edgecolor="white")
    b.set_yscale("log")
    b.set_xticks(range(5))
    b.set_xlabel("flux sector $n$")
    b.set_ylabel("$w_n$")
    b.set_title("(b) sector weights at $\\beta=1$  ($w_0=99.69\\%$)", fontsize=8.5)
    for k, w in enumerate(ws):
        b.text(k, w * 1.5, "%.1e" % w, ha="center", fontsize=6.5, color="0.3")
    fig.tight_layout()
    save(fig, "g4_thermo")


# ---------------------------------------------------------------- figure 5
def fig_duality():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a = ax[0]
    Ff = np.linspace(2.5, 21, 200)
    a.plot(Ff, Ff / (2 * np.pi), "-", color=BLUE, lw=1.4,
           label=r"$\beta_*=F/2\pi$")
    Fs = np.array([4, 6, 8, 12, 20])
    a.plot(Fs, Fs / (2 * np.pi), "o", ms=5, color=BLUE)
    a.axhline(1.0, color=GRAY, ls="--", lw=1, label=r"$\beta=1$")
    a.axvline(2 * np.pi, color=RED, ls=":", lw=1)
    a.annotate("$F=2\\pi$", (2 * np.pi, 0.45), color=RED, fontsize=8,
               ha="center")
    for F in Fs:
        a.annotate("$%d$" % F, (F, F / (2 * np.pi) + 0.08), fontsize=6.5,
                   ha="center", color="0.4")
    a.set_xlabel("number of faces $F$")
    a.set_ylabel(r"self-dual coupling $\beta_*$")
    a.set_ylim(0, 3.4)
    a.legend(loc="upper left", fontsize=7.5)
    a.set_title("(a) self-duality is extensive", fontsize=8.5)

    b = ax[1]
    vals = [0.619915, 1.0, 0.0]
    labels = ["ratio\nmatch", "curvature\nmatch", "amplitude\n(no solution)"]
    cols = [BLUE, "#9dc3e6", "#d9d9d9"]
    b.bar(range(3), vals, color=cols, edgecolor="0.3", lw=0.5, width=0.55)
    b.axhline(4 / np.pi, color=RED, ls="--", lw=1.2,
              label=r"$\beta_*=4/\pi$")
    b.axhline(1.0, color=GRAY, ls=":", lw=1)
    b.set_xticks(range(3)); b.set_xticklabels(labels)
    b.set_ylabel(r"$\beta_V$ at $\beta_W=1$")
    b.set_ylim(0, 1.45)
    b.legend(loc="upper left", fontsize=7.5)
    b.text(2, 0.05, "max coeff 0.242\n$<I_1(1)=0.446$", ha="center",
           fontsize=6.5, color="0.35")
    b.set_title("(b) three natural dictionaries", fontsize=8.5)
    fig.tight_layout()
    save(fig, "g5_duality")


# ---------------------------------------------------------------- figure 6
def fig_fermion():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a = ax[0]
    ev = [0, 0, 0, 0, 2 * np.sqrt(3), -2 * np.sqrt(3)]
    a.axhline(0, color="0.85", lw=0.7)
    a.vlines(range(6), 0, ev, color=BLUE, lw=1.6)
    a.plot(range(6), ev, "o", ms=5, color=BLUE)
    a.fill_between([-0.5, 3.5], -0.35, 0.35, color=RED, alpha=0.12)
    a.text(1.5, 0.55, "4 zero modes", color=RED, ha="center", fontsize=8)
    a.set_xticks(range(6))
    a.set_xticklabels(["$v_1$", "$v_2$", "$v_3$", "$v_4$", "$v_5$", "$v_6$"])
    a.set_ylabel(r"eigenvalue of $\Gamma$")
    a.set_ylim(-4.3, 4.3)
    a.set_title(r"(a) $\Gamma=\varepsilon\otimes J_2$, rank 2", fontsize=8.5)

    b = ax[1]
    ms = np.linspace(0, 1.6, 200)
    b.plot(ms, ms ** 4 * (ms ** 2 + 12), color=TEAL, lw=1.4)
    b.plot([0], [0], "o", color=RED, ms=5)
    b.set_xlabel("mass $m$")
    b.set_ylabel(r"$\det(m+\Gamma)$")
    b.set_title(r"(b) $m^4(m^2+12)$: four massless modes", fontsize=8.5)
    fig.tight_layout()
    save(fig, "g6_fermion")


# ---------------------------------------------------------------- figure 7
def fig_algebra():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a = ax[0]
    a.set_xlim(0, 5.6); a.set_ylim(0, 5.6); a.set_aspect("equal")
    a.add_patch(Rectangle((0.2, 1.0), 4.0, 4.0, facecolor=LIGHT,
                          edgecolor="0.4", lw=0.8))
    a.add_patch(Rectangle((4.2, 1.0), 1.0, 4.0, facecolor="#f6c99f",
                          edgecolor="0.4", lw=0.8))
    a.add_patch(Rectangle((0.2, 0.2), 4.0, 0.8, facecolor="#c9e2d6",
                          edgecolor="0.4", lw=0.8))
    a.text(2.2, 3.0, r"$\mathfrak{g}_0=\mathfrak{gl}(4)$" + "\ndim 16",
           ha="center", va="center", fontsize=9)
    a.text(4.7, 3.0, r"$\mathfrak{g}_1$" + "\n$\\bar4$", ha="center",
           va="center", fontsize=9, rotation=90)
    a.text(2.2, 0.6, r"$\mathfrak{g}_{-1}=4$", ha="center", va="center",
           fontsize=9)
    a.annotate("", xy=(4.25, 2.2), xytext=(4.25, 1.8),
               arrowprops=dict(arrowstyle="-", lw=0.8, color="0.4"))
    a.set_axis_off()
    a.set_title(r"(a) $\mathfrak{sl}(5)=\mathfrak{g}_{-1}\oplus\mathfrak{g}_0\oplus\mathfrak{g}_1$"
                "\n$16+4+4=24$", fontsize=8.5)

    b = ax[1]
    items = ["mixed\ntrilinear", "bracket\nfamily", "chiral\ncontent", "SM algebra\nin $\\mathfrak{u}(4)$"]
    vals = [0, 1, 0, 0]
    cols = [RED, ORANGE, RED, RED]
    b.barh(range(4), vals, color=cols, height=0.5)
    b.set_yticks(range(4)); b.set_yticklabels(items, fontsize=7.5)
    b.set_xlim(0, 2)
    b.set_xticks([0, 1, 2])
    b.set_xticklabels(["0", "1 parameter", "2"])
    b.set_title("(b) what survives the audit", fontsize=8.5)
    b.invert_yaxis()
    for k, v in enumerate(vals):
        b.text(v + 0.05, k, "none" if v == 0 else "one", va="center",
               fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, "g7_algebra")




# ---------------------------------------------------------------- graphical abstract
def graphical_abstract():
    import mpmath as mp
    mp.mp.dps = 30

    def I(n):
        return mp.besseli(int(n), 1)

    def face(F):
        Z = sum(I(n) ** F for n in range(-30, 31))
        return sum(I(n) * I(n + 1) ** (F - 1) for n in range(-30, 31)) / Z

    fig = plt.figure(figsize=(10.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.25)

    ax = fig.add_subplot(gs[0], projection="3d")
    octahedron(ax, highlight_faces=(7,), alpha=0.20, labels=False)
    ax.set_title("exact $U(1)$ gauge theory\non the octahedron", fontsize=11, pad=2)

    ax2 = fig.add_subplot(gs[1])
    r = float(I(1) / I(0))
    A = np.arange(1, 5)
    Z = sum(I(n) ** 8 for n in range(-30, 31))
    W = [float(sum(I(n) ** a * I(n + 1) ** (8 - a) for n in range(-30, 31)) / Z)
         for a in A]
    ax2.semilogy(A, W, "o-", ms=5, lw=1.2, color=BLUE)
    ax2.semilogy(A, r ** A, "--", color=GRAY, lw=1)
    ax2.set_xticks(A)
    ax2.set_xlabel("enclosed area $A$")
    ax2.set_ylabel(r"$\langle W(C)\rangle$")
    ax2.set_title("exact area law, shape independent", fontsize=11)
    ax2.set_ylim(0.05, 0.8)

    ax3 = fig.add_subplot(gs[2])
    Fs = np.array([4, 6, 8, 12, 20, 32])
    sig = [float(-mp.log(face(int(F)))) for F in Fs]
    sinf = float(-mp.log(I(1) / I(0)))
    ax3.plot(Fs, sig, "o", ms=5, color=BLUE, label=r"$\sigma_F$ exact")
    Ff = np.linspace(3.5, 34, 100)
    bracket = 1 + float(I(2) / I(1)) * r - 2 * r ** 2
    ax3.plot(Ff, [sinf - float(r ** (F - 2) * bracket) for F in Ff], "-",
             color=ORANGE, lw=1.2, label="analytic")
    ax3.axhline(sinf, color=GRAY, ls="--", lw=1)
    ax3.set_xlabel("number of faces $F$")
    ax3.set_ylabel(r"$\sigma_F$")
    ax3.set_title("exponential finite-size law", fontsize=11)
    ax3.legend(fontsize=7.5, loc="lower right")

    fig.savefig(os.path.join(OUT, "graphical_abstract.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_geometry()
    fig_wilson()
    fig_finite_size()
    fig_thermo()
    fig_duality()
    fig_fermion()
    fig_algebra()
    graphical_abstract()
    print("guide figures written to", OUT)
