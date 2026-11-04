#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary 3-D visualisations for radialflux.

Every figure is produced from the same functions that ship in src/, so the
pictures cannot drift away from the code.  Output: supplement/fig/*.pdf|png
"""
import os
import sys
import math
import warnings
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, PowerNorm, Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from radialflux import exact_radius, exact_bias, q_star, omega_d, enclosed_charge  # noqa: E402
import lattice96  # noqa: E402

FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# house style
# --------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.linewidth": 0.6,
    "figure.dpi": 110,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "legend.frameon": False,
})

INK = "#1b1b1b"
GOLD = "#c8952b"
BLUE = "#1f4e79"
RED = "#b22222"
TEAL = "#0f7b7b"

# a dark, saturated map for surfaces and a warm one for contrast
CM_MAIN = mpl.colors.LinearSegmentedColormap.from_list(
    "deepsea", ["#0b1d33", "#123a5c", "#1f6f8b", "#3fa7a0", "#a8d8a0", "#f2e6b3"])
CM_WARM = mpl.colors.LinearSegmentedColormap.from_list(
    "ember", ["#2b0b13", "#6b1220", "#b22222", "#e07b39", "#f5c26b", "#fff6da"])


def style3d(ax, elev=22, azim=-58, dist=10.4, panes=True):
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.dist = dist
    if panes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0.0))
            axis.pane.set_edgecolor((1, 1, 1, 0.0))
            axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.35)
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis._axinfo["axisline"]["color"] = (0.55, 0.55, 0.55, 0.6)
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")


def save(fig, name, pad=0.02):
    fig.savefig(os.path.join(FIG, name + ".pdf"), transparent=False,
                facecolor="white", pad_inches=pad)
    fig.savefig(os.path.join(FIG, name + ".png"), facecolor="white", pad_inches=pad)
    plt.close(fig)
    print("  wrote", name)


def sphere(R, n=90, cut=None):
    """Sphere surface grid; if cut=(phi0,phi1) only that lune is kept."""
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, 2 * n)
    if cut is not None:
        v = np.linspace(cut[0], cut[1], n)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = R * np.sin(U) * np.cos(V)
    Y = R * np.sin(U) * np.sin(V)
    Z = R * np.cos(U)
    return X, Y, Z


# ==========================================================================
# S1 -- the geometry of one shell: r_n, r_{n+1}, and the exact radius rho_*
# ==========================================================================
def fig_s1():
    d, q, rn = 3, 1.6, 1.0
    rho = float(exact_radius(rn, q * rn, d))
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    cut = (-0.55 * np.pi, 0.72 * np.pi)

    # outer shell: translucent, orange-tinted
    X, Y, Z = sphere(q, 96, cut)
    ax.plot_surface(X, Y, Z, color="#d9822b", alpha=0.13, linewidth=0,
                    antialiased=True, rstride=2, cstride=2, shade=False, zorder=1)
    # inner shell
    X, Y, Z = sphere(rn, 96, cut)
    ax.plot_surface(X, Y, Z, color="#1f4e79", alpha=0.16, linewidth=0,
                    antialiased=True, rstride=2, cstride=2, shade=False, zorder=2)

    # the exact flux sphere: bright gold, fine wireframe
    X, Y, Z = sphere(rho, 80, cut)
    ax.plot_surface(X, Y, Z, color=GOLD, alpha=0.34, linewidth=0, rstride=3,
                    cstride=3, antialiased=True, shade=True, zorder=3)
    # meridians of the exact sphere for readability
    for phi in np.linspace(cut[0], cut[1], 7):
        t = np.linspace(0, np.pi, 200)
        ax.plot(rho * np.sin(t) * np.cos(phi), rho * np.sin(t) * np.sin(phi),
                rho * np.cos(t), color=GOLD, lw=0.9, alpha=0.9, zorder=4)
    for th in np.linspace(0.25, np.pi - 0.25, 5):
        ph = np.linspace(cut[0], cut[1], 200)
        ax.plot(rho * np.sin(th) * np.cos(ph), rho * np.sin(th) * np.sin(ph),
                np.full_like(ph, rho * np.cos(th)), color=GOLD, lw=0.7, alpha=0.8, zorder=4)

    # flux arrows crossing the shell along +x, coloured by 1/r^2 weight
    for z in (-0.55, -0.2, 0.2, 0.55):
        for y in (-0.5, 0.5):
            r0, r1 = 0.72, 1.55
            xs = np.array([r0, r1])
            ax.plot(xs, [y, y], [z, z], color="#8b1a1a", lw=1.4, alpha=0.85)
            ax.quiver(r1 - 0.02, y, z, 0.14, 0, 0, color="#8b1a1a",
                      arrow_length_ratio=0.9, lw=1.2, zorder=6)

    # radii markers on the equatorial plane
    for R, lab, col in ((rn, r"$r_n$", BLUE), (rho, r"$\rho_\ast$", "#8a5a00"),
                        (q, r"$r_{n+1}$", "#a5551a")):
        ax.plot([0, R], [0, 0], [0, 0], color=col, lw=1.1, ls=(0, (4, 2)), zorder=7)
        ax.scatter([R], [0], [0], s=22, color=col, depthshade=False, zorder=9)
        ax.text(R * 1.04, 0.03, 0.05, lab, color=col, fontsize=11)

    ax.scatter([0], [0], [0], s=30, color=INK, depthshade=False, zorder=10)
    ax.text(0.02, -0.06, -0.16, r"$Q$", fontsize=11, color=INK)

    # the quoted identity, placed in space
    ax.text2D(0.03, 0.955,
              r"$\rho_\ast^{\,d-1}=\dfrac{(d-2)\,(r_{n+1}-r_n)}"
              r"{r_n^{\,2-d}-r_{n+1}^{\,2-d}}$", transform=ax.transAxes,
              fontsize=15, color=INK, va="top")
    ax.text2D(0.03, 0.80,
              r"$d=3:\ \ \rho_\ast=\sqrt{r_n r_{n+1}}=" + f"{rho:.4f}" + r"$",
              transform=ax.transAxes, fontsize=10.5, color=GOLD, va="top")
    ax.text2D(0.03, 0.745,
              "one sphere carries the entire shell flux\n"
              "exactly, for every mesh ratio",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")

    lim = 1.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel(r"$x$", labelpad=-6); ax.set_ylabel(r"$y$", labelpad=-8)
    ax.set_zlabel(r"$z$", labelpad=-10)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    style3d(ax, elev=20, azim=-62, dist=9.4)
    save(fig, "s1_shell_geometry")


# ==========================================================================
# S2 -- bias landscape kappa_L(q,d): one picture for the whole closed form
# ==========================================================================
def fig_s2():
    q = np.linspace(1.002, 3.0, 260)
    d = np.linspace(2.02, 8.0, 220)
    Q, D = np.meshgrid(q, d)
    # exact_bias() is scalar in d by design; this grid uses the same closed form
    A = D - 2.0
    K = np.where(A == 0.0, np.log(Q) / (Q - 1.0),
                 (1.0 - Q ** (-A)) / (A * (Q - 1.0)))
    K = np.clip(K, 0.0, 1.25)

    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.25)
    surf = ax.plot_surface(Q, D, K, cmap=CM_MAIN, norm=norm, rstride=2, cstride=2,
                           linewidth=0.08, edgecolor=(0, 0, 0, 0.10),
                           antialiased=True, shade=True, alpha=0.98)

    # exact-rule surface (kappa = 1) -- a translucent red ceiling
    qq = np.array([1.002, 3.0]); dd = np.array([2.02, 8.0])
    QQ, DD = np.meshgrid(qq, dd)
    ax.plot_surface(QQ, DD, np.ones_like(QQ), color=RED, alpha=0.22,
                    linewidth=0, shade=False)

    # floor projection
    ax.contourf(Q, D, K, zdir="z", offset=0.42, levels=18, cmap=CM_MAIN,
                norm=norm, alpha=0.85, antialiased=True)

    # the four dimensions quoted in Table 1 at q = 1.1
    for dd_ in (3, 4, 5, 6):
        k = float(exact_bias("left", 1.1, dd_))
        ax.scatter([1.1], [dd_], [k], s=42, color=GOLD, edgecolor="white",
                   linewidth=0.8, depthshade=False, zorder=12)
        ax.text(1.13, dd_ - 0.12, k + 0.03, f"$d={dd_}$: {k:.3f}",
                fontsize=8.2, color="#7a4a00")

    # the d=2 ridge (logarithmic mean) drawn as a bright curve
    k2 = np.log(q) / (q - 1.0)
    ax.plot(q, np.full_like(q, 2.02), np.clip(k2, 0.42, 1.25), color="#e8e1c0", lw=2.0)

    ax.set_xlabel(r"mesh ratio  $q=r_{n+1}/r_n$", labelpad=6)
    ax.set_ylabel(r"dimension  $d$", labelpad=8)
    ax.set_zlabel(r"$\kappa_L=\hat Q/Q$", labelpad=6)
    ax.set_zlim(0.42, 1.14)
    ax.text2D(0.02, 0.95, r"$\kappa_L=\dfrac{1-q^{-(d-2)}}{(d-2)(q-1)}$",
              transform=ax.transAxes, fontsize=16, va="top")
    ax.text2D(0.02, 0.79,
              "closed form, no root-finding: the left-endpoint rule\n"
              "is off by up to 55 % on coarse meshes, and the error\n"
              "shrink only as $q\\to1$; the red plane is the exact rule",
              transform=ax.transAxes, fontsize=8.4, color="#555555", va="top")
    ax.view_init(elev=25, azim=-56)
    ax.set_box_aspect((1.25, 1.0, 0.85))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.dist = 10.2
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.3)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")
    save(fig, "s2_bias_landscape")


# ==========================================================================
# S3 -- the graded mesh: nested shells and where the exact radius sits
# ==========================================================================
def fig_s3():
    q = 1.35
    r = 1.0 * q ** np.arange(0, 7)
    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    cut = (-0.35 * np.pi, 0.85 * np.pi)

    for i, rn in enumerate(r):
        ro = rn * q
        rho = float(exact_radius(rn, ro, 3))
        X, Y, Z = sphere(rn, 64, cut)
        alpha = 0.10 + 0.05 * (i % 2)
        ax.plot_surface(X, Y, Z, color="#1f4e79", alpha=alpha, linewidth=0,
                        rstride=3, cstride=3, shade=False)
        # sampling sphere for this shell
        X, Y, Z = sphere(rho, 48, cut)
        ax.plot_surface(X, Y, Z, color=GOLD, alpha=0.20, linewidth=0,
                        rstride=3, cstride=3, shade=False)
        # one meridian arc: the radial "wire"
        t = np.linspace(0, np.pi, 300)
        ax.plot(rho * np.sin(t) * np.cos(cut[1]), rho * np.sin(t) * np.sin(cut[1]),
                rho * np.cos(t), color=GOLD, lw=0.8, alpha=0.85)

    # the outermost shell, drawn as a fine mesh
    X, Y, Z = sphere(r[-1] * q, 80, cut)
    ax.plot_wireframe(X, Y, Z, color="#8a8a8a", lw=0.35, alpha=0.5,
                      rstride=4, cstride=4)

    # radial ray with the mesh nodes marked
    ray = np.linspace(0.0, r[-1] * q, 400)
    ax.plot(ray * 0.06, ray * 0.06, ray, color=INK, lw=1.0, alpha=0.5)
    for rn in r:
        ax.scatter([0.06 * rn], [0.06 * rn], [rn], s=18, color=RED,
                   depthshade=False, zorder=12)

    ax.text2D(0.02, 0.95,
              f"graded mesh,  $q={q}$: shells $[r_n,r_{{n+1}}]$\n"
              "gold = the single exact sampling sphere of each shell",
              transform=ax.transAxes, fontsize=9.6, va="top", color=INK)
    ax.text2D(0.02, 0.845,
              r"$r_{n+1}/r_n$ constant $\Rightarrow$ $\rho_\ast/r_n$ constant:"
              "\nthe relative position of the sample never drifts",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")

    lim = r[-1] * q * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.1, lim)
    ax.set_box_aspect((1, 1, 1.25))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    style3d(ax, elev=14, azim=-70, dist=9.2)
    save(fig, "s3_graded_mesh")


# ==========================================================================
# S4 -- the 96^3 lattice solve: a quarter-cutaway cloud of the potential
# ==========================================================================
def fig_s4():
    L = 96
    V = lattice96.solve(L)
    idx = np.arange(L) - L // 2
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")

    # keep one octant removed so that the interior is visible
    keep = ~((X > 0) & (Y > 0))
    # thin out: every second site along each axis
    thin = (X % 2 == 0) & (Y % 2 == 0) & (Z % 2 == 0)
    sel = keep & thin
    x, y, z, v = X[sel], Y[sel], Z[sel], V[sel]

    # logarithmic colour scale: the 1/r dynamic range spans six decades
    w = v - v.min() + 1e-12
    lw = np.log10(w)
    norm = Normalize(vmin=np.percentile(lw, 0.4), vmax=np.percentile(lw, 99.7))
    cmap = CM_WARM

    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=lw, cmap=cmap, norm=norm, s=0.9, alpha=0.55,
               depthshade=False, rasterized=True)

    # the measurement region: shells R = 8..24
    Xs, Ys, Zs = sphere(24, 46)
    ax.plot_wireframe(Xs, Ys, Zs, color="#ffffff", lw=0.4, alpha=0.55,
                      rstride=3, cstride=3)
    Xs, Ys, Zs = sphere(8, 40)
    ax.plot_wireframe(Xs, Ys, Zs, color=GOLD, lw=0.5, alpha=0.8,
                      rstride=2, cstride=2)

    h = L / 2 - 1
    for s_ in (-h, h):
        for t_ in (-h, h):
            ax.plot([-h, h], [s_, s_], [t_, t_], color="#bbbbbb", lw=0.4, alpha=0.5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.52, pad=0.02, aspect=26)
    cb.set_label(r"$\log_{10}[V(\mathbf{x})-V_{\min}]$", fontsize=8.6)
    cb.ax.tick_params(labelsize=7)

    ax.text2D(0.012, 0.965,
              "one FFT solve on $" + "96^3$" + ", 0.14 s, 120 MB",
              transform=ax.transAxes, fontsize=11.5, va="top")
    ax.text2D(0.012, 0.905,
              r"$-\Delta_{\rm lat}V=\delta_0-L^{-3}$,   "
              r"symbol $4\sum_i\sin^2(k_i/2)$",
              transform=ax.transAxes, fontsize=8.8, color="#555555", va="top")
    ax.text2D(0.012, 0.855,
              "cutaway octant;  gold shell $R=8$,  white shell $R=24$",
              transform=ax.transAxes, fontsize=8.2, color="#7a7a7a", va="top")

    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([-48, 0, 48]); ax.set_yticks([-48, 0, 48]); ax.set_zticks([-48, 0, 48])
    style3d(ax, elev=20, azim=-48, dist=9.0)
    save(fig, "s4_lattice96")


# ==========================================================================
# S5 -- q*_d: the n-nacci thresholds as a space curve on its own surface
# ==========================================================================
def fig_s5():
    d = np.linspace(3.0, 9.0, 240)
    q = np.linspace(1.0, 2.6, 240)
    D, Qg = np.meshgrid(d, q)
    F = Qg + Qg ** (2.0 - D) - 2.0

    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    norm = TwoSlopeNorm(vmin=-0.9, vcenter=0.0, vmax=1.2)
    ax.plot_surface(D, Qg, F, cmap=CM_MAIN, norm=norm, alpha=0.55,
                    linewidth=0, rstride=3, cstride=3, antialiased=True, shade=True)
    # zero level = the threshold curve
    ax.contour(D, Qg, F, levels=[0.0], zdir="z", offset=0.0,
               colors=[GOLD], linewidths=2.0)

    ds = np.arange(4, 10)
    qs = np.array([q_star(int(x)) for x in ds])
    ax.plot(ds, qs, np.zeros_like(qs), "-", color=GOLD, lw=1.6, zorder=10)
    ax.scatter(ds, qs, np.zeros_like(qs), s=60, color=GOLD, edgecolor="white",
               linewidth=0.9, depthshade=False, zorder=11)
    ax.plot(ds, qs, np.full_like(qs, -0.9), "-", color="#9a9a9a", lw=0.8,
            ls=(0, (4, 2)))
    for x, y in zip(ds, qs):
        ax.text(x, y + 0.05, 0.02, f"{y:.4f}", fontsize=7.6, color="#7a4a00")

    ax.set_xlabel(r"dimension $d$", labelpad=6)
    ax.set_ylabel(r"mesh ratio $q$", labelpad=8)
    ax.set_zlabel(r"$q+q^{\,2-d}-2$", labelpad=6)
    ax.set_zlim(-0.9, 0.9)
    ax.text2D(0.02, 0.95,
              r"$q^\ast_d+q^{\ast\,2-d}_d=2$", transform=ax.transAxes,
              fontsize=16, va="top")
    ax.text2D(0.02, 0.80,
              "$d=4$: golden ratio $\\varphi=1.6180$\n"
              "$d=5$: plastic constant  $1.8393$\n"
              "$d=6$: $1.9276$   (n-nacci constants)\n"
              "$d=3$: double root at $q=1$ -- no threshold",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")
    ax.view_init(elev=20, azim=-62)
    ax.set_box_aspect((1.2, 1.0, 0.8))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.3)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")
    save(fig, "s5_thresholds")


# ==========================================================================
# S6 -- convergence funnel: error surface over (N, domain ratio)
# ==========================================================================
def fig_s6():
    Ns = np.array([24, 32, 48, 64, 96, 128, 192, 256, 320])
    ratios = np.logspace(np.log10(2.0), np.log10(400.0), 26)
    E = np.zeros((len(Ns), len(ratios)))
    Qs = np.zeros_like(E)
    for i, N in enumerate(Ns):
        for j, ratio in enumerate(ratios):
            r = np.geomspace(1e-3, ratio * 1e-3, int(N))
            V = 1.0 / (4 * math.pi * r)                     # point charge, d = 3
            Qh = enclosed_charge(V, r, 3)
            E[i, j] = np.max(np.abs(Qh - 1.0))
            Qs[i, j] = (r[-1] / r[0]) ** (1.0 / (N - 1)) - 1.0

    fig = plt.figure(figsize=(8.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    C = np.log10(np.clip(E, 1e-15, None))
    norm = Normalize(vmin=-13, vmax=-1)
    ax.plot_surface(np.log10(Qs), np.log10(Ns)[:, None] * np.ones_like(Qs),
                    C, cmap=CM_WARM.reversed(), norm=norm,
                    rstride=1, cstride=1, linewidth=0.1,
                    edgecolor=(0, 0, 0, 0.12), antialiased=True, shade=True)
    # second-order guide plane
    ax.plot_surface(np.log10(Qs), np.log10(Ns)[:, None] * np.ones_like(Qs),
                    -2.0 + 2.0 * np.log10(Qs), color=TEAL, alpha=0.18,
                    linewidth=0, shade=False)
    ax.contour(np.log10(Qs), np.log10(Ns)[:, None] * np.ones_like(Qs), C,
               zdir="z", offset=-15, levels=12, cmap=CM_WARM.reversed())

    ax.set_xlabel(r"$\log_{10}(q-1)$", labelpad=6)
    ax.set_ylabel(r"$\log_{10}N$", labelpad=8)
    ax.set_zlabel(r"$\log_{10}\max|\hat Q/Q-1|$", labelpad=6)
    ax.set_zlim(-15, -1)
    ax.text2D(0.02, 0.95, "measured order: 2.01, 2.00, 2.00",
              transform=ax.transAxes, fontsize=15, va="top", color=INK)
    ax.text2D(0.02, 0.865,
              "error collapses along the teal guide\n"
              r"$\propto(q-1)^2$: refining the mesh and\n"
              "flattening the grading are interchangeable",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")
    ax.view_init(elev=24, azim=-64)
    ax.set_box_aspect((1.15, 1.0, 0.9))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.3)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")
    save(fig, "s6_convergence")


# ==========================================================================
# S7 -- outlook: shells as symbol intervals, sampling points as a constellation
# ==========================================================================
def fig_s7():
    fig = plt.figure(figsize=(7.4, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    radii = [1.0, 1.45, 2.10, 3.05, 4.42]
    colors = ["#1f4e79", "#1f6f8b", "#2f8f8f", "#6fae8f", "#c8b46a"]

    for R, c in zip(radii, colors):
        X, Y, Z = sphere(R, 70)
        ax.plot_surface(X, Y, Z, color=c, alpha=0.10, linewidth=0,
                        rstride=4, cstride=4, shade=False)
        th = np.linspace(0, np.pi, 300)
        ax.plot(R * np.sin(th) * np.cos(0.4), R * np.sin(th) * np.sin(0.4),
                R * np.cos(th), color=c, lw=0.8, alpha=0.7)

    # Fibonacci constellation: the sampled directions
    n = 220
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    for R, c in zip(radii, colors):
        x = R * np.sin(phi) * np.cos(theta)
        y = R * np.sin(phi) * np.sin(theta)
        z = R * np.cos(phi)
        ax.scatter(x, y, z, s=5.5, color=c, alpha=0.85, depthshade=False)

    # a modulated radial "signal": amplitude drawn as tube thickness
    t = np.linspace(0.6, 4.9, 500)
    ax.plot(t * np.cos(0.55), t * np.sin(0.55), 0.12 * np.sin(6.0 * t),
            color=GOLD, lw=2.0)
    ax.text2D(0.02, 0.95,
              "an outlook, not a result:\n"
              "each shell = one symbol interval;  $\\rho_\\ast$ = the\n"
              "bias-free decision point inside it",
              transform=ax.transAxes, fontsize=9.4, va="top", color=INK)
    ax.text2D(0.02, 0.815,
              "drawn only to make the analogy visible.\n"
              "Nothing in this figure is used in the paper's claims.",
              transform=ax.transAxes, fontsize=8.0, va="top", color="#8a8a8a",
              style="italic")

    lim = 4.9
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    style3d(ax, elev=18, azim=-60, dist=8.6)
    save(fig, "s7_outlook")




# ==========================================================================
# S8 -- multipole leakage: how much of a higher multipole the monopole reads
# ==========================================================================
def fig_s8():
    q = np.linspace(1.02, 1.60, 240)
    l = np.arange(0, 13)
    Q, L = np.meshgrid(q, l)
    # P_l(q) = sum_{j=0..l} q^{-j}
    P = np.zeros_like(Q)
    for j in range(0, 13):
        P += np.where(L >= j, Q ** (-j), 0.0)

    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=1.0, vmax=float(P.max()))
    ax.plot_surface(Q, L, P, cmap=CM_WARM, norm=norm, rstride=1, cstride=2,
                    linewidth=0.15, edgecolor=(0, 0, 0, 0.10), antialiased=True,
                    shade=True)
    # the l -> infinity limit q/(q-1), drawn as a bright curve
    ax.plot(q, np.full_like(q, 12.0), q / (q - 1.0), color="#2f6f8f", lw=2.2)
    ax.plot([1.25], [10], [sum(1.25 ** -j for j in range(11))], "o",
            ms=8, color=GOLD, markeredgecolor="white", markeredgewidth=1.0)

    ax.set_xlabel(r"$q$", labelpad=6)
    ax.set_ylabel(r"multipole order $\ell$", labelpad=8)
    ax.set_zlabel(r"prefactor   $\sum_{j\leq\ell} q^{-j}$", labelpad=6)
    ax.text2D(0.02, 0.95, r"leakage $=\sum_{j\leq\ell}q^{-j}$",
              transform=ax.transAxes, fontsize=14, va="top")
    ax.text2D(0.02, 0.855,
              "$q=1.25$, $\\ell=10$: 1.375, i.e. 138 %.\n"
              "At fixed $q$ the sum saturates at $q/(q-1)$:\n"
              "the highest multipoles matter least.",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")
    ax.view_init(elev=23, azim=-58)
    ax.set_box_aspect((1.2, 1.0, 0.85))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.3)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")
    save(fig, "s8_multipole")


# ==========================================================================
# S9 -- the null test as a waterfall: four estimators, one lattice
# ==========================================================================
def fig_s9():
    L = 96
    V = lattice96.solve(L)
    idx = np.arange(L) - L // 2
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    Rm = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    n = np.rint(Rm).astype(int)
    st = {}
    for s in range(1, L // 2):
        sel = n == s
        if sel.any():
            st[s] = (float(V[sel].mean()), float((Rm[sel] ** 2).mean()))

    Rs = np.arange(8, 25)
    Om = 4.0 * math.pi
    series = [
        ("left endpoint", "#d9822b", lambda s: Om * s * s * (st[s][0] - st[s + 1][0])),
        ("midpoint", "#6a3d9a", lambda s: Om * (s + .5) ** 2 * (st[s][0] - st[s + 1][0])),
        ("exact $\\rho_\\ast$", "#1f4e79", lambda s: Om * s * (s + 1) * (st[s][0] - st[s + 1][0])),
        ("$\\rho_\\ast$ - background", "#b22222",
         lambda s: Om * s * (s + 1) * ((st[s][0] - st[s + 1][0])
                                       - (st[s][1] - st[s + 1][1]) / (6 * L ** 3))),
    ]

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    yy = np.arange(len(series))
    for k, (name, col, fn) in enumerate(series):
        z = np.array([fn(int(s)) for s in Rs])
        ax.plot(Rs, np.full_like(Rs, float(k), dtype=float), z, "-o", ms=3.2,
                color=col, lw=1.6)
        for x, zz in zip(Rs, z):
            ax.plot([x, x], [k, k], [1.0, zz], color=col, lw=0.5, alpha=0.35)

    qq, yy2 = np.meshgrid(np.array([6, 26.0]), np.array([-0.6, 3.6]))
    ax.plot_surface(qq, yy2, np.ones_like(qq), color="#7a7a7a", alpha=0.18,
                    linewidth=0, shade=False)

    ax.set_xlabel(r"shell radius $R$", labelpad=6)
    ax.set_ylabel("estimator", labelpad=10)
    ax.set_zlabel(r"$\hat Q/Q$", labelpad=6)
    ax.set_yticks(yy)
    ax.set_yticklabels([s[0] for s in series], fontsize=7.2)
    ax.set_zlim(0.85, 1.06)
    ax.text2D(0.02, 0.95, "one lattice, four ways to read it",
              transform=ax.transAxes, fontsize=13, va="top")
    ax.text2D(0.02, 0.87,
              "the grey plane is the truth.\n"
              "Only the background-corrected $\\rho_\\ast$ row\n"
              "stays on it across the whole range.",
              transform=ax.transAxes, fontsize=8.6, color="#555555", va="top")
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.25, 0.9, 0.85))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.3)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=7.5, pad=1.5, colors="#444444")
    save(fig, "s9_waterfall")


if __name__ == "__main__":
    which = sys.argv[1:] or ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    for w in which:
        globals()["fig_" + w]()
