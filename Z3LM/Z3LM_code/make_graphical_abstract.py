#!/usr/bin/env python3
"""make_graphical_abstract.py

Elsevier graphical abstract (English). 1500 x 600 px at 300 dpi (2.5:1),
also exported as vector PDF.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans"})

W_CM, H_CM = 12.7, 5.08          # 1500 x 600 px at 300 dpi
fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")

BLUE = "#2b5d8a"
RED = "#b03a2e"
GRAY = "#555555"


def box(x, y, w, h, ec=BLUE, fc="#f4f7fb", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=lw, edgecolor=ec, facecolor=fc))


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.4, color=GRAY,
                                 shrinkA=1, shrinkB=1))


def stack(x, y, lines, dy=2.5, size=8.5, color="k", weight="normal"):
    for i, t in enumerate(lines):
        ax.text(x, y - i * dy, t, ha="center", va="center", fontsize=size,
                color=color, weight=weight)


# ------------------------------------------------------------------ panel 1
box(1, 4, 30, 32)
ax.text(16, 33.4, "1. Texture", ha="center", fontsize=11.5, weight="bold", color=BLUE)
stack(16, 30.4, ["phases locked to"], dy=0, size=8)
stack(16, 27.9, [r"$\omega=e^{2\pi i/3}$"], dy=0, size=8.6)
rows = [
    r"$-\frac{1}{2}a_{12}$", r"$a_{12}$", r"$\frac{1}{3}a_{12}\omega$",
    r"$a_{12}$", r"$\frac{9}{11}d_3$", r"$-\frac{45}{22}a_{12}$",
    r"$\frac{1}{3}a_{12}\omega$", r"$-\frac{45}{22}a_{12}$", r"$d_3$",
]
x0, y0, dx, dy = 8.0, 20.6, 7.4, 4.3
for i in range(3):
    for j in range(3):
        ax.add_patch(plt.Rectangle((x0 + j * dx - 3.2, y0 - i * dy - 1.75), 6.4, 3.5,
                                   facecolor="white", edgecolor="#9bb7d4", lw=0.8))
        ax.text(x0 + j * dx, y0 - i * dy, rows[3 * i + j], ha="center", va="center",
                fontsize=8.2)
stack(16, 7.4, ["five real moduli", "fixed by data"], dy=2.2, size=7.8, color=GRAY)

# ------------------------------------------------------------------ panel 2
box(34, 4, 30, 32)
ax.text(49, 33.4, "2. Fit and scan", ha="center", fontsize=11.5, weight="bold", color=BLUE)
stack(49, 30.0, ["NuFIT 5.2", "joint statistic"], dy=2.2, size=7.8)
ax.text(49, 24.6, r"$\chi^2_{\rm block}=1.44$", ha="center", fontsize=13, color=RED)
stack(49, 20.8, ["447 rational", "alternatives"], dy=2.2, size=7.8)
stack(49, 16.2, [r"15 below $\chi^2=2$", "one fits better"], dy=2.2, size=7.8)
stack(49, 11.4, ["moduli are fitted,", "not predicted"], dy=2.2, size=8.6,
      weight="bold", color=RED)
rng = np.random.default_rng(3)
xs = rng.normal(49, 6.0, 110)
ys = 6.0 + 0.7 * np.abs(rng.normal(0, 1, 110))
ax.scatter(xs, ys, s=3.5, color="#8fb3d9", alpha=0.85, zorder=3)
ax.plot([46.5, 51.5], [5.2, 5.2], color=RED, lw=1.2, zorder=4)

# ------------------------------------------------------------------ panel 3
box(67, 4, 32, 32)
ax.text(83, 33.4, "3. What survives", ha="center", fontsize=11.5, weight="bold", color=BLUE)
stack(83, 29.8, ["predictions (not fitted)"], dy=0, size=7.8)
vals = [r"$\delta_{CP}=-83.4^\circ$", r"$J=-0.033$", "normal ordering",
        r"$\sum m_\nu=70\ {\rm meV}$", r"$m_{\beta\beta}=5\ {\rm meV}$"]
for i, t in enumerate(vals):
    ax.text(83, 26.2 - i * 3.0, t, ha="center", va="center", fontsize=8.6)
ax.plot([69, 97], [10.2, 10.2], color="#cccccc", lw=0.8)
stack(83, 8.8, ["no-go:", "moduli not derivable", r"$11\cdot19\mid|G|$ required"],
      dy=2.1, size=7.4, color=RED, weight="bold")

arrow(31.5, 20, 33.7, 20)
arrow(64.5, 20, 66.7, 20)

fig.savefig("figures/graphical_abstract.png", dpi=300)
fig.savefig("figures/graphical_abstract.pdf")
print("saved")
