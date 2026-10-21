#!/usr/bin/env python3
"""make_figures.py -- figures for the Chinese supplementary document.

Outputs (figures/*.png, 300 dpi):
  fig1_texture.png      |M_nu| and phases of the reference texture
  fig2_pmns.png         |U_PMNS| with values
  fig3_masses.png       mass spectrum, sum m, m_bb
  fig4_profile.png      chi2_block profile along m0
  fig5_scan.png         look-elsewhere scan (rational alternatives)
  fig6_phases.png       discrete phase-pattern scan
  fig7_nogo.png         denominator theorem: group order vs needed primes
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import z3lm
from z3lm import M6, obs_vec, obs_full, takagi_all, OMEGA
from nufit_parser import NuFIT52

os.makedirs("figures", exist_ok=True)
nf = NuFIT52()
plt.rcParams.update({"font.size": 10, "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight"})

Q = np.array([1.0, -3/19, 9/11, 1.0, 6/19, 135/209])
M0 = M6(Q)
U, m = takagi_all(M0)
MREF = M0 * 0.031593
o = obs_full(MREF)


def block_scalar(M):
    y = obs_vec(M)
    return nf.chi2_block(np.array(y))


# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
mod = np.abs(M0)
arg = np.angle(M0) / np.pi
im0 = ax[0].imshow(mod, cmap="viridis")
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{mod[i, j]:.3f}", ha="center", va="center",
                   color="w", fontsize=11)
ax[0].set_title(r"$|M_\nu|/m_0$")
im1 = ax[1].imshow(arg, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, f"{arg[i, j]:+.2f}" if abs(arg[i, j]) > 1e-9 else "0",
                   ha="center", va="center", fontsize=11,
                   color="k" if abs(arg[i, j]) < 0.5 else "w")
ax[1].set_title(r"$\arg M_\nu/\pi$")
for a in ax:
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels([r"$e$", r"$\mu$", r"$\tau$"])
    a.set_yticklabels([r"$e$", r"$\mu$", r"$\tau$"])
    a.grid(False)
fig.colorbar(im0, ax=ax[0], fraction=0.046)
fig.colorbar(im1, ax=ax[1], fraction=0.046)
fig.suptitle(r"Reference texture: $|d_1|/a_{12}=1/2$, $d_2/d_3=9/11$, "
             r"$a_{23}/a_{12}=45/22$, $a_{13}/a_{12}=1/3$, $a_{12}/d_3=6/19$",
             fontsize=9)
fig.savefig("figures/fig1_texture.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
A = np.abs(U[np.argsort(m)][:, np.argsort(m)])
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                color="k" if A[i, j] < 0.6 else "w", fontsize=12)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"])
ax.set_yticklabels([r"$e$", r"$\mu$", r"$\tau$"])
ax.grid(False)
ax.set_title(r"$|U_{\rm PMNS}|$ of the reference texture")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig("figures/fig2_pmns.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
ms = np.array([o["m1"], o["m2"], o["m3"]]) * 1e3
fig, ax = plt.subplots(figsize=(5.6, 3.6))
bars = ax.bar([r"$m_1$", r"$m_2$", r"$m_3$"], ms,
              color=["#4c72b0", "#55a868", "#c44e52"], width=0.55)
for b, v in zip(bars, ms):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f} meV",
            ha="center", fontsize=10)
ax.set_ylabel("mass [meV]")
ax.set_ylim(0, 60)
ax.axhline(o["sum_m"] * 1e3, ls="--", c="k", lw=1)
ax.text(2.35, o["sum_m"] * 1e3 + 1.2,
        r"$\sum m_\nu = %.1f$ meV" % (o["sum_m"] * 1e3), ha="right", fontsize=9)
ax.axhline(o["mbb"] * 1e3, ls=":", c="crimson", lw=1.4)
ax.text(-0.35, o["mbb"] * 1e3 + 1.2,
        r"$m_{\beta\beta} = %.1f$ meV" % (o["mbb"] * 1e3), fontsize=9,
        color="crimson")
ax.set_title("Normal ordering; scale fixed by $\\Delta m^2_{31}$")
fig.savefig("figures/fig3_masses.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 4
grid = np.linspace(0.0295, 0.0338, 90)
chis = np.array([block_scalar(M0 * mm) for mm in grid])
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(grid, chis, lw=1.6)
ax.axhline(chis.min() + 1, ls="--", lw=0.9, c="gray")
ax.axhline(chis.min() + 4, ls=":", lw=0.9, c="gray")
ax.axvline(0.031593, ls="-.", lw=0.9, c="crimson")
ax.set_xlabel(r"$m_0$ [eV]")
ax.set_ylabel(r"$\chi^2_{\rm block}$")
ax.set_title(r"Profile along the single scale ($\chi^2_{\min}=%.2f$)" % chis.min())
ax.text(0.0318, chis.min() + 1.15, r"$\Delta\chi^2=1$", fontsize=8, color="gray")
fig.savefig("figures/fig4_profile.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 5
data = []
try:
    for line in open("rational_scan_refined.txt"):
        if line.startswith("chi2"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        c = float(parts[0])
        r3 = parts[3]
        data.append((c, r3))
except FileNotFoundError:
    pass

fig, ax = plt.subplots(figsize=(6.0, 3.8))
if data:
    xs, ys = [], []
    for c, frac in data:
        p, q = frac.split("/")
        xs.append(int(p) / int(q))
        ys.append(c)
    ax.scatter(xs, ys, s=14, alpha=0.55, c="#4c72b0", label="rational alternatives")
    ax.axhline(1.436, ls="--", c="crimson", lw=1.2, label=r"reference point (1.44)")
    ax.set_xlabel(r"$a_{23}/a_{12}$")
    ax.set_ylabel(r"$\chi^2_{\rm block}$")
    ax.set_title("Look-elsewhere scan: 447 alternatives, 15 below $\\chi^2=2$")
    ax.legend(fontsize=8)
    ax.set_ylim(0.8, 3.2)
fig.savefig("figures/fig5_scan.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 6
import json
try:
    v = json.load(open("phase_scan_block.json"))
except Exception:
    v = {}
if v:
    labels = list(v.keys())
    vals = [v[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.barh(range(len(vals)), vals, color="#4c72b0")
    bars[1].set_color("#c44e52")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([l.replace("(", "(").replace(")", "°)") for l in labels],
                       fontsize=8)
    ax.set_xlabel(r"$\chi^2_{\rm block}$")
    ax.set_title("Discrete phase-pattern scan (red: adopted pattern)")
    for i, vv in enumerate(vals):
        ax.text(vv + 0.05, i, f"{vv:.2f}", va="center", fontsize=8)
    fig.savefig("figures/fig6_phases.png")
    plt.close(fig)

# ---------------------------------------------------------------- fig 7
groups = [r"$\Delta(27)$", r"$A_4$", r"$S_4$", r"$\Sigma(36)$",
          r"$\Delta(54)$", r"$\Delta(96)$"]
orders = [27, 12, 24, 36, 54, 96]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.bar(groups, orders, color="#8c8c8c", width=0.6)
ax.axhline(209, ls="--", c="crimson", lw=1.4)
ax.text(4.6, 214, r"needed: $11\cdot19=209$", color="crimson", fontsize=9,
        ha="right")
ax.set_yscale("log")
ax.set_ylabel(r"$|G|$")
ax.set_title("Denominator theorem: no minimal flavour group reaches 209")
fig.savefig("figures/fig7_nogo.png")
plt.close(fig)

print("figures written to figures/")
