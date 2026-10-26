#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py — paper figures 3 and 4 (pulls and octant sum rule)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

lam = 73 / 324
s13 = 0.022057741290
s12 = 0.314355357877
s23 = 0.450221543608
dcp = 240.0

# ------------------------------------------------------------------ pulls
labels = [r"$\sin^2\theta_{13}$", r"$\sin^2\theta_{12}$", r"$\sin^2\theta_{23}$",
          r"$\delta_{CP}$", r"$\lambda=|V_{us}|$", r"$|V_{ud}|$"]
vals = [s13, s12, s23, dcp, lam, 0.974287]
exps = [0.02225, 0.304, 0.450, 230.0, 0.22500, 0.97373]
errs = [0.00060, 0.012, 0.017, 35.0, 0.00067, 0.00031]
pulls = [(v - e) / s for v, e, s in zip(vals, exps, errs)]

fig, ax = plt.subplots(figsize=(6.4, 3.2))
cols = ["#4e79a7" if abs(p) < 2 else "#e15759" for p in pulls]
ax.barh(range(len(pulls)), pulls, color=cols, height=0.6)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(1, color="gray", ls=":", lw=0.8)
ax.axvline(-1, color="gray", ls=":", lw=0.8)
ax.set_yticks(range(len(pulls)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel(r"pull $=({\rm predicted}-{\rm experiment})/\sigma$", fontsize=10)
ax.set_xlim(-2.2, 2.2)
for i, p in enumerate(pulls):
    ax.text(p + (0.08 if p >= 0 else -0.08), i, f"{p:+.2f}", va="center",
            ha="left" if p >= 0 else "right", fontsize=9)
ax.set_title("predictions vs. experiment (NuFIT 5.2 / PDG 2024)", fontsize=11)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_pulls.pdf"))
fig.savefig(os.path.join(ROOT, "figures", "fig_pulls.png"), dpi=220)

# ------------------------------------------------------------ octant curve
d = np.linspace(0, 360, 2000)
s13_ = 0.022058
curve = 0.5 + np.sqrt(s13_) * np.cos(np.deg2rad(d)) / np.sqrt(2)
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(d, curve, lw=2, color="#1f4e79", label=r"$\sin^2\theta_{23}=\frac{1}{2}+s_{13}\cos\delta/\sqrt{2}$")
ax.axhspan(0.450 - 0.017, 0.450 + 0.017, color="#4e79a7", alpha=0.18,
           label=r"NuFIT 5.2, $1\sigma$")
ax.axhline(0.5, color="gray", ls=":", lw=0.8)
ax.plot([240], [s23], "o", ms=7, color="#e15759", label=r"$\delta=240^\circ$ (this work)")
ax.plot([230], [0.450], "s", ms=6, color="k", label="global best fit")
ax.errorbar([230], [0.450], yerr=[0.017], color="k", capsize=3, lw=1)
ax.set_xlabel(r"$\delta_{CP}$ [deg]", fontsize=10)
ax.set_ylabel(r"$\sin^2\theta_{23}$", fontsize=10)
ax.set_xlim(0, 360)
ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)
ax.set_title("the octant is tied to the CP phase", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "fig_sumrule.pdf"))
fig.savefig(os.path.join(ROOT, "figures", "fig_sumrule.png"), dpi=220)

print("wrote fig_pulls and fig_sumrule")
print("pulls:", [round(float(p), 3) for p in pulls])
