#!/usr/bin/env python3
"""Figures for the CPC manuscript (vector PDF)."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.4, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})
C1, C2, C3, C4 = "#1f4e79", "#d97b29", "#2a9d8f", "#b03a2e"

qs = np.exp(np.linspace(math.log(1.0005), math.log(5.0), 400))
left, right = qs, 1 / qs
arith = 4 * qs / (1 + qs) ** 2

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
a = ax[0]
a.loglog(qs - 1, np.abs(left - 1), color=C1, lw=1.3, label="left endpoint")
a.loglog(qs - 1, np.abs(right - 1), color=C2, lw=1.3, ls="--", label="right endpoint")
a.loglog(qs - 1, np.abs(arith - 1), color=C3, lw=1.3, label="arithmetic mean")
a.loglog(qs - 1, np.full_like(qs, 1e-16), color=C4, lw=1.6, label="geometric mean (exact)")
x = np.array([3e-3, 3e-1])
a.loglog(x, x, color="0.55", lw=0.7, ls=":")
a.loglog(x, 0.25 * x ** 2, color="0.55", lw=0.7, ls="-.")
a.text(0.02, 0.02, "slope 1", fontsize=6.8, color="0.45", rotation=32)
a.text(0.30, 0.010, "slope 2", fontsize=6.8, color="0.45", rotation=17)
a.set_xlabel(r"$q-1$"); a.set_ylabel(r"$|Q_{\rm rec}/Q-1|$")
a.set_ylim(1e-17, 5)
a.legend(loc="lower right", frameon=False)
a.set_title("(a) estimator bias versus mesh ratio")
for s in ("top", "right"): a.spines[s].set_visible(False)

b = ax[1]
ps = np.linspace(-0.5, 2.5, 200)
for q, c, ls in ((1.2, C1, "-"), (math.sqrt(3), C2, "--"), (2.0, C3, "-.")):
    b.semilogy(ps, q ** (ps - 1), color=c, ls=ls, lw=1.3, label=rf"$q={q:.3f}$")
b.axvline(1.0, color="0.5", lw=0.7, ls=":"); b.axhline(1.0, color="0.5", lw=0.7, ls=":")
b.text(1.05, 3.2, r"marginal $p=d-2$", fontsize=6.8, color="0.35")
b.set_xlabel(r"trial exponent $p$"); b.set_ylabel(r"$A_{n+1}/A_n=q^{\,p-d+2}$")
b.legend(loc="upper left", frameon=False)
b.set_title("(b) shell-to-shell ratio, $d=3$")
for s in ("top", "right"): b.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_bias.pdf")); plt.close(fig)

ks = np.arange(1, 9); qk = 2 ** (1 / ks)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.plot(ks, qk, "o-", color=C1, ms=4, label=r"left endpoint $=q_k$")
ax.plot(ks, 4 * qk / (1 + qk) ** 2, "s-", color=C3, ms=4, label=r"arithmetic $=4q_k/(1+q_k)^2$")
ax.plot(ks, np.ones_like(ks), "^-", color=C2, ms=4, label="geometric (exact)")
ax.set_xlabel(r"refinement level $k$  ($q_k=2^{1/k}$)")
ax.set_ylabel(r"$\alpha_{\rm eff}/\alpha$")
ax.set_yscale("log"); ax.set_ylim(0.85, 2.2)
ax.legend(loc="lower left", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_refinement.pdf")); plt.close(fig)

data = json.load(open(os.path.join(ROOT, "logs", "cpc_analysis.json")))
est = {int(k): v for k, v in data["D_lattice"]["estimator"].items()}
Rs = np.array(sorted(est))
Ql = np.array([est[r]["Q_left"] for r in Rs])
Qe = np.array([est[r]["Q_exact"] for r in Rs])
fac = np.array([est[r]["endpoint_factor"] for r in Rs])

fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.plot(Rs, Ql, "o-", color=C1, ms=3.5, lw=1.1, label=r"left endpoint, $4\pi R^2$")
ax.plot(Rs, Qe, "s-", color=C2, ms=3.5, lw=1.1, label=r"exact radius, $4\pi R(R+1)$")
ax.plot(Rs, fac, color="0.55", lw=0.8, ls="--", label=r"$R/(R+1)$ (continuum)")
ax.axhline(1.0, color="k", lw=0.5, ls=":")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"$Q(R)$  (unit charge)")
ax.set_ylim(0.7, 1.1)
ax.legend(loc="lower right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_lattice.pdf")); plt.close(fig)
print("wrote figures:", sorted(os.listdir(FIG)))
