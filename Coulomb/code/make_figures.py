#!/usr/bin/env python3
"""Figures for the letter. Output: fig1.pdf/png, fig2.pdf/png."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
    "figure.dpi": 160, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans",
})


def coeff_curve(rule, qs, r_max=1e6):
    out = []
    for q in qs:
        r = q ** np.arange(int(math.ceil(math.log(r_max) / math.log(q))) + 1)
        if rule == "geom":
            rho = np.sqrt(r[:-1] * r[1:])
        elif rule == "left":
            rho = r[:-1]
        else:
            rho = 0.5 * (r[:-1] + r[1:])
        d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
        V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
        ref = (1 / r - 1 / r[-1]) / (4 * math.pi)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (V - V[-1]) / ref
        good = np.isfinite(ratio) & (ref > 0)
        out.append(float(np.median(ratio[good])))
    return np.array(out)


qs = np.logspace(np.log10(1.002), np.log10(4.0), 60)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

c_left = coeff_curve("left", qs)
c_ar = coeff_curve("arith", qs)
c_geo = coeff_curve("geom", qs)
ax[0].loglog(qs - 1, np.abs(c_left - 1), "o-", ms=2.5, lw=1.0, label="left endpoint")
ax[0].loglog(qs - 1, np.abs(c_ar - 1), "s-", ms=2.5, lw=1.0, label="arithmetic mean")
ax[0].loglog(qs - 1, np.maximum(np.abs(c_geo - 1), 1e-16), "^-", ms=2.5, lw=1.0,
             label="geometric mean")
ax[0].axhline(2.2e-16, color="0.6", ls=":", lw=0.8)
ax[0].set_xlabel(r"$q-1$")
ax[0].set_ylabel(r"$|Q_{\rm rec}/Q-1|$")
ax[0].set_ylim(1e-17, 3)
ax[0].legend(loc="lower right", frameon=False)
ax[0].set_title("(a) charge bias vs mesh ratio", fontsize=9)

ps = np.linspace(0.3, 2.6, 160)
for q, ls in [(1.2, "-"), (math.sqrt(3), "--"), (2.0, "-.")]:
    r = q ** np.arange(0.0, 40.0)
    rho = np.sqrt(r[:-1] * r[1:])
    d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
    V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
    ratios = []
    for p in ps:
        with np.errstate(invalid="ignore", divide="ignore"):
            A = (V - V[-1]) / (r ** (-p) - r[-1] ** (-p))
        ratios.append(float(np.median(A[1:-2] / A[:-3])))
    ax[1].semilogy(ps, ratios, ls, lw=1.0, label=r"$q=%.3f$" % q)
ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
ax[1].axvline(1.0, color="0.7", lw=0.8, ls=":")
ax[1].set_xlabel(r"trial power $p$   ($V\propto r^{-p}$)")
ax[1].set_ylabel(r"$A_{n+1}/A_n$")
ax[1].set_ylim(0.3, 3)
ax[1].legend(loc="upper left", frameon=False)
ax[1].set_title(r"(b) zero running only at $p=1$", fontsize=9)
fig.savefig("fig1.pdf")
fig.savefig("fig1.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ks = np.arange(1, 9)
qk = 2 ** (1 / ks)
left = np.array([coeff_curve("left", [q])[0] for q in qk])
geo = np.array([coeff_curve("geom", [q])[0] for q in qk])
ax.plot(ks, left, "o-", ms=3, lw=1.0, label=r"$\rho_n=r_n$")
ax.plot(ks, geo, "^-", ms=3, lw=1.0, label=r"$\rho_n^2=r_nr_{n+1}$")
ax.plot(ks, 1 + np.log(2) / ks, "k:", lw=0.9, label=r"$1+\ln2/k$")
ax.set_xlabel(r"refinement level $k$   ($q_k=2^{1/k}$)")
ax.set_ylabel(r"$\alpha_{\rm eff}/\alpha$")
ax.set_xticks(ks)
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0.99, 2.06)
fig.savefig("fig2.pdf")
fig.savefig("fig2.png", dpi=300)
plt.close(fig)
print("wrote fig1.pdf/png, fig2.pdf/png")
