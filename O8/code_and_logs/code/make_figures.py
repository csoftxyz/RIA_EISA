#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the NPB manuscript (vector PDF + PNG preview)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, os.pardir, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.linewidth": 0.8, "xtick.direction": "in",
                     "ytick.direction": "in", "legend.frameon": False})


def In(n, b=1.0):
    return mp.besseli(int(n), b)


def ZF(F, b=1.0):
    return sum(In(n, b) ** F for n in range(-30, 31))


def face(F, b=1.0):
    return sum(In(n, b) * In(n + 1, b) ** (F - 1) for n in range(-30, 31)) / ZF(F, b)


def Wsplit(F, A, b=1.0):
    return sum(In(n, b) ** A * In(n + 1, b) ** (F - A) for n in range(-30, 31)) / ZF(F, b)


def cbeta(b, F=8):
    """specific heat in double precision from the exact series"""
    from scipy.special import iv
    n = np.arange(0, 40)
    def lZ(x):
        w = iv(n, x) ** F
        w = np.concatenate([w[:1], 2 * w[1:]])
        return np.log(w.sum())
    h = 1e-4 * max(b, 0.1)
    d2 = (lZ(b + h) - 2 * lZ(b) + lZ(b - h)) / h ** 2
    return b ** 2 * d2


# ---------------------------------------------------------------- figure 1
def fig1():
    Fs = [4, 6, 8, 12, 20, 32]
    sig = [float(-mp.log(face(F))) for F in Fs]
    sinf = float(-mp.log(In(1) / In(0)))
    d = [sinf - s for s in sig]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a = ax[0]
    Fgrid = np.linspace(3.2, 34, 200)
    a.plot(Fgrid, [sinf] * len(Fgrid), "--", color="0.55", lw=1.0,
           label=r"$\sigma_\infty$")
    a.plot(Fs, sig, "o", ms=4.5, color="#1f4e79", label=r"$\sigma_F$ (exact)")
    a.set_xlabel(r"number of faces $F$")
    a.set_ylabel(r"finite-volume tension $\sigma_F$")
    a.set_xlim(3, 34)
    a.set_ylim(0.66, 0.83)
    a.legend(loc="lower right")
    a.text(0.04, 0.92, "(a)", transform=a.transAxes)

    b = ax[1]
    idx = [i for i, F in enumerate(Fs) if F >= 4]
    fit = np.polyfit([Fs[i] for i in idx if Fs[i] >= 8],
                     np.log([d[i] for i in idx if Fs[i] >= 8]), 1)
    Ff = np.linspace(4, 33, 200)
    r = float(In(1) / In(0))
    i21 = float(In(2) / In(1))
    bracket = 1 + i21 * r - 2 * r ** 2
    asym = [float(r ** (F - 2) * bracket) for F in Ff]
    b.semilogy(Ff, asym, "-", color="#1f4e79", lw=1.2,
               label=r"analytic: $r^{F-2}[1+(I_2/I_1)r-2r^2]$")
    b.semilogy([Fs[i] for i in idx], [d[i] for i in idx], "s", ms=4.5,
               color="#a63d2f", label=r"exact $\sigma_\infty-\sigma_F$")
    b.set_xlabel(r"$F$")
    b.set_ylabel(r"$\sigma_\infty-\sigma_F$")
    b.legend(loc="upper right")
    b.text(0.04, 0.92, "(b)", transform=b.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_finite_volume.pdf"))
    fig.savefig(os.path.join(FIG, "fig1_finite_volume.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a = ax[0]
    A = np.arange(1, 5)
    W = [float(Wsplit(8, int(k))) for k in A]
    r = float(In(1) / In(0))
    a.semilogy(A, W, "o", ms=5, color="#1f4e79", label=r"exact $\langle W_A\rangle$")
    a.semilogy(A, r ** A, "--", color="0.45", lw=1.0, label=r"$r^{A}$, $r=I_1/I_0$")
    for k, w in zip(A, W):
        a.annotate("%.4f" % w, (k, w), textcoords="offset points", xytext=(4, 6),
                   fontsize=7.5)
    a.set_xlabel(r"enclosed faces $A$")
    a.set_ylabel(r"$\langle W(C)\rangle$")
    a.set_xticks(A)
    a.set_ylim(0.06, 0.65)
    a.legend(loc="lower left")
    a.text(0.04, 0.92, "(a)", transform=a.transAxes)

    b = ax[1]
    labels = ["ratio\nmatch", "curvature\nmatch", "amplitude\nmatch"]
    vals = [0.61991, 1.0, 1.0]
    b.bar(np.arange(3), vals, width=0.55, color=["#1f4e79", "#8faadc", "#c9d6ea"],
          edgecolor="0.3", lw=0.5)
    b.axhline(4 / np.pi, color="#a63d2f", ls="--", lw=1.1,
              label=r"Villain fixed point $4/\pi$")
    b.axhline(1.0, color="0.6", ls=":", lw=0.9, label=r"$\beta_W=1$")
    b.set_xticks(np.arange(3))
    b.set_xticklabels(labels)
    b.set_ylabel(r"$\beta_V$")
    b.set_ylim(0, 1.45)
    b.legend(loc="upper left", fontsize=7.5)
    b.text(0.04, 0.92, "(b)", transform=b.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_loops_duality.pdf"))
    fig.savefig(os.path.join(FIG, "fig2_loops_duality.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a = ax[0]
    bs = np.linspace(0.02, 6, 120)
    cs = [float(cbeta(float(b))) for b in bs]
    a.plot(bs, cs, "-", color="#1f4e79", lw=1.3)
    a.axhline(3.5, color="0.55", ls="--", lw=1.0, label=r"$(F-1)/2=3.5$")
    a.axvline(1.0, color="0.75", ls=":", lw=1.0)
    a.plot([2.05], [5.7452], "o", ms=4.5, color="#a63d2f")
    a.annotate(r"$\beta\simeq2.05$", (2.05, 5.7452), textcoords="offset points",
               xytext=(-46, 6), fontsize=7.5, color="#a63d2f")
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$c(\beta)=\beta^2\,\mathrm{Var}(\sum_f\cos\theta_f)$")
    a.set_xlim(0, 6)
    a.set_ylim(0, 6.4)
    a.legend(loc="lower right")
    a.text(0.04, 0.92, "(a)", transform=a.transAxes)

    b = ax[1]
    ev = [0, 0, 0, 0, 2 * np.sqrt(3), -2 * np.sqrt(3)]
    b.axhline(0, color="0.8", lw=0.6)
    b.vlines(range(6), 0, ev, color="#1f4e79", lw=1.4)
    b.plot(range(6), ev, "o", ms=4, color="#1f4e79")
    b.fill_between([-0.5, 3.5], -0.3, 0.3, color="#a63d2f", alpha=0.15)
    b.text(1.5, 0.45, "4 zero modes", color="#a63d2f", fontsize=8, ha="center")
    b.set_xticks(range(6))
    b.set_xticklabels([r"$v_1$", r"$v_2$", r"$v_3$", r"$v_4$", r"$v_5$", r"$v_6$"])
    b.set_ylabel(r"eigenvalue of $\Gamma$  (units of $t$)")
    b.set_ylim(-4.2, 4.2)
    b.set_title(r"$\Gamma=\varepsilon\otimes J_2$,  rank 2", fontsize=8.5)
    b.text(0.04, 0.92, "(b)", transform=b.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_thermo_fermion.pdf"))
    fig.savefig(os.path.join(FIG, "fig3_thermo_fermion.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("figures written to", os.path.abspath(FIG))
