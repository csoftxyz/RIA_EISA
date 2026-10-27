#!/usr/bin/env python3
"""Figures for the illustrated guides (Chinese and English).
Writes guide/g*.pdf|png (zh) and guide/en_g*.pdf|png (en)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

NAVY, TEAL, ORANGE, GREY, RED, GREEN = ("#12294a", "#0f7b7b", "#d1741f", "#8a9099",
                                        "#b02a37", "#2e7d32")

TXT = {
    "zh": {
        "g1a": "(a) 壳层与通量球：\n只有橙色虚线上取通量，$1/r$ 才精确",
        "g1b": "(b) 几何平均落在两壳的\n对数中点，任何 $q$ 都成立",
        "radius": "半径（对数轴）",
        "shell": "$n{=}%d$",
        "g2a": "(a) 电荷偏差 vs 网格比：只有几何平均恒为零",
        "g2b": "(b) 只有 $p=1$ 与壳层无关（零反常量纲）",
        "g2c": "(c) 伪跑动：几何平均恒为 1",
        "g2d": "(d) 有限层外推的系统偏差",
        "power": "试探势的幂 $p$   ($V\\propto r^{-p}$)",
        "level": "加细层数 $k$   ($q_k=2^{1/k}$)",
        "kmax": "拟合所用到的最大层数 $k_{\\max}$",
        "extrap": "外推得到的 $\\alpha$",
        "truth": "真值 1",
        "bias4": "$k_{\\max}=4$：偏 10%",
        "left": "左端点 $\\rho=r_n$",
        "arith": "算术平均",
        "right": "右端点 $\\rho=r_{n+1}$",
        "geom": "几何平均",
        "g3a": "(a) 离散通量（散度定理）：恒为 1，误差为零",
        "g3b": "(b) 同一数据、连续式估计子：10% 偏差全属估计",
        "region": "区域半径 n（96³ 立方格点）",
        "enclosed": "包围电荷",
        "shellR": "球壳半径 $R/a$",
        "mean": "均值 0.900",
        "g4t": "一次变换 + 一次求和，即可判定",
        "cand": "候选涌现耦合\n$\\hat Q^{(i)}_k$",
        "recomp": "同一数据用几何平均\n估计子重算 $\\hat Q^{(ii)}_k$",
        "b1": "$\\hat Q^{(ii)}$ 与 $k$ 无关\n$\\hat Q^{(i)}$ 漂移",
        "b2": "两者同步漂移",
        "b3": "$\\hat Q^{(ii)}$ 也漂移",
        "r1": "伪跑动\n偏差有闭式",
        "r2": "真跑动\n= 反常量纲",
        "r3": "库仑解释\n不适用",
        "g5a": "(a) 只有 marginal 幂给常数",
        "g5b": "(b) 两种“漂移”必须区分开",
        "measR": "测量半径 $r$",
        "amp": "测得振幅 $A(r)$",
        "estbias": "估计子偏差（伪）",
        "screen": "屏蔽（真）",
        "g6a": "(a) 加细流：改网格与估计子，物理不变",
        "g6b": "(b) 重整化群流：改观测尺度，积掉自由度",
        "g6c": "(c) 二者之差 = 估计子偏差（有闭式）",
        "meshcoarse": "粗网格",
        "meshfine": "细网格",
        "fixedphys": "物理不变",
        "scale": "观测尺度 $\\ln\\mu$",
        "cloud": "极化云被积掉",
        "comm": "换序之差",
    },
    "en": {
        "g1a": "(a) shells and flux sphere:\nonly the orange dotted radius is exact",
        "g1b": "(b) the geometric mean sits at the\nlog midpoint, for every $q$",
        "radius": "radius (log axis)",
        "shell": "$n{=}%d$",
        "g2a": "(a) charge bias vs mesh ratio: only the geometric mean vanishes",
        "g2b": "(b) shell-independent only at $p=1$ (zero anomalous dimension)",
        "g2c": "(c) spurious running: geometric mean is flat at 1",
        "g2d": "(d) systematic bias of finite-refinement extrapolation",
        "power": "trial power $p$   ($V\\propto r^{-p}$)",
        "level": "refinement level $k$   ($q_k=2^{1/k}$)",
        "kmax": "largest level used, $k_{\\max}$",
        "extrap": "extrapolated $\\alpha$",
        "truth": "true value 1",
        "bias4": "$k_{\\max}=4$: 10\\% low",
        "left": "left endpoint $\\rho=r_n$",
        "arith": "arithmetic mean",
        "right": "right endpoint $\\rho=r_{n+1}$",
        "geom": "geometric mean",
        "g3a": "(a) discrete flux (divergence theorem): exactly 1",
        "g3b": "(b) same data, continuum-style estimator: 10\\% all estimator",
        "region": "region radius n ($96^3$ lattice)",
        "enclosed": "enclosed charge",
        "shellR": "shell radius $R/a$",
        "mean": "mean 0.900",
        "g4t": "one transform and one sum settle it",
        "cand": "candidate coupling\n$\\hat Q^{(i)}_k$",
        "recomp": "recompute from the same\ndata with $\\hat Q^{(ii)}_k$",
        "b1": "$\\hat Q^{(ii)}$ flat in $k$,\n$\\hat Q^{(i)}$ drifts",
        "b2": "both drift equally",
        "b3": "$\\hat Q^{(ii)}$ drifts too",
        "r1": "artifact:\nbias in closed form",
        "r2": "real running\n= $\\gamma(\\alpha)$",
        "r3": "Coulomb reading\nunavailable",
        "g5a": "(a) only the marginal power gives a constant",
        "g5b": "(b) two kinds of drift, kept apart",
        "measR": "measurement radius $r$",
        "amp": "measured amplitude $A(r)$",
        "estbias": "estimator bias (fake)",
        "screen": "screening (real)",
        "g6a": "(a) refinement flow: new mesh, same physics",
        "g6b": "(b) RG flow: new scale, modes integrated out",
        "g6c": "(c) their difference = estimator bias (closed form)",
        "meshcoarse": "coarse",
        "meshfine": "fine",
        "fixedphys": "physics fixed",
        "scale": "observation scale $\\ln\\mu$",
        "cloud": "polarisation cloud removed",
        "comm": "order matters",
    },
}


def coeff(q, rule, r_max=1e6):
    r = q ** np.arange(int(math.ceil(math.log(r_max) / math.log(q))) + 1)
    rho = {"geom": np.sqrt(r[:-1] * r[1:]), "left": r[:-1],
           "arith": 0.5 * (r[:-1] + r[1:]), "right": r[1:]}[rule]
    d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
    V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
    ref = (1 / r - 1 / r[-1]) / (4 * math.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (V - V[-1]) / ref
    good = np.isfinite(ratio) & (ref > 0)
    return float(np.median(ratio[good]))


def build(lang):
    L = TXT[lang]
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
        "axes.edgecolor": "#4a4a4a", "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "dejavusans", "savefig.bbox": "tight", "figure.dpi": 150,
        "font.family": "Noto Sans CJK JP",
    })
    pre = "guide/" if lang == "zh" else "guide/en_"

    # ---------------- g1 schematic
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    q = 1.6
    rr = q ** np.arange(0, 6)
    for i, r in enumerate(rr):
        ax.add_patch(Circle((0, 0), r, fill=False, lw=1.0, color=NAVY if i % 2 == 0 else TEAL,
                            ls="-" if i % 2 == 0 else "--"))
    for i in range(len(rr) - 1):
        ax.add_patch(Circle((0, 0), math.sqrt(rr[i] * rr[i + 1]), fill=False, lw=1.2,
                            color=ORANGE, ls=":"))
    ax.plot([0], [0], "o", ms=5, color=RED)
    ax.annotate("$+Q$", (0, 0), textcoords="offset points", xytext=(6, -10), color=RED, fontsize=9)
    ax.annotate("", xy=(rr[3], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9))
    ax.text(rr[3] * 0.55, 0.12, "$r_n$", color=NAVY, fontsize=9)
    ax.annotate("", xy=(math.sqrt(rr[1] * rr[2]), rr[0] * 0.35), xytext=(0, rr[0] * 0.35),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.text(math.sqrt(rr[1] * rr[2]) * 0.45, rr[0] * 0.42,
            r"$\rho_n=\sqrt{r_nr_{n+1}}$", color=ORANGE, fontsize=9)
    ax.set_xlim(-rr[-1] * 1.05, rr[-1] * 1.05)
    ax.set_ylim(-rr[-1] * 0.85, rr[-1] * 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(L["g1a"], fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    r = q ** np.arange(0, 6)
    for i, x in enumerate(r):
        ax2.axvline(x, color=NAVY if i % 2 == 0 else TEAL, lw=1.0, ls="-" if i % 2 == 0 else "--")
    for i in range(len(r) - 1):
        ax2.plot([math.sqrt(r[i] * r[i + 1])], [0.5], "|", ms=14, color=ORANGE, mew=2)
        ax2.annotate(L["shell"] % i, (0.5 * (r[i] + r[i + 1]), 0.18), ha="center",
                     fontsize=7.5, color=GREY)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel(L["radius"])
    ax2.set_title(L["g1b"], fontsize=9)
    ax2.spines["left"].set_visible(False)
    fig.savefig(pre + "g1_scheme.pdf")
    fig.savefig(pre + "g1_scheme.png", dpi=300)
    plt.close(fig)

    # ---------------- g2 four panels
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    qs = np.logspace(np.log10(1.002), np.log10(4.0), 50)
    for rule, key, col, mk in [("left", "left", ORANGE, "o"), ("arith", "arith", GREY, "s"),
                               ("right", "right", TEAL, "v"), ("geom", "geom", GREEN, "^")]:
        c = np.array([coeff(x, rule) for x in qs])
        ax[0, 0].loglog(qs - 1, np.maximum(np.abs(c - 1), 1e-17), mk + "-", ms=2, lw=1,
                        color=col, label=L[key])
    ax[0, 0].axhline(2.2e-16, color="0.6", ls=":", lw=0.8)
    ax[0, 0].set_xlabel("$q-1$")
    ax[0, 0].set_ylabel("$|Q_{\\rm rec}/Q-1|$")
    ax[0, 0].legend(loc="lower right", frameon=False)
    ax[0, 0].set_title(L["g2a"], fontsize=9)
    ax[0, 0].set_ylim(1e-17, 5)
    ps = np.linspace(0.3, 2.6, 120)
    for qq, col, ls in [(1.2, TEAL, "-"), (math.sqrt(3), NAVY, "--"), (2.0, ORANGE, "-.")]:
        r = qq ** np.arange(0.0, 40.0)
        rho = np.sqrt(r[:-1] * r[1:])
        d = (r[1:] - r[:-1]) / (4 * math.pi * rho ** 2)
        V = np.concatenate([np.cumsum(d[::-1])[::-1], [0.0]])
        ratio = []
        for p in ps:
            with np.errstate(invalid="ignore", divide="ignore"):
                A = (V - V[-1]) / (r ** (-p) - r[-1] ** (-p))
            ratio.append(float(np.median(A[1:-2] / A[:-3])))
        ax[0, 1].semilogy(ps, ratio, ls, color=col, lw=1.1, label="$q=%.2f$" % qq)
    ax[0, 1].axhline(1, color="k", ls=":", lw=0.8)
    ax[0, 1].axvline(1, color=RED, ls=":", lw=0.9)
    ax[0, 1].set_xlabel(L["power"])
    ax[0, 1].set_ylabel("$A_{n+1}/A_n$")
    ax[0, 1].legend(frameon=False, loc="upper left")
    ax[0, 1].set_title(L["g2b"], fontsize=9)
    ks = np.arange(1, 9)
    qk = 2 ** (1 / ks)
    ax[1, 0].plot(ks, [coeff(x, "left") for x in qk], "o-", ms=3.2, color=ORANGE, lw=1.1,
                  label=L["left"])
    ax[1, 0].plot(ks, [coeff(x, "geom") for x in qk], "^-", ms=3.2, color=GREEN, lw=1.1,
                  label=L["geom"])
    ax[1, 0].plot(ks, 1 + np.log(2) / ks, "k:", lw=0.9, label="$1+\\ln2/k$")
    ax[1, 0].set_xlabel(L["level"])
    ax[1, 0].set_ylabel("$\\alpha_{\\rm eff}/\\alpha$")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].set_title(L["g2c"], fontsize=9)
    ax[1, 0].set_xticks(ks)
    kmax = [2, 3, 4, 6, 8]
    bias = [0.8284, 0.8710, 0.8957, 0.9236, 0.9390]
    ax[1, 1].plot(kmax, bias, "s-", ms=3.5, color=NAVY, lw=1.1, label=L["extrap"])
    ax[1, 1].axhline(1.0, color=GREEN, lw=1.2, ls="--", label=L["truth"])
    ax[1, 1].fill_between(kmax, bias, 1.0, color=RED, alpha=0.10)
    ax[1, 1].annotate(L["bias4"], (4, 0.8957), textcoords="offset points", xytext=(10, -16),
                      fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax[1, 1].set_xlabel(L["kmax"])
    ax[1, 1].set_ylabel(L["extrap"])
    ax[1, 1].set_ylim(0.78, 1.04)
    ax[1, 1].legend(frameon=False, loc="lower right")
    ax[1, 1].set_title(L["g2d"], fontsize=9)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    fig.savefig(pre + "g2_four.pdf")
    fig.savefig(pre + "g2_four.png", dpi=300)
    plt.close(fig)

    # ---------------- g3 lattice
    flux_n = [2, 3, 5, 8, 12]
    Rs = np.arange(8, 16)
    Q = np.array([0.907143, 0.917805, 0.864661, 0.875100, 0.923950, 0.920667, 0.883202, 0.908724])
    fig, ax = plt.subplots(2, 1, figsize=(4.6, 4.4))
    ax[0].plot(flux_n, np.ones(len(flux_n)), "o", ms=6, color=GREEN)
    ax[0].axhline(1.0, color=GREEN, ls="--", lw=0.9)
    ax[0].set_ylim(0.8, 1.02)
    ax[0].set_xticks(flux_n)
    ax[0].set_xlabel(L["region"])
    ax[0].set_ylabel(L["enclosed"])
    ax[0].set_title(L["g3a"], fontsize=9)
    ax[1].plot(Rs, Q, "s-", ms=3.5, color=ORANGE, lw=1.0)
    ax[1].axhline(1.0, color=GREEN, ls="--", lw=1.0, label=L["truth"])
    ax[1].axhline(Q.mean(), color=ORANGE, ls=":", lw=1.0, label=L["mean"])
    ax[1].set_ylim(0.8, 1.03)
    ax[1].set_xlabel(L["shellR"])
    ax[1].set_ylabel("$4\\pi R^2[\\bar V(R)-\\bar V(R+1)]$")
    ax[1].legend(frameon=False, loc="lower right")
    ax[1].set_title(L["g3b"], fontsize=9)
    fig.tight_layout()
    fig.savefig(pre + "g3_lattice.pdf")
    fig.savefig(pre + "g3_lattice.png", dpi=300)
    plt.close(fig)

    # ---------------- g4 flow chart
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=NAVY)

    def arrow(x1, y1, x2, y2, col=NAVY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.0, color=col, shrinkA=2, shrinkB=2))

    box(0.2, 2.1, 2.0, 1.0, L["cand"], "#eef3f8", NAVY)
    box(2.8, 2.1, 2.2, 1.0, L["recomp"], "#eef3f8", NAVY)
    arrow(2.2, 2.6, 2.8, 2.6)
    box(5.5, 3.4, 2.1, 1.1, L["b1"], "#fdf1e3", ORANGE)
    box(5.5, 1.8, 2.1, 1.1, L["b2"], "#fdf1e3", ORANGE)
    box(5.5, 0.2, 2.1, 1.1, L["b3"], "#fdf1e3", ORANGE)
    arrow(5.0, 2.75, 5.5, 3.95, ORANGE)
    arrow(5.0, 2.6, 5.5, 2.35, ORANGE)
    arrow(5.0, 2.45, 5.5, 0.75, ORANGE)
    box(8.0, 3.4, 1.9, 1.1, L["r1"], "#eaf4ec", GREEN)
    box(8.0, 1.8, 1.9, 1.1, L["r2"], "#eaf4ec", GREEN)
    box(8.0, 0.2, 1.9, 1.1, L["r3"], "#fbeaec", RED)
    arrow(7.6, 3.95, 8.0, 3.95, GREEN)
    arrow(7.6, 2.35, 8.0, 2.35, GREEN)
    arrow(7.6, 0.75, 8.0, 0.75, RED)
    ax.text(5.0, 4.75, L["g4t"], ha="center", fontsize=9.5, color=NAVY)
    fig.savefig(pre + "g4_test.pdf")
    fig.savefig(pre + "g4_test.png", dpi=300)
    plt.close(fig)

    # ---------------- g5 marginality
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    r = np.logspace(-1, 1.4, 200)
    ax[0].loglog(r, np.ones_like(r), color=GREEN, lw=1.6, label="$p=1$: $\\gamma=0$")
    ax[0].loglog(r, r / r[0], color=ORANGE, lw=1.2, label="$p=2$: $A\\propto r$")
    ax[0].loglog(r, (r / r[0]) ** -0.5, color=TEAL, lw=1.2, label="$p=1/2$: $A\\propto r^{-1/2}$")
    ax[0].set_xlabel(L["measR"])
    ax[0].set_ylabel(L["amp"])
    ax[0].legend(frameon=False)
    ax[0].set_title(L["g5a"], fontsize=9)
    ax[1].plot(r, np.ones_like(r), color=ORANGE, ls="--", lw=1.2, label=L["estbias"])
    ax[1].plot(r, 1 + 0.30 / (1 + r ** 2), color=TEAL, lw=1.4, label=L["screen"])
    ax[1].set_xscale("log")
    ax[1].set_xlabel(L["measR"])
    ax[1].set_ylabel("$\\alpha_{\\rm meas}(r)/\\alpha$")
    ax[1].legend(frameon=False)
    ax[1].set_title(L["g5b"], fontsize=9)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(pre + "g5_marginal.pdf")
    fig.savefig(pre + "g5_marginal.png", dpi=300)
    plt.close(fig)

    # ---------------- g6 the two flows
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for a in ax:
        a.axis("off")
    a = ax[0]
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    for i, x in enumerate([0.12, 0.30, 0.52, 0.78]):
        a.add_patch(FancyBboxPatch((x - 0.075, 0.42 - 0.045), 0.15, 0.09,
                                   boxstyle="round,pad=0.01", fc="#eef3f8", ec=NAVY, lw=0.8))
        for k in range(3):
            a.plot([x - 0.06 + 0.045 * k], [0.50], "|", ms=6, color=TEAL)
    a.annotate("", xy=(0.95, 0.20), xytext=(0.05, 0.20),
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    a.text(0.5, 0.10, "$q_k\\to q_{k+1}$", ha="center", fontsize=8.5, color=ORANGE)
    a.text(0.5, 0.72, L["fixedphys"], ha="center", fontsize=8.5, color=GREEN)
    a.set_title(L["g6a"], fontsize=8.5)
    a = ax[1]
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.annotate("", xy=(0.05, 0.5), xytext=(0.95, 0.5),
               arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))
    a.text(0.5, 0.28, L["scale"], ha="center", fontsize=8.0)
    for i, x in enumerate([0.25, 0.45, 0.65, 0.85]):
        a.add_patch(Circle((x, 0.62), 0.055 + 0.02 * i, fill=False, color=ORANGE, lw=0.8, ls=":"))
    a.text(0.5, 0.86, L["cloud"], ha="center", fontsize=8.0, color=ORANGE)
    a.set_title(L["g6b"], fontsize=8.5)
    a = ax[2]
    a.set_xlim(0, 2)
    a.set_ylim(0, 2)
    a.add_patch(FancyBboxPatch((0.15, 0.75), 0.7, 0.5, boxstyle="round,pad=0.05",
                               fc="#eef3f8", ec=NAVY, lw=0.8))
    a.add_patch(FancyBboxPatch((1.15, 0.75), 0.7, 0.5, boxstyle="round,pad=0.05",
                               fc="#eef3f8", ec=NAVY, lw=0.8))
    a.text(0.5, 1.0, "refine", ha="center", fontsize=8.0)
    a.text(1.5, 1.0, "measure", ha="center", fontsize=8.0)
    a.annotate("", xy=(1.15, 0.6), xytext=(0.85, 0.6),
               arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    a.text(1.0, 0.30, L["comm"], ha="center", fontsize=8.0, color=RED)
    a.text(1.0, 1.55, "$\\beta_{\\rm refine}=0$", ha="center", fontsize=8.5, color=GREEN)
    a.text(1.0, 1.30, "$\\beta_{\\rm phys}\\neq0$", ha="center", fontsize=8.5, color=ORANGE)
    a.set_title(L["g6c"], fontsize=8.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(pre + "g6_twoflows.pdf")
    fig.savefig(pre + "g6_twoflows.png", dpi=300)
    plt.close(fig)
    print("built", lang)


for lang in ("zh", "en"):
    build(lang)
