import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5})

MODELS = [
 ("SU(5)  5+45",            4, 3, 0, "SU5"),
 ("SU(5)  5+45+50",         4, 3, 0, "SU5"),
 ("SO(10)  10+126",         5, 3, 0, "SO10"),
 ("SO(10)  120+126",        5, 3, 1, "SO10"),
 ("SO(10)  10+120+126",     5, 3, 0, "SO10"),
 ("PS  (1,2,2)+(15,2,2)",   4, 1, 0, "PS"),
]
COL = {"SU5":"#2E86AB","SO10":"#A23B72","PS":"#E8871E"}

fig = plt.figure(figsize=(15.5,8.6))
ax = fig.add_axes([0.015,0.03,0.60,0.90], projection="3d")

n = len(MODELS)
xs = np.arange(n, dtype=float)
ys = 0.45*np.sin(np.linspace(0, 2.4, n))          # gentle arc for depth
base = np.array([m[2] for m in MODELS], float)
cols = [COL[m[4]] for m in MODELS]

# stage floor
ax.bar3d(-0.8, -0.9, -0.28, n+0.6, 1.8, 0.26, color="#e9edf2",
         edgecolor="#cfd6de", linewidth=0.5, shade=True)

# pillars
for i,(name,rank,P,extra,g) in enumerate(MODELS):
    ax.bar3d(xs[i]-0.30, ys[i]-0.30, 0.0, 0.60, 0.60, P-extra,
             color=COL[g], edgecolor="white", linewidth=0.9, shade=True, alpha=0.95)
    if extra:
        ax.bar3d(xs[i]-0.30, ys[i]-0.30, P-extra, 0.60, 0.60, extra,
                 color="#1B9E77", edgecolor="white", linewidth=0.9, shade=True, alpha=0.98)
    ax.text(xs[i], ys[i], P+0.22, f"{P}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=("#1B9E77" if extra else COL[g]))

# the stage plane at 3 (simple groups) and at 1 (Pati-Salam)
gx,gy = np.meshgrid([-0.45, 4.45], [-0.55, 0.75])
ax.plot_surface(gx, gy, np.full_like(gx, 3.0), color="#2E86AB", alpha=0.10,
                linewidth=0, shade=False, zorder=1)
ax.plot([-0.45,4.45,4.45,-0.45,-0.45],[-0.55,-0.55,0.75,0.75,-0.55],[3,3,3,3,3],
        color="#2E86AB", lw=1.0, ls=(0,(4,3)), alpha=0.75)
gx2,gy2 = np.meshgrid([4.55, 5.45], [-0.55, 0.75])
ax.plot_surface(gx2, gy2, np.full_like(gx2, 1.0), color="#E8871E", alpha=0.12,
                linewidth=0, shade=False)
ax.plot([4.55,5.45,5.45,4.55,4.55],[-0.55,-0.55,0.75,0.75,-0.55],[1,1,1,1,1],
        color="#E8871E", lw=1.0, ls=(0,(4,3)), alpha=0.8)

# empty "script" volume above the stage
ax.plot([-0.45,4.45,4.45,-0.45,-0.45],[-0.55,-0.55,0.75,0.75,-0.55],[5.4]*5,
        color="#7a7a7a", lw=0.8, ls=(0,(1,3)))
for x0 in (-0.45,4.45):
    ax.plot([x0,x0],[-0.55,-0.55],[3,5.4],color="#7a7a7a",lw=0.8,ls=(0,(1,3)))
ax.text(2.0,0.10,5.55,"script: $P_{\\rm add}\\leq 1$",ha="center",va="bottom",
        fontsize=12,color="#4a4a4a",style="italic")
ax.text(2.0,0.10,3.12,"stage: baseline $\\mathcal{B}=2+1$",ha="center",va="bottom",
        fontsize=11,color="#1b5e82")

ax.set_xlim(-0.9,6.0); ax.set_ylim(-1.0,1.3); ax.set_zlim(0,6.2)
ax.set_xticks([]); ax.set_yticks([])
ax.set_zticks([0,1,2,3,4,5])
ax.set_zlabel("exact reach  $P_{\\rm ex}$",labelpad=10,fontsize=11)
ax.view_init(elev=17, azim=-72)
for pane in (ax.xaxis,ax.yaxis,ax.zaxis): pane.set_alpha(0.0)
ax.grid(color="#e3e6ea",lw=0.5)

# ---- right panel: stage vs script ----
ax2 = fig.add_axes([0.665,0.10,0.315,0.78])
labels = [m[0] for m in MODELS][::-1]
stage  = np.array([m[2] for m in MODELS][::-1], float)
script = np.array([m[3] for m in MODELS][::-1], float)
y = np.arange(len(labels))
ax2.barh(y, stage, 0.55, color="#2E86AB", edgecolor="white", linewidth=0.8,
         label="stage (baseline)")
ax2.barh(y, script, 0.55, left=stage, color="#1B9E77", edgecolor="white",
         linewidth=0.8, label="script (additional)")
for i,g in enumerate([m[4] for m in MODELS][::-1]):
    v = script[i]
    ax2.text(stage[i]+v+0.09, i, ("0" if v==0 else "%d"%int(v)), va="center", ha="left",
             fontsize=10.5, color="#1B9E77", fontweight="bold")
ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=9.5)
for tl,g in zip(ax2.get_yticklabels(),[m[4] for m in MODELS][::-1]):
    tl.set_color(COL[g])
ax2.set_xlim(0,4.3); ax2.set_xticks([0,1,2,3])
ax2.set_xlabel("exact reach")
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
ax2.grid(axis="x",color="#eeeeee",lw=0.7); ax2.set_axisbelow(True)
ax2.set_title("six splitting models: five on the baseline, one above it",fontsize=11,loc="left",pad=8)
ax2.legend(frameon=False,fontsize=9,loc="lower right")
ax2.annotate("Pati–Salam: product group,\nno gauge unification",
             xy=(1.0,0), xytext=(1.75,1.05), fontsize=8.5, color="#7a4a10",
             arrowprops=dict(arrowstyle="->",lw=0.8,color="#E8871E"))

fig.text(0.015,0.965,"Grand unification builds the stage, not the script",
         fontsize=16,fontweight="bold",color="#1a1a1a")
fig.text(0.015,0.925,
 "Five of six spectrum-splitting models reach exactly their baseline; the sixth adds one relation, and the antisymmetric 120 is its only source.",
 fontsize=10.5,color="#555555")

for ext in ("pdf","png"):
    fig.savefig(f"fig_stage.{ext}",dpi=300,bbox_inches=None,facecolor="white")
print("ok")
