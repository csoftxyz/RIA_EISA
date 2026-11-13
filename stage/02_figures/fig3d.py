import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.linewidth":0.6,"axes.edgecolor":"#333333"})

# ---------------- data (from model.py) ----------------
# name, rankG, nops, P, splitting, baseline
D = [("E6 27^3",6,1,13,0,3),("SO(10) 10",5,1,13,0,3),("SO(10) 120",5,1,13,0,3),
     ("SO(10) 126",5,1,13,0,3),("PS (15,2,2)",4,1,11,0,1),
     ("SU(5) 5",4,2,6,0,3),("SU(5) 45",4,2,6,0,3),("SU(5) 45+50",4,3,6,0,3),
     ("SU(5) 5+50",4,3,6,0,3),("PS (1,2,2)+(15,2,2)",4,2,1,1,1),
     ("SU(5) 50",4,1,3,0,3),
     ("SU(5) 5+45",4,4,3,1,3),("SU(5) 5+45+50",4,5,3,1,3),
     ("SO(10) 10+120",5,2,3,1,3),("SO(10) 10+126",5,2,3,1,3),
     ("SO(10) 120+126",5,2,3,1,3),("SO(10) 10+120+126",5,3,3,1,3)]

fig = plt.figure(figsize=(11.4,5.0))
ax = fig.add_subplot(121, projection="3d")

xs=np.array([d[1] for d in D]); ys=np.array([d[2] for d in D]); zs=np.array([d[3] for d in D])
sp=np.array([d[4] for d in D],dtype=bool)
ss=np.array([d[5] for d in D])

# glass ceiling plane at the simple-group baseline
gx,gy=np.meshgrid([3.6,6.4],[0.6,5.4])
ax.plot_surface(gx,gy,np.full_like(gx,3.0),color="#4C9BE8",alpha=0.13,
                linewidth=0,shade=False)
ax.plot([3.6,6.4,6.4,3.6,3.6],[0.6,0.6,5.4,5.4,0.6],[3,3,3,3,3],
        color="#4C9BE8",lw=0.9,ls="--",alpha=0.85)

# drop lines
for x,y,z,s in zip(xs,ys,zs,sp):
    ax.plot([x,x],[y,y],[0,z],color="#9aa4b2",lw=0.55,alpha=0.55,zorder=0)

cs = np.where(sp,"#1B9E77","#D95F02")
ax.scatter(xs[sp],ys[sp],zs[sp],s=95,c="#1B9E77",edgecolor="white",linewidth=0.8,
           depthshade=False,label="splitting  ($P_{add}=0$)")
ax.scatter(xs[~sp],ys[~sp],zs[~sp],s=95,c="#D95F02",edgecolor="white",linewidth=0.8,
           depthshade=False,marker="^",label="excluded")

for x,y,z,s in zip(xs,ys,zs,sp):
    if z>=6 or s:
        ax.text(x,y,z+0.55,f"{z}",ha="center",va="bottom",
                fontsize=8,color="#1B9E77" if s else "#8a3b00",fontweight="bold")

ax.set_xlabel("rank $G$",labelpad=6); ax.set_ylabel("operators",labelpad=6)
ax.set_zlabel("exact reach $P_{ex}$",labelpad=6)
ax.set_zlim(0,15); ax.set_xticks([4,5,6]); ax.set_yticks([1,2,3,4,5])
ax.view_init(elev=21,azim=-58)
ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
ax.grid(color="#e6e6e6",lw=0.5)
ax.set_title("(a) reach grows with operator count, not with the group",
             fontsize=9.5,pad=2,loc="left")
ax.legend(loc="upper left",bbox_to_anchor=(0.0,0.98),frameon=False,fontsize=8)

# ---------------- panel b: stage vs script ----------------
ax2 = fig.add_subplot(122)
names = ["SU(5)\n5+45","SU(5)\n5+45+50","SO(10)\n10+120","SO(10)\n10+126",
         "SO(10)\n120+126","SO(10)\n10+120+126","PS\n(1,2,2)+(15,2,2)"]
base  = [3,3,3,3,3,3,1]
extra = [0,0,0,0,0,0,0]
idx = np.arange(len(names)); w=0.62
ax2.bar(idx,base,w,color="#4C9BE8",edgecolor="white",linewidth=0.7,label="stage (baseline)")
ax2.bar(idx,extra,w,bottom=base,color="#1B9E77",edgecolor="white",linewidth=0.7,
        label="script (additional reach)")
for i,(b,e) in enumerate(zip(base,extra)):
    ax2.text(i,b+e+0.12,"0",ha="center",va="bottom",fontsize=9,color="#1B9E77",fontweight="bold")
ax2.set_xticks(idx); ax2.set_xticklabels(names,fontsize=7)
ax2.set_ylabel("exact reach $P_{ex}$"); ax2.set_ylim(0,4.6)
ax2.set_yticks([0,1,2,3])
for s in ("top","right"): ax2.spines[s].set_visible(False)
ax2.grid(axis="y",color="#eeeeee",lw=0.6); ax2.set_axisbelow(True)
ax2.set_title("(b) every splitting model sits on its baseline",fontsize=9.5,loc="left")
ax2.legend(frameon=False,fontsize=8,loc="upper right")
ax2.annotate("Pati–Salam: product group,\nno gauge unification (baseline 1)",
             xy=(6,1.0),xytext=(4.05,2.55),fontsize=7,color="#444444",
             arrowprops=dict(arrowstyle="->",lw=0.7,color="#777777"))

fig.tight_layout(pad=1.0)
for ext in ("pdf","png"):
    fig.savefig(f"fig_reach3d.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
print("saved fig_reach3d.pdf / .png")
