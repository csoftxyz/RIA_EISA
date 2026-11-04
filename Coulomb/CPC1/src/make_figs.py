import numpy as np, matplotlib as mpl, math
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'dejavuserif','font.size':9,'axes.linewidth':0.8,
 'axes.labelsize':9.5,'legend.fontsize':7.6,'xtick.labelsize':8,'ytick.labelsize':8,
 'figure.dpi':150,'savefig.bbox':'tight','legend.frameon':False,'lines.linewidth':1.4})
C=dict(blue='#1f4e79',red='#b22222',orange='#d9822b',green='#2e7d32',
       grey='#6b6b6b',purple='#6a3d9a',teal='#0f7b7b')
def save(fig,name):
    fig.savefig(f'fig/{name}.pdf'); fig.savefig(f'fig/{name}.png',dpi=230); plt.close(fig)

Om=lambda d: 2*math.pi**(d/2)/math.gamma(d/2)
def rho_star(d,q,rn=1.0):
    return (rn**(2-d)*(d-2)*(q-1)/(1-q**(2-d)))**(1/(d-1))
def rho_naive(d,q,rn=1.0):
    return (rn**(2-d)*(q-1)/(1-q**(2-d)))**(1/(d-1))

# ============================================ FIG 1 : the idea in one picture
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.3,2.85))
d,q,rn=3,2.0,1.0
rr=np.linspace(rn,q,500); y=rr**(1-d); rho=rho_star(d,q)
a1.fill_between(rr,y,color=C['blue'],alpha=0.13,lw=0)
a1.plot(rr,y,color=C['blue'],lw=1.7)
a1.add_patch(Rectangle((rn,0),q-rn,rho**(1-d),fc='none',ec=C['red'],lw=1.6,ls=(0,(5,2))))
a1.axvline(rho,color=C['red'],lw=1.0,alpha=.55,ls=':')
a1.annotate('',xy=(rho,0.60),xytext=(rn,0.60),arrowprops=dict(arrowstyle='<->',color=C['red'],lw=1.0))
a1.text((rn+rho)/2+0.02,0.63,r'$\rho_\ast-r_n$',color=C['red'],fontsize=8.5)
a1.text(1.55,0.42,r'$r^{1-d}$',color=C['blue'],fontsize=11)
a1.text(1.06,0.09,'equal areas',color=C['red'],fontsize=8.2,style='italic')
a1.set_xlim(0.97,2.08); a1.set_ylim(0,0.80); a1.set_yticks([])
a1.set_xlabel(r'$r$'); a1.set_ylabel(r'$r^{\,1-d}$')
a1.set_title('(a)  the exact rule is a one-point Gauss rule',fontsize=9.2)

q=1.5; rs=rho_star(3,q)
a2.axvspan(1,q,color=C['blue'],alpha=.08,lw=0)
a2.text(1.25,1.32,'shell $[r_n,r_{n+1}]$',fontsize=8,color=C['blue'],ha='center')
pts=[(1.0,C['orange'],r'left  $\rho^\prime=r_n$',1.46),
     ((1+q)/2,C['purple'],r'midpoint',1.46),
     (q,C['green'],r'right  $\rho^\prime=r_{n+1}$',1.46)]
for x,c,lab,yy in pts:
    a2.plot([x,x],[0,1.16],color=c,lw=1.7)
    a2.annotate(lab,xy=(x,1.16),xytext=(x,yy),ha='center',fontsize=7.4,color=c)
a2.plot([rs,rs],[0,1.16],color=C['red'],lw=3.0,solid_capstyle='round')
a2.annotate(r'$\rho_\ast=\sqrt{r_nr_{n+1}}$'+'\n exact for all $q$',xy=(rs,0.55),
            xytext=(rs+0.035,0.30),fontsize=7.8,color=C['red'],
            arrowprops=dict(arrowstyle='->',color=C['red'],lw=.8))
a2.set_xlim(0.86,1.66); a2.set_ylim(0,1.60); a2.set_yticks([])
a2.set_xlabel(r'$r/r_n$      ($q=1.5$)')
a2.set_title('(b)  four candidate radii, one exact',fontsize=9.2)
for s in ('top','right','left'): a2.spines[s].set_visible(False)
fig.tight_layout(); save(fig,'fig1_concept')

# ============================================ FIG 2 : the naive generalisation
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.3,2.85),gridspec_kw={'width_ratios':[1.5,1]})
cols={3:C['blue'],4:C['red'],5:C['green'],6:C['purple'],7:C['orange']}
qq=np.linspace(1.001,2.2,1500)
qstar={3:1.0,4:1.6180339887,5:1.8392867552,6:1.9275619755,7:1.9659482366}
a1.fill_between(qq,1,qq,color=C['grey'],alpha=.09,lw=0)
for dd,c in cols.items():
    a1.semilogy(qq,[rho_naive(dd,x) for x in qq],color=c,lw=1.5,label=fr'$d={dd}$')
    a1.semilogy([qstar[dd]],[1.0],'o',ms=4,color=c,mec='w',mew=.7)
a1.axhline(1,color='k',lw=.6,ls=':')
a1.text(1.63,1.30,'inside the shell',fontsize=7.8,color=C['grey'])
a1.text(1.03,1.70,'sampling point leaves the shell\n'
        r'$\Rightarrow$ no continuum limit',fontsize=7.8,color=C['red'],va='top')
a1.annotate(r'$q^\ast_4=\varphi$',xy=(1.618,1.0),xytext=(1.66,1.12),fontsize=8,
            color=C['red'],arrowprops=dict(arrowstyle='->',color=C['red'],lw=.8))
a1.set_xlim(1,2.2); a1.set_ylim(0.73,1.78)
a1.set_xlabel(r'mesh ratio $q=r_{n+1}/r_n$'); a1.set_ylabel(r'$\rho_{\rm naive}/r_n$')
a1.legend(loc='lower right',ncol=2,fontsize=7.2,columnspacing=.9,handlelength=1.4)
a1.set_title(r'(a)  the $d$-general form without the $(d-2)$ factor',fontsize=9.2)

ds=np.arange(3,9); ks=1/(ds-2)
a2.plot(ds,ks,'o-',color=C['red'],ms=4.5)
for d_,k in zip(ds,ks): a2.annotate(f'{k:.2f}',xy=(d_,k),xytext=(0,7),
    textcoords='offset points',ha='center',fontsize=7)
a2.axhline(1,color=C['green'],lw=1.3,ls='--')
a2.text(8.4,1.04,'exact rule $=1$',color=C['green'],fontsize=7.5,ha='right')
a2.set_xlabel(r'dimension $d$'); a2.set_ylabel(r'$\hat Q/Q$   as $q\to1$')
a2.set_ylim(0,1.22); a2.set_title('(b)  what it converges to',fontsize=9.2)
fig.tight_layout(); save(fig,'fig2_naive')

# ============================================ FIG 3 : order on a uniform lattice
fig,ax=plt.subplots(figsize=(4.3,2.95))
n=np.arange(2,80)
ax.loglog(n,1/n,color=C['green'],label=r'left endpoint   $|b|=1/n$')
ax.loglog(n,np.abs(1/(4*n*(n+1))),'--',color=C['purple'],
          label=r'midpoint   $|b|=1/[4n(n+1)]$')
ax.loglog(n,0.12/n**2,':',color=C['grey'],lw=1.0,label=r'$\propto n^{-2}$ guide')
ax.set_xlabel(r'$n=r_n/a$'); ax.set_ylabel(r'$|\hat Q/Q-1|$')
ax.set_title('uniform lattice, $d=3$: exact bias',fontsize=9.2)
ax.legend(loc='lower left',fontsize=7.4); ax.set_ylim(1e-6,1)
ax.annotate(r'$+50\%$ at $n=2$',xy=(2,0.5),xytext=(2.35,0.62),fontsize=7.4,color=C['green'])
ax.annotate(r'$-4\%$ at $n=2$',xy=(2,1/24),xytext=(2.5,0.012),fontsize=7.4,color=C['purple'])
fig.tight_layout(); save(fig,'fig3_lattice')
print('ok 1-3')
