import numpy as np, matplotlib as mpl, math, sys, time
mpl.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src')
mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'dejavuserif','font.size':9,'axes.linewidth':0.8,
 'axes.labelsize':9.5,'legend.fontsize':7.2,'xtick.labelsize':8,'ytick.labelsize':8,
 'figure.dpi':150,'savefig.bbox':'tight','legend.frameon':False,'lines.linewidth':1.4})
C=dict(blue='#1f4e79',red='#b22222',orange='#d9822b',green='#2e7d32',
       grey='#6b6b6b',purple='#6a3d9a',teal='#0f7b7b')
from lattice96 import solve

L=96; Om=4*math.pi
V=solve(L)
idx=np.arange(L)-L//2
X,Y,Z=np.meshgrid(idx,idx,idx,indexing='ij')
Rm=np.sqrt(X**2+Y**2+Z**2); n=np.rint(Rm).astype(int)
st={}
for s in range(1,L//2):
    sel=n==s
    if sel.sum()==0: continue
    st[s]=dict(V=V[sel].mean(), r2=float((Rm[sel]**2).mean()))
Rs=list(range(8,25))
def series(fn):
    return [fn(s) for s in Rs]
Ql=series(lambda s: Om*s*s*(st[s]['V']-st[s+1]['V']))
Qm=series(lambda s: Om*(s+0.5)**2*(st[s]['V']-st[s+1]['V']))
Qe=series(lambda s: Om*s*(s+1)*(st[s]['V']-st[s+1]['V']))
Qb=series(lambda s: Om*s*(s+1)*((st[s]['V']-st[s+1]['V'])-(st[s]['r2']-st[s+1]['r2'])/(6*L**3)))

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(10.4,2.85))
a1.axhline(1,color='k',lw=.7,ls=':')
a1.plot(Rs,Ql,'-o',ms=2.6,color=C['orange'],label='left endpoint')
a1.plot(Rs,Qm,'-s',ms=2.4,color=C['purple'],label='midpoint')
a1.plot(Rs,Qe,'-^',ms=2.6,color=C['blue'],label=r'exact radius $\rho_\ast$')
a1.plot(Rs,Qb,'-d',ms=3.0,color=C['red'],lw=2.0,label=r'$\rho_\ast$, background removed')
a1.set_xlabel(r'$R$'); a1.set_ylabel(r'$\hat Q/Q$')
a1.set_title(r'(a) measured on a $96^3$ periodic lattice',fontsize=9)
a1.legend(loc='lower left',ncol=1,handlelength=1.7,fontsize=6.9)
a1.set_ylim(0.86,1.05)
a1.annotate(r'$+3\%$ artifact',xy=(10,0.951),xytext=(11.3,0.875),fontsize=7,
            color=C['grey'],arrowprops=dict(arrowstyle='->',color=C['grey'],lw=.7))

Rr=np.arange(4,49)
a2.plot(Rr,100*(1-(2*math.pi/(3*L**3))*Rr*(Rr+1)*(2*Rr+1)),color=C['red'],lw=1.8)
for R0 in (8,12,24):
    y=100*(1-(1-(2*math.pi/(3*L**3))*R0*(R0+1)*(2*R0+1)))
    a2.plot([R0],[y],'o',color=C['red'],ms=4)
    a2.annotate(f'{y:.2f}%',xy=(R0,y),xytext=(3,-9),textcoords='offset points',fontsize=7)
a2.set_xlabel(r'$R$'); a2.set_ylabel(r'background bias  $[\%]$')
a2.set_title(r'(b) the background is closed-form',fontsize=9)
a2.text(17,5.4,r'$-\frac{2\pi}{3L^3}R(R+1)(2R+1)$',fontsize=8.4,color=C['red'])

RR=np.logspace(np.log10(4),np.log10(48),200)
a3.loglog(RR,(2*math.pi/(3*L**3))*RR*(RR+1)*(2*RR+1)*100,color=C['red'],
          label=r'artefact $\propto R^3/L^3$')
a3.loglog(RR,np.ones_like(RR)*0.5,'-',color=C['blue'],lw=1.6,label='signal: constant')
a3.loglog(RR,0.5*np.log(RR/4)/np.log(12),'--',color=C['green'],lw=1.4,label=r'signal $\propto\ln R$')
a3.set_xlabel(r'$R$'); a3.set_ylabel(r'$100\,|\hat Q/Q-1|$')
a3.set_title('(c) null test: the two scale differently',fontsize=9)
a3.legend(loc='lower right',handlelength=1.8,fontsize=6.9)
a3.set_ylim(0.05,60)
fig.tight_layout(); fig.savefig('fig/fig5_lattice96.pdf'); fig.savefig('fig/fig5_lattice96.png',dpi=230); plt.close()

# numbers for the text
print("raw   mean(8..24) = %.4f  range [%.4f, %.4f]"%(np.mean(Qe),min(Qe),max(Qe)))
print("bgsub mean(8..24) = %.4f  range [%.4f, %.4f]"%(np.mean(Qb),min(Qb),max(Qb)))
Rv=np.array(Rs,float); Vv=np.array([st[s]['V'] for s in Rs])
M=np.vstack([1/Rv,np.ones_like(Rv),Rv**2]).T
co,*_=np.linalg.lstsq(M,Vv,rcond=None)
print("fit Q=%.4f  C=%.3e (pred %.3e)  rms=%.2e"%(4*math.pi*co[0],co[2],1/(6*L**3),(Vv-M@co).std()))
