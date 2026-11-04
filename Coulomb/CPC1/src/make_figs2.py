exec(open('src/make_figs.py').read().split('# ============================================ FIG 1')[0])
import numpy as np

# ============================================ FIG 4
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.3,2.9))
# (a) distributed source
d=3; QT=1.0; R=1.0
V=lambda r: QT*(3*R**2-r**2)/(8*math.pi*R**3)
rn=0.35
eps=np.logspace(-2.6,-0.3,60)
res={k:[] for k in ('left','mid','right','exact')}
for e in eps:
    q=1+e; r2=rn*q
    dV=V(rn)-V(r2)
    rho=rho_star(d,q,rn)
    Qenc=lambda x: QT*(x/R)**3
    ref=Qenc(rho)
    for k,rp in (('left',rn),('mid',(rn+r2)/2),('right',r2),('exact',rho)):
        Qh=Om(d)*rp**(d-1)*dV/(r2-rn)
        res[k].append(abs(Qh/ref-1))
a1.loglog(eps,res['left'],color=C['orange'],label=r'left endpoint  $\propto\varepsilon$')
a1.loglog(eps,res['right'],color=C['green'],label=r'right endpoint  $\propto\varepsilon$')
a1.loglog(eps,res['mid'],'--',color=C['purple'],label=r'midpoint  $\propto\varepsilon^2$')
a1.loglog(eps,res['exact'],'-',color=C['red'],lw=1.9,label=r'exact radius $\rho_\ast$  $\propto\varepsilon^2$')
a1.loglog(eps,3.0*eps,':',color=C['grey'],lw=1.0)
a1.loglog(eps,1.1e-3*eps**2,':',color=C['grey'],lw=1.0)
a1.set_xlabel(r'$\varepsilon=q-1$'); a1.set_ylabel(r'$|\hat Q/Q_{\rm enc}(\rho_\ast)-1|$')
a1.set_title('(a)  distributed source (uniform ball)',fontsize=9.2)
a1.legend(loc='lower right',fontsize=7.2,handlelength=1.8)
a1.set_ylim(1e-9,1)
a1.text(2.2e-2,4e-8,'second-order\ncoefficient\n(d-2)/24 vs (d-1)/12',fontsize=6.8,color=C['red'])

# (b) running field
qq=np.linspace(1.0005,2.0,900)
for d_,c in ((3,C['blue']),(4,C['red'])):
    exact=[1/(d_-2)-math.log(q)/(q**(d_-2)-1) for q in qq]
    a2.plot(qq,exact,color=c,lw=1.6,label=fr'exact, $d={d_}$')
a2.plot(qq,(qq-1)/2,'--',color=C['grey'],lw=1.3,label=r'$\gamma(q-1)/2$')
a2.set_xlabel(r'mesh ratio $q$'); a2.set_ylabel(r'$(\hat Q-Q_{\rm enc})/\gamma$')
a2.set_title('(b)  logarithmically running field',fontsize=9.2)
a2.legend(loc='upper left',fontsize=7.4)
a2.set_ylim(0,0.32)
a2.text(1.03,0.02,r'residual $=\gamma\,[1/(d-2)-\ell_d(q)]$, first order in $q-1$',
        fontsize=7.2,color=C['grey'])
fig.tight_layout(); save(fig,'fig4_source_running')

print('fig4 done')
