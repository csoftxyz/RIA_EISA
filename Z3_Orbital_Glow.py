"""
Z3_Orbital_Glow.py
═══════════════════════════════════════════════════════════════════
Beautiful multi-angle orbital montages with volumetric glow effect.

Multi-layer rendering: inner bright core + outer glow halo + background mist.
4 viewing angles per orbital. Nature/Science cover-quality.
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy import linalg
from scipy.special import iv, sph_harm_y
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)

# ═══ DATA ═══

def get_radial():
    S=(iv(1,1.)/iv(0,1.))**4; alpha_geom=S/(np.pi*np.sqrt(3)); alpha_phys=1/(1/alpha_geom-(S-S**3)/(4*np.sqrt(3))); a0=1/alpha_phys
    k=12; q=3**(1./(2*k)); h=np.log(q); N=int(np.ceil(np.log(25*a0/(a0/200))/np.log(q)))+5
    t=np.log(a0/200)+np.arange(N)*h; r=np.exp(t); rad={}
    for l in [0,1,2]:
        H=np.zeros((N,N)); Md=np.zeros(N)
        for j in range(N): H[j,j]=1/h**2+.125*(2*l+1)**2-alpha_phys*r[j]; Md[j]=r[j]**2
        for j in range(N-1): H[j,j+1]=H[j+1,j]=-.5/h**2
        if l==0: H[0,0]=1/h**2+.125-alpha_phys*r[0]; H[0,1]=-1/h**2
        ev,evc=linalg.eigh(H,np.diag(Md)); bd=ev<0; Eb=ev[bd]; evb=evc[:,bd]
        idx=np.argsort(Eb)
        for i in range(min(4,len(Eb))):
            n=i+l+1; g=evb[:,idx[i]]; Rf=np.exp(t/2)*g/r
            nm=np.sqrt(np.sum(Rf**2*r**2*h))
            if nm>1e-15: Rf/=nm
            rad[(n,l)]=(r,Rf)
    return rad,alpha_phys,a0

def ang_fn(l,sel):
    def Yc(lv,mv,th,ph): return sph_harm_y(lv,mv,th,ph)
    if l==0: return lambda th,ph: Yc(0,0,th,ph).real
    if sel=='pz': return lambda th,ph: Yc(1,0,th,ph).real
    if sel=='px': return lambda th,ph: (Yc(1,-1,th,ph)-Yc(1,1,th,ph)).real/np.sqrt(2)
    if sel=='py': return lambda th,ph: (1j*(Yc(1,-1,th,ph)+Yc(1,1,th,ph))).real/np.sqrt(2)
    if sel=='dz2': return lambda th,ph: Yc(2,0,th,ph).real
    if sel=='dxy': return lambda th,ph: (1j*(Yc(2,-2,th,ph)-Yc(2,2,th,ph))).real/np.sqrt(2)
    if sel=='dx2y2': return lambda th,ph: (Yc(2,-2,th,ph)+Yc(2,2,th,ph)).real/np.sqrt(2)
    return lambda th,ph: Yc(0,0,th,ph).real

def cloud(n,l,sel,rad,a0,n_pts=250000):
    rg,Rf=rad[(n,l)]; fn=ang_fn(l,sel)
    rp=rg[np.argmax(np.abs(Rf)*rg)]
    rs=np.logspace(np.log10(max(rg[0],rp*.02)),np.log10(min(rg[-1],rp*11)),220)
    Ri=np.interp(rs,rg,Rf)
    th=np.linspace(.01,np.pi-.01,160); ph=np.linspace(0,2*np.pi,320)
    TH,PH=np.meshgrid(th,ph,indexing='ij'); av=fn(TH,PH)
    apt=[]; aw=[]
    for i,rv in enumerate(rs):
        psq=(Ri[i]*av)**2; pm=np.max(psq)
        if pm<1e-25: continue
        mk=psq>pm*.004
        if np.sum(mk)>0:
            x=rv*np.sin(TH[mk])*np.cos(PH[mk])
            y=rv*np.sin(TH[mk])*np.sin(PH[mk])
            z=rv*np.cos(TH[mk])
            apt.append(np.column_stack([x,y,z])); aw.append(psq[mk])
    if not apt: return np.zeros((0,3)),np.array([])
    pts=np.vstack(apt); w=np.concatenate(aw)
    if len(pts)>n_pts: 
        wn=w/w.sum(); idx=np.random.choice(len(pts),n_pts,replace=False,p=wn)
        pts=pts[idx]; w=w[idx]
    return pts,w

# ═══ RENDER ═══

def render_glow(pts, w, ax, lim):
    """Multi-layer volumetric glow — core + halo + mist."""
    wn = np.clip(w/np.percentile(w, 98), 0, 1)
    colors = plt.cm.inferno(wn*0.5+0.5)
    sidx = np.argsort(wn)
    
    # Layer 1: Background mist (large, very transparent)
    mask_mist = wn < 0.2
    if np.sum(mask_mist) > 0:
        ax.scatter(pts[mask_mist,0], pts[mask_mist,1], pts[mask_mist,2],
                  c=colors[mask_mist], s=1.5, alpha=0.06, rasterized=True)
    
    # Layer 2: Outer halo (medium)
    mask_halo = (wn >= 0.2) & (wn < 0.6)
    if np.sum(mask_halo) > 0:
        ax.scatter(pts[mask_halo,0], pts[mask_halo,1], pts[mask_halo,2],
                  c=colors[mask_halo], s=0.4, alpha=0.25, rasterized=True)
    
    # Layer 3: Inner glow (dense, bright)
    mask_core = (wn >= 0.6) & (wn < 0.9)
    if np.sum(mask_core) > 0:
        ax.scatter(pts[mask_core,0], pts[mask_core,1], pts[mask_core,2],
                  c=colors[mask_core], s=0.15, alpha=0.50, rasterized=True)
    
    # Layer 4: Hot core (brightest)
    mask_hot = wn >= 0.9
    if np.sum(mask_hot) > 0:
        ax.scatter(pts[mask_hot,0], pts[mask_hot,1], pts[mask_hot,2],
                  c=colors[mask_hot], s=0.10, alpha=0.70, rasterized=True)
    
    ax.scatter([0],[0],[0], c='white', s=180, marker='*', zorder=1000)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_axis_off()

# ═══ MONTAGE ═══

def montage(n,l,sel,title,rad,a0,filename):
    """4-angle glow montage."""
    print(f'  {title}...')
    pts,w=cloud(n,l,sel,rad,a0,200000)
    if len(w)<=1: return
    
    fig=plt.figure(figsize=(14,14),facecolor='black')
    views=[(22,35),(22,125),(70,45),(5,80)]
    labels=['Perspective','Side','Top','Front']
    lim=a0*(1.5+n*2.5)
    
    for i,(el,az) in enumerate(views):
        ax=fig.add_subplot(2,2,i+1,projection='3d',facecolor='black')
        render_glow(pts,w,ax,lim)
        ax.view_init(elev=el,azim=az); ax.dist=7
        ax.text2D(.05,.93,labels[i],transform=ax.transAxes,fontsize=13,
                 color='white',fontweight='bold',fontfamily='serif')
    
    fig.suptitle(title,fontsize=18,color='white',fontweight='bold',
                fontfamily='serif',y=.98)
    outpath=os.path.join(OUT,filename)
    plt.savefig(outpath,dpi=300,bbox_inches='tight',facecolor='black',pad_inches=.1)
    plt.close()
    print(f'    [OK] {outpath}')
    return outpath

# ═══ CURVE PLOTS (clean radial + energy) ═══

def curves(rad,alpha,a0):
    """Clean radial wavefunction curves on dark background."""
    fig=plt.figure(figsize=(20,8),facecolor='black')
    
    for li,l in enumerate([0,1,2]):
        ax=fig.add_subplot(1,3,li+1,facecolor='black')
        colors=['#ff4444','#44aaff','#44ff44','#ffaa44']
        for i in range(3):
            n=i+l+1
            if (n,l) in rad:
                rg,Rf=rad[(n,l)]
                ax.plot(rg*alpha,Rf,color=colors[i],lw=2.5,alpha=.9,label=f'n={n}')
                rs=np.logspace(np.log10(rg[0]*alpha),np.log10(rg[-1]*alpha),300)
                Re=np.zeros_like(rs)
                for j,rv in enumerate(rs):
                    rho=2*rv/n
                    from scipy.special import eval_genlaguerre,factorial
                    nm=np.sqrt((2./n)**3*factorial(n-l-1)/(2*n*factorial(n+l)))
                    Lp=eval_genlaguerre(n-l-1,2*l+1,rho)
                    Re[j]=nm*np.exp(-rho/2)*rho**l*Lp
                he=np.log(rg[1]/rg[0])
                nr=np.sqrt(np.sum(Re**2*(rs/alpha)**2*he))
                if nr>1e-15: Re/=nr
                ax.plot(rs,Re,'--',color=colors[i],lw=1.2,alpha=.5)
        ax.set_title(f'l = {l}',fontsize=16,color='white',fontweight='bold')
        ax.set_xlabel('r (a₀)',fontsize=13,color='white')
        if li==0: ax.set_ylabel('R(r)',fontsize=13,color='white')
        ax.legend(fontsize=10,loc='upper right',facecolor='black',
                 edgecolor='#444',labelcolor='white')
        ax.tick_params(colors='white',labelsize=10)
        for sp in ax.spines.values(): sp.set_color('#444')
        ax.grid(True,alpha=.15)
        ax.set_facecolor('black')
    
    plt.tight_layout()
    outpath=os.path.join(OUT,'Z3_Radial_Curves.png')
    plt.savefig(outpath,dpi=300,bbox_inches='tight',facecolor='black')
    plt.close()
    print(f'  [OK] {outpath}')

# ═══ MAIN ═══

if __name__=='__main__':
    print('Z3 Glow Orbital Visualisation')
    print('='*45)
    rad,alpha,a0=get_radial()
    print(f'1/α={1/alpha:.6f}, a₀={a0:.1f}')
    
    print('\n--- 4-Angle Glow Montages ---')
    montage(1,0,0,'1s Orbital',rad,a0,'Z3_Glow_1s.png')
    montage(2,0,0,'2s Orbital',rad,a0,'Z3_Glow_2s.png')
    montage(2,1,'pz','2p_z Orbital',rad,a0,'Z3_Glow_2pz.png')
    montage(2,1,'px','2p_x Orbital',rad,a0,'Z3_Glow_2px.png')
    montage(2,1,'py','2p_y Orbital',rad,a0,'Z3_Glow_2py.png')
    montage(3,2,'dz2','3d_{z^2} Orbital',rad,a0,'Z3_Glow_3dz2.png')
    montage(3,2,'dxy','3d_{xy} Orbital',rad,a0,'Z3_Glow_3dxy.png')
    montage(3,2,'dx2y2','3d_{x^2-y^2} Orbital',rad,a0,'Z3_Glow_3dx2y2.png')
    
    print('\n--- Radial Curves ---')
    curves(rad,alpha,a0)
    
    print('\nDone.')
