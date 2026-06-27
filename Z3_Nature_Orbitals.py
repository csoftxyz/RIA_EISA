"""
Z3_Nature_Orbitals.py
═══════════════════════════════════════════════════════════════════
Publication-quality 3D orbital visualisation — Nature journal standard.

Output: 8 individual orbital renders + 1 composite figure.
Style:   Black background, smooth |ψ|²-weighted point clouds,
         inferno colormap, nucleus star marker, no clutter.
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy import linalg
from scipy.special import iv, sph_harm_y, eval_genlaguerre, factorial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Radial wavefunctions
# ═══════════════════════════════════════════════════════════════

def get_radial():
    # Geometric bare coupling + exact 42 ppm topological correction
    S = (iv(1,1.0)/iv(0,1.0))**4
    alpha_geom = S/(np.pi*np.sqrt(3))  # bare coupling
    delta = (S - S**3)/(4*np.sqrt(3))   # exact 42 ppm correction
    alpha_phys = 1.0/(1.0/alpha_geom - delta)  # physical coupling
    a0 = 1.0/alpha_phys; k=12; q=3**(1.0/(2*k)); h=np.log(q)
    r_min=a0/200; r_max=25*a0
    N=int(np.ceil(np.log(r_max/r_min)/np.log(q)))+5
    t=np.log(r_min)+np.arange(N)*h; r=np.exp(t)
    radial = {}
    for l in [0,1,2]:
        H=np.zeros((N,N)); Md=np.zeros(N)
        for j in range(N):
            H[j,j]=1/h**2+0.125*(2*l+1)**2-alpha_phys*r[j]; Md[j]=r[j]**2
        for j in range(N-1): H[j,j+1]=-0.5/h**2; H[j+1,j]=-0.5/h**2
        if l==0: H[0,0]=1/h**2+0.125-alpha_phys*r[0]; H[0,1]=-1/h**2
        evals,evecs=linalg.eigh(H,np.diag(Md))
        bound=evals<0; Eb=evals[bound]; ev=evecs[:,bound]
        idx=np.argsort(Eb)
        for i in range(min(4,len(Eb))):
            n=i+l+1; g=ev[:,idx[i]]
            Rf=np.exp(t/2)*g/r; nm=np.sqrt(np.sum(Rf**2*r**2*h))
            if nm>1e-15: Rf/=nm
            radial[(n,l)]=(r,Rf)
    return radial, alpha_phys, a0

# ═══════════════════════════════════════════════════════════════
# 2. Angular functions — real spherical harmonics
# ═══════════════════════════════════════════════════════════════

def ang_fn(l, sel):
    """
    Real tesseral spherical harmonics.
    Returns purely real angular function for any (l, sel).
    
    IMPORTANT: Real linear combinations of Y_{l,m} may involve
    the IMAGINARY parts. Each orbital definition handles this
    explicitly via 1j*... or .imag as needed.
    """
    # Raw complex spherical harmonic (scipy 1.17: n, m, theta, phi)
    def Yc(lv, mv, th, ph):
        return sph_harm_y(lv, mv, th, ph)
    
    if l==0:
        return lambda th,ph: Yc(0,0,th,ph).real  # Y_00 is real
    if l==1:
        # p_z = Y_{1,0} — purely real
        if sel=='pz': return lambda th,ph: Yc(1,0,th,ph).real
        # p_x = (Y_{1,-1} - Y_{1,1})/√2 — real combination
        if sel=='px': return lambda th,ph: (Yc(1,-1,th,ph)-Yc(1,1,th,ph)).real/np.sqrt(2)
        # p_y = i(Y_{1,-1} + Y_{1,1})/√2 — imaginary combination, made real by i×
        if sel=='py': return lambda th,ph: (1j*(Yc(1,-1,th,ph)+Yc(1,1,th,ph))).real/np.sqrt(2)
    if l==2:
        # d_{z²} = Y_{2,0} — real
        if sel=='dz2':  return lambda th,ph: Yc(2,0,th,ph).real
        # d_{xz} = (Y_{2,-1} - Y_{2,1})/√2 — pure imaginary, use .imag
        if sel=='dxz':  return lambda th,ph: (Yc(2,-1,th,ph)-Yc(2,1,th,ph)).imag/np.sqrt(2)
        # d_{yz} = i(Y_{2,-1} + Y_{2,1})/√2
        if sel=='dyz':  return lambda th,ph: (1j*(Yc(2,-1,th,ph)+Yc(2,1,th,ph))).real/np.sqrt(2)
        # d_{xy} = i(Y_{2,-2} - Y_{2,2})/√2
        if sel=='dxy':  return lambda th,ph: (1j*(Yc(2,-2,th,ph)-Yc(2,2,th,ph))).real/np.sqrt(2)
        # d_{x²-y²} = (Y_{2,-2} + Y_{2,2})/√2 — real
        if sel=='dx2y2': return lambda th,ph: (Yc(2,-2,th,ph)+Yc(2,2,th,ph)).real/np.sqrt(2)
    return lambda th,ph: Yc(0,0,th,ph).real

# ═══════════════════════════════════════════════════════════════
# 3. Point cloud generation
# ═══════════════════════════════════════════════════════════════

def generate_cloud(n, l, sel, radial, n_pts=180000):
    """Generate dense |ψ|²-weighted 3D point cloud."""
    r_grid, R = radial[(n,l)]
    fn = ang_fn(l, sel)
    
    # Find radial peak for adaptive sampling
    r_peak = r_grid[np.argmax(np.abs(R)*r_grid)]
    
    # Dense radial sampling around peak
    r_samp = np.logspace(
        np.log10(max(r_grid[0], r_peak*0.02)),
        np.log10(min(r_grid[-1], r_peak*12)),
        300
    )
    R_interp = np.interp(r_samp, r_grid, R)
    
    # Angular grid
    nth, nph = 200, 400
    th = np.linspace(0.01, np.pi-0.01, nth)
    ph = np.linspace(0, 2*np.pi, nph)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    av = fn(TH, PH)
    
    # Build cloud
    all_pts=[]; all_w=[]
    for i, rv in enumerate(r_samp):
        psq = (R_interp[i]*av)**2
        pmax = np.max(psq)
        if pmax < 1e-25: continue
        mask = psq > pmax*0.008
        if np.sum(mask) > 0:
            x=rv*np.sin(TH[mask])*np.cos(PH[mask])
            y=rv*np.sin(TH[mask])*np.sin(PH[mask])
            z=rv*np.cos(TH[mask])
            all_pts.append(np.column_stack([x,y,z]))
            all_w.append(psq[mask])
    
    if not all_pts: return np.zeros((0,3)), np.array([])
    pts=np.vstack(all_pts); w=np.concatenate(all_w)
    if len(pts)>n_pts:
        wn=w/w.sum()
        idx=np.random.choice(len(pts), n_pts, replace=False, p=wn)
        pts=pts[idx]; w=w[idx]
    return pts, w

# ═══════════════════════════════════════════════════════════════
# 4. Rendering
# ═══════════════════════════════════════════════════════════════

def render_orbital(n, l, sel, label, radial, a0, filename, n_pts=250000):
    """Single orbital — Nature-quality render."""
    pts, w = generate_cloud(n, l, sel, radial, n_pts)
    
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    if len(w) > 1:
        wn = np.clip(w/np.percentile(w, 98), 0, 1)
        colors = plt.colormaps['inferno'](wn*0.65+0.35)
        sidx = np.argsort(wn)
        ax.scatter(pts[sidx,0], pts[sidx,1], pts[sidx,2],
                  c=colors[sidx], s=0.18, alpha=0.55, rasterized=True)
    
    # Nucleus — white star
    ax.scatter([0],[0],[0], c='white', s=250, marker='*',
              edgecolors='none', linewidth=0, zorder=100)
    
    # Clean limits
    lim = a0 * (1.5 + n*2.5)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=48)
    ax.dist = 7.5
    
    # Subtle label
    ax.text2D(0.05, 0.95, label, transform=ax.transAxes,
             fontsize=18, color='white', fontweight='bold',
             fontfamily='serif', fontstyle='italic')
    
    plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='black',
               pad_inches=0.1)
    plt.close()
    return filename


def render_composite(radial, a0):
    """Nature-style composite figure: 8 orbitals in grid."""
    orbitals = [
        (1,0,0,  r'$1s$'),
        (2,0,0,  r'$2s$'),
        (2,1,'pz', r'$2p_z$'),
        (2,1,'px', r'$2p_x$'),
        (2,1,'py', r'$2p_y$'),
        (3,2,'dz2', r'$3d_{z^2}$'),
        (3,2,'dxy', r'$3d_{xy}$'),
        (3,2,'dx2y2', r'$3d_{x^2-y^2}$'),
    ]
    
    fig = plt.figure(figsize=(16, 8.5), facecolor='black')
    
    for idx, (n,l,sel,label) in enumerate(orbitals):
        ax = fig.add_subplot(2, 4, idx+1, projection='3d', facecolor='black')
        
        pts, w = generate_cloud(n, l, sel, radial, n_pts=120000)
        
        if len(w) > 1:
            wn = np.clip(w/np.percentile(w, 98), 0, 1)
            colors = plt.colormaps['inferno'](wn*0.6+0.4)
            sidx = np.argsort(wn)
            ax.scatter(pts[sidx,0], pts[sidx,1], pts[sidx,2],
                      c=colors[sidx], s=0.25, alpha=0.60, rasterized=True)
        
        ax.scatter([0],[0],[0], c='white', s=80, marker='*', zorder=100)
        
        lim = a0*(1.5+n*2.0)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_axis_off()
        ax.view_init(elev=22, azim=35+idx*5)
        ax.dist = 8
        
        # Label
        ax.text2D(0.05, 0.92, label, transform=ax.transAxes,
                 fontsize=14, color='white', fontweight='bold',
                 fontfamily='serif')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    filename = os.path.join(OUT, 'Z3_Orbitals_Nature_Composite.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='black',
               pad_inches=0.05)
    plt.close()
    print(f'[OK] {filename}')
    return filename

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Z3 Nature-Quality Orbital Visualisation')
    print('='*50)
    
    radial, alpha_phys, a0 = get_radial()
    print(f'alpha = {alpha_phys:.8f}, alpha^-1 = {1/alpha_phys:.6f}, a0 = {a0:.1f}')
    
    # Individual high-res renders
    print('\nIndividual orbitals:')
    specs = [
        (1,0,0,  '1s'),
        (2,0,0,  '2s'),
        (2,1,'pz','2p_z'),
        (2,1,'px','2p_x'),
        (2,1,'py','2p_y'),
        (3,2,'dz2','3d_{z^2}'),
        (3,2,'dxy','3d_{xy}'),
        (3,2,'dx2y2','3d_{x^2-y^2}'),
    ]
    for n,l,sel,label in specs:
        fname = os.path.join(OUT, f'Z3_Orbital_{label}.png')
        render_orbital(n, l, sel, f'${label}$', radial, a0, fname)
        print(f'  [OK] {fname}')
    
    # Composite
    print('\nComposite figure:')
    render_composite(radial, a0)
    
    print('\nDone.')
