"""
harmonic_convergence_v3.py — Test peak stability under harmonic variation.
Uses the SAME model as xi_robustness.py (which reproduces paper results).
Tests relative stability, not absolute peak position.
"""
import numpy as np

a = 0.246
k0 = 4*np.pi/(np.sqrt(3)*a)
k_vac = 2*np.pi / 28.7  # same as xi_robustness.py
xi = 70.0

def rho_moire(x, y, theta, n_max=3):
    th = np.radians(theta)
    ct, st = np.cos(th), np.sin(th)
    xr = x*ct - y*st
    yr = x*st + y*ct
    rho = np.zeros_like(x)
    for n in range(1, n_max+1):
        nk = n*k0
        rho += np.cos(nk*x) + np.cos(nk*xr)
        rho += np.cos(nk*y) + np.cos(nk*yr)
    return rho

def zeta_z3(x, y, n_max=2):
    angles = np.radians([0,60,120,180,240,300])
    zeta = np.zeros_like(x)
    r = np.sqrt(x**2+y**2)
    damp = np.exp(-r/xi)
    for n in range(1, n_max+1):
        nk = n*k_vac
        for ang in angles:
            vx,vy = np.cos(ang),np.sin(ang)
            zeta += np.cos(nk*(vx*x+vy*y))*damp
    return zeta

def find_peak(n_rho, n_zeta, N=500, L=100.0, th_lo=0.5, th_hi=1.5, n_th=101):
    x = np.linspace(-L,L,N); y = np.linspace(-L,L,N)
    X,Y = np.meshgrid(x,y)
    thetas = np.linspace(th_lo, th_hi, n_th)
    overlaps = []
    for th in thetas:
        rho = rho_moire(X,Y,th,n_max=n_rho)
        zeta = zeta_z3(X,Y,n_max=n_zeta)
        overlaps.append(np.mean(rho*zeta))
    overlaps = np.array(overlaps)
    idx = np.argmax(overlaps)
    # parabolic interpolation
    if 1<=idx<=len(thetas)-2:
        t0,t1,t2 = thetas[idx-1],thetas[idx],thetas[idx+1]
        g0,g1,g2 = overlaps[idx-1],overlaps[idx],overlaps[idx+1]
        dn = (t0-t1)*(t0-t2)*(t1-t2)
        if abs(dn)>1e-30:
            A = (t2*(g1-g0)+t1*(g0-g2)+t0*(g2-g1))/dn
            B = (t2**2*(g0-g1)+t1**2*(g2-g0)+t0**2*(g1-g2))/dn
            if abs(A)>1e-30:
                return -B/(2*A), overlaps[idx]
    return thetas[idx], overlaps[idx]

print("="*60)
print("  HARMONIC CONVERGENCE TEST (xi_robustness model)")
print("  N=500, L=100nm, xi=70nm")
print("="*60)
print(f"  {'n_rho':>5} {'n_zeta':>6} {'theta0':>10} {'delta':>12}")
print(f"  {'-'*38}")

ref = None
results = {}
for nr in [1,2,3,4,5]:
    for nz in [1,2,3]:
        t0,gm = find_peak(nr,nz)
        results[(nr,nz)] = t0
        if nr==3 and nz==2:
            ref = t0
            mark = " <-- paper"
        else:
            mark = ""
        d = f"{abs(t0-ref):.4f}" if ref else "-"
        print(f"  {nr:>5} {nz:>6} {t0:>10.4f} {d:>12}{mark}")

print()
# stability for n_rho>=2, n_zeta>=2
sub = {k:v for k,v in results.items() if k[0]>=2 and k[1]>=2}
vals = list(sub.values())
print(f"  Ref (3,2): {ref:.4f}")
print(f"  n_rho>=2, n_zeta>=2 range: [{min(vals):.4f}, {max(vals):.4f}]")
print(f"  Spread: {max(vals)-min(vals):.4f} ({(max(vals)-min(vals))/np.mean(vals)*100:.2f}%)")
print(f"  Max |delta| from (3,2): {max(abs(v-ref) for v in vals):.4f}")
