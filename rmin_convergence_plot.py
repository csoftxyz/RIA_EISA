"""r_min convergence plot — O(r_min) boundary truncation proof."""
import numpy as np
from scipy import linalg
from scipy.special import iv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use GEOMETRIC alpha (not CODATA) — the bare coupling from Z3 framework
S = (iv(1,1.)/iv(0,1.))**4
alpha_geom = S/(np.pi*np.sqrt(3))  # geometric bare coupling
alpha_inv_geom = 1/alpha_geom

# Apply exact 42 ppm correction: delta = (S - S^3)/(4*sqrt(3))
delta = (S - S**3)/(4*np.sqrt(3))
alpha_inv_phys = alpha_inv_geom - delta
alpha_phys = 1/alpha_inv_phys

a0 = 1./alpha_phys; Eh = -0.5*alpha_phys**2
k=12; qk=3**(1./(2*k)); h=np.log(qk)

r_min_factors = [100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
deltas, r_min_vals = [], []

for factor in r_min_factors:
    r_min = a0/factor; r_max = 25*a0
    N = int(np.ceil(np.log(r_max/r_min)/np.log(qk))) + 5
    t = np.log(r_min) + np.arange(N)*h; r = np.exp(t)
    H = np.zeros((N,N)); Md = np.zeros(N)
    for j in range(N):
        H[j,j] = 1/h**2 + 0.125 - alpha_phys*r[j]; Md[j] = r[j]**2
    for j in range(N-1):
        H[j,j+1] = -0.5/h**2; H[j+1,j] = -0.5/h**2
    H[0,0] = 1/h**2 + 0.125 - alpha_phys*r[0]; H[0,1] = -1/h**2
    ev,_ = linalg.eigh(H, np.diag(Md))
    E1s = ev[ev<0][np.argsort(ev[ev<0])[0]]
    deltas.append(abs(E1s-Eh)/abs(Eh)*100)
    r_min_vals.append(r_min/a0)

fig, ax = plt.subplots(figsize=(6,4.5), facecolor='white')
ax.loglog(r_min_vals, deltas, 'o-', color='#e74c3c', lw=2, ms=8, 
          markerfacecolor='white', markeredgewidth=2)
ax.loglog(r_min_vals, 0.4*np.array(r_min_vals), '--', color='gray', lw=1, alpha=0.7, 
          label=r'$\propto r_{\min}$')
ax.set_xlabel(r'$r_{\min}/a_0$', fontsize=13)
ax.set_ylabel(r'$|\Delta E|/|E_{\rm H}|$ (%)', fontsize=13)
ax.set_title(r'$E_{1s}$ Convergence with Inner Cutoff', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.03, 5)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('rmin_convergence.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: rmin_convergence.png')
