"""Re-check every numerical claim that appears in radialflux.tex."""
import math, numpy as np, sys
sys.path.insert(0,'src')
from radialflux import *

ok=lambda c,msg: print(("  PASS  " if c else "  FAIL  ")+msg) or c
res=[]
def chk(c,msg): res.append(ok(c,msg))

Om=lambda d: omega_d(d)
print("Eq.(18)  kappa_A = (2n+1)^2/(4n(n+1)) = 1+1/(4n(n+1))")
for n in (2,4,12):
    chk(abs((2*n+1)**2/(4*n*(n+1)) - (1+1/(4*n*(n+1))))<1e-14, f"n={n}")

print("atomic solver: midpoint vs r_n e^{h/2} is h^2/8")
for h in (0.01,0.05,0.1):
    chk(abs(((1+math.exp(h))/(2*math.exp(h/2))-1)/(h*h/8)-1)<1e-3, f"h={h}")

print("Table 1 (bias, q=1.1)")
tab={3:(0.909091,1.100000,1.002273),4:(0.867769,1.155000,1.004551),
     5:(0.828951,1.213667,1.007595),6:(0.792466,1.276275,1.011410)}
for d,(a,b,c) in tab.items():
    chk(abs(exact_bias('left',1.1,d)-a)<1e-6 and abs(exact_bias('right',1.1,d)-b)<1e-6
        and abs(exact_bias('midpoint',1.1,d)-c)<1e-6, f"d={d}")

print("Table 2 (uniform lattice, d=3)")
ref={2:(-33.3,50.0,4.17),4:(-20.0,25.0,1.25),6:(-14.3,16.7,0.60),
     8:(-11.1,12.5,0.35),12:(-7.7,8.3,0.16),20:(-4.8,5.0,0.06)}
for n,(a,b,c) in ref.items():
    u=uniform_rule_values(n)
    chk(abs(100*(u['left']-1)-a)<0.06 and abs(100*(u['right']-1)-b)<0.06
        and abs(100*(u['midpoint']-1)-c)<0.006, f"n={n}")

print("q*_d  (n-nacci)")
for d,v in {4:1.618034,5:1.839287,6:1.927562,7:1.965948}.items():
    chk(abs(q_star(d)-v)<1e-6, f"d={d}")

print("rho*/r_n = 1+eps/2-d eps^2/24 ; kappa series")
for d in (3,4,5,6,7):
    e=1e-4; q=1+e
    r_=(exact_radius(1.0,q,d)-1)/e
    chk(abs(r_-(0.5-d*e/24))<1e-6, f"d={d} slope")
for d in (3,4,5,6,7):
    for e in (1e-3,1e-4):
        q=1+e
        rL=(exact_bias('left',q,d)-(1-(d-1)/2*e+d*(d-1)/6*e*e))/e**3
        rA=(exact_bias('midpoint',q,d)-(1+d*(d-1)/24*e*e))/e**3
        chk(abs(rL)<25 and abs(rA)<25, f"kL/kA d={d} e={e}: res/e^3 = {rL:+.3f} / {rA:+.3f}")

print("difference formulas: 6.7% at r=4a")
chk(abs(1/(1-1/16)-1-0.0667)<1e-3, "central @ r=4a")

print("forward diff quoted at midpoint: kappa-1 = 1/(4n(n+1))")
for n in (2,4,6,12):
    chk(abs(((n+0.5)**2/(n*(n+1)))-1-1/(4*n*(n+1)))<1e-12, f"n={n}")

print("layered shell: arith/harm bias")
for chi,val in ((2,0.125),(10,2.025),(1.1,0.00227)):
    chk(abs((chi-1)**2/(4*chi)-val)<2e-5, f"chi={chi}")

print("running residual: 4.7% at q=1.1, gamma=1, d=3")
chk(abs(running_residual(1.1,3)-0.04690)<1e-4, "q=1.1")
chk(abs((math.log(1.1)/2-1*(math.log(1.1))**2/12)-running_residual(1.1,3))<5e-7,"two-term")

print("multipole amplification at q=1.25, l=10, d=3")
chk(abs((10+1)*0.125-1.375)<1e-12, "138%")

print("96^3 background Q_bg(R)")
for R,v in ((8,-0.29),(12,-0.92),(24,-6.96)):
    chk(abs(100*(background_charge(R,96)-1)-v)<0.01, f"R={R}")

print("d=2 logarithmic mean")
chk(abs(exact_radius(1.0,1.5,2)-0.5/math.log(1.5))<1e-14, "d=2")

print("Table 4 (96^3 measurement) + fit")
from lattice96 import solve
L=96; V=solve(L); i=np.arange(L)-L//2
X,Y,Z=np.meshgrid(i,i,i,indexing='ij'); Rm=np.sqrt(X**2+Y**2+Z**2); nn=np.rint(Rm).astype(int)
st={}
for s in range(1,L//2):
    sel=nn==s
    if sel.sum(): st[s]=dict(V=V[sel].mean(), r2=float((Rm[sel]**2).mean()))
tab4={8:(0.9071,1.0241,1.0205,1.0235),10:(0.8647,0.9533,0.9511,0.9565),
      12:(0.9240,1.0026,1.0009,1.0104),16:(0.9447,1.0046,1.0037,1.0256),
      20:(0.9240,0.9708,0.9702,1.0116),24:(0.8971,0.9349,0.9345,1.0045)}
for s,(a,b,c,e) in tab4.items():
    dV=st[s]['V']-st[s+1]['V']
    g=(Om(3)*s*s*dV, Om(3)*(s+.5)**2*dV, Om(3)*s*(s+1)*dV,
       Om(3)*s*(s+1)*(dV-(st[s]['r2']-st[s+1]['r2'])/(6*L**3)))
    chk(max(abs(np.array(g)-np.array([a,b,c,e])))<1e-4, f"R={s}")
Rs=list(range(8,25)); A=np.array([[Om(3)*s*s*(st[s]['V']-st[s+1]['V']) for s in Rs],
    [Om(3)*(s+.5)**2*(st[s]['V']-st[s+1]['V']) for s in Rs],
    [Om(3)*s*(s+1)*(st[s]['V']-st[s+1]['V']) for s in Rs],
    [Om(3)*s*(s+1)*((st[s]['V']-st[s+1]['V'])-(st[s]['r2']-st[s+1]['r2'])/(6*L**3)) for s in Rs]])
for j,v in enumerate((0.9053,0.9695,0.9682,0.9951)):
    chk(abs(A[j].mean()-v)<1e-4, f"mean col{j+1}")
Rv=np.array(Rs,float); Vv=np.array([st[s]['V'] for s in Rs])
M=np.vstack([1/Rv,np.ones_like(Rv),Rv**2]).T; co,*_=np.linalg.lstsq(M,Vv,rcond=None)
chk(abs(4*math.pi*co[0]-0.9936)<1e-4,"fit Q")
chk(abs(co[2]/1e-7-1.94)<0.02,"fit C")
chk(abs(co[1]+0.002334)<1e-6,"fit B")

print("Cornell worked example (Table 5) and \hbar c convention")
hb = 0.1973269804; sig = 0.18; als = 0.30
REF = {(0.05,0.2):-13.7,(0.05,0.3):-7.0,(0.05,0.5):-2.3,(0.05,0.7):-1.0,(0.05,1.0):-0.4,
       (0.10,0.2):-22.8,(0.10,0.3):-12.3,(0.10,0.5):-4.3,(0.10,0.7):-1.9,(0.10,1.0):-0.7}
for (a_fm, r_fm), ref in REF.items():
    r = r_fm/hb; a = a_fm/hb
    true = als + 0.75*sig*r*r
    quoted = als*r/(r + a) + 0.75*sig*r*r        # r^2/rho_F^2 = r/(r+a), exact
    val = 100.0*(quoted/true - 1.0)
    chk(abs(val - ref) < 0.06, f"a={a_fm} r={r_fm}: {val:+.2f}% (table {ref:+.1f}%)")
    chk(abs(val - 100.0*(-(a/(r+a))*als/true)) < 1e-9, f"a={a_fm} r={r_fm}: closed form == -[a/(r+a)] alpha_s/alpha")
chk(abs(0.75*sig*(0.2/hb)**2 - 0.1387) < 5e-4, "hbar*c normalisation: 0.75*sigma*(0.2fm)^2 = 0.14")

print("\n%d checks, %d failed"%(len(res),res.count(False)))
sys.exit(0 if all(res) else 1)
