import numpy as np, sympy as sp, math
print("="*72)
print("[D'] lattice Laplacian of |x|^2 is exactly 2d (no periodic wrap in the stencil)")
L=96; i=np.arange(L)-L//2
X,Y,Z=np.meshgrid(i,i,i,indexing='ij'); f=X**2+Y**2+Z**2
# explicit neighbour coordinates (not np.roll!)
def lap(f,X,Y,Z):
    out=np.zeros_like(f,dtype=float)
    for A,B,C in ((X,Y,Z),(Y,Z,X),(Z,X,Y)):
        fplus  = B**2+C**2+(A+1)**2
        fminus = B**2+C**2+(A-1)**2
        out += fplus - 2*f + fminus
    return out
print(f"  max|Delta_lat |x|^2 - 6| over the whole box = {np.max(np.abs(lap(f,X,Y,Z)-6)):.3e}")
print(f"  => V_bg = |x|^2/(2dL^d):  max|Delta V_bg - 1/L^3| = {np.max(np.abs(lap(f,X,Y,Z)/(2*3*L**3)-1/L**3)):.3e}")
print(f"  continuous prediction for Q_bg(R) = 1 - (2pi/3L^3) R(R+1)(2R+1):  R=8: {1-2*math.pi/(3*L**3)*8*9*17:.6f}, R=24: {1-2*math.pi/(3*L**3)*24*25*49:.6f}")

print()
print("[F] exact two-term identity + shell identity, closed forms (sympy, d=3 and d=4)")
rr=sp.Symbol('r',positive=True)
for dd,Rs,rho0 in [(3,sp.Integer(3),sp.Rational(17,10)),(4,sp.Integer(2),sp.Rational(13,10))]:
    Om=sp.simplify(2*sp.pi**sp.Rational(dd,2)/sp.gamma(sp.Rational(dd,2)))
    rho=sp.Piecewise((rho0, rr<Rs),(0,True))
    Qr=sp.integrate(rho*Om*rr**(dd-1),(rr,0,sp.Symbol('R_',positive=True)))
    # inside branch
    Qin=sp.simplify((Om*rho0*sp.Symbol('R_',positive=True)**dd/dd))
    F=Qin/((dd-2)*Om*sp.Symbol('R_',positive=True)**(dd-2))
    G=sp.integrate(rho0*rr,(rr,sp.Symbol('R_',positive=True),Rs))/(dd-2)
    V=sp.simplify(F+G)
    print(f"  d={dd}: V_in(r) = {sp.simplify(V)}")
    # shell identity check inside the ball
    rn,r2=sp.symbols('r_n r_2',positive=True)
    Vf=lambda x: V.subs(sp.Symbol('R_',positive=True),x)
    lhs=sp.simplify(Vf(rn)-Vf(r2))
    Qn=Om*rho0*rn**dd/dd; Qm=Om*rho0*r2**dd/dd
    rhs=(Qn*rn**(2-dd)-Qm*r2**(2-dd))/((dd-2)*Om)+sp.integrate(rho0*rr,(rr,rn,r2))/(dd-2)
    print(f"    shell identity  V_n-V_(n+1) - [boxed] = {sp.simplify(lhs-rhs)}")

print()
print("[G] distributed source: bias of the point-rule when Q varies across the shell")
print("    bias = (rho_*-<r>_w) * s/Q_n  -> second order in (q-1); endpoints are first order")
for dd in [3,4,5]:
    qv=1.1; rn_=1.0
    rho_s=(((dd-2)*(qv-1))/(1-qv**(2-dd)))**(1/(dd-1))
    def meanr(dd,qv):
        n=sp.integrate(rr**(2-dd),(rr,1,qv)); d_=sp.integrate(rr**(1-dd),(rr,1,qv)); return float(n/d_)
    print(f"  d={dd}: rho_*={rho_s:.6f}  <r>_w={meanr(dd,qv):.6f}  diff={rho_s-meanr(dd,qv):+.3e}"
          f"   (d-2)(q-1)^2/24 = {(dd-2)*(qv-1)**2/24:.3e}")

print()
print("[H] running field: exact residual and its two-term expansion")
for dd in [3,4,5]:
    g=1.0
    for qv in [1.05,1.10,1.25,2.0]:
        ell=math.log(qv)/(qv**(dd-2)-1)
        ex=g*(1/(dd-2)-ell)
        ser=g*(math.log(qv)/2-(dd-2)*math.log(qv)**2/12)
        ser2=g*((qv-1)/2-(qv-1)**2*(0.25+(dd-2)/12))
        print(f"  d={dd} q={qv}: exact {ex:.6f} | ln-series {ser:.6f} | (q-1)-series {ser2:.6f} | gamma(q-1)/2 {g*(qv-1)/2:.4f}")

print()
print("[I] stability: condition number of the difference form  ~ 1/((d-2)*eps)")
for dd in [3,4]:
    for qv in [1.1,1.01,1.001]:
        kap=1/((dd-2)*(qv-1))
        print(f"  d={dd} q={qv}: predicted amplification 1/((d-2)eps) = {kap:.1f}")
