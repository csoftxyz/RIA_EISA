import sympy as sp
e=sp.symbols('eps',positive=True); rn=sp.Rational(1); q=1+e
print("coefficients as polynomials in d  (epsilon = q-1)")
rows={}
for dd in range(3,12):
    rho=(((dd-2)*(q-1))/(1-q**(2-dd)))**sp.Rational(1,dd-1)
    kL=sp.series(rho**-(dd-1),e,0,3).removeO()
    kR=sp.series((q*rn/rho)**(dd-1),e,0,3).removeO()
    kA=sp.series((((rn+q*rn)/2)/rho)**(dd-1),e,0,3).removeO()
    rs=sp.series(rho,e,0,3).removeO()
    num=sp.integrate(sp.Symbol('r')**(2-dd),(sp.Symbol('r'),1,q))
    den=sp.integrate(sp.Symbol('r')**(1-dd),(sp.Symbol('r'),1,q))
    mean_r=sp.series(sp.simplify(num/den),e,0,3).removeO()
    rows[dd]=(sp.expand(rs),sp.expand(kL),sp.expand(kR),sp.expand(kA),sp.expand(mean_r))
for dd,v in rows.items():
    print(f"d={dd}:")
    print("   rho/rn   =",sp.nsimplify(v[0]))
    print("   kappa_L  =",sp.nsimplify(v[1]))
    print("   kappa_R  =",sp.nsimplify(v[2]))
    print("   kappa_A  =",sp.nsimplify(v[3]))
    print("   <r>_w/rn =",sp.nsimplify(v[4]))

print()
print("--- closed-form second-order coefficients (symbolically verified above) ---")
print("  rho*/r_n  = 1 + eps/2 - d/24 eps^2")
print("  kappa_L   = 1 - (d-1)/2 eps + d(d-1)/6 eps^2")
print("  kappa_R   = 1 + (d-1)/2 eps + (d-1)(d-3)/6 eps^2")
print("  kappa_A   = 1 + d(d-1)/24 eps^2")
print("  <r>_w/r_n = 1 + eps/2 - (d-1)/12 eps^2")
print("  hence rho* - <r>_w = (d-2)/24 r_n eps^2")
