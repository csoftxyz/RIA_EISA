import sys
import sympy as sp
import numpy as np

q, d, rn, a = sp.symbols('q d r_n a', positive=True)
eps = sp.symbols('eps', positive=True)

Om = lambda dd: 2*sp.pi**(dd/2)/sp.gamma(dd/2)

def rho_star(dd, r1, r2):
    val = (dd-2)*(r2-r1)/(r1**(2-dd)-r2**(2-dd))
    return val**sp.Rational(1, dd-1)

print("="*70)
print("[1] exact radius == power mean of order 1-d of r over the shell (uniform in r)")
for dd in [3,4,5,6]:
    lhs = rho_star(dd, rn, q*rn)**(dd-1)
    # (1/Delta) int r^{1-d} dr  -> its reciprocal
    integ = sp.integrate(sp.Symbol('r')**(1-dd), ('r' if False else sp.Symbol('r'), rn, q*rn))/((q-1)*rn)
    print(f"  d={dd}: rho^(d-1) = {sp.simplify(lhs)}   |   1/<r^(1-d)> = {sp.simplify(1/integ)}   equal={sp.simplify(lhs-1/integ)==0}")

print()
print("[2] naive rule (drop the (d-2) factor): rho_naive, threshold q*_d, continuum limit")
for dd in [3,4,5,6,7]:
    rho_naive = ((q-1)*rn/(rn**(2-dd)-(q*rn)**(2-dd)))**sp.Rational(1,dd-1)
    lim = sp.limit(rho_naive/rn, q, 1)
    print(f"  d={dd}: lim_{{q->1}} rho_naive/r_n = {sp.nsimplify(sp.simplify(lim))} = {float(lim):.6f}"
          f"   (exact rule -> 1, naive ratio = (d-2)^(-1/(d-1)) = {float((dd-2)**(-1/(dd-1))):.6f})")
print("  q*_d solves q + q^(2-d) = 2  <=>  q^(d-1) - 2 q^(d-2) + 1 = 0  <=>  (q-1)(q^(d-2)-...-1)=0")
for dd in [3,4,5,6,7,10]:
    poly = sp.Poly(q**(dd-1) - 2*q**(dd-2) + 1, q)
    print("   d=%2d factor: %s" % (dd, sp.factor(poly.as_expr())))
sys.path.insert(0,'src')
from radialflux import q_star
for dd in [3,4,5,6,7]:
    qs=q_star(dd)
    print(f"   q*_{dd} = {'none (d=3: rule is exact)' if qs is None else f'{qs:.8f}'}")

print()
print("[3] recovered fraction kappa = Qhat/Q for the four rules, any d  (kappa exact, then series in eps=q-1)")
def kappas(dd):
    rho = rho_star(dd, rn, q*rn)
    kL = (rn/rho)**(dd-1)
    kR = (q*rn/rho)**(dd-1)
    kA = ((rn+q*rn)/2/rho)**(dd-1)
    return sp.simplify(kL), sp.simplify(kR), sp.simplify(kA)

e = sp.symbols('e', positive=True)
for dd in [3,4,5]:
    kL,kR,kA = kappas(dd)
    kL = kL.subs(q, 1+e); kR = kR.subs(q,1+e); kA = kA.subs(q,1+e)
    print(f"  d={dd}")
    print(f"    kappa_L exact = {sp.simplify(kL)}")
    print(f"    kappa_L series = {sp.series(kL, e, 0, 3)}")
    print(f"    kappa_R series = {sp.series(kR, e, 0, 3)}")
    print(f"    kappa_A exact = {sp.simplify(kA)}")
    print(f"    kappa_A series = {sp.series(kA, e, 0, 3)}")

print()
print("[4] uniform lattice r_n = n a : exact values")
n = sp.symbols('n', positive=True, integer=True)
for dd in [3]:
    rho = rho_star(dd, n*a, (n+1)*a)
    print(f"  d=3 rho/a = {sp.simplify(rho/a)}  (=sqrt(n(n+1)))")
    print(f"    kappa_L = {sp.simplify((n*a/rho)**2)}")
    print(f"    kappa_R = {sp.simplify(((n+1)*a/rho)**2)}")
    print(f"    kappa_A = {sp.simplify((((n+sp.Rational(1,2))*a)/rho)**2)}")
print("  midpoint: kappa_A - 1 = 1/(4n(n+1)) = 1/((2n+1)^2-1);  report's 4n(n+1)/(2n+1)^2 is its reciprocal")

print()
print("[5] forward / backward / central differences of C/r are EXACT at sqrt(r(r+a)), sqrt(r(r-a)), sqrt(r^2-a^2)")
r_ = sp.symbols('r', positive=True)
V = lambda x: 1/x
print("   (V(r)-V(r+a))/a - 1/(r(r+a)) =", sp.simplify((V(r_)-V(r_+a))/a - 1/(r_*(r_+a))))
print("   (V(r-a)-V(r))/a - 1/(r(r-a)) =", sp.simplify((V(r_-a)-V(r_))/a - 1/(r_*(r_-a))))
print("   (V(r-a)-V(r+a))/(2a)-1/(r^2-a^2)=", sp.simplify((V(r_-a)-V(r_+a))/(2*a) - 1/(r_**2-a**2)))

print()
print("[6] distributed source: rho_* vs w-weighted mean radius <r>_w, any d")
rr = sp.Symbol('r', positive=True)
for dd in [3,4,5,6]:
    num = sp.integrate(rr**(2-dd), (rr, rn, q*rn))
    den = sp.integrate(rr**(1-dd), (rr, rn, q*rn))
    meanr = sp.simplify(num/den)
    rho = rho_star(dd, rn, q*rn)
    diff = sp.series(sp.simplify(rho - meanr)/rn, q, 1, 3)
    print(f"  d={dd}: <r>_w/r_n = {sp.simplify(meanr/rn)} ; rho_*/r_n - <r>_w/r_n = {diff}")

print()
print("[7] running field A(r) = A0 + gamma ln r")
g = sp.symbols('gamma', real=True)
A = lambda x: sp.log(x)
for dd in [3,4]:
    A1 = A(rn); A2 = A(q*rn)
    Qhat = (A1*rn**(2-dd) - A2*(q*rn)**(2-dd))/(rn**(2-dd)-(q*rn)**(2-dd))
    ell = sp.simplify(A1 - Qhat)
    Qenc = A1 - g/(dd-2)
    bias = sp.simplify(Qhat - Qenc)
    print(f"  d={dd}:  ell_d(q) = A(r_n)-Qhat = {sp.simplify(ell)}")
    print(f"         Qhat - Qenc = {sp.simplify(bias)}")
    print(f"         1/(d-2) - ell = {sp.series(sp.simplify(1/(dd-2)-ell), q, 1, 3)}")
    print(f"         check vs ln q/2 - (d-2)(ln q)^2/12 : {sp.simplify(sp.series(sp.simplify(1/(dd-2)-ell) , q,1,2) - sp.series(sp.log(q)/2, q,1,2))}")

print()
print("[8] d=4 uniform ball: exact V and Q_enc, and the factor-2 failure of the missing (d-2)")
Rd, Qd = sp.symbols('R Q', positive=True)
print("   V_in(r) = Q/(4 pi^2 R^2) (2 - r^2/R^2)   [d=4, Omega_4 = 2 pi^2]")
print("   Omega_4 =", sp.simplify(Om(4)), " Omega_3 =", sp.simplify(Om(3)))
