"""radialflux: exact flux-sampling radii for radial Coulomb/Poisson problems.
Reference implementation accompanying the paper.  MIT licence."""
from __future__ import annotations
import math
import numpy as np

__all__ = ["omega_d","exact_radius","recovered_fraction","exact_bias","enclosed_charge",
           "q_star","naive_radius","uniform_rule_values","shell_identity",
           "source_bias","running_residual","background_charge","Sphere"]

def omega_d(d: float) -> float:
    """Surface area of the unit (d-1)-sphere: 2 pi^{d/2} / Gamma(d/2)."""
    return 2.0*math.pi**(d/2.0)/math.gamma(d/2.0)

def exact_radius(r_in, r_out, d):
    """rho_* with rho_*^{d-1} = (d-2)(r_out-r_in)/(r_in^{2-d}-r_out^{2-d}).

    Evaluated through the stable form rho^{d-1} = (r_out-r_in)/int_{r_in}^{r_out} r^{1-d}dr,
    i.e. via expm1, so that q -> 1 loses no significant digits."""
    r_in = np.asarray(r_in, dtype=float); r_out = np.asarray(r_out, dtype=float)
    a = d - 2.0
    lq = np.log(r_out/r_in)
    if a == 0.0:
        # d = 2: logarithmic mean rho = (r_out-r_in)/ln(r_out/r_in)
        return (r_out - r_in)/lq
    # int r^{1-d} dr = (r_in^{-a}-r_out^{-a})/a = r_in^{-a}(-expm1(-a ln q))/a
    with np.errstate(divide="ignore", invalid="ignore"):
        integ = r_in**(-a) * (-np.expm1(-a*lq))/a
        rho = (r_out - r_in)/integ
    return rho**(1.0/(d-1.0))

def recovered_fraction(rule: str, q, d):
    """kappa = Qhat/Q for rule in {'left','right','midpoint','exact'}."""
    q = np.asarray(q, dtype=float)
    rho = exact_radius(1.0, q, d)
    rp = {"left": np.ones_like(q), "right": q, "midpoint": 0.5*(1.0+q), "exact": rho}[rule]
    return (rp/rho)**(d-1.0)

def exact_bias(rule: str, q, d):
    """Closed form, no root needed.  kappa_L etc. (Eq. 12)."""
    q = np.asarray(q, dtype=float); e = q - 1.0
    a = d - 2.0
    if a == 0.0:                       # d = 2 limit: kappa_L = ln q/(q-1)
        kL = np.log(q)/e
    else:
        kL = (1.0 - q**(-a))/(a*e)
    if rule == "left":     return kL
    if rule == "right":    return kL*q**(d-1.0)
    if rule == "midpoint": return kL*(0.5*(1.0+q))**(d-1.0)
    if rule == "exact":    return np.ones_like(q)
    raise ValueError(rule)

def naive_radius(r_in, r_out, d):
    """The (d-2)-less generalisation; equals rho_* * (d-2)^{-1/(d-1)}."""
    return exact_radius(r_in, r_out, d)*(d-2.0)**(-1.0/(d-1.0))

def enclosed_charge(V, r, d, rule="exact", rho_prime=None):
    """Qhat from a radial potential sampled at r[0..N].  Uses the exact radius by default."""
    r = np.asarray(r, float); V = np.asarray(V, float)
    rp = exact_radius(r[:-1], r[1:], d) if rho_prime is None else np.asarray(rho_prime, float)
    dV = V[:-1] - V[1:]
    return omega_d(d)*rp**(d-1.0)*dV/np.diff(r)

def q_star(d, tol=1e-15):
    """Smallest q for which the (d-2)-less rule still samples inside the shell:
    the (d-2)-nacci constant, the root in (1, inf) of  q + q^{2-d} - 2 = 0.

    Returns None for d = 3, where the factor (d-2) equals one, the equation
    degenerates to the double root q = 1, and the rule is exact for every q,
    so that no threshold exists.  Pure NumPy/Python: no SciPy required.
    """
    if d == 3:
        return None
    f = lambda x: x + x**(2.0-d) - 2.0
    lo, hi = 1.0 + 1e-9, 2.0
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5*(lo + hi)
        if mid in (lo, hi):
            break
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol*hi:
            break
    return 0.5*(lo + hi)

def uniform_rule_values(n):
    """Exact kappa on a uniform lattice r_n = n a, d = 3 (Eq. 18)."""
    n = np.asarray(n, float)
    return {"left": n/(n+1.0), "right": (n+1.0)/n,
            "midpoint": (2*n+1)**2/(4*n*(n+1.0))}

def shell_identity(Q_in, Q_out, r_in, r_out, d, rho_integral=None):
    """d=3-any-d exact two-term shell identity (Eq. 24)."""
    a = d - 2.0
    return (Q_in*r_in**(2-d) - Q_out*r_out**(2-d))/(a*omega_d(d)) \
           + (0.0 if rho_integral is None else rho_integral/a)

def source_bias(r_n, q, d, s_over_Q=1.0):
    """Leading bias of the exact-radius rule for a source that varies across the shell."""
    return s_over_Q*(d-2.0)/24.0*(q-1.0)**2

def running_residual(q, d, gamma=1.0):
    """Exact finite-q residual for A(r) = A0 + gamma ln r."""
    return gamma*(1.0/(d-2.0) - np.log(q)/(np.asarray(q, float)**(d-2.0)-1.0))

def background_charge(R, L, d=3):
    """Charge reading polluted by a neutralising background, relative to Q:
    1 - (2 pi/3) R(R+1)(2R+1)/L^3  for d = 3."""
    R = np.asarray(R, float)
    return 1.0 - (2.0*math.pi/(3.0*L**3))*R*(R+1.0)*(2*R+1.0)

class Sphere:
    """Closed-form radial profiles used in the verification suite (Sec. 9)."""
    def __init__(self, kind, d=3, **kw):
        self.kind, self.d, self.p = kind, d, kw
    def Q(self, r):
        k, p = self.kind, self.p
        if k == "ball":
            return p["Q"]*np.minimum(r, p["R"])**self.d/p["R"]**self.d
        if k == "hydrogen1s":
            Z = p.get("Z", 1.0)
            return 1.0-np.exp(-2*Z*r)*(1+2*Z*r+2*(Z*r)**2)
        if k == "hernquist":                     # V = +GM/(r+a)  (electric sign convention)
            return omega_d(3)*p["M"]*r**2/(r+p["a"])**2
        if k == "plummer":
            return omega_d(3)*p["M"]*r**3/(r**2+p["a"]**2)**1.5
        raise ValueError(k)
    def V(self, r):
        k, p, d = self.kind, self.p, self.d
        O = omega_d(d)
        if k == "ball":
            Q, R = p["Q"], p["R"]
            inside = Q*(d*R**2-(d-2)*r**2)/(2*(d-2)*O*R**d)
            outside = Q/((d-2)*O*r**(d-2))
            return np.where(r <= R, inside, outside)
        if k == "hydrogen1s":
            Z = p.get("Z", 1.0)
            return (1.0-(1+Z*r)*np.exp(-2*Z*r))/(4*math.pi*r)
        if k == "hernquist":
            return p["M"]/(r+p["a"])
        if k == "plummer":
            return p["M"]/np.sqrt(r**2+p["a"]**2)
        raise ValueError(k)
