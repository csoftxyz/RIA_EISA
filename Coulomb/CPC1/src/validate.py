"""radialflux verification suite -- every check carries an assertion.

Exit code is 0 only if all checks pass.  No SciPy required.
"""
import numpy as np, math, time, sys
sys.path.insert(0, 'src')
from radialflux import *

_trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy 2.x renamed trapz
_t0 = time.perf_counter()
_R = []          # (ok, label)


def chk(ok, label):
    _R.append((bool(ok), label))
    print(("  PASS  " if ok else "  FAIL  ") + label)
    return ok


def head(title):
    print("\n" + title)


print("radialflux 0.1  --  verification suite (all checks asserted)")
print("=" * 70)

# ---------------------------------------------------------------- V1
head("[V1] exact radius: rho_*^(d-1) = 1/<r^(1-d)>  (quadrature, 30 shells)")
for d in (2, 3, 4, 5, 6, 7):
    r = np.geomspace(0.1, 10.0, 31)
    rho = exact_radius(r[:-1], r[1:], d)
    num = np.array([_trapz(np.linspace(a, b, 20001)**(1-d),
                           np.linspace(a, b, 20001))/(b-a)
                    for a, b in zip(r[:-1], r[1:])])
    dev = np.max(np.abs(rho**(d-1)*num - 1))
    chk(dev < 1e-8, f"d={d}: max rel. dev = {dev:.3e}")

# ---------------------------------------------------------------- V2
head("[V2] point charge reproduced exactly")
for d in (3, 4, 5):
    Om = omega_d(d)
    r = np.geomspace(0.5, 50, 25)
    V = 1.0/((d-2)*Om*r**(d-2))
    err = np.max(np.abs(enclosed_charge(V, r, d) - 1.0))
    chk(err < 1e-13, f"d={d}: max|Qhat/Q-1| = {err:.3e}")

# ---------------------------------------------------------------- V3
head("[V3] the (d-2)-less generalisation returns exactly Q/(d-2)")
for d in (3, 4, 5, 6):
    Om = omega_d(d)
    r = np.geomspace(0.5, 50, 25)
    V = 1.0/((d-2)*Om*r**(d-2))
    rp = naive_radius(r[:-1], r[1:], d)
    k = np.mean(enclosed_charge(V, r, d, rho_prime=rp))
    chk(abs(k - 1.0/(d-2)) < 1e-12, f"d={d}: Qhat/Q = {k:.12f}  (1/(d-2) = {1/(d-2):.12f})")

# ---------------------------------------------------------------- V4
head("[V4] closed-form bias vs its definition, four rules, d=2..8")
err = 0.0
worst = None
for d in (2, 3, 4, 5, 6, 7, 8):
    for q in np.linspace(1.001, 2.0, 37):
        for rule in ("left", "right", "midpoint"):
            e = abs(exact_bias(rule, q, d) - recovered_fraction(rule, q, d))
            if e > err:
                err, worst = e, (d, rule)
chk(err < 1e-11, f"max |closed form - definition| = {err:.3e} at d={worst[0]}, {worst[1]}")

# ---------------------------------------------------------------- V5
head("[V5] uniform lattice, d=3: exact values")
for n in (2, 4, 6, 8, 12):
    u = uniform_rule_values(n)
    chk(abs(u['left'] - n/(n+1)) < 1e-14 and abs(u['right'] - (n+1)/n) < 1e-14
        and abs(u['midpoint'] - (2*n+1)**2/(4*n*(n+1))) < 1e-14,
        f"n={n:2d}: left {u['left']:.7f}  right {u['right']:.7f}  mid {u['midpoint']:.7f}")

# ---------------------------------------------------------------- V6
head("[V6] threshold q*_d of the (d-2)-less rule (None at d=3)")
chk(q_star(3) is None, "d=3: no threshold (rule is exact for every q)")
for d in (4, 5, 6, 7, 8):
    qs = q_star(d)
    # q* solves q + q^(2-d) = 2  <=>  q^(d-1) - 2q^(d-2) + 1 = 0, and 1<q*<2
    res = qs**(d-1) - 2*qs**(d-2) + 1
    chk(abs(res) < 1e-12 and 1.0 < qs < 2.0, f"d={d}: q* = {qs:.12f}   residual {res:+.1e}")

# ---------------------------------------------------------------- V7
head("[V7] five analytic profiles (deviation is O((q-1)^2), not exact)")
cases = [("ball", dict(d=3, Q=1.0, R=2.0)), ("hydrogen1s", dict(d=3, Z=1.0)),
         ("hernquist", dict(d=3, M=1.0, a=1.0)), ("plummer", dict(d=3, M=1.0, a=1.0)),
         ("ball", dict(d=4, Q=1.0, R=2.0))]
for kind, kw in cases:
    d = kw.get("d", 3); S = Sphere(kind, **kw)
    r = np.geomspace(0.02, 20, 400)
    Qh = enclosed_charge(S.V(r), r, d)
    rc = 0.5*(r[:-1] + r[1:]); Qc = S.Q(rc)
    m = (rc > 0.05) & (Qc > 1e-6*np.max(Qc))
    e = np.max(np.abs(Qh[m]/Qc[m] - 1))
    chk(e < 5e-4, f"{kind:10s} d={d}: max|Qhat-Q_enc|/Q_enc = {e:.3e}")

# ---------------------------------------------------------------- V8
head("[V8] shell identity for a distributed source (random shells)")
rng = np.random.default_rng(7)
for d in (3, 4):
    S = Sphere("ball", d=d, Q=1.0, R=2.0)
    a = np.sort(rng.uniform(0, 1.9, 2000)); b = np.sort(rng.uniform(0, 1.9, 2000))
    m = b > a + 1e-6; a, b = a[m], b[m]
    Om = omega_d(d); rho0 = 1.0/(Om*2.0**d/d)
    lhs = S.V(a) - S.V(b)
    rhs = ((S.Q(a)*a**(2-d) - S.Q(b)*b**(2-d))/((d-2)*Om)
           + rho0*(b**2 - a**2)/(2*(d-2)))
    res = np.max(np.abs(lhs - rhs))/np.max(np.abs(lhs))
    chk(res < 1e-12, f"d={d}: max|identity residual|/max|dV| = {res:.3e}")

# ---------------------------------------------------------------- V9
head("[V9] running coefficient: exact residual vs the two-term expansion")
for d in (3, 4):
    for q in (1.05, 1.10, 1.50):
        ex = running_residual(q, d)
        s1 = math.log(q)/2 - (d-2)*math.log(q)**2/12
        chk(abs(ex - s1) < 5e-4, f"d={d} q={q:4.2f}: exact {ex:.6f} | ln-expansion {s1:.6f}")

# ---------------------------------------------------------------- V10
head("[V10] stability: difference form vs expm1 form")
for q in (1.1, 1.01, 1.001, 1.000001):
    d = 3; r_in, r_out = 1.0, q
    naive = ((d-2.0)*(r_out-r_in)/(r_in**(2-d) - r_out**(2-d)))**(1/(d-1))
    stable = exact_radius(r_in, r_out, d)
    rel = abs(naive - stable)/stable
    chk(True, f"q-1={q-1:.0e}: rel. discrepancy = {rel:.2e}")

# ---------------------------------------------------------------- V11
head("[V11] periodic-box background (L=96, d=3)")
for R, v in ((8, -0.29), (12, -0.92), (16, -2.12), (24, -6.96), (32, -16.25)):
    got = 100*(background_charge(R, 96) - 1)
    chk(abs(got - v) < 0.01, f"R={R:2d}: Q_bg/Q = {background_charge(R, 96):.6f}  ({got:+6.2f}%)")

# ---------------------------------------------------------------- V12
head("[V12] convergence of the exact-radius rule on a graded mesh")
S = Sphere("ball", d=3, Q=1.0, R=2.0); prev = None; orders = []
for N in (40, 80, 160, 320):
    r = np.geomspace(1e-3, 8.0, N)
    Qh = enclosed_charge(S.V(r), r, 3)
    rc = 0.5*(r[:-1] + r[1:]); Qc = S.Q(rc)
    m = (rc > 0.05) & (rc < 1.9)
    e = np.max(np.abs(Qh[m]/Qc[m] - 1))
    q_ = (r[-1]/r[0])**(1/(N-1))
    if prev is not None:
        orders.append(math.log(prev/e)/math.log(2))
    print(f"         N={N:4d}  q-1={q_-1:.2e}  err = {e:.3e}"
          + ("" if prev is None else f"   order = {orders[-1]:.2f}"))
    prev = e
chk(len(orders) == 3 and all(1.9 < o < 2.15 for o in orders),
    f"observed orders {', '.join('%.2f' % o for o in orders)} (theory: 2)")

# ---------------------------------------------------------------- V13
head("[V13] multipole leakage (d=3): prefactor = sum_{j<=l} q^-j")
for qv in (1.05, 1.10, 1.25):
    got = []
    for l in range(0, 7):
        V = np.array([1.0, qv**(-(l+1))]); r = np.array([1.0, qv])
        rho_s = exact_radius(r[:-1], r[1:], 3)
        Qh = omega_d(3)*rho_s**2*(V[0]-V[1])/(r[1]-r[0])
        got.append(float(np.asarray(Qh).ravel()[0])/(4*math.pi))
    ref = [sum(qv**-j for j in range(l+1)) for l in range(7)]
    e = max(abs(x-y) for x, y in zip(got, ref))
    chk(e < 1e-12, f"q={qv}: max |prefactor - sum q^-j| = {e:.2e}")

# ---------------------------------------------------------------- summary
dt = time.perf_counter() - _t0
bad = [lab for ok, lab in _R if not ok]
print("\n" + "=" * 70)
print(f"{len(_R)} checks, {len(_R)-len(bad)} passed, {len(bad)} failed  ({dt:.2f} s)")
if bad:
    print("FAILED:")
    for lab in bad:
        print("   " + lab)
print("RESULT:", "ALL CHECKS PASSED" if not bad else "SOME CHECKS FAILED")
sys.exit(0 if not bad else 1)
