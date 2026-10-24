#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Villain (periodic Gaussian) formulation on a closed surface with F faces.

    v(theta) = sum_k exp[-beta/2 (theta-2 pi k)^2]

Poisson resummation + the Gauss-law constraint (all dual-connected faces carry
the same integer charge) give the closed form

    Z_V(beta) = (2 pi beta)^(-F/2) sum_m exp[-F m^2/(2 beta)]
              = (2 pi beta)^(-F/2) theta_3(0, e^{-F/(2 beta)}).

Modular transformation gives an exact duality beta -> beta_*^2/beta with
beta_* = F/(2 pi), i.e. the self-dual coupling is *extensive* in F.

Also compares the three natural Wilson->Villain dictionaries.
"""
import json
import os
import mpmath as mp

mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__))


def ZV(F, beta, M=40):
    beta = mp.mpf(beta)
    pref = (2 * mp.pi * beta) ** (-mp.mpf(F) / 2)
    s = mp.mpf(1)
    for m in range(1, M + 1):
        t = mp.e ** (-F * m * m / (2 * beta))
        s += 2 * t
        if t < mp.mpf("1e-60"):
            break
    return pref * s


def main():
    res = {}
    print("== Villain partition function, F = 8, beta = 1 ==")
    zv = ZV(8, 1)
    print("Z_V(1)      =", mp.nstr(zv, 16))
    print("theta3 form =", mp.nstr((2 * mp.pi) ** -4 * mp.jtheta(3, 0, mp.e ** -4), 16))

    print("\n== self-dual coupling beta_* = F/(2 pi) ==")
    for F in (4, 6, 8, 12, 20):
        bs = mp.mpf(F) / (2 * mp.pi)
        # exact duality at beta = 1: Z_V(1) = C * Z_V(bs^2)
        lhs = ZV(F, 1)
        rhs = ZV(F, bs ** 2 / 1)
        C = mp.sqrt(2 * mp.pi / F) * bs ** F * mp.mpf(1) ** (mp.mpf(1) / 2 - F)
        print(f"F={F:>2}  beta_*={mp.nstr(bs,10):>12}  "
              f"Z_V(1)/Z_V(beta_*^2)={mp.nstr(lhs/rhs,10):>12}  C={mp.nstr(C,10):>12}")
        res[f"duality_F{F}"] = dict(beta_star=mp.nstr(bs, 12),
                                    ratio=mp.nstr(lhs / rhs, 12),
                                    C=mp.nstr(C, 12))
    # full identity check at generic beta, F=8
    F = 8
    bs = mp.mpf(F) / (2 * mp.pi)
    for b in ("0.7", "1.0", "2.3"):
        b = mp.mpf(b)
        lhs = ZV(F, b)
        rhs = (mp.sqrt(2 * mp.pi / F) * bs ** F * b ** (mp.mpf(1) / 2 - F)
               * ZV(F, bs ** 2 / b))
        res[f"identity_beta_{mp.nstr(b,4)}"] = mp.nstr(lhs / rhs, 12)
        print(f"identity beta={mp.nstr(b,4)}: Z_V(beta)/[C(beta) Z_V(beta_*^2/beta)]"
              f" - 1 = {mp.nstr(lhs/rhs-1, 6)}")

    print("\n== Wilson -> Villain dictionaries at beta_W = 1 ==")
    r = mp.besseli(1, 1) / mp.besseli(0, 1)
    bv_ratio = -1 / (2 * mp.log(r))
    bv_curv = mp.mpf(1)
    # first Fourier coefficient of the Villain weight: max over beta
    f = lambda b: (2 * mp.pi * b) ** mp.mpf("-0.5") * mp.e ** (-1 / (2 * b))
    bmax = mp.findroot(lambda b: mp.diff(f, b), 1.0)
    print("ratio match:        beta_V =", mp.nstr(bv_ratio, 8), " (4/pi =",
          mp.nstr(4 / mp.pi, 8), ")")
    print("curvature match:    beta_V =", mp.nstr(bv_curv, 8))
    print("amplitude match:    max Fourier coeff =", mp.nstr(f(bmax), 8),
          "at beta_V =", mp.nstr(bmax, 6), "vs I_1(1) =", mp.nstr(r, 8))
    # Wilson coupling whose ratio-matched Villain coupling equals beta_* (F=8)
    target = 4 / mp.pi
    g = lambda b: -1 / (2 * mp.log(mp.besseli(1, b) / mp.besseli(0, b))) - target
    bw = mp.findroot(g, 2.0)
    print("Wilson beta_W that maps onto beta_* = 4/pi under ratio matching:",
          mp.nstr(bw, 8))
    res["dictionaries"] = dict(ratio=mp.nstr(bv_ratio, 10),
                               curvature=mp.nstr(bv_curv, 10),
                               amp_max=mp.nstr(f(bmax), 10),
                               amp_at=mp.nstr(bmax, 10),
                               I1_over_I0=mp.nstr(r, 10),
                               beta_W_to_beta_star=mp.nstr(bw, 10))

    # Villain pressure / plaquette at beta=1, and comparison with Wilson
    print("\n== Villain vs Wilson at beta = 1 (F=8) ==")
    ZW = sum(mp.besseli(n, 1) ** 8 for n in range(-30, 31))
    print("Z_W(1) =", mp.nstr(ZW, 12), " Z_V(1) =", mp.nstr(zv, 12),
          " ratio =", mp.nstr(ZW / zv, 10))
    res["Z_W"] = mp.nstr(ZW, 14)
    res["Z_V"] = mp.nstr(zv, 14)

    with open(os.path.join(HERE, os.pardir, "logs", "villain_general.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
