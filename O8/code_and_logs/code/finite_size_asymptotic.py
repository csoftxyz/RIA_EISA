#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finite_size_asymptotic.py

Analytic large-F expansion of the closed-surface Wilson loop, and its numerical
verification (with error control).

For a loop enclosing A faces of a closed surface with F faces,

    <W_A> = sum_n I_n^A I_{n+1}^{F-A} / sum_n I_n^F .

The sum is dominated by n = -1.  Expanding the subleading terms gives, for
A < F/2,

    <W_A> = r^A { 1 + r^{F-2A} [ 1 + (I_2 r / I_1)^A - 2 r^{2A} ]
                  + O(r^{2(F-2A)}) } ,

    r = I_1(beta)/I_0(beta) = exp(-sigma_infinity).

For A = 1 this gives the finite-volume tension

    sigma_inf - sigma_F = r^{F-2} [1 + (I_2/I_1) r - 2 r^2] + O(r^{2F-4})
                        = 3.5565... e^{-0.80656 F}   at beta = 1.

The script prints the exact and asymptotic values, their relative error, and
the ratio R_F = (<W>/r - 1)/r^{F-2}, which must tend to the analytic bracket.
"""
import json
import os
import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "logs", "finite_size_asymptotic.json")


def I(n, b=1):
    return mp.besseli(int(n), b)


def ZF(F, b=1):
    return sum(I(n, b) ** F for n in range(-30, 31))


def W(F, A, b=1):
    return sum(I(n, b) ** A * I(n + 1, b) ** (F - A)
               for n in range(-25, 25)) / ZF(F, b)


def main():
    b = mp.mpf(1)
    r = I(1, b) / I(0, b)
    s = -mp.log(r)
    i21 = I(2, b) / I(1, b)

    # A = 1: exact, asymptotic, error, ratio
    bracket1 = 1 + i21 * r - 2 * r ** 2
    pref = mp.e ** (2 * s) * bracket1
    print("beta = 1")
    print("r = I1/I0 =", mp.nstr(r, 14))
    print("sigma_inf =", mp.nstr(s, 14))
    print("I2/I1 =", mp.nstr(i21, 10))
    print("bracket (A=1) =", mp.nstr(bracket1, 12))
    print("analytic prefactor e^{2 sigma} * bracket =", mp.nstr(pref, 12))
    print()
    print(f"{'F':>3} {'sigma_F exact':>18} {'sigma_F asym':>18} "
          f"{'rel. error':>12} {'R_F':>12}")
    rows = []
    for F in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        ex = -mp.log(W(F, 1, b))
        asym = s - r ** (F - 2) * bracket1
        RF = (mp.e ** (-ex) / r - 1) / r ** (F - 2)
        rel = (asym - ex) / ex
        rows.append(dict(F=F, exact=mp.nstr(ex, 16), asym=mp.nstr(asym, 16),
                         rel=mp.nstr(rel, 6), RF=mp.nstr(RF, 10)))
        print(f"{F:>3} {mp.nstr(ex,14):>18} {mp.nstr(asym,14):>18} "
              f"{mp.nstr(rel,4):>12} {mp.nstr(RF,8):>12}")
    print("\nlimit of R_F =", mp.nstr(bracket1, 10))

    # general A: coefficient c_A
    print("\nA-dependence: c_A = 1 + (I2 r/I1)^A - 2 r^{2A}")
    arows = []
    for A in (1, 2, 3, 4):
        cA = 1 + (i21 * r) ** A - 2 * r ** (2 * A)
        vals = []
        for F in (12, 20, 32):
            if F - 2 * A <= 0:
                continue
            R = (W(F, A, b) / r ** A - 1) / r ** (F - 2 * A)
            vals.append((F, mp.nstr(R, 8)))
        arows.append(dict(A=A, cA=mp.nstr(cA, 12), limits=vals))
        print(f"  A={A}: c_A = {mp.nstr(cA,10)}  R_F -> {vals}")

    # error at the octahedron
    ex8 = -mp.log(W(8, 1, b))
    as8 = s - r ** 6 * bracket1
    print("\nF=8: exact %.12f, asymptotic %.12f, abs. error %.2e, rel. %.2e"
          % (float(ex8), float(as8), float(abs(as8 - ex8)), float(abs(as8 - ex8) / ex8)))

    json.dump(dict(r=mp.nstr(r, 16), sigma=mp.nstr(s, 16), I2I1=mp.nstr(i21, 12),
                   bracket=mp.nstr(bracket1, 14), prefactor=mp.nstr(pref, 14),
                   table=rows, general_A=arows), open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
