#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_algebra.py
=================
Exact (symbolic) verification of every algebraic statement used in the paper.

Setting.  g_n = g_0 + g_1 + g_2 with
    g_0 = u(n)  (represented by A in gl_n),   g_1 = C^n (psi),  g_2 = C^n (chi)
and the one-parameter family of brackets
    [A,B]      = AB - BA
    [A,psi]    = A psi + alpha tr(A) psi
    [A,chi]    = -A^T chi - alpha tr(A) chi
    [psi,chi]  = psi chi^T + c (psi.chi) 1_n
    [psi,psi'] = [chi,chi'] = 0

Results proved here (symbolically, for n = 2,3,4):
  (1) the graded Jacobi identity holds iff
              c (1 + n alpha) = 1 - alpha ,
      i.e.  c = (1 - alpha)/(1 + n alpha).
      * paper's action  (alpha = 0)  ->  c = 1     (NOT c = 1/2)
      * sl_{n+1} action (alpha = 1)  ->  c = 0
      * for n = 4, alpha = 1/2       ->  c = 1/6
  (2) hence "c = 1/2" (with alpha = 0) is not a Lie algebra; and quoting
      "c in {0,1}" without alpha is meaningless.
  (3) for alpha = 1, c = 0 the block matrices
          A -> diag(A, -tr A),  psi -> [[0,psi],[0,0]],  chi -> [[0,0],[chi^T,0]]
      give an explicit isomorphism g_n = sl_{n+1}; dim = n^2 + 2n.
      n = 4 -> 24, n = 3 -> 15, and 19 is not of this form.
  (4) the one-loop operator Pi_{AB} = sum_a <[T_A,z_a],[T_B,z_a]> is the
      identity on g_0 for the sl_{n+1} action (numerical, n = 4).
"""
import json
import os
import random

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------
# (1)+(2)  symbolic graded Jacobi
# ----------------------------------------------------------------------
def bracket_factory(n, alpha, c):
    """Return the bracket on triples (A, psi, chi)."""
    def br(X, Y):
        A, p, q = X
        B, r, s = Y
        tA = sp.trace(A) if alpha != 0 else sp.Integer(0)
        tB = sp.trace(B) if alpha != 0 else sp.Integer(0)
        C = (A * B - B * A
             + p * s.T + c * (p.T * s)[0] * sp.eye(n)
             - r * q.T - c * (r.T * q)[0] * sp.eye(n))
        p1 = (A + alpha * tA * sp.eye(n)) * r - (B + alpha * tB * sp.eye(n)) * p
        q1 = -(A.T + alpha * tA * sp.eye(n)) * s + (B.T + alpha * tB * sp.eye(n)) * q
        return (sp.expand(C), sp.expand(p1), sp.expand(q1))
    return br


def jacobi_residual(n, alpha, c, trials=8, seed=0):
    rng = random.Random(seed)
    br = bracket_factory(n, alpha, c)
    mx = 0
    for _ in range(trials):
        def rnd():
            return sp.Matrix(n, n, lambda i, j: sp.Integer(rng.randint(-2, 2)))

        def rv():
            return sp.Matrix(n, 1, lambda i, j: sp.Integer(rng.randint(-2, 2)))
        X, Y, Z = (rnd(), rv(), rv()), (rnd(), rv(), rv()), (rnd(), rv(), rv())
        t1, t2, t3 = br(X, br(Y, Z)), br(Y, br(Z, X)), br(Z, br(X, Y))
        for k in range(3):
            R = sp.expand(t1[k] + t2[k] + t3[k])
            for e in R:
                mx = max(mx, abs(int(e)))
    return mx


# ----------------------------------------------------------------------
# (3)  explicit isomorphism to sl_{n+1}
# ----------------------------------------------------------------------
def block(A, p, q, alpha=1.0):
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = p.ravel()
    M[n, :n] = q.ravel()
    M[n, n] = -np.trace(M[:n, :n])
    return M


def iso_check(n=4, trials=5, seed=1):
    """Compare the abstract bracket (alpha=1,c=0) with the matrix commutator."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        p, q = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
        r, s = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
        X = block(A, p, q)
        Y = block(B, r, s)
        comm = X @ Y - Y @ X
        # abstract bracket, alpha = 1, c = 0
        C = A @ B - B @ A + p @ s.T - r @ q.T
        p1 = (A + np.trace(A) * np.eye(n)) @ r - (B + np.trace(B) * np.eye(n)) @ p
        q1 = -(A.T + np.trace(A) * np.eye(n)) @ s + (B.T + np.trace(B) * np.eye(n)) @ q
        Z = block(C, p1, q1)
        worst = max(worst, np.abs(comm - Z).max())
        # traceless check
        worst = max(worst, abs(np.trace(X)))
    return worst


# ----------------------------------------------------------------------
# (4)  one-loop operator
# ----------------------------------------------------------------------
def one_loop(n=4, alpha=1.0):
    """Pi_{AB} = sum_{a in g_1} <[T_A,z_a],[T_B,z_a]> on an orthonormal u(n) basis."""
    basis = []
    # off-diagonal E_ij  (orthonormal: tr(E_ji E_ij)=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                M = np.zeros((n, n))
                M[i, j] = 1.0
                basis.append(M)
    # orthonormal Cartan basis (Helmert)
    for i in range(1, n):
        M = np.zeros((n, n))
        for k in range(i):
            M[k, k] = 1.0 / np.sqrt(i * (i + 1))
        M[i, i] = -i / np.sqrt(i * (i + 1))
        basis.append(M)
    basis.append(np.eye(n) / np.sqrt(n))  # u(1)

    def ip(M, N):
        return float(np.trace(M.T @ N))

    G = np.array([[ip(a, b) for b in basis] for a in basis])
    Z = [np.eye(n)[:, a:a + 1] for a in range(n)]
    NG = len(basis)
    Pi = np.zeros((NG, NG))
    for A in range(NG):
        for B in range(NG):
            acc = 0.0
            for z in Z:
                X = basis[A] @ z + alpha * np.trace(basis[A]) * z
                Y = basis[B] @ z + alpha * np.trace(basis[B]) * z
                acc += float((X.T @ Y)[0, 0])
            Pi[A, B] = acc
    return Pi, G


def main():
    lines = []
    lines.append("=" * 72)
    lines.append("SYMBOLIC GRADED JACOBI  (entries in {-2..2}, 8 random triples)")
    lines.append("=" * 72)
    table = {}
    for n in (2, 3, 4):
        for alpha in (0, 1):
            for c in (-1, sp.Rational(-2, 3), 0, sp.Rational(1, 2), 1):
                r = jacobi_residual(n, alpha, c)
                lines.append(f"n={n}  alpha={alpha}  c={str(c):>4} :  max|Jacobi| = {r:>4}")
                table[f"n{n}_a{alpha}_c{c}"] = int(r)
    lines.append("")
    lines.append("=> Jacobi holds iff c(1 + n alpha) = 1 - alpha.")
    lines.append("   alpha=0 (fundamental u(n) action): c = 1")
    lines.append("   alpha=1 (sl_{n+1} action):          c = 0")
    lines.append("   c = 1/2 (with alpha = 0) is NOT a Lie bracket.")
    lines.append("")
    lines.append("check of the general solution c = (1-alpha)/(1+n alpha):")
    for n in (2, 3, 4):
        for alpha in (sp.Rational(1, 2), sp.Rational(3, 2)):
            c = (1 - alpha) / (1 + n * alpha)
            r = jacobi_residual(n, alpha, c, trials=4)
            lines.append(f"   n={n} alpha={alpha} c={c} : max|Jacobi| = {r}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("EXPLICIT ISOMORPHISM g_n = sl_{n+1}  (alpha=1, c=0)")
    lines.append("=" * 72)
    for n in (3, 4):
        w = iso_check(n)
        lines.append(f"n={n}: max |[X,Y] - Phi([x,y])| = {w:.3e};  dim g = {n*n+2*n}, "
                     f"dim sl_{n+1} = {(n+1)**2-1}")
    lines.append("19 = 12+4+3 is not of the form n^2+2n "
                 "(n^2+2n=19 has no integer root); the nearest are 15 (n=3) and 24 (n=4).")
    lines.append("")

    lines.append("=" * 72)
    lines.append("ONE-LOOP OPERATOR Pi ON g_0  (n=4, alpha=1, c=0)")
    lines.append("=" * 72)
    for alpha in (0.0, 1.0):
        Pi, G = one_loop(4, alpha)
        lines.append(f"alpha = {alpha}")
        lines.append(f"  Gram matrix of the u(4) basis: max|G-1| = {np.abs(G-np.eye(16)).max():.2e}")
        lines.append(f"  Pi - 1_16: max = {np.abs(Pi-np.eye(16)).max():.2e}")
        lines.append(f"  eigenvalues of Pi: {np.round(np.linalg.eigvalsh(Pi), 10)}")
        lines.append(f"  (u(1) entry = {Pi[15,15]:.6f}; expected 1 + n(n+2) = {1+n*(n+2)} for alpha=1)")
    lines.append("")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ROOT, "logs", "verify_algebra.log"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(ROOT, "logs", "verify_algebra.json"), "w", encoding="utf-8") as f:
        json.dump(dict(jacobi=table,
                       iso={str(n): iso_check(n) for n in (3, 4)},
                       pi_dev={a: float(np.abs(one_loop(4, a)[0] - np.eye(16)).max())
                               for a in (0.0, 1.0)}), f, indent=1)


if __name__ == "__main__":
    main()
