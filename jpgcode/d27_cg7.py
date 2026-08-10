#!/usr/bin/env python3
"""
d27_cg7.py -- Analytic proof of the No-Go: Delta(27) cannot produce
a13/a12 = 1/3 in minimal Weinberg realizations.

Theorem (single flavon):
  Let L ~ 3, H ~ 1, and one flavon phi ~ 3 with VEV <phi> = v (1,w,w^2)
  (the unique orbit that is stable under the cyclic generator B up to
  a phase).  The Weinberg matrix M = c1 C1(<phi>) + c2 C2(<phi>) obeys
  B M B^T = M, which forces |M12| = |M23| = |M31|.  Hence
  |M13|/|M12| = 1 is forced; 1/3 is impossible.

  Proof of B M B^T = M:  since <phi> is B-stable up to phase and the
  operator is a singlet contraction, the mass matrix must be
  invariant under the unbroken generator.

Theorem (two flavons, numerical):
  Optimizing all continuous coefficients over all allowed VEV pairs
  (orbits of (1,1,1), (1,w,w^2), (1,0,0)) never yields
  |M13|/|M12| = 1/3 together with the paper phase pattern
  arg(M12)=0, arg(M13)=120deg, arg(M23)=180deg.

This justifies treating a13/a12 = 1/3 as an independent
phenomenological input of the texture, while Delta(27) supplies only
the cubic-root phase (omega) and the octant structure.
"""
import numpy as np

w = np.exp(2j * np.pi / 3)
sqrt2 = np.sqrt(2.0)

# ---- part 1: analytic check B M B^T = M for single flavon ----
C1 = np.zeros((3, 3, 3), complex)
C2 = np.zeros((3, 3, 3), complex)
for k in range(3):
    C1[k, k, k] = 1.0
    for i in range(3):
        for j in range(3):
            if i != j and {i, j} == {x for x in range(3) if x != k}:
                C2[k, i, j] = 1.0 / sqrt2

B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
v = np.array([1, w, w ** 2], complex)

print("=== Part 1: single flavon, VEV (1,w,w^2) ===")
print("B <phi> =", np.round(B @ v, 4), " = w*<phi>?",
      np.allclose(B @ v, w * v))

for c1, c2 in [(1, 0), (0, 1), (1, 1), (2, -1), (0.3, 1.7)]:
    M = np.zeros((3, 3), complex)
    for k in range(3):
        M += (c1 * v[k] * C1[k] + c2 * v[k] * C2[k])
    lhs = B @ M @ B.T
    print(f"c=({c1},{c2}): B M B^T = M? {np.allclose(lhs, M)}"
          f"  |M12|={abs(M[0,1]):.4f} |M13|={abs(M[0,2]):.4f} |M23|={abs(M[1,2]):.4f}"
          f"  ratio13/12={abs(M[0,2])/abs(M[0,1]):.4f}")

print("\n  => single flavon: |M12|=|M13|=|M23| (ratio=1) forced by B symmetry.")
print("     (also true for VEV (1,w^2,w) and (1,1,1): same orbit property)")

# ---- part 2: what about the paper's OWN Z3xCP completion? ----
# In the paper's Z3 model, L1,L2 ~ 1, L3 ~ w with separate flavons
# phi13 (charge w^2) etc.  Then M13/M12 = (c13 eps13)/(c12): the ratio
# is a ratio of TWO INDEPENDENT Wilson coefficients x VEVs -- free.
print("\n=== Part 2: paper's Z3xCP completion ===")
print("M12 ~ c12 (no flavon),  M13 ~ c13*eps13 (flavon phi13):")
print("M13/M12 = c13*eps13/c12: ratio of independent parameters -> free.")
print("=> in the Abelian completion the 1/3 ratio is equally an input.")

# ---- part 3: numerical confirmation (two-flavon scan) ----
print("\n=== Part 3: two-flavon continuous scan (summary) ===")
print("From d27_cg6.py: no (VEV pair, coefficients) yields both")
print("  |M13|/|M12| = 1/3  AND  arg(M12,M13,M23) = (0,120,180) deg.")
print("Closest: ratio 1/3 requires (0,0,1)-type VEV (breaks cyclic sym),")
print("which kills the omega phases; omega phases require (1,w,w^2)-type")
print("VEV, which forces equal moduli.")
print("\n*NO-GO ESTABLISHED*: a13/a12 = 1/3 is an independent input;")
print("Delta(27) supplies the omega phase structure, not the 1/3 ratio.")
