# TEST REPORT — octahedron submission package (v3)

**Date:** 2026-10-24
**Run command:** `MPLBACKEND=Agg bash run_all_tests.sh`
**Result:** **ALL 20 STEPS PASSED**, `verify_all.py` **28/28 checks passed**
**Total wall time:** ≈ 4–5 min

## Environment

| item | version |
|---|---|
| OS | Linux x86_64 (container `ed4eae63e5aa`) |
| Python | 3.12.3 |
| numpy / scipy / mpmath / matplotlib | 1.26.4 / 1.11.4 / 1.2.1 / 3.6.3 |
| pdfTeX | 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian) |
| XeTeX | 3.141592653-2.6-0.999995 |

## Step-by-step

| # | step | exit | time | log |
|---|---|---|---|---|
| 1 | `exact_gauge.py` | 0 | 19 s | `logs/exact_gauge.log` |
| 2 | `finite_size_asymptotic.py` | 0 | 31 s | `logs/finite_size.log` |
| 3 | `wilson_loops.py` | 0 | 40 s | `logs/wilson_loops.log` |
| 4 | `villain_general.py` | 0 | 2 s | `logs/villain_general.log` |
| 5 | `triangulation_uniqueness.py` | 0 | 0 s | `logs/triangulation.log` |
| 6 | `dirac_operator_checks.py` | 0 | 0 s | `logs/dirac_checks.log` |
| 7 | `fermion_spectrum.py` | 0 | 0 s | `logs/fermion_spectrum.log` |
| 8 | `fermion_exact.py` | 0 | 6 s | `logs/fermion_exact.log` |
| 9 | `fermion_scan.py` | 0 | 55 s | `logs/fermion_scan.log` |
| 10 | `algebra_census.py` | 0 | 1 s | `logs/algebra_census.log` |
| 11 | `verify_jacobi.py` | 0 | 0 s | `logs/verify_jacobi.log` |
| 12 | `make_figures.py` | 0 | 12 s | `logs/figures.log` |
| 13 | `make_guide_figures.py` | 0 | 22 s | `logs/guide_figures.log` |
| 14 | `verify_all.py` | 0 | 19 s | `logs/verify_all.log` |
| 15 | pdfLaTeX paper (1) | 0 | 0 s | `logs/latex_paper1.log` |
| 16 | pdfLaTeX paper (2) | 0 | 1 s | `logs/latex_paper2.log` |
| 17 | XeLaTeX guide-zh (1) | 0 | 1 s | `logs/latex_guide_zh1.log` |
| 18 | XeLaTeX guide-zh (2) | 0 | 1 s | `logs/latex_guide_zh2.log` |
| 19 | pdfLaTeX guide-en (1) | 0 | 0 s | `logs/latex_guide_en1.log` |
| 20 | pdfLaTeX guide-en (2) | 0 | 0 s | `logs/latex_guide_en2.log` |

## Headline assertions (from `logs/verify_all.log`)

```
Z(1) = sum I_n^8                6.622434805   PASS
<e^{i theta_f}>(1)              0.4488851448  PASS
sigma_oct(1)                    0.8009882263  PASS
sigma_inf(1)                    0.8065623460  PASS
W(A=1..4)                       0.4488851 / 0.2066158 / 0.1063616 / 0.0791731  PASS
sigma_F(F=4,6,8,12,20)          0.6828077 / 0.7792269 / 0.8009882 / 0.8063398 / 0.8065620  PASS
c(1); c(2.05)                   2.9326556; 5.7452222  PASS
beta_* = 4/pi                   1.273239545   PASS
Villain duality residual        1.8e-46       PASS
rank Gamma; det Gamma           2; 0          PASS
det(m+Gamma) at m=0.3, 1        0.097929; 13  PASS
<|det D0|^2>                    53.1808082    PASS
asymptotic prefactor            3.556548393   PASS
asym vs exact, F=8 / F=12       4.1e-05 / 6.5e-08  PASS
dirac_operator_checks verdict   PASS
triangulation_uniqueness verdict PASS
```

## Independent cross-checks

* **Wilson loops:** exact character expansion versus Metropolis on the 12 link
  angles (20 batches × 12 000 sweeps):  
  `0.44863 ± 0.00073` vs exact `0.448885`;  
  `0.20718 ± 0.00242` vs exact `0.206616`;  
  `0.08136 ± 0.00284` vs exact `0.079173`.
* **Villain duality:** `Z_V(beta) = C_F(beta) Z_V(beta_*^2/beta)` holds to a
  relative accuracy `1.8e-46` at `beta=1`.
* **Fermions:** the exact cycle-cover/Bessel integration gives
  `<det D0> = 4.257698`, `<|det D0|^2> = 53.180808`; the second is also asserted
  in `verify_all.py`.
* **Dirac algebra:** `D0 = Gamma o U` is anti-Hermitian for 200 random
  configurations; `det D0` is real for all of them and nonzero for all of them;
  `det D0 = 0` exactly at the flat connection; no diagonal grading exists.

## Output integrity

| file | SHA-256 (first 16 hex) | pages |
|---|---|---|
| `octahedron_NPB.pdf` | `d47c878accd0d8e4` | 28 |
| `guide_en.pdf` | `09fe1ceddcac29c6` | 10 |
| `cover_letter_NPB.pdf` | `33673cc88e447d17` | 2 |

## Known, non-blocking warnings

* pdfLaTeX reports a few `Overfull \hbox` warnings in the long data-availability
  paragraph and in two appendix lines.  Cosmetic only; no undefined references.
* `fermion_scan.py` prints `nan` in the `DeltaF` column for some beta points.
  Expected: the sector free-energy difference is not a positive observable once
  fermions are included (`f0 > 1`), as stated in the paper.
* Matplotlib may emit font-substitution messages when the figures are
  regenerated on a machine without the DejaVu serif family; the PDFs are
  unaffected.

## What changed in v3

1. **Fermion operator, unified.**
   - Explicit split between the massless operator `D0 = t Gamma o U` and the
     full operator `D(m) = m 1 + D0`; a new operator-dictionary table.
   - `D0` is anti-Hermitian; `det D0` is real but sign-indefinite; `det D(m)`
     is complex for `m>0`.  The positive weight is the doubled one in both
     cases, for different reasons.
   - All determinant/zero-mode formulas restated and checked:
     `det(m+D0)^* = det(m-D0)`, `det(m 1 + Gamma) = m^4(m^2+12)`,
     `ker Gamma` four-dimensional, `det D0 = 0` only at the flat connection,
     `det D0 != 0` generically (8.375 for a random configuration).
2. **Genuinely new theorems highlighted.**  A dedicated paragraph lists
   (T1) F-universality, (T2) the loop law, (T3) the asymptotic expansion with
   remainder bound, (T4) `beta_* = F/2pi`, (T5) triangulation uniqueness, and
   separates them from the known character expansion, area law and Villain
   solution.  Table 1 gives the item-by-item status.
3. **Finite-size error control.**  The remainder is now bounded explicitly:
   `|remainder|/x <= 2 r^F + O(r^{2F}) + O((I2/I0)^{F-1}/x)`, with numbers at
   `F=8` (bound 3.2e-3, measured 2.2e-3) and `F=12` (1.3e-4 vs 1.2e-4).  The
   expansion's range of validity is stated, and the exact/asymptotic/error table
   is retained.
4. **Emergence statements scoped.**  A "Scope" paragraph opens the algebra
   section; all claims are restricted to `u(4) + 4 + 4bar` with the stated
   bracket conventions; the five criteria are "necessary conditions within this
   class", and the convention dependence of the intermediate Jacobi locus is
   stated.
5. **Code and data at submission.**  Scripts, dependency list, logs and this
   report are included and archived at
   <https://github.com/csoftxyz/RIA_EISA/O8>.  The English guide replaces the
   Chinese guide as the primary illustrated document (`guide_en.pdf`); the
   Chinese version is kept as `guide_zh.pdf`.
