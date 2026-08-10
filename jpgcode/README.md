# Z3LM Paper Code

Reproducible code for the paper
*A One-Parameter Phase-Constrained Majorana Neutrino Texture with a
\(Z_3\) CP Seed* (Y. Zhang, W. Hu, W. Zhang).

## Contents
https://github.com/csoftxyz/RIA_EISA/tree/main/jpgcode
| File | Purpose |
|------|---------|
| `z3lm.py` | Core: texture matrix, Takagi decomposition, observable extraction |
| `z3lm_1param.py` | **Final model**: one-parameter texture (all five ratios fixed), chi2, predictions, 1-sigma ranges |
| `z3lm_3param.py` | Three-parameter texture (ratios 1/2, 9/11, 45/22, 1/3 fixed) prediction table |
| `nufit_parser.py` | Parser/interpolators for the official NuFIT 5.2 release tables |
| `fit_official.py` | Re-fit against official tables (6/7/9-param) + discrete phase scan |
| `hi_precision.py` | High-precision refit to test the structural ratio hypotheses |
| `recon_scan.py` | Numerical reconnaissance that identified the ratios |
| `eval_plb.py` | Evaluation of the alternative three-parameter PLB texture |
| `d27_check.py` | Numerical verification of Appendix C (Delta(27)xCP) claims |
| `d27_cg*.py`, `d27_seesaw.py` | Clebsch-Gordan / seesaw analyses behind the No-Go result |
| `test_paper.py` | End-to-end test of every quantitative claim in the paper |
| `v52-NO` | Official NuFIT 5.2 chi2 table (NO, with SK), see below |

## Requirements

- Python >= 3.9, numpy, scipy
- The official NuFIT 5.2 table:
  - Download: `https://www.nu-fit.org/sites/default/files/v52.release-SKyes-NO.txt.xz`
  - Decompress: `xz -dk v52.release-SKyes-NO.txt.xz`
  - Place as `v52-NO` in this directory (or edit `DEFAULT_TABLE` in
    `nufit_parser.py`)

## Usage

```bash
python3 z3lm_1param.py   # final one-parameter model (fast, ~30 s)
python3 z3lm_3param.py   # three-parameter texture table (~10-30 min)
python3 z3lm.py          # best-fit matrix, observables, Takagi residual
python3 nufit_parser.py  # official best-fit values + chi2 at texture point
python3 fit_official.py  # full re-fit + phase scan (takes ~10-30 min)
python3 hi_precision.py  # ratio-hypothesis precision test
python3 d27_check.py     # Delta(27) group-theory verification
python3 test_paper.py    # end-to-end verification (all numbers in paper)
```

## Key results (reproduced by the code)

### Final one-parameter model (`z3lm_1param.py`)

Fixed ratios (phenomenological inputs): |d1|/a12 = 1/2,
d2/d3 = 9/11, a23/a12 = 45/22, a13/a12 = 1/3, a12/d3 = 6/19.
Phases locked: M12 in R+, M13 = (a12/3) omega, M23 in R-.

| Quantity | Value |
|----------|-------|
| Free parameter | m0 = 0.0316 eV (d3 gauge fixed) |
| chi2_min (official NuFIT 5.2, dof = 5) | 1.41 (p ~ 0.92) |
| sin^2 t12, sin^2 t23, sin^2 t13 | 0.3029, 0.4497, 0.02228 |
| delta_CP | -83.4 deg (276.6 deg; standard PDG/Jarlskog) |
| dm21, dm31 | 7.45e-5, 2.505e-3 eV^2 |
| J_PMNS | -0.0331 |
| m1, m2, m3 | 0.0076, 0.0115, 0.0506 eV |
| Sum m | 0.0698 eV (1 sigma: [0.0694, 0.0701]) |
| m_bb | 0.0050 eV |
| Ordering | NO |

### Reference versions (for context)

- 3-parameter texture (a12/d3 free): chi2 = 1.34 (dof = 4).
- 6-parameter texture (all ratios free, phases locked):
  m0=0.02243, d1=-0.2226, d2=1.1539, d3=1.4095, a12=0.4456,
  a23=0.9119; chi2 = 1.33 (dof = 0, no goodness-of-fit statement).

## Notes on the fit objective

The fit minimizes the sum of the six official one-dimensional
Delta-chi^2 projections interpolated at the texture point, the
standard model-evaluation procedure recommended for the NuFIT release
tables (`NuFIT52.chi2_1d_sum`).  A cross-check using the 3D+2D
combination (`chi2_block`) gives consistent results.  The
differential-evolution fit is multi-seeded; the seed set used in
`test_paper.py` reaches the global minimum.

## Delta(27) appendix facts (verified in `d27_check.py`,
`d27_definitive_check.py`)

- Delta(27) has 11 conjugacy classes -> 11 irreps: nine 1-dim
  singlets + 3 + 3bar (9*1^2 + 2*3^2 = 27).  There is NO 8-dim
  irrep.
- Monomial representation A=diag(1,w,w^2), B=cyclic shift, with
  B A B^-1 = w A and (AB)^3 = 1.
- Complex conjugation is an automorphism (A->A^2, B->B): generalized
  CP is consistent with X=I (and 162 monomial solutions, all with
  X X* = 1).
- 3 x 3bar = direct sum of all nine singlets (1_1 + ... + 1_9);
  3 x 3 = 3bar + 3bar + 3bar.
- Vacuum (1,w,w^2): projective stabilizer of size 9 (Z3 modulo the
  center), orbit of 3 branches (1,w,w^2),(1,w^2,w),(1,1,1).

## No-Go result (App. C, `d27_affine_nogo_v2.py`)

The ratio a13/a12 = 1/3 cannot be derived from Delta(27)
representation theory: in the single-flavon B-preserving realization
the Clebsch-Gordan coefficients enforce |M12| = |M13| = |M23|
(equal moduli, ratio = 1), and the ratio remains a free combination
of Wilson coefficients and VEVs otherwise.  The No-Go persists under
affine extensions of the mass matrices and under VEV translations.
1/3 is therefore an irreducible phenomenological input of the
texture.

Author: Yuxuan Zhang, 2026-08-09 (final version).

## v1.2 change note (2026-08-09)

The Dirac phase is extracted from REPHASING-INVARIANT quantities
(sin delta = J/D, cos delta from |U_mu1|^2), guaranteeing
J = D sin(delta) exactly.  Earlier versions fixed U_t3 real and read
delta = -arg(U_e3), which is NOT the PDG convention and gave
delta = 237.6 deg inconsistent with J = -0.033 (J/D = -0.993 requires
delta = 276.6 deg).  All numbers in this version use the standard
convention: delta = -83.4 deg (276.6 deg), chi2(1-param) = 1.41
(dof = 5, p ~ 0.92), chi2(3-param) = 1.34 (dof = 4).  The 1.7-sigma
pull of delta versus the NuFIT best fit (-130 deg) is a genuine,
falsifiable prediction of the texture.

## Reproducing Table 6 (phase scan)

Table 6 uses the CONSTRAINED phase scan: six real moduli fitted with
the rephasing-invariant delta extraction, restricted to the Z3LM
phase branch d2,d3 > 0 (the unconstrained fit drifts to basins with
d2,d3<0 that lie outside the texture branch).  Reproduce with:

    python3 - << 'PY'
    import numpy as np
    from scipy.optimize import differential_evolution
    from nufit_parser import NuFIT52
    from z3lm import obs_vec, OMEGA
    nf = NuFIT52()
    def M_phase(phi12, phi13, phi23, ratio13=1/3):
        e12, e13, e23 = (np.exp(1j*np.radians(p)) for p in (phi12, phi13, phi23))
        def build(q):
            m0, d1, d2, d3, a12, a23 = q
            return m0*np.array([[d1, a12*e12, ratio13*a12*e13],
                                [a12*e12, d2, a23*e23],
                                [ratio13*a12*e13, a23*e23, d3]], dtype=complex)
        return build
    def fit_c(build, seeds=6):
        B6 = [(0.005,0.10),(-1.5,1.5),(0.01,1.5),(0.01,1.5),(0.1,1.0),(0.1,2.0)]
        def fo(q):
            if q[2] <= 0 or q[3] <= 0: return 1e6
            try: return nf.chi2_1d_sum(obs_vec(build(q)))
            except Exception: return 1e6
        best = None
        for sd in range(seeds):
            r = differential_evolution(fo, B6, seed=sd, maxiter=600, tol=1e-9, polish=True)
            if best is None or r.fun < best.fun: best = r
        return best
    for pat in [(0,90,180),(0,120,180),(240,0,180),(0,120,90),(240,240,180)]:
        r = fit_c(M_phase(*pat))
        print(pat, round(r.fun, 2))
    PY

Expected: (0,90,180) 8.84, (0,120,180) 1.32, (240,0,180) 0.30,
(0,120,90) 162.3, (240,240,180) 58.9.
