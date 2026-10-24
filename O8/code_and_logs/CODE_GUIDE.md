# CODE GUIDE — octahedron exact lattice gauge theory

This package contains every script needed to reproduce the numbers, tables and
figures of the manuscript

> *Exact lattice gauge theory on the octahedron: a rigorous laboratory for
> confinement, topological rigidity, and the limits of emergent model building*

and of the Chinese illustrated guide `guide_zh.pdf`.

Code archive: <https://github.com/csoftxyz/RIA_EISA/O8>

## 1. Environment

| item | version used |
|---|---|
| Python | 3.12 |
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| mpmath | 1.2.1 |
| matplotlib | 3.6.3 |
| LaTeX | pdfTeX + `elsarticle`; XeLaTeX + `ctex` (guide) |

`requirements.txt` lists the Python dependencies.  Closed-form sums are
evaluated at 40–60 decimal digits with `mpmath`; double precision is used only
for plotting and for the Monte Carlo cross-check.

## 2. Quick start

```bash
cd NPB_octahedron
python3 code/verify_all.py       # 27 assertions; the gatekeeper
bash run_all_tests.sh            # every script + both documents, logs in logs/
```

`run_all_tests.sh` writes one log per step and prints a PASS/FAIL summary.

## 3. Scripts

### `exact_gauge.py`
Exact partition function `Z_F(beta)=sum_n I_n(beta)^F`, single-face expectation,
finite-volume tension `sigma_F`, Wilson loops for `A=1..4`, sector weights and
the Z3 character, specific heat, plaquette-plaquette correlator.
Outputs `logs/exact_gauge.json`.  Runtime ~20 s.
Paper: Table 2, Table 3 (loops), Fig. 1, Sec. 3.3.

### `finite_size_asymptotic.py`
Derives and verifies the large-`F` expansion
`<W_A> = r^A {1 + r^{F-2A}[1 + (I2 r/I1)^A - 2 r^{2A}] + ...}`,
the analytic prefactor `3.5565` of the tension deficit, the ratio `R_F`, and the
relative error table.  Outputs `logs/finite_size_asymptotic.json`.
Runtime ~30 s.  Paper: Eqs. (12)–(13), Table 4.

### `wilson_loops.py`
Exact character-expansion loops versus a direct Metropolis Monte Carlo on the
twelve link angles (no gauge fixing).  Batch-means errors.  Outputs
`logs/wilson_loops.json`.  Runtime ~40 s.  Paper: Sec. 3.1.

### `villain_general.py`
Villain partition function, exact Jacobi duality, self-dual coupling
`beta_* = F/(2 pi)`, the three Wilson–Villain dictionaries, and the Wilson
coupling that ratio-matches onto `4/pi`.  Outputs `logs/villain_general.json`.
Runtime ~3 s.  Paper: Sec. 4.

### `triangulation_uniqueness.py`
Exhaustive check that the octahedron is the unique six-vertex triangulation of
`S^2`: enumerates all 4-regular graphs on six vertices (15 labelled, 1
isomorphism class), their eight triangles, edge multiplicities, vertex links and
Euler characteristic.  Outputs `logs/triangulation_uniqueness.json`.
Runtime <1 s.  Paper: Lemma 1.

### `dirac_operator_checks.py`
Checks `D0 = Gamma o U` anti-Hermiticity, reality and generic non-vanishing of
`det D0`, `det(m+D0)^* = det(m-D0)`, complexity of `det D(m)` for `m>0`,
absence of a diagonal grading, bipartiteness, flat-connection spectrum,
`det(m+Gamma)=m^4(m^2+12)`, and the Dirac gap for random configurations.  Outputs `logs/dirac_operator_checks.json`.  Runtime <1 s.
Paper: Table 5 (operator dictionary), Lemmas 2–3, Results 5–6.

### `fermion_exact.py`
Exact gauge integration for the minimal fermion model at `m=0`, `t=1`, using
permutation cycle covers and Bessel character series under the Gauss-law
constraints.  Prints `Z_g`, `Z_ferm`, `<|det|^2>` and sector fractions.
Runtime ~6 s.  Paper: Sec. 5.3.

### `fermion_scan.py`
Beta scan of the same exact computation.  Column `<det^2>` denotes
`<|det D|^2>`.  Outputs a table used for the smoothness/derivative statements.
Runtime ~50 s.  Paper: Eq. (21) and the slope values.

### `fermion_spectrum.py`
Builds `Gamma = epsilon x J2`, prints eigenvalues, rank, kernel basis, the
determinant formula and its check at several masses.  Outputs
`logs/fermion_spectrum.json`.  Runtime <1 s.  Paper: Result 6.

### `algebra_census.py`
Numerical null-space count of `u(4)`-invariant bilinears and trilinears for
`g1=4`, `g2=4bar`, and of the bracket targets `Lambda^2 g1 -> g2` and
`Lambda^2 g2 -> g1`.  Runtime <1 s.  Paper: Result 7.

### `verify_jacobi.py`
Super-Jacobi constraint on the mixed bracket
`[F,zeta] = p F zeta^T + q (zeta^T F) 1`: the complex-linear dual gives
`q = -p`; the `sl(4)` (traceless) requirement gives `p = q = 0`.
Runtime ~1 s.  Paper: Result 8.

### `verify_all.py`
Runs all headline numbers as assertions (27 checks), including subprocess
verdicts of the uniqueness and Dirac scripts.  Exit code 0 iff all pass.
Runtime ~20 s.

### `make_figures.py`, `make_guide_figures.py`
Paper figures (`figures/`, 3 two-panel figures) and guide figures
(`guide_figures/`, 7 figures), each as vector PDF and 300 dpi PNG.
Runtime ~10–15 s each.

## 4. Conventions and caveats

* The lattice measure is `prod_l dtheta_l / (2 pi)`.
* `r = I_1(beta)/I_0(beta) = exp(-sigma_infinity)`.
* `A` always denotes the number of faces enclosed by the loop, not its length.
* In `fermion_scan.py` the column printed as `<det^2>` is `<|det D|^2>`.
* The earlier `Z3_algebra_audit.py` real-form implementation is superseded by
  `verify_jacobi.py`; only the latter is used for the paper's statement.
* Monte Carlo is used only in `wilson_loops.py`, as an independent check of the
  exact loop formula; all other numbers are deterministic.

## 5. Reproducibility checklist

| claim | script | log |
|---|---|---|
| `Z(1)=6.6224348046478` | `exact_gauge.py` | `logs/exact_gauge.log` |
| `<e^{i theta_f}>=0.4488851447705` | `exact_gauge.py` | idem |
| `sigma_oct=0.8009882263` | `exact_gauge.py` | idem |
| loop values `A=1..4` | `exact_gauge.py` | idem |
| asymptotic prefactor `3.5565` | `finite_size_asymptotic.py` | `logs/finite_size.log` |
| `beta_*=F/2 pi` | `villain_general.py` | `logs/villain_general.log` |
| uniqueness of the 6-vertex triangulation | `triangulation_uniqueness.py` | `logs/triangulation.log` |
| anti-Hermiticity, no grading, zero modes | `dirac_operator_checks.py` | `logs/dirac_checks.log` |
| `<det D>=4.257698`, `<|det D|^2>=53.180808` | `fermion_scan.py` | `logs/fermion_scan.log` |
| census and Jacobi locus | `algebra_census.py`, `verify_jacobi.py` | `logs/*.log` |

## 6. Troubleshooting

* `mpmath` slow: the 40–60 digit sums in `exact_gauge.py` and
  `finite_size_asymptotic.py` take tens of seconds; that is expected.
* `elsarticle.cls` missing: install `texlive-publishers`.
* Chinese guide: compile with `xelatex` twice; `ctex` and Noto CJK fonts are
  required.
