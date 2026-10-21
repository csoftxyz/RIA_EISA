

https://github.com/csoftxyz/RIA_EISA/tree/main/Z3LM


# Z3LM — code and figures

This directory contains the complete analysis code and all figures for the
manuscript

> **\(Z_3\)-Locked Majorana Textures: Phases, Moduli, and the Limits of
> Predictivity**
> Yuxuan Zhang, Weitong Hu, Wei Zhang
> (originally NPB-D-26-01786; transfer to *Chinese Journal of Physics*)

The paper studies a Majorana neutrino mass matrix whose complex phases are
locked to the cubic root of unity \(\omega=e^{2\pi i/3}\), while its five real
modulus ratios are parameters determined by data. It reports (i) a fit to the
official NuFIT 5.2 release tables, (ii) an exhaustive *look-elsewhere* scan
showing that the moduli are not predicted, and (iii) two no-go theorems stating
that the ratios cannot be derived from the CG structure of a minimal discrete
flavour group.

Everything in the paper — every number, table and figure — is produced by the
scripts below.

---

## Quick start

```bash
# 1. dependencies (Python >= 3.9)
pip install numpy scipy matplotlib

# 2. NuFIT 5.2 release table (normal ordering, with Super-K data)
curl -O https://www.nu-fit.org/sites/default/files/v52.release-SKyes-NO.txt.xz
xz -dk v52.release-SKyes-NO.txt.xz
#    then set DEFAULT_TABLE in nufit_parser.py to the decompressed file,
#    or export it / edit the one line at the top of that module.

# 3. reproduce the headline numbers
python3 verify_all.py            # 20-point check (~10 min, includes scans)
python3 z3lm_1param.py           # reference point
python3 make_figures.py          # regenerate all paper figures
```

The NuFIT table is ~46 MB and is **not** included in this repository.

---

## Layout

```
Z3LM/
├── Z3LM_code/          all Python scripts + machine-readable outputs
│   ├── z3lm.py                     core: texture, Takagi, observables
│   ├── nufit_parser.py             NuFIT 5.2 tables + joint statistic
│   ├── z3lm_1param.py              reference-point fit
│   ├── z3lm_3param.py              three-parameter version
│   ├── fit_official.py             6/7/9-parameter fits + phase scan
│   ├── rational_scan2.py           look-elsewhere scan  (paper App. D)
│   ├── smooth_rational_scan.py     |G|-smooth denominator scan
│   ├── integer_scan.py             small-integer enumeration
│   ├── d54_invariants.py           Δ(27)/Δ(54) invariant tensors (App. E)
│   ├── d27_cg*.py, d27_check.py    Δ(27) CG machinery and No-Go checks
│   ├── verify_all.py               20-point verification
│   ├── make_figures.py             paper figures
│   ├── make_graphical_abstract.py  graphical abstract
│   ├── phase_scan_block.json       phase-scan results (block statistic)
│   ├── rational_scan_refined.txt   447 alternatives, ranked
│   ├── verification_report.md      latest verification output
│   └── verification_values.json    same, machine-readable
└── Z3LM_figures/       fig1 … fig7 + graphical abstract (PNG and PDF)
```

---

## Core modules

### `z3lm.py` — the model

Defines the texture and everything downstream of it.

| Function | Purpose |
|---|---|
| `M6(q)` | builds \(M_\nu\) from `q = (m0, d1, d2, d3, a12, a23)`; phases hard-wired to \(\omega\) |
| `takagi_all(M)` | Autonne–Takagi decomposition \(M=U\,\mathrm{diag}(m)\,U^T\) |
| `takagi_residual(M)` | numerical diagnostic, expected \(<10^{-15}\) |
| `obs_vec(M)` | returns `(sin²θ12, sin²θ13, sin²θ23, δ_CP, Δm²21, Δm²31)` |
| `obs_full(M)` | same plus \(J\), \(m_i\), \(\sum m_\nu\), \(m_{\beta\beta}\), ordering |
| `BEST_FIT` | the six-parameter reference point of the paper |

Two conventions matter and are easy to get wrong:

* **Takagi, not `U†MU`.** A Majorana matrix is complex *symmetric*; the PMNS
  matrix is the Takagi factor, not the eigenvector matrix of \(MM^\dagger\).
* **δ_CP is extracted from rephasing invariants**, `sin δ = J/D` and `cos δ`
  from \(|U_{\mu1}|^2\), not from `arg U_e3`. The latter is convention dependent
  and was the source of an error in early versions of the analysis.

### `nufit_parser.py` — data and statistic

Parses the 23 sections of the official release (1 three-dimensional, 15
two-dimensional, 6 one-dimensional projections) and provides:

* `chi2_1d(name, x)`, `proj2d`, `proj3d` — interpolators;
* `chi2_1d_sum(y)` — **legacy** sum of the six 1D projections. Not a
  \(\chi^2\) distribution; kept only for comparison with the original
  submission;
* `chi2_block(y)` — the joint 3D+2D statistic used throughout the paper.

### `z3lm_1param.py`, `z3lm_3param.py`

Reference-point and three-parameter fits. They scan the single scale \(m_0\)
(or \(m_0\) and \(a_{12}/d_3\)) and print observables, masses, \(m_{\beta\beta}\)
and the \(1\sigma\) interval on the scale.

> **Statistics caveat.** These two scripts still report `chi2_1d_sum`, i.e. the
> legacy number \(1.41\). The paper quotes the joint statistic
> \(\chi^2_{\rm block}=1.44\), produced by `verify_all.py` and
> `refine_point.py`. Both values correspond to the same best-fit point; only
> the statistic differs.

---

## Analysis scripts

### Fits, phases, controls

| Script | What it does |
|---|---|
| `fit_official.py` | six/seven/nine-parameter fits; discrete phase-pattern scan; documents that the free fit drifts to \(d_2,d_3<0\) |
| `perturb_analysis.py` | looks for perturbative relations among the ratios |
| `recon_scan.py` | tries to fix the six parameters with 44-vector lattice invariants |
| `source_hunt.py` | combined structural test of candidate rational sources |
| `variational.py` | asks whether a flavon potential can produce the ratios |
| `wkb_lattice.py` | path-weight interpretation on the 44-vector lattice |
| `hi_precision.py` | high-precision refit used to test ratio hypotheses |
| `eval_plb.py` | evaluates the companion PLB predictions against NuFIT 5.2 |

### Look-elsewhere and naturalness (paper Appendix D)

| Script | Output |
|---|---|
| `rational_scan2.py` | the 447 alternatives, denominator \(q\le22\), \(\pm3\%\) window; produces `rational_scan_refined.txt` |
| `rational_alternative_scan.py` | earlier, Gaussian-metric version of the same count |
| `ratio_uniqueness.py` | 1σ windows on each ratio, and how many rationals fall inside |
| `smooth_rational_scan.py` | restricts denominators to \(|G|\)-smooth values (primes 2, 3) |
| `srs_run.py` | working copy of the same scan with different denominator sets |
| `integer_scan.py` | \(4.69\times10^6\) small-integer textures; none reaches \(\chi^2<6\) |

### Group theory / no-go (paper Appendix E)

| Script | Output |
|---|---|
| `d27_cg.py` … `d27_cg7.py` | successive versions of the Δ(27) CG analysis; `d27_cg7.py` states the analytic No-Go |
| `d27_check.py` | numerical check of every group-theoretic claim in App. C |
| `d27_seesaw.py` | whether a type-I seesaw with Δ(27) can give \(a_{13}/a_{12}=1/3\) |
| `d54_invariants.py` | invariant-tensor dimensions for Δ(27) and Δ(54); the one-flavon theorem |

`d54_invariants.py` is the script behind the statement “a single triplet
flavon gives a diagonal symmetric mass matrix”: it builds the group as 3×3
monomial matrices and solves for invariant tensors \(T_{ijk}\). Result:
\(\dim = 3\) for Δ(27) (two antisymmetric, one symmetric
\(T_{ijk}\propto\delta_{ij}\delta_{ik}\)) and \(\dim = 2\) for Δ(54), both
antisymmetric.

### Verification

| Script | Output |
|---|---|
| `verify_all.py` | the 20-point check quoted in the supplementary notes; also reads `phase_scan_block.json` |
| `test_paper.py` | earlier end-to-end check (1D statistic era) |
| `refine_point.py` | locates the block-statistic optimum, \(m_0=0.031592\) eV |
| `phase_scan_refine.py`, `phase_scan2.py`, `phase_scan3.py`, `phase_scan4.py` | iterations of the discrete phase scan; `phase_scan4.py` produced the final values |

### Figures

| Script | Output |
|---|---|
| `make_figures.py` | `fig1_texture`, `fig2_pmns`, `fig3_masses`, `fig4_profile`, `fig5_scan`, `fig6_phases`, `fig7_nogo` |
| `make_graphical_abstract.py` | `graphical_abstract.png` (1500×600, 300 dpi) and `.pdf` |

---

## Results files

| File | Contents |
|---|---|
| `phase_scan_block.json` | joint statistic for the five phase patterns: 7.98, 1.32, 0.27, 130.01, 94.29 |
| `rational_scan_refined.txt` | the 447 alternatives with their \(\chi^2_{\rm block}\), ranked |
| `verification_report.md` | the latest 20-point verification, human-readable |
| `verification_values.json` | the same in machine-readable form |

---

## Mapping to the paper

| Paper item | Script / file |
|---|---|
| Table 2 (reference point) | `z3lm_1param.py`, `refine_point.py` |
| Table 3 (pulls) | `verify_all.py` |
| Table 4 (model comparison) | `verify_all.py`, `fit_official.py` |
| Table 5 (447 alternatives) | `rational_scan2.py` → `rational_scan_refined.txt` |
| Table 6 (stability in \(a_{12}/d_3\)) | `z3lm_3param.py` |
| Table 7 (phase scan) | `phase_scan4.py` → `phase_scan_block.json` |
| Appendix D (look-elsewhere, integer scan) | `rational_scan2.py`, `smooth_rational_scan.py`, `integer_scan.py` |
| Appendix E (two theorems) | `d54_invariants.py`, `d27_cg7.py`, `d27_check.py` |
| fig1–fig7 | `make_figures.py` |
| Graphical abstract | `make_graphical_abstract.py` |

Headline numbers, for reference:

```
chi2_block (reference)         1.44      m0 = 0.031592 eV
sin^2 theta_12 / 23 / 13       0.3029 / 0.4497 / 0.02228
delta_CP                       -83.4 deg      J = -0.0331
sum m_nu                       69.6 meV       m_bb = 4.99 meV
rational alternatives          447   (15 below chi2 = 2; best 1.35)
small-integer textures         4.69e6, none below chi2 = 6
phase scan (block)             7.98 / 1.32 / 0.27 / 130.0 / 94.3
```

---

## Reproducibility

* Every script is standalone and reads only the NuFIT table plus its own
  module imports; there is no hidden state.
* Randomised optimisers (`differential_evolution`) are seeded. Where a scan
  depends on a global search, the result is stored as a JSON/txt file so that
  the paper can be checked without re-running the optimiser.
* Typical runtimes on a laptop: `verify_all.py` ≈ 10 min, `rational_scan2.py`
  ≈ 2 min, `integer_scan.py` ≈ 3 min, everything else < 1 min.
* `verify_all.py` is the single entry point that reproduces every number
  quoted in the manuscript.

### Known legacy outputs

Three scripts still print numbers from the original submission, which used the
sum of 1D projections:

* `z3lm_1param.py` prints `chi2_min = 1.41` (the paper uses 1.44, block);
* `fit_official.py` prints the phase-scan table 8.84 / 1.32 / 0.30 / 162.3 /
  58.9 (the paper uses 7.98 / 1.32 / 0.27 / 130.0 / 94.3);
* `test_paper.py` checks the older values.

These are kept for the audit trail. The current values come from
`verify_all.py` and `phase_scan_block.json`.

---

## Requirements

```
python >= 3.9
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4      (figures only)
```

No other dependencies; no compiled extensions.

---

## Citation

If you use this code, please cite the manuscript:

```
Y. Zhang, W. Hu, W. Zhang,
"Z3-Locked Majorana Textures: Phases, Moduli, and the Limits of Predictivity",
submitted to Chinese Journal of Physics (2026).
```

## License

Released under the repository license (see `../LICENSE`).
