# Radial-flux estimators of Coulomb couplings on discrete substrates

Code and data for the manuscript submitted to *Computer Physics Communications*
(`paper/main.pdf`, 15 pp.).

## Reproduce everything

```bash
./code/run_all.sh      # analysis -> verification -> figures; exits 0 only if all checks pass
```

Writes `logs/cpc_analysis.json`, `logs/cpc_analysis.log`, and the three figure
PDFs in `figures/`. `code/verify_claims.py` asserts every number printed in the
manuscript against the JSON log.

## Contents

| Path | What it is |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | manuscript, elsarticle, CPC format |
| `figures/fig1_bias.pdf` | estimator bias vs mesh ratio; shell-ratio vs trial exponent |
| `figures/fig2_lattice.pdf` | 96³ lattice: endpoint vs exact-radius estimator |
| `figures/fig3_refinement.pdf` | apparent coupling vs refinement level |
| `code/cpc_analysis.py` | analysis driver (sections A–G of the log) |
| `code/verify_claims.py` | asserts every headline number; non-zero exit on drift |
| `code/make_figures_cpc.py` | regenerates the figures from the JSON log |
| `code/run_all.sh` | the three steps above |
| `logs/cpc_analysis.log` | human-readable log of the run |
| `logs/cpc_analysis.json` | machine-readable results |
| `guide/guide_cpc.pdf` | 7-page illustrated companion for the editor and referees |
| `guide/gc*.pdf` | its seven figures (geometry, bias, lattice, stability, pipeline, decision tree, dashboard) |
| `notes/referee_response.md` | point-by-point response to the reports that triggered this revision |

## Requirements and runtime

Python 3, NumPy; matplotlib for the figures only. No data files: the 96³
lattice Green function is computed by FFT at run time. Measured on one core of
an x86-64 container (Python 3.12, NumPy 1.26): 0.4 s analysis, 0.9 s figures.

## Headline numbers

* Exact radius: ρ^{d−1} = (d−2)(r_{n+1}−r_n)/(r_n^{2−d}−r_{n+1}^{2−d}); interior by the
  mean-value theorem, unique because x ↦ x^{d−1} is injective. Only in d = 3 is
  ρ² = r_n r_{n+1}.
* The naive generalisation omits (d−2) and leaves the shell for q below
  q* = 1 (d=3), 1.6180 (d=4), 1.8393 (d=5), 1.9276 (d=6).
* Bias: q (left), 1/q (right), −(q−1)²/(1+q)² (arithmetic), 0 (geometric).
* 96³: Q_left = [R/(R+1)]·Q_exact exactly, per shell; R = 8–15 averages
  0.900 ± 0.021 (left) and 0.982 ± 0.029 (exact radius).
* Deep mesh: q ∈ [1.0001, 5], up to 2×10⁵ shells; worst residual 4.4×10⁻¹⁶ occurs at
  q = 2 (39 shells), the 2×10⁵-shell case being at round-off.
