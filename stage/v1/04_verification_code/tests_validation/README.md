# Tests & Validation Suite

This folder packages every **test, validation, audit and control** script used to
check the paper's claims. The production truth-source (`verify_prd.py`,
`strict_obs.py`, `exact_core.py`, `grade_a_fast.py`, `viability_fit.py`,
`charged_grade_a.py`, `rg_lowenergy_match.py`) lives one level up; the scripts
here are the *independent checks* that either confirm or falsify those claims.

Each script is self-contained and runs with the locked environment
(Python 3.6.6, numpy 1.19.5, scipy 1.5.4, matplotlib 3.3.4). Run any of them with:

```
python <script>.py
```

A script that exits non-zero or prints a FAILED control is a genuine regression
signal — none of these are hard-coded to pass.

**Prerequisite for summary views:** `exclusion_taxonomy.py` run standalone only
summarises an existing `06_results/results.json`; if that record is absent it
prints a notice and exits 0 (run `verify_prd.py` first). The master verifier
itself needs no such file — it passes its own census/RG into `compute()`.

## Categories

### A. Independent re-derivations (do not import the paper's logic)
| Script | What it checks |
|---|---|
| `audit_independent.py` | Re-derives every census rank by separate exact Gaussian elimination; rebuilds mass matrices from scratch; tests seesaw surjectivity with a rank-deficient input. |
| `audit5_tests.py` | Counting-trap, RG-sensitivity and type-II diagnostics on the correct parameterisation. |
| `dichotomy_check.py` | The viable/excluded dichotomy and the transpose criterion. |
| `constrained_seesaw_test.py` | The correlated SO(10) seesaw map and its pencil structure. |
| `census_viability.py` | Spectrum-splitting viability of all 18 contents. |
| `so10_numerics.py` | SO(10) Clebsch numerics and the bidoublet counting. |

### B. Attribution controls (the "is it really the pencil?" tests)
| Script | What it checks |
|---|---|
| `pencil_attribution_control.py` | **Decisive control:** the weaker-hierarchy preference appears in 10+126 (which has NO pencil identity) and in pure type-I (vL=0). Proves the preference is a single-S M_R structure effect, not a pencil consequence. Also counts 10+126 eigenvalues (3, no repeated root). |
| `pencil_prior_sensitivity.py` | Re-runs the preference under 6 priors + 3 breaking strengths; direction robust 6/6. |
| `singleS_spectral_theorem.py` | 120+126 single-S Mnu has 2 distinct eigenvalues (A^2 eigs {0,l,l}); broken-S has 3. Records this as a 120+126-only observation, NOT the preference mechanism. |
| `pencil_analytic_origin.py` | Verifies every link of the pencil algebraic derivation chain to 1e-13. |
| `pencil_mining.py` | Pencil scope across contents, eps^2 breaking scale, PMNS statistics. |

### C. Grade-(a) certificates & obstruction evidence
| Script | What it checks |
|---|---|
| `rational_route_exhaustive.py` | Exhaustive screen of 40 trace words + fraction-field argument: no rational invariant reaches rank 22. |
| `exclusion_taxonomy.py` | 9/11 excluded contents are CKM-locked; exclusion RG-robust. |
| `seven_content_scoreboard.py` | Declaration-cost scoreboard for all 7 viable contents + Pfaffian generalisation test (SU(5) 45 fails). |
| `declaration_scoreboard.py` | The original 2-content declaration scoreboard. |

### D. Numerical diagnostics
| Script | What it checks |
|---|---|
| `seed42_diagnosis.py` | The seed-42 rank-21 point is a Jarlskog-near-zero coordinate degeneracy, not a relation. |
| `fiber_proof.py` | Tangent-space demonstration: Sigma varies along the observable fibre. |
| `rg_sensitivity.py` | Indicative-coupling RG example (the wall is not by itself a low-energy exclusion). |
| `measure_dims.py` | **DEPRECATED** observable construction (kept for provenance; do NOT use for dimensions). |
| `explicit_relation.py` | The explicit E1/E2 matrix-level constraints. |

### E. Charged-sector fits
| Script | What it checks |
|---|---|
| `item1_charged_fit.py` | Levenberg-Marquardt fit of the six charged ratios. |
| `item1_scale_free.py` | Scale-invariance of the fit. |
| `item23_seesaw_and_ten.py` | Seesaw + the 10-containing content fit. |

## How these map to the audit history

The controls in category B were written specifically to catch **misattribution**
errors found during self-audit (rounds 1 and 3): a statistical effect was twice
wrongly blamed on the pencil identity / its repeated root, and both times the
10+126 and pure-type-I controls falsified the attribution. See the consolidated
audit report (`../../07_audit_report/`) for the full story.
