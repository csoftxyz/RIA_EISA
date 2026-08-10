# 1- and 3-Parameter Texture Scripts (final paper)

Scripts for the structural-relations analysis (Sec. 4.3-4.4 of the
final version):

- `z3lm_1param.py`  **final model**: all five ratios fixed
  (1/2, 9/11, 45/22, 1/3, 6/19), one free parameter m0,
  chi2 = 1.41 (dof = 5), 1-sigma ranges on sum m / m_bb / delta
  (delta = -83.4 deg, standard PDG/Jarlskog convention)
- `recon_scan.py`    ratio scan: texture parameters vs lattice invariants
- `source_hunt.py`   combined 3/4-param fits + source matching
- `hi_precision.py`  high-precision refit (8 sig digits) + ratio tests
- `z3lm_3param.py`   full 3-parameter texture prediction table
- `eval_plb.py`      strict NuFIT 5.2 evaluation of the old PLB paper

Key results:

- 3-parameter texture (d1=-a12/2, d2=(9/11)d3, a13=a12/3,
  a23=(45/22)a12 imposed): free params (m0,a12,d3),
  chi2 = 1.34 (dof = 4), predictions unchanged.
- One-parameter texture (additionally a12/d3 = 6/19): only m0 free,
  chi2 = 1.41 (dof = 5). The alternative a12/d3 = 1/sqrt(10) gives
  chi2 = 1.38; the two are indistinguishable by data
  (Delta chi2 = 0.03) and 6/19 is an explicit structural selection.

Honest caveat: 1/2, 9/11, 45/22 are data-verified structural
relations; 1/3 and 6/19 are phenomenological/structural choices. None
is derived from the flavor symmetry (see App. C No-Go result).
