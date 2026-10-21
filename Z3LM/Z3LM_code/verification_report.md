# Verification report — Z3LM manuscript

| quantity | paper | recomputed | tol | status |
|---|---|---|---|---|
| Takagi residual max|U^T M U - diag(m)| | 0.0 | 4.88503e-16 | 1e-12 | PASS |
| unitarity max|U^dag U - 1| | 0.0 | 4.44089e-16 | 1e-12 | PASS |
| sin^2 theta12 | 0.3029 | 0.30292 | 0.0005 | PASS |
| sin^2 theta23 | 0.4497 | 0.449736 | 0.0005 | PASS |
| sin^2 theta13 | 0.02228 | 0.0222789 | 5e-05 | PASS |
| delta_CP [deg] | -83.4deg | -83.3997 | 0.2 | PASS |
| dm21 [1e-5 eV^2] | 7.45 | 7.43066 | 0.05 | PASS |
| dm31 [1e-3 eV^2] | 2.497 | 2.49744 | 0.005 | PASS |
| J_PMNS | -0.0331 | -0.0331393 | 0.0005 | PASS |
| sum m_nu [eV] | 0.0698 | 0.0696495 | 0.0005 | PASS |
| m_beta_beta [eV] | 0.005 | 0.00498837 | 0.0002 | PASS |
| chi2_block reference | 1.44 | 1.43646 | 0.01 | PASS |
| chi2_block 6-param | 1.34 | 1.33723 | 0.02 | PASS |
| chi2_block 3-param | 1.34 | 1.34146 | 0.02 | PASS |
| a12/d3 (3-param best fit) | 0.3161 | 0.316106 | 0.0005 | PASS |
| phase (0, 90, 180) | 7.98 | 7.97709 | 0.05 | PASS |
| phase (0, 120, 180) | 1.32 | 1.32279 | 0.05 | PASS |
| phase (240, 0, 180) | 0.27 | 0.271827 | 0.05 | PASS |
| phase (0, 120, 90) | 130.0 | 130.012 | 0.05 | PASS |
| phase (240, 240, 180) | 94.3 | 94.2887 | 0.05 | PASS |

## Discrete phase scan (block statistic)

| pattern (deg) | chi2_block |
|---|---|
| (0, 90, 180) | 7.98 |
| (0, 120, 180) | 1.32 |
| (240, 0, 180) | 0.27 |
| (0, 120, 90) | 130.01 |
| (240, 240, 180) | 94.29 |

Reference chi2_block = 1.436

Integer scan (N=12): # 0 shapes with chi2_shape < 6.0

## Summary: 20/20 checks passed