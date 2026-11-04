# radialflux — exact flux-sampling radii for radial Coulomb / Poisson problems

Companion code for the CPC submission
*Exact flux-sampling radii for radial Coulomb and Poisson problems on graded
meshes: closed-form estimator bias in any dimension, and a null test*.

## What is here

| path | contents |
|---|---|
| `radialflux.tex` / `.pdf` | the manuscript (24 pp., elsarticle, 49 references) |
| `src/radialflux.py` | the module (147 lines; NumPy only) |
| `src/validate.py` | verification suite, 13 groups / 51 asserted checks, exits non-zero on failure |
| `src/lattice96.py` | periodic 96³ Coulomb solve (exact lattice symbol, FFT) |
| `src/audit_claims.py` | re-computes every number quoted in the paper (84 checks) |
| `src/check_tex.py` | LaTeX integrity: duplicate labels, dangling refs, uncited/phantom bibitems, unused figures |
| `src/verify_core.py`, `verify_shells.py`, `series_d.py` | symbolic cross-checks (SymPy) |
| `src/make_figs*.py` | the four figure scripts |
| `fig/` | figures, PDF (for LaTeX) and PNG (for slides) |
| `requirements.txt` | NumPy (the only dependency) |

## Quick start

```bash
pip install -r requirements.txt
python3 src/validate.py        # 51 asserted checks, ~0.06 s, exit code 0/1
python3 src/audit_claims.py    # 63 checks, 0 failed
python3 src/lattice96.py       # 96^3 solve, 0.14 s, 120 MB
```

```python
from radialflux import exact_radius, enclosed_charge, exact_bias, q_star

rho = exact_radius(r[:-1], r[1:], d=3)   # the exact sampling radius
Q   = enclosed_charge(V, r, d=3)         # uses it by default
print(exact_bias('midpoint', q=1.1, d=4))  # 1.004551
print(q_star(4))                           # 1.6180339887...
```

## The one line that matters

Replace the sampling radius in your radial estimator by

  rho_star^{d-1} = (d-2)(r_{n+1}-r_n) / (r_n^{2-d} - r_{n+1}^{2-d})

evaluated through `expm1` (see `exact_radius`), and quote the result at
`rho_star`. In three dimensions `rho_star = sqrt(r_n r_{n+1})`.

## Licence

MIT.
