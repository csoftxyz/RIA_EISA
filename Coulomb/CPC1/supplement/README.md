# CPC Supplementary Materials v1 (2026-11-04)

Supplementary file for "Exact flux-sampling radii for radial Coulomb and Poisson problems on graded meshes: closed-form estimator bias in any dimension, and a null test."

## What's in the package

| Path | Contents |
|---|---|
| `supplement/supplement.pdf` | Supplementary material main text, 56 pages |
| `supplement/supplement.tex` | LaTeX source of the above (two pdflatex passes suffice) |
| `supplement/fig/` | PDFs of the 9 three-dimensional figures (vector, for typesetting) + PNGs (320 dpi, for display) |
| `supplement/make_figs3d.py` | Script that generates these 9 figures (each calls functions from `src/`; no formulas are duplicated) |
| `supplement/annotations.py` | Line-by-line annotation database (267 annotations) |
| `supplement/check_annotations.py` | Machine verification of coverage: 1383/1383 executable lines fully covered; returns 0 only when there are zero gaps |
| `supplement/make_supplement.py` | Automatically generates supplement.tex from the source code + annotation database |
| `supplement/logs/` | Raw output of the five scripts (reproduced verbatim in Section 5 of this PDF) |
| `src/` | Code accompanying the paper (12 files, unmodified) |
| `radialflux.tex` / `.pdf` | Main manuscript, for cross-reference (the supplementary material cites its equation numbers) |

## Three parts

1. **Code**: All 12 files in `src/`, annotated line by line (267 annotations); each annotation states why the line is written that way. External functions (`np.expm1`, `np.fft.fftfreq`, `np.linalg.lstsq`, etc.) have their provenance and rationale for selection listed in a separate table. Coverage is machine-verified, not self-declared.
2. **Three-dimensional figures**: 9 figures, each corresponding to a section of code (Table 3 in the supplementary material is the mapping table).
3. **Communication physical meaning**: Section 5, explicitly marked as conjecture, not a conclusion of the paper, and stating that the author's graduate field was communications and that the idea for this paper originated there.

## Reproduction

```bash
cd CPC_radialflux
python3 src/validate.py          # 51 items, exit 0
python3 src/audit_claims.py      # 84 items, exit 0
python3 src/check_tex.py         # structure check, exit 0
python3 src/lattice96.py         # 96^3, about 0.14 s
python3 supplement/check_annotations.py   # coverage
python3 supplement/make_figs3d.py         # 9 figures
python3 supplement/make_supplement.py     # generate tex
pdflatex supplement/supplement.tex        # two passes
```

Dependencies: NumPy (Matplotlib additionally required for figures; SymPy additionally required for symbolic verification). No data files needed.