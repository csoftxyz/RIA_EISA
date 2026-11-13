# Catalogue of contents (English edition)

**Paper**: *Grand unification builds the stage, not the script*  
**Journal**: Physics Letters B　**Packaged**: 13 November 2026　**Version**: v9

---

## Where to start

| If you want to | open |
|---|---|
| submit | `01_paper/reach_plb.pdf` + `01_paper/supplement.pdf` |
| read the cover letter | `00_cover_letter/cover_letter.pdf` |
| read the code, illustrated | `04_documentation/illustrated.pdf`, 19 pp |
| run the verification | `cd 03_code/main && python3 annotated_verification_en.py` -> 21/21 |
| check Table 1 row by row | `cd 03_code/support && python3 verify_v2.py` -> 20/20 |
| see the figure | `02_figures/fig_stage.pdf` |

## 00_cover_letter/　Cover letter

| File | Size | Pages | Description |
|---|---|---|---|
| `c1.txt` | 5 KB |  | -- |
| `c2.txt` | 5 KB |  | -- |
| `cover_letter.md` | 4 KB |  | Markdown edition of the cover letter. |
| `cover_letter.out` | 0 B |  | -- |
| `cover_letter.pdf` | 91 KB | 2 pp | Letter to the Editors of Physics Letters B, 2 pp. |
| `cover_letter.tex` | 5 KB |  | LaTeX source of the cover letter. |

## 01_paper/　Manuscript

| File | Size | Pages | Description |
|---|---|---|---|
| `reach_plb.pdf` | 101 KB | 5 pp | Letter, 5 pp. |
| `reach_plb.spl` | 0 B |  | -- |
| `reach_plb.tex` | 20 KB |  | LaTeX source (elsarticle 3p). |
| `supplement.pdf` | 197 KB | 9 pp | Supplementary Material, 9 pp. |
| `supplement.tex` | 28 KB |  | LaTeX source; figures via \graphicspath to ../02_figures. |

## 02_figures/　Figures

| File | Size | Pages | Description |
|---|---|---|---|
| `fig3d.py` | 4 KB |  | Script that draws the two-panel figure. |
| `fig_reach3d.pdf` | 28 KB | 1 pp | Two-panel figure (vector). |
| `fig_reach3d.png` | 421 KB |  | Two-panel figure, 300 dpi. |
| `fig_stage.pdf` | 36 KB | 1 pp | Stage figure (vector). |
| `fig_stage.png` | 555 KB |  | Stage figure, 300 dpi. |
| `fig_stage.py` | 5 KB |  | Script that draws the stage figure. |

## 03_code/　Code

| File | Size | Pages | Description |
|---|---|---|---|
| `corrected/VERIFICATION.md` | 3 KB |  | Summary of the four scripts and the list of fixes. |
| `corrected/d5_theorems_fixed.py` | 8 KB |  | Fixed: Theorem C correctly stated and asserted; step 4 tests Corollary D. |
| `corrected/d5_trace_identity_fixed.py` | 8 KB |  | Fixed: tabulated Antusch-Spinrath values with an assertion. |
| `corrected/independent_check2_fixed.py` | 9 KB |  | Fixed: three configurations side by side. |
| `corrected/independent_check_fixed.py` | 5 KB |  | Fixed: docstring and dead code. |
| `independent/d5_derivation_check.py` | 8 KB |  | Structural equivalence and the redundancy count. |
| `independent/d5_theorems.py` | 6 KB |  | Theorems A/B/C; steps 4 and 5 defective, see corrected/. |
| `independent/d5_trace_identity.py` | 9 KB |  | Trace identities, both bounds over 6000 points, GUT comparison. |
| `independent/independent_check.py` | 5 KB |  | Rank of 120+126 over 300 points; the relation is not linear. |
| `independent/independent_check2.py` | 8 KB |  | Parameter counts and ranks of the six splitting models. |
| `independent/independent_check3.py` | 5 KB |  | Real-vev control: only 120+126 loses rank. |
| `main/annotated_verification_en.py` | 85 KB |  | Main program, English annotations, 21 assertions, census 6/12. |
| `main/annotated_verification_v2.py` | 81 KB |  | Same program, Chinese annotations, also 21 assertions. |
| `results/figures/fig_0_infrastructure.png` | 59 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_1_census.png` | 81 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_2_observables.png` | 39 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_3_jacobian.png` | 56 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_4_identity_bank.png` | 92 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_5_witness.png` | 83 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_6_reach.png` | 161 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_7_verification.png` | 83 KB |  | Figure produced by the module of the same number. |
| `results/figures/fig_8_walkthrough.png` | 81 KB |  | Figure produced by the module of the same number. |
| `results/verification_results.json` | 21 KB |  | Output of the main program, including the two generic controls. |
| `support/d5_check.py` | 4 KB |  | Independent check of Theorems A/B and Corollaries D/E. |
| `support/identity_scan.py` | 4 KB |  | Search for exception contractions. |
| `support/model2.py` | 3 KB |  | Single source of truth for the census. |
| `support/verify_v2.py` | 3 KB |  | 20 assertions over Table 1 and the counting rules. |

## 04_documentation/　Code documentation

| File | Size | Pages | Description |
|---|---|---|---|
| `illustrated.pdf` | 897 KB | 18 pp | Illustrated code description, 19 pp. |
| `illustrated.tex` | 10 KB |  | Source; compile with lualatex. |
| `index.html` | 1 MB |  | Self-contained reading page. |

## 05_notes/　Working notes and history

| File | Size | Pages | Description |
|---|---|---|---|
| `CEILING_SEARCH.md` | 7 KB |  | Working note 3 (Chinese). |
| `CEILING_SEARCH_EN.md` | 2 KB |  | Working note 3, English edition. |
| `GUT_REACH.md` | 12 KB |  | Working note 1 (Chinese). |
| `GUT_REACH_EN.md` | 2 KB |  | Working note 1, English edition. |
| `NEUTRINO_TEST.md` | 9 KB |  | Working note 2 (Chinese). |
| `NEUTRINO_TEST_EN.md` | 2 KB |  | Working note 2, English edition. |
| `VEV_LOOPHOLE.md` | 6 KB |  | Working note 4 (Chinese). |
| `VEV_LOOPHOLE_EN.md` | 2 KB |  | Working note 4, English edition. |
| `annotated_verification_v1_7-11.py` | 78 KB |  | First version, census 7/11. Do not submit. |
