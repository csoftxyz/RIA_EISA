# Response to the referee report

The report was checked line by line, with independent recomputation. Verdicts:

| # | Referee point | Verdict | Action |
|---|---|---|---|
| 1 | Eq. (2)/(3)/(5) miss a factor (d−2); the naive radius leaves the shell for d ≥ 4 | **correct** | corrected radius ρ^{d−1} = (d−2)(r_{n+1}−r_n)/(r_n^{2−d}−r_{n+1}^{2−d}); uniqueness proved by MVT; failure table added (q* = 1, 1.618, 1.839, 1.928 for d = 3…6) |
| 2 | Eq. (6) q-factor (q−1)/q holds only in d = 3 | **correct** | replaced by 1−q^{2−d}; superscript audited independently |
| 3 | Fig. 1(a) caption says "linear" but the arithmetic mean is strictly second order | **correct** | caption rewritten; slope-2 guide line added |
| 4 | Lemma 2 states "any gauge-field configuration" but the proof assumes E has no root components | **correct** | non-Abelian section removed from this paper (methods paper); to be restored only with the hypothesis in the statement |
| 5 | The 96³ "10% geometry bias" is mostly the left-endpoint bias | **correct in substance** | exact factorisation Q_left = [R/(R+1)]·Q_exact; recomputed: 0.900±0.021 → 0.982±0.029 |
| 5b | "...so the residual should be ≈1.00±0.01" | **not confirmed** | measured residual is 1.8% with ±2.9% shell oscillation (cubic-shell averaging), not 1% |
| 6 | The "10% at k=4" depends on the fit window | **correct** | 0.8957 (k≤4) / 0.9876 (3–8) / 0.9912 (4–8); closed form q_k−1 quoted instead (18.9% at k=4) |
| 7 | Theorem 2 conflates the anomalous dimension of a conserved current with the β function | **correct** | no-go theorem removed; the residual slope is now called the running of the static force, explicitly not β |
| 8 | The asymptotic-safety paragraph is a category error | **correct** | paragraph removed |
| 9 | Novelty overstated: the radius rule is the standard exact-transmissibility / q-symmetric point | **partly correct** | reframed as a methods paper; cites Aziz–Settari and Kac–Cheung; the iff-characterisation over arbitrary meshes and the closed-form bias table are retained as the contribution |
| 10 | The no-go names no target; 8 of 16 references are never cited | **correct** | no-go removed; bibliography rebuilt, all entries cited |
| 11 | "1/k ≃ 0.12–0.25" vs (ln2)/k | **correct** | closed form q_k−1 used throughout |
| 12 | The criterion degrades numerically as q→1 by cancellation | **correct** | stable factored form added; digit-loss table included |

Everything else in the report (Table I closed forms, Table III values, the
least-squares check 0.89575/0.92360/0.93906, the exponent p−d+2) was
re-verified and confirmed.

---

## Second round (internal review of the CPC version)

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | Table 5 used r₀√q as the reference radius, so the first two rows were a (q−1)²/24 artefact | **correct** | errors re-measured against a 30-digit Decimal reference; factored form now shows 6×10⁻¹⁷…2×10⁻¹⁷, difference form degrades 7×10⁻¹⁶ → 5.5×10⁻¹⁰ |
| 2 | "5×10⁻¹³, 1.4×10⁵ shells" had no provenance in the package | **correct** | deep-mesh sweep added to the driver: worst 4.4×10⁻¹⁶ over 2×10⁵ shells; text updated |
| 3 | "eleven-per-cent endpoint factor" contradicts the window mean (0.9171) | **correct** | now "eight per cent averaged over R=8–15 (eleven per cent at R=8)"; exactness restricted to the per-shell identity |
| 4 | 0.941 quoted at R=15 (it is R=16) | **correct** | 0.9375 at R=15, 0.941 noted as R=16 |
| 5 | duplicated phrase "it says that the potential it says that the potential" | **correct** | removed |
| 6 | "logarithmic mean of r²" misnamed | **correct** | geometric mean, stated correctly |
| 7 | Table 3 labelled "fitted slopes" but no fit existed; (d,p) sets mismatched | **correct** | real log–log polyfit added for exactly the eight (d,p) pairs in the table; slopes agree to 4×10⁻¹⁶ |
| 8 | fig2 orphaned, unreferenced | **correct** | figure included in Sec. 6.3; files renamed so numbers match (fig2_lattice, fig3_refinement) |
| 9 | Data-availability URL points at the old package | **correct** | text now says the archive is authoritative and names the CPC subdirectory; the repository must be updated before submission |
| 10 | ε_mach/(q−1) overstates the loss; "unusable below 1e−6" too strong | **correct** | replaced by "about one digit per decade", with measured values |
| 11 | "(2×10⁻¹⁶ or below)" is format/version dependent | **correct** | "a few times 10⁻¹⁶ (worst 4.4×10⁻¹⁶)" |
| 12 | "uniqueness" claimed but only interiority proved | **correct** | uniqueness sentence added to Lemma 1 (injectivity of x↦x^{d−1}) |
| 13 | running time 3 s / 12 s not reproducible | **correct** | measured 0.4 s / 0.9 s, environment stated |
| 14 | verify_claims section numbers swapped; table numbers mismatch | **correct** | cards relabelled by LaTeX label |
| 15 | Program summary missing standard CPC fields | **correct** | Classification, External routines, No. of lines, Distribution format, supersedes added |
| 16 | bibitem{flow}, bibitem{bali} to be checked | **no change** | both are correct as printed (Lüscher, JHEP 06 (2014) 105; Bali–Schilling, PRD 47 (1993) 661) |

---

## Third round (review of v2)

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 1 | Table 6 rows said `k=3--8` / `k=4--8`, but the fits used {3,4,6,8} / {4,6,8} | **correct** | labels changed to `k=3,4,6,8` and `k=4,6,8`, in the table and in the body text |
| 2 | "Number of lines: 517" not reproducible | **correct** | measured 539 raw lines (343 + 97 + 83 + 16); field updated |
| 3 | "worst 4.4e−16 over 2e5 shells" merges two rows | **correct** | now states that the worst case occurs at q = 2 on 39 shells, and that the 2e5-shell run is at round-off |
| 4 | underfull hbox in the Program summary ("Programming lan-guage") | cosmetic | first column set ragged-right; warning gone |
| 5 | Data-availability repository still holds the old package | open | requires the authors to push `Coulomb/CPC`; the archive is declared authoritative |

Verification added: the stability sentence now also states the observed crossover ("below
q−1 ≈ 1e−9 the difference form returns ρ = 1 exactly"), which was checked independently
(ε = 8.6e−9 is the last scale at which the in-shell offset is still accurate).
