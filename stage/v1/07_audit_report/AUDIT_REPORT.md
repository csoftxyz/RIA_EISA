# The C21 Paper & Code Audit Report

**Project:** *Grand unification builds the stage, not the script: the exact reach of unified flavour models, with neutrinos*
**Authors:** Yuxuan Zhang, Weitong Hu, Wei Zhang
**Project start:** February 2025 (the Z₃-graded algebraic framework)
**This report covers:** February 2025 – November 2026
**Final state:** package v24, 70/70 assertions, four clean submission PDFs (cover letter 4 pp, main 14 pp, supplement 12 pp, code documentation 61 pp) plus this internal transparency report (8 pp, not submitted with the manuscript)

---

## How to read this report

This is an honest, plain-language account of everything that went wrong, everything
that was tried, and how each problem was finally solved. It is written so that a
non-specialist can follow the story. Technical terms are explained the first time
they appear. The companion flowchart (`audit_journey_flowchart.png`) shows the whole
journey on one page; this document tells it in detail.

The project went through **13 external audits**, **6 self-audits**, and **8 "mining"
directions** (deliberate searches for new results). Along the way the paper's
headline conclusion was **withdrawn and rewritten three times**. That is not a sign
of failure — it is the sign of a project that kept checking itself until the answer
was right. This report records every turn.

### The one-sentence story

> We asked "how many laws does a unified theory *force*, with nothing fitted?"
> The answer turned out to be **zero extra flavour laws** in the charged sector,
> and **zero observable neutrino laws** — but the simplest model hides one *exact
> matrix-level identity* (the "pencil identity") that is real, provable, and yet
> invisible to any low-energy measurement. Getting to that honest sentence required
> throwing away three earlier, wrong answers.

---

## Part 0 — What the paper is actually about (in plain words)

A **grand unified theory (GUT)** tries to describe quarks and leptons as different
faces of one object, related by a big symmetry group (SU(5), SO(10), E₆, or
Pati–Salam). The hope is that the bigger group *explains* more — for example, why
the bottom quark and the tau lepton have similar masses.

The paper asks a sharper question than "can this model fit the data?". It asks:

> **What relations does the model force to be true for *every* allowed choice of
> its parameters?**

Fitting data with adjustable knobs is not prediction. A relation is a *prediction*
only if it holds no matter how you turn the knobs. The paper calls the number of
such forced relations the **exact reach** (a codimension — a count of constraints,
not a count of parameters).

It splits the reach into three parts:
- **B_g** — gauge-coupling unification (a boundary condition),
- **B_c** — charge quantisation (a structural fact),
- **P_add** — the model-specific *flavour* part (the interesting bit).

The headline finding: **P_add = 0 for every viable content** in the charged sector.
The group "builds the stage" (the baseline) but "does not write the script" (no
extra flavour lines). Any further predictivity must be *bought* by declaring extra
inputs — texture zeros, discrete symmetries, a special form of the heavy-neutrino
matrix M_R. The framework's job is to make that price explicit.

---

## Part 1 — The chronological journey

### Stage 0 (Feb 2025): the seed idea

The project began as the **Z₃-graded algebraic framework**, a theoretical programme
of the authors. The flavour question ("how many relations are forced?") grew out of
it. The first concrete deliverable was a draft (v5) defining *exact reach* and
enumerating **18 standard Higgs contents** across the four groups. The census found
**7 viable / 11 excluded**. This census survived every later audit — it is the
sturdy backbone of the paper.

### Stage 1 (Audit 1): "Is the neutrino sector really free?"

The first external audit liked the idea but flagged a key theoretical issue: the
paper treated the heavy Majorana matrix M_R as a *free* matrix. In a real SO(10)
model M_R is **not** free — it is `M_R = v_R · Y₁₂₆`, sharing the very Yukawa matrix
that feeds the charged fermions.

**Route taken:** prove the *free* case first as a baseline. The type-I seesaw map
`M_R ↦ −M_Dᵀ M_R⁻¹ M_D` was shown **surjective** (it can hit any symmetric target).
This is Theorem 2. It says a *free* Majorana sector adds nine parameters and **no**
relations — predictivity built on a declared M_R structure is *purchased*, not
supplied by the group.

**Pit avoided:** conflating "free M_R" with "unified M_R". The audit was right that
the unified case is correlated and must be treated separately.

### Stage 2 (Audit 2): the demand for exact proof — and the first withdrawal

Audit 2 insisted: *random-point numerical full rank is not a proof.* Provide an
exact witness or an analytic argument.

**Route taken:** build a **pure rational-arithmetic** witness (`exact_witness.py`):
rational parameters, Gaussian-rational matrix entries, dual-number automatic
differentiation over ℚ(i), fraction Gaussian elimination — **no floats, no
tolerance**.

**Pit #1 (a real bug):** the first exact run gave rank 17 where floats gave 22.
Column-by-column comparison exposed that **every odd (imaginary) derivative column
was wrong**: for a complex parameter `z = p_i + i p_j`, the seed for `∂z/∂p_j` must
be `i`, but it had been set to `1`. Fixed; all 34 columns then agreed with an
independent exact finite difference. *Lesson: a derivative-seed bug can manufacture
an entirely wrong physical conclusion.*

**The first wrong headline:** using exact matrix rank **minus a guessed U(3) orbit**
(10 "redundancy" directions), the count came out **18 physical directions < 22
observables ⇒ ≥4 forced relations** for 120+126. The paper was rewritten to claim
"the minimal content forces at least four relations" (0 → 4, which *looked* like
progress).

**Why it was wrong (found later):** the orbit subtraction is invalid — six of the
ten "redundancy" directions actually *change* physical observables. You cannot
subtract directions you have not verified are truly redundant. (A related early
mistake: assuming a real rescaling `A→λA, vev→/λ` was a redundancy; an exact finite
difference along it gave 81.97 ≠ 0 — *not* redundant, because the seesaw weight
changes.)

### Stage 3 (Audits 3–7): fits, optimisers, running, and the counting trap

These audits attacked the phenomenology and the bookkeeping.

- **Weak-optimiser pit:** an early damped Gauss–Newton fit with few starts returned
  large residuals, and the paper wrongly declared "no content admits a precision
  fit." **Route:** a proper Levenberg–Marquardt fit with batched Jacobians and 120
  starts reached **exactly zero** residual for 10+126 on all six ratios. The blanket
  claim was withdrawn; the obstruction is *specific* to 120+126, quantified as a
  **wall near m_τ/m_b = 2.76**. *Lesson: an under-powered numerical failure can be
  mistaken for a physical obstruction.*
- **RG-sensitivity pit:** the GUT-scale wall is *not* by itself a low-energy
  exclusion — an indicative one-loop example runs a wall point down into the
  observed window. The relation count was downgraded from "Theorem" to "Result" to
  match the evidence. *Lesson: a computation should be allowed to weaken the claim
  it was meant to support.*
- **The bidoublet counting trap (the editor's catch):** treating the SO(10)
  **120** as a single operator (one irrep = one operator) imposes `v₁₅ = 0` and
  invents the identity `M_e = −2M_d − M_dᵀ`, wrongly excluding 10+120. In reality
  the 120 carries **two** electroweak bidoublets with independent vevs. Correctly
  counted, 10+120 is **viable**. This error was found by **Professor Csaba Csáki**
  acting as editor of Physics Letters B — *not* by any automated check. *Lesson:
  code–text consistency can be checked mechanically; fidelity to the group theory
  cannot.*

### Stage 4 (Audits 8–9): THE CORE REVERSAL

This is the most important turn. Audit 8 asked whether the claimed relations
descend to the *observable* space. Audit 9 did the decisive experiment and
**overturned the core conclusion**.

**The pit (the deepest one):** the neutrino "observable" vector was **not
basis-independent** — i.e. it was not a physical observable at all. Three concrete
defects:
1. it used `eigh(M_ν M_ν†)` eigenvectors, which land on the **wrong (conjugate)
   side** of the Takagi decomposition;
2. it never did the **transpose congruence** to the charged-lepton mass basis
   (`M_ν_mass = L_eᵀ M_ν L_e`);
3. its "Majorana phase" was `angle[(U_nᵀ M_ν U_n)₀₀]`, but that matrix had
   off-diagonal norm ≈ 100 — it was **not a phase**.

Under a common left-handed flavour rotation (a change of basis that must leave
physics unchanged) the nine neutrino components changed by **0.6–5**. A true
observable changes by ~0.

**Route taken:** rebuild from scratch (`strict_obs.py`, written independently of the
auditor's scripts): SVD *left* unitaries for the charged sector; transpose
congruence + strict Autonne–Takagi with proper column-phase fixing for the neutrino
sector; rephasing-invariant Majorana phases `φ_i = arg(P₁ᵢ² P₁₁*²)`. Verified
basis-independent to **~10⁻¹⁴**.

**The result:** generic observable rank **22**, i.e. **ZERO** forced low-energy
relations for *every* content (charged 13). The earlier "18/4" *and* an intermediate
"19/3" were **both withdrawn**.

**What survived:** the **pencil identity** — `B = A S⁻¹` is traceless of rank ≤ 2,
so the pencil (M_ν, S) has a repeated generalised eigenvalue, `disc(M_ν S⁻¹) = 0`.
This is exact (grade a). But it is a **matrix-level** identity: projected onto the
strict observable space it leaves **no** relation. The headline became: *one exact
matrix-level identity, invisible to low-energy data.*

**The meta-lesson (recorded in the report):** "exact" was repeatedly attached to the
wrong statement. Exact arithmetic guarantees the *identity* is correct; it does
**not** guarantee the *inference* from identity to "number of observable relations"
is correct. And **basis-independence is a prerequisite** for calling anything an
observable — it must be tested *before* any dimension count is trusted.

### Stage 5 (Audits 10–13): tightening, grading, and synchronising

With the core conclusion now correct, these audits hardened the evidence and the
prose:
- **Evidence grades** were separated: grade (a) = exact/symbolic or rigorous
  certificate; grade (b) = generic numerical measurement. The charged dominance was
  upgraded to **grade (a) for all four SO(10) contents** (`charged_grade_a.py`,
  exact rational invariant Jacobian rank 13).
- **The neutrino dimension 22** was upgraded from grade (b) to **grade (a)** via a
  rigorous **interval-arithmetic Neumann certificate** (`grade_a_fast.py`): at three
  rational points a 22×22 submatrix satisfies ρ = ‖A⁻¹‖∞‖E‖∞ < 1 (≈ 0.28/0.41/0.10,
  margins ≥ 0.59, stable up to 64 ULP padding), proving every matrix in the interval
  enclosure is nonsingular; E carries a conservative Richardson-FD inflation and the containment
  |A−J_fd|≤E is independently verified (audit 14).
- **The rational route was closed:** an exhaustive screen of 40 trace words plus a
  fraction-field argument (`d(f/g) ∈ span{df,dg}`) shows no rational bi-unitary
  invariant of the trace-word fraction field reaches rank 22 — the rational lower
  bound stalls at 17. This explains *why* the certificate must use the full
  non-rational observable map.
- **Measured-boundary RG matching** (`rg_lowenergy_match.py`): evolving the
  *measured* spectrum to the GUT scale gives (m_τ/m_b)_GUT ≈ 1.07, far below the 2.76
  wall — so the minimal-content exclusion **survives running**, scheme-independently.
- **Prose synchronisation:** many rounds of making the abstract, cover letter,
  supplement, AI-use statement and CONTENTS agree with the code (assertion counts,
  page counts, withdrawn claims).

### Stage 6 (Mining + self-audits): new theorems, and catching our own mistakes

Eight mining directions were pursued. The genuine breakthroughs:
- **Pencil algebraic origin theorem** (`pencil_analytic_origin.py`): the identity is
  *derived*, not just observed. With `M_D = aA + bS` (A antisymmetric), the cross
  term cancels and `M_ν S⁻¹ = c₁I + c₂B²`; since `det A = 0`, B is traceless rank-2,
  B² has a repeated eigenvalue ⇒ disc = 0. The repeated root is forced by the
  **Pfaffian structure** of the antisymmetric Yukawa. Every link verified to 10⁻¹³.
- **Pfaffian generalisation test:** the mechanism does **not** extend to SU(5) 45
  (general down Yukawa, no renormalisable seesaw singlet). Only SO(10) 120+126 meets
  all three structural requirements.
- **CKM-locking exclusion theorem:** 9 of the 11 exclusions follow from one
  cross-group mechanism — a single Yukawa forces aligned left rotations ⇒
  V_CKM = I, which data refuse.
- **Seven-content declaration scoreboard:** 5 contents fit with zero declared
  inputs; 10+120 and 120+126 need ≥ 1. The minimal content, despite its exact
  identity, is among the **least economical**.

Then the **self-audits** caught three errors that the mining itself had introduced:

**Self-audit round 1 (misattribution V1):** a statistical preference (single-S
models favour a weaker neutrino hierarchy) had been blamed on the pencil identity.
**Control:** repeat on 10+126, which has *no* pencil identity — the *same*
preference appears (m₃/m₁ KS p ~ 10⁻¹²⁸), and it persists in pure type-I (v_L = 0).
**Conclusion:** the preference is a **single-S M_R structure effect**, not a pencil
consequence. The claim "the pencil has a falsifiable statistical consequence" was
**withdrawn**.

**Self-audit round 2 (misattribution V7, a relapse):** the preference was then
blamed on the antisymmetry-induced 3→2 repeated-root reduction. **Control:** 10+126
has *three* distinct eigenvalues (no repeated root) yet the same preference.
**Conclusion:** the 3→2 reduction is a 120+126-only observation, **not** the
preference mechanism. Corrected again.

**Self-audit round 3 (code bug V8):** the generic builder represented a general
complex matrix as `sym + i·sym`, which spans only complex *symmetric* matrices
(Jacobian rank **12**, not 18). **Fix:** `sym + anti` (rank 18). This also removed an
optimiser-sensitive false positive (5+45+50 read cost ≥ 1 under the buggy builder,
cost 0 once correct).

**Self-audit (disclosure sync):** the abstract/cover/AI statement still said "we do
no RG" and "the scoreboard has not been computed" after those had been done. All
five carriers were synchronised.

**The recurring lesson:** misattribution has a *relapse* risk. Every attempt to
blame the statistical effect on a 120+126-specific feature (the identity, the
repeated root, antisymmetry) was falsified by the 10+126 and pure-type-I controls.
The only correct general mechanism is the **single-S M_R structure**.

---

## Part 2 — The routes we took (and the ones we abandoned)

| Goal | Route tried | Outcome |
|---|---|---|
| Prove neutrino dimension | Random float Jacobian full rank | **Abandoned** — not a proof; branch jumps fake directions |
| Prove neutrino dimension | Exact rational witness + orbit subtraction | **Abandoned** — orbit subtraction invalid (18/4 wrong) |
| Prove neutrino dimension | Richer invariant words (19/3) | **Abandoned** — observable vector not basis-independent |
| Prove neutrino dimension | Strict Autonne–Takagi, generic rank | **Kept** (grade b), then upgraded |
| Prove neutrino dimension | Rational invariant Jacobian → 22 | **Abandoned** — stalls at 17; fraction field adds nothing |
| Prove neutrino dimension | Interval-arithmetic Neumann certificate | **Kept** (grade a) |
| Explain the preference | Pencil identity causes it | **Abandoned** (V1 control) |
| Explain the preference | Repeated-root 3→2 causes it | **Abandoned** (V7 control) |
| Explain the preference | Single-S M_R structure | **Kept** |
| Fit the spectrum | Damped Gauss–Newton, few starts | **Abandoned** — weak optimiser false obstruction |
| Fit the spectrum | Levenberg–Marquardt, 120 starts | **Kept** |
| Generalise the pencil | Any single-S content / SU(5) 45 | **Abandoned** — needs M_D ∈ span{A,S} + seesaw singlet |

---

## Part 3 — The pits, catalogued

1. **Imaginary derivative seed = 1 instead of i** (Audit 2). Corrupted every odd
   Jacobian column. Fixed by setting the seed to i; verified column-by-column.
2. **Orbit subtraction without verifying redundancy** (Audits 2, 8). Gave 18/4.
   Withdrawn; six of ten "redundancy" directions change observables.
3. **Non-basis-independent observable vector** (Audits 8–9). The core reversal.
   Fixed by strict Autonne–Takagi; basis-independence tested to 10⁻¹⁴.
4. **"exact" attached to the wrong statement** (recurring). Exact identity ≠ exact
   dimension inference. Fixed by separating evidence grades.
5. **Weak optimiser mistaken for physical obstruction** (Audit 4). Fixed by LM +
   many starts.
6. **Bidoublet counting trap** (editor Csáki). 120 = two bidoublets, not one operator.
7. **Hard-coded "robustness" assertions** (programmer self-audit). Three assertions
   were literals that always passed. Replaced by real measurements.
8. **Float-vs-Fraction parameterisation mismatch** (grade-a certificate). The
   certificate used free v_L while the paper used tied v_L; M_ν differed by ~43.
   Fixed by using the same tied parameterisation; verified to 10⁻¹³.
9. **Misattribution of the statistical preference** (V1, V7). Twice blamed on
   pencil-specific features; twice falsified by controls.
10. **gen builder underparameterisation** (V8). sym+i·sym spans rank 12, not 18.
    Fixed to sym+anti.
11. **Disclosure drift** (self-audit). Abstract/cover/AI lagged the code. Synchronised.
12. **Float-stuck tables** (revtex). Fixed with the float package and [H].

---

## Part 4 — What is solid, and what remains open

### Solid (grade a unless noted)
- The 18-content census, 7 viable / 11 excluded; bidoublet counting rule.
- Theorem 2 (free-M_R seesaw surjectivity).
- The pencil identity, now with a complete algebraic origin theorem.
- Charged dominance grade (a) × 4 SO(10) contents.
- Neutrino observable dimension 22 / zero relations, grade (a) Neumann certificate.
- CKM-locking exclusion theorem (9/11, cross-group).
- Measured-boundary RG: minimal-content exclusion survives running.
- Seven-content declaration scoreboard (5 free / 2 need input).
- 70/70 assertions; four clean submission PDFs plus this internal transparency report; deterministic, basis-independent truth source.

### Open (honestly declared, not hidden)
- **The pencil identity has no independent falsifiable low-energy consequence.** It
  is a matrix-level constraint on model space. This is the paper's central
  limitation and the main desk-reject risk for a regular article.
- **The m_τ/m_b wall (2.76) has no analytic derivation** — it is an emergent
  joint-fit ceiling, not a single-ratio algebraic bound.
- **Exclusions are a controlled one-loop matching**, not a full precision global fit
  (no GUT thresholds, no two-loop Yukawa).
- **SU(5)/Pati–Salam census dominance is grade (b)** (sampled), not grade (a).
- **The GitHub repository is not yet uploaded** ("will be deposited upon
  acceptance"); this must be done before submission.
- Result 1 is close to tautological; P_ex mixes gauge boundary conditions with
  charge quantisation. These narrative weaknesses are acknowledged.

---

## Part 5 — Methodology lessons (the durable takeaways)

1. **A floating-point full rank is not a proof.** Especially when observables
   involve eigenvector branches, p±h can jump branches and fake independent
   directions.
2. **Basis-independence is a prerequisite, not an afterthought.** Test it before
   trusting any dimension count.
3. **"Exact" must attach to the specific statement it proves.** An exact identity
   does not license an exact dimension inference.
4. **An under-powered numerical failure is not a physical obstruction.** Strengthen
   the optimiser before declaring exclusion.
5. **A computation must be allowed to weaken the claim it was meant to support.**
   The RG example and the core reversal both did this.
6. **Misattribution relapses.** Any claim that a 120+126-specific feature causes a
   general effect must pass the 10+126 (no-pencil) and pure-type-I controls.
7. **Verify representation dimensions by Jacobian rank.** A "general matrix"
   representation must have image rank equal to its intended degrees of freedom.
8. **Synchronise every carrier after each major update.** Code-generated tables
   stay correct; hand-written prose drifts. Grep all five carriers.
9. **Automated consistency checks are not a substitute for a physicist reading the
   group theory.** The bidoublet error was caught by a human editor, not a script.

---

## Part 6 — Reproducibility & the control panel

Everything is reproduced by a single program (`verify_prd.py`, 70 assertions) plus
the independent test/validation suite in `04_verification_code/tests_validation/`.
A graphical control panel (`runall.bat` → `08_gui/runall_gui.py`) presents every
program as a button with a plain-language explanation and an illustrative figure;
the interface screenshot is embedded in the code documentation. The locked
environment is Python 3.6.6 / numpy 1.19.5 / scipy 1.5.4 / matplotlib 3.3.4.

**Files in this report folder:**
- `AUDIT_REPORT.md` — this document.
- `audit_journey_flowchart.png` — the one-page journey map.
- `audit_statistics.md` — counts of audits, scripts, lines, pages.

*End of report.*
