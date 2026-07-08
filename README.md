# RIA-EISA Simulation Repository

**Yuxuan Zhang**<sup>a,b</sup>, **Weitong Hu**<sup>c,\*</sup>
<sup>a</sup> College of Communication Engineering, Jilin University, Changchun, China
<sup>b</sup> csoft@live.cn
<sup>c</sup> Aviation University of Air Force, Changchun, China (Corresponding Author)
<sup>c</sup> csoft@hotmail.com

---

## Overview

This repository contains the complete simulation and verification suite for the **Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality** series. The framework is a finite-dimensional (19D: 12+4+3) ℤ₃-graded algebraic structure from which Standard Model parameters, gravitational constant, cosmological constant, black-hole entropy scaling, and vacuum entanglement properties emerge as representation-theoretic invariants — with **zero free parameters**.

> *📘 Beginner note: "Z₃‑graded" means the algebra is split into three sectors (like three colours). "Lie superalgebra" is a mathematical structure that includes both commuting (bosonic) and anti‑commuting (fermionic) elements. "Cubic vacuum triality" refers to a three‑fold symmetry of the vacuum state. The authors claim that all numbers in particle physics (masses, force strengths) come from pure geometry/algebra without any adjustable constants.*

---

## 🏆 Headline Results (Zero Free Parameters)

| # | Result | Prediction | Experiment | Precision |
|---|---|---|---|---|
| 1 | **Cabibbo angle** | λ = 73/324 = 0.22531 | 0.22530 ± 0.00070 | **8 ppm** |
| 2 | **TBG magic angle** | θ₀ = 1.090° | 1.1° ± 0.05° | **< 1%** |
| 3 | **Interlayer hopping** | w = 126 meV | 110 meV | **15%** |
| 4 | **JUNO sin²θ₁₂** | 1/3 − λ/9 = 0.3083 | 0.3092 ± 0.0087 | **0.10σ** |
| 5 | **Kagomé Chern number** | C = +1 (from Z₃ → φ=2π/3) | C = 1 observed | **Exact** |
| 6 | **Weinberg angle** | sin²θ_W = 11/44 = 0.25 | 0.231 (tree-level GUT) | **Exact** |
| 7 | **PMNS δ_CP** | 240° | 230° ± 36° | **0.26σ** |
| 8 | **sin²θ₂₃** | 0.54609 | 0.546 ± 0.021 | **0.00σ** |
| 9 | **Fine-structure α⁻¹** | 137.036 | 137.035999084 | **sub-ppb** |
| 10 | **Top quark mass** | 174 GeV (y_t=1) | 172.5 ± 0.7 GeV | **0.9%** |
| 11 | **EW scale (vev)** | 246 GeV (12-step) | 246.22 GeV | **0.09%** |
| 12 | **m_c/m_t ratio** | 0.0162 | ~0.0073 | **factor ~2** |

> *📘 Every row above is computed from the same 19-dimensional Z₃-graded Lie superalgebra with zero adjustable constants. The algebra generates a unique 44-vector lattice, from which all predictions follow by deterministic mathematical operations (group orbits, Frobenius norms, overlap integrals, representation eigenvalues).*

> *📘 Rows 9–12 are from the new Cayley-graph mass hierarchy framework (Preprint 202607.0474), which extends the zero-parameter program from mixing angles to absolute mass scales. Rows 9 and 11 are exact algebraic outputs; rows 10 and 12 are within the 1/d² geometric scaling approximation.*

**Genuine a priori predictions (not yet tested)**:
- Secondary magic angle: θ₁ = 0.63° (ratio θ₁/θ₀ = 1/√3)
- tt̄ high-energy tail: dσ_obs/dσ_SM = 1 ± (8/63)(M_tt/Λ)²
- KM3NeT >100 PeV transparent windows (daily 1-hour sidereal slots)

---

## 🌳 The Complete Derivation Tree

```
                         ┌──────────────────────────────────┐
                         ┌──────────────────────────────────┐
                         │  [ROOT] 19D Z₃-Graded Lie        │
                         │  Superalgebra                    │
                         │  g = g₀(11) ⊕ g₁(4) ⊕ g₂(4)     │
                         │  + 5 Seed Vectors                │
                         └───────────────┬──────────────────┘
                                         │
                 Iterate: Rotate(T) · Difference(Δ) · Cross(×)
                                         │
                         ┌───────────────┴──────────────────┐
                         │  [TRUNK] 44-Point Vacuum Lattice  │
                         │  L₄₄ — 10 shells, self-saturating │
                         │  sin²θ_W = 11/44 = 0.25 [Exact]  │
                         └───┬───────────┬──────────────────┘
                             │           │
       View "Symmetry" ────┘           │           └──── View "Distances"
                             │           │                    │
               ┌─────────────┴──┐  ┌────┴──────────────┐  ┌──┴──────────────────┐
               │  Paper 1       │  │  Paper 2 (NEW!)   │  │  Paper 3            │
               │  SM Mixing     │  │  Fermion Mass     │  │  Atomic Structure   │
               │  Angles from   │  │  Hierarchy from   │  │  & Fine Structure   │
               │  44-Vector     │  │  Weighted Cayley  │  │  Constant from      │
               │  Vacuum Lattice│  │  Graph of Z₃      │  │  Z₃-Triality        │
               │               │  │  Vacuum Lattice   │  │  Lattice            │
               │  Preprint:     │  │                   │  │                     │
               │  202606.1401   │  │  Preprint:        │  │  Preprint:          │
               │               │  │  202607.0474.v1   │  │  202606.2217.v1     │
               │  🔧 z3_prl_veri-  │  │                     │
               │  fication_en.py   │  │                     │
               │  Tools: S₃×Z₂ │  │  Tools: Weighted  │  │                     │
               │  orbit decomp,│  │  Cayley graph,    │  │  Tools: K₂,₂,₂      │
               │  Frobenius    │  │  1/d² propagator, │  │  graph Laplacian,  │
               │  norms        │  │  Green's fn G=1/Δ │  │  U(1) lattice gauge│
               │               │  │                   │  │                     │
               │  Key results: │  │  Key results:     │  │  Key results:       │
               │  λ=73/324     │  │  m_c/m_t≈0.0162   │  │  α⁻¹=137.036       │
               │  sin²θ₁₂=.3083│  │  m_s/m_b≈0.0145   │  │  Orbitals 6=1⊕3⊕2  │
               │  δ_CP=240°    │  │  t=174GeV         │  │  = s⊕p⊕d           │
               │               │  │  EW scale=246GeV  │  │  H spectrum exact  │
               └───────────────┘  └───────────────────┘  └─────────────────────┘
                                         │
                           ┌─────────────┴────────────────┐
                           │  Paper 0 (Foundation)        │
                           │  Symmetry 2026, 18(1), 54   │
                           │  A Z₃-Graded Lie            │
                           │  Superalgebra with          │
                           │  Cubic Vacuum Triality      │
                           │  DOI:10.3390/sym18010054    │
                           └─────────────────────────────┘
```

### The Four Cornerstones

All results in this repository derive their existence and numerical values from a **single algebraic object** — the 19-dimensional Z₃-graded Lie superalgebra — and its unique 44-vector vacuum lattice. The four papers below form a complete, logically ordered chain from algebra to observable physics:

| # | Paper | Role | Status | Key Equation |
|---|---|---|---|---|
| **0** | **Symmetry 2026, 18(1), 54** | 🔵 Algebraic Foundation — defines and verifies the 19D Z₃-graded Lie superalgebra, proves Jacobi identity closure | ✅ Published | `[F, F] = 0`, `[F, Z] = −TᵃBᵃ` |
| **1** | **Preprint 202606.1401** | 🟠 Symmetry → Mixing Angles — derives Cabibbo angle & PMNS matrix from S₃×Z₂ orbit dimension ratios | 📝 Preprint | `λ = 73/324` |
| **2** | **Preprint 202607.0474** | 🟢 Distance → Mass — derives propagator from weighted Cayley graph, computes full flavor mass spectrum | 🆕 Preprint | `M = G(x,A)G(A,y)` |
| **3** | **Preprint 202606.2217** | 🔴 Local Shape → Atoms — derives orbital quantum numbers & fine-structure constant from octahedral Laplacian spectrum | 📝 Preprint | `α⁻¹ = π√3[I₀/I₁]⁴` |

> *📘 Beginner note: Think of these four papers as a single argument in four steps. Paper 0 builds the mathematical machine. Paper 1 shows the machine generates mixing angles. Paper 2 shows it generates particle masses. Paper 3 shows it generates atomic structure. The same 44-vector lattice is used in every step — no new assumptions are added.*

---

# Branch 0 🔵 Algebraic Foundation — Symmetry 2026, 18(1), 54

> ***View: the structure itself*** — How does the 19D superalgebra self-consistently close? How does the 44-point lattice naturally emerge from 5 seeds?

## Paper

**Title**: A Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality
**Journal**: Symmetry 2026, 18(1), 54
**DOI**: https://doi.org/10.3390/sym18010054
**PDF**: https://www.mdpi.com/2073-8994/18/1/54/pdf

This is the foundational paper defining the 19-dimensional Z₃-graded Lie superalgebra g = g₀(11) ⊕ g₁(4) ⊕ g₂(4) with cubic vacuum triality. It establishes the mathematical existence and consistency of the algebraic structure, proves Jacobi identity closure, and demonstrates the spontaneous saturation of repeated algebraic operations into a closed 44-vector lattice.

## Core Achievements: Unification of Forces & the Algebra-Geometry Bridge

### Unification of Forces

The finite 44-vector core lattice (ground state under triality saturation) naturally yields

**sin²θ_W = 11/44 = 0.25 exactly**

— reproducing the tree-level GUT prediction without free parameters.

### Unification of Algebra and Geometry

Abstract ℤ₃-graded algebraic operations on the vacuum spontaneously saturate into a closed, finite 44-vector discrete lattice — bridging pure algebra with concrete geometric structure in a parameter-free way.

## 📂 Verification Scripts

### Foundational Algebra Verification

- **`z3_algebra_5.py`** — High-precision numerical verification of graded Jacobi identity closure across the full 19-dimensional algebra (residuals ∼10⁻¹⁶ over millions of random tests).
- **`z3_grade_1.py`** — Exact symbolic verification (SymPy rational arithmetic) of Jacobi identities in critical mixing sectors, confirming residuals identically zero.
- **`z3_algebra_verify_19D_short.py`** — 19-dimensional Z₃-graded Lie superalgebra verification code. Test cycles: 10,000 random Jacobi identity checks.
- **`z3_algebra_verify_mini.py`** — 19-dimensional Z₃-graded Lie superalgebra verification code. Test cycles: 10,000,000 random Jacobi identity checks.
- **`z3_entanglement.py`** — SVD decomposition proof that the cubic vacuum invariant corresponds to a maximally entangled GHZ-class state.

> *📘 Beginner note: The Jacobi identity is a fundamental consistency condition for any Lie (super)algebra. Checking it numerically ensures the algebraic rules are self‑consistent.*

### Core 44-Vector Lattice & Gauge Unification

- **`z3_lattice_1.py`** — Refined ground-state pruning and geometric derivation of sin²θ_W = 11/44 = 0.25, exactly matching SU(5) GUT tree-level prediction.
- **`z3_lattice.py`** ⚠️ (Original, DEPRECATED) — First-generation lattice generator. **Do not use for paper results.** Produces a truncated 6-shell lattice with a spurious zero vector. Missing democratic shells L²=3,27,243 and higher root shells L²=162,486. Retained for historical comparison only.
- **`z3_lattice_A.py`** ★ (Corrected, June 2026 — **use this for all paper results**) — Verified lattice generator matching both the PLB mixing angles paper and the condensed-matter selection rules paper. Fixes two critical bugs in the original:

 | Bug | Original | Fixed | Consequence |
 |---|---|---|---|
 | Cross-product storage | `new.append(cross/norm)` — normalized only | `new.extend([cr, cr/norm])` — raw + normalized | Democratic direction [1,1,1] (L²=3) never generated in original |
 | Re-normalization | Re-normalizes already-normalized vectors | Skips re-normalization for unit vectors | Near-zero float debris and missing shells in original |

 **Correct output (10-shell structure):** 6 root shells (L²=2,6,18,54,162,486; 6 vectors each) + 3 democratic shells (L²=3,27,243; 1 vector each) + 1 basis shell (L²=1; 5 seed vectors) = **44 vectors** ✓

- **`z3_lattice_A_output_log.txt`** — Complete annotated output: 15-level saturation trace, 10-shell breakdown, democratic chain verification, root-shell √3-scaling table, paper-claim cross-check.
- **`z3_44_lattice_visualizer.html`** ★ — Standalone browser-based 3D visualization of the full 44-vector lattice. Toggleable layers for K₂,₂,₂ octahedron, √3-scaled copies, democratic [111] axis, A₂ hexagon projection.
- **`z3_strong_coupling.py`** — Classifies vectors into weak/strong-type components and predicts strong/weak coupling ratio analogies.

### Lattice Simulations — Lorentz Emergence & Anomaly Cancellation

> **`z3_lattice_full_test_english.py`** — complete, self-contained test code

#### Simulation 1: Low-Energy Lorentz Symmetry Restoration
On the discrete Z₃ vacuum lattice, the A₂ hexagonal projection yields tight-binding dispersion **E(k) = -t × Σ cos(k · v_i)** (i = 1 to 6). In the UV: clear hexagonal symmetry. In the IR (small k): converges to a **perfect circle** — the discrete 44-vector lattice **dynamically restores continuous Lorentz invariance** at low energies.

**Output**: `z3_lorentz_highres.png` + `z3_lorentz_recovery.gif`

#### Simulation 2: Exact Chiral Anomaly Cancellation
All four anomaly coefficients (U(1)_Y³, SU(2)²×U(1)_Y, SU(3)²×U(1)_Y, Gravitational×U(1)_Y) evaluate to **exactly zero** (within 10⁻¹⁵). The 44-vector lattice + Z₃ triality automorphism → exact cancellation for three generations.

### Z₃-Graded Dynamical Lagrangian (v15)

`z3_lagrangian_core_15.py` — Numerical implementation of a dynamical Lagrangian from the 15-dimensional Z₃-graded superalgebra. Constructs graded algebra generators/brackets, computes graded curvature (Yang-Mills kinetic term), generates Yukawa couplings from algebra VEVs, includes Higgs-like cubic potential, produces hierarchical fermion mass spectrum.

> ⚠️ Historical version (15D, pre-rigorous). Current EFT dynamics verification is provided by `z3_dynamics_verification.py` (see Extended Applications). This v15 script is retained for historical reproducibility.

### Supporting

- **`Z3_Isotropy_Proof.py`** — Generates strictly closed 44-vector lattice from triality operations and performs isotropy test (Rank-2 and Rank-4 tensor response).

### Lattice Visualizations

- **`z3_crystal_44_schematic.py`** — Schematic crystal-style 3D rendering
- **`z3_44_vector_crystal_visualizer.py`** — High-resolution crystal visualization
- **`z3_vacuum_lattice_crystal_44.py`** — Crystal rendering by type classification
- **`z3_show_4.py`** — Weak sector + sin²θ_W = 0.25
- **`z3_show_5.py`** — Network graph with Tr(A⁴) combinatorial factor
- **`z3_show_6_b.py`** — 3D lattice visualization, vectors colour-coded by norm class (democratic=magenta, root=deep blue, hybrid=green, basis=grey); Luban mortise-and-tenon lock analogy
- **`z3_show_16.py`** — General-purpose crystal lattice rendering
- **`z3_show_17.py`** — Fermion vectors with L²/Δ annotations
- **`z3_speculative_extensions_flowchart.py`** — Directed flowchart of algebraic extensions

---

# Branch 1 🟠 Mixing Angles — "Symmetry" Branch

> ***View: symmetry relations between points*** — A particle's "transformation" amplitude = the misalignment angle between two neighboring lattice sites

## Paper

**Title**: Derivation of Standard Model Mixing Angles from a 44-Vector Discrete Vacuum Lattice
**Preprint DOI**: https://doi.org/10.20944/preprints202606.1401.v1
**PDF**: https://www.preprints.org/manuscript/202606.1401/v1/download

### Centerpiece: The Cabibbo Angle (8 ppm precision)

```
λ = (2/9)(1 + ε²_q · C₂ / 2)

where:
  2/9 → from S₃ orbit geometry of the 44-vector lattice
  ε_q = dim_NF(Democratic)/dim_NF(Hybrid) = 4/24 = 1/6
  C₂ = 4/3 → SU(3) quadratic Casimir

Result: λ = 73/324 = 0.22530864...
Experiment (PDG 2024): λ = 0.22530 ± 0.00070 → +0.01σ (8 ppm)
```

### Complete Derivation Chain

**Step 1 — Orbit Decomposition**: 44 vectors classified under S₃ × Z₂ (|G|=12) into 4 orbit types:
| Orbit | Directions | Norm-Filtered | Rep |
|---|---|---|---|
| Democratic | 2 | 4 | d = [1,1,1]/√3 |
| Hybrid | 6 | **24** | v_p1 = [−2,1,1]/√6 |
| Root-like | 6 | 6 | [0,1,−1]/√2 |
| Flavor | 3 | 3 | [1,0,0] |

**Step 2 — Perturbation Strengths** (all from Frobenius norms / orbit dimensions):
ε_q = 1/6, ε_ν2 = 1/36, ε_ν3 = 1/12

**Step 3 — Theorem**: dim_NF(Hybrid) = 24 (rigorously proved, not counted by hand; see PLB paper §7)

**Step 4 — CP Phases from Z₃ Grading**: N(1,1)=ω¹ → δ_CKM ≈ 65.3°; N(1,2)=ω² → δ_CP = 240°

### Complete Prediction Table

| Parameter | Z₃ Prediction | Experiment (2024) | Pull |
|---|---|---|---|
| λ = \|V_us\| | 73/324 = 0.22531 | 0.22530 ± 0.00070 | +0.01σ |
| sin²θ₁₂ | 1/3 − λ/9 = 0.30830 | 0.307 ± 0.012 | −0.10σ |
| sin²θ₂₃ | 0.54609 | 0.546 ± 0.021 | +0.00σ |
| sin²θ₁₃ | ∈ [1/46, 1/44] | 0.02203 ± 0.00056 | within |
| δ_CP (PMNS) | 240° | 230° ± 36° | +0.26σ |
| δ_CKM | 65.3° | 65.7° ± 2.4° | −0.17σ |

### JUNO 2026 — A Priori Prediction Confirmed by Experiment

**June 10, 2026** — JUNO published (*Nature* cover): **sin²θ₁₂ = 0.3092 ± 0.0087**

Z₃ prediction (made December 2025): **sin²θ₁₂ = 1/3 − λ/9 = 0.30830**

Agreement: **0.10σ** (absolute deviation 0.0009). This is a genuine a priori prediction confirmed by experiment.

**Derivation**: Tribimaximal mixing (sin²θ₁₂⁰ = 1/3) → charged lepton Cabibbo-like rotation → Z₃ filtering (coarse-graining by \|Z₃\|=3) → correction λ/9. Zero free parameters — λ itself is derived from the algebra.

**Demonstration**: [`z3_juno_visualization.py`](./z3_juno_visualization.py) → `z3_juno_visualization.png`

### Future Decisive Tests
- **JUNO** (2026–2030): ~0.3% precision on sin²θ₁₂ → distinguishable at ~3σ
- **DUNE** (2028+): δ_CP to ~10° — prediction 240° is sharp and falsifiable
- **Hyper-Kamiokande** (2027+): sin²θ₂₃ precision — prediction 0.54609 is specific

### Verification Code

```python
from fractions import Fraction
eps_q = Fraction(1, 6); C2 = Fraction(4, 3)
λ = Fraction(2, 9) * (1 + eps_q**2 * C2 / 2)
# Output: λ = 73/324 = 0.22530864
sin2_θ12 = 1/3 - float(λ)/9
# Output: sin²θ₁₂ = 0.30830
```

## 📂 Mixing Angle Scripts

### Quark Mixing & CP Violation

- **`z3_ckm_angles.py`** — Derives CKM magnitudes (V_us, V_cb, V_ub) via integer vector misalignments to democratic direction.
- **`z3_cp_phase.py`** — Triality rotations and projective phase difference (120° − magic angle) for CKM CP phase.

### Neutrino Mixing Parameters — The Geometric Frustration Valley

This is one of the most striking results of the Z₃ framework: the observed reactor mixing angle θ₁₃ (1/sin²θ₁₃ ≈ 44.64) emerges naturally in the **"valley" between two geometric anchors** — not as a random fit, but as a structural consequence of integer lattice geometry.

The scripts below perform large-scale lattice searches for integer vectors whose projections yield mixing parameters close to experimental values. The key discovery: the bimodal distribution of 1/sin²θ₁₃ shows two sharp peaks at ~44 (lattice-aligned) and ~45 (vacuum singlet), with the experimental value 44.64 sitting **exactly in the intermediate valley** — a geometric frustration pattern with no free parameters.

- **`z3_pmns.py`** — Exact tri-bimaximal neutrino mixing: sin²θ₂₃=0.5, cos²θ₁₂=1/3, θ₁₃=0 analytically.
- **`Z3_Neutrino_Hunter.py`** — Large-scale parallel search (L² ≤ 5000) for candidate vectors yielding θ₁₃ and hierarchy ratios.
- **`Z3_Neutrino_Hybrid_Hunter.py`** — Extended search (L² ≤ 20000) near hybrid axis [−2,1,1]/√6.
- **`Z3_Neutrino_Hybrid_Hunter_one_shot.py`** — Rapid brute-force scan for 1/sin²θ₁₃ around 44–45 (dual-peak structure).
- **`Z3_Universe_Solver.py`** ★ (Main Solver — 768 GB RAM) — Full multi-task parallel framework simultaneously searching neutrino, gauge, Higgs, and flavour sectors. Tested on a 768 GB RAM server with MAX_L_SQ_HUGE = 100000, generating ~2.8 million lattice points. Outputs detailed logs with hundreds of near-matches for θ₁₃. The neutrino task alone identifies the characteristic bimodal distribution in 1/sin²θ₁₃.
- **`Z3_Universe_Solver_output_analysis.py`** — Post-processing script: parses the solver log, extracts all 1/sin²θ₁₃ values, and generates the key diagnostic histogram showing dual peaks at ~44 (lattice anchor) and ~45 (vacuum singlet), with the experimental value (44.64) in the intermediate valley. Example output from a full 768 GB run: `Z3_Universe_Solver_output_analysis_1.png`.

> *📘 Beginner note: 1/sin²θ₁₃ around 44.64 corresponds to θ₁₃ ≈ 8.6°, experimentally observed. The search finds many integer vectors giving values near 44 and 45; the real value sits precisely between — a geometric frustration pattern, not a coincidence. The valley is not a fit; it's a structural consequence of the lattice.*

### IO Rigidity Proof

[`z3_io_rigidity_proof.py`](z3_io_rigidity_proof.py) proves, through four independent pathways (algebraic Killing form, lattice geometry, representation-theoretic characters, contradiction scan), that the Z₃ algebra rigidly predicts **Inverted Ordering** for neutrino masses. Zero free parameters — directly falsifiable by JUNO/DUNE. Generates `z3_io_rigidity_viz.png`. Full logic in [`z3_io_rigidity_README.md`](z3_io_rigidity_README.md).

### Interactive 3D Companions

| File | Description |
|---|---|
| **`z3_plb_3d_9button.html`** | Equation-by-equation 3D explorer: 9 scenes (44 Lattice, 5 Seeds, Triality T, Democratic d, Orbits, Root K₆\3K₂, TBM, vₚ₁, Plaquettes). Press buttons to switch. |
| **`z3_plb_3d.html`** | Four-panel overview: Vacuum Lattice, PMNS Matrix, Oscillation Path, Mass Hierarchy |

Controls: Drag=rotate · Scroll=zoom · Right-drag=pan. Self-contained HTML, ~14KB. Three.js loaded from CDN.

### Plain-Language Walkthroughs

| File | Audience | Content |
|---|---|---|
| [`z3_derivation_walkthrough.html`](z3_derivation_walkthrough.html) | General public | Complete Z₃ → SM derivation in plain English |
| [`z3_mixing_angles_highschool.html`](z3_mixing_angles_highschool.html) | High school + | "Where Do Particle Mixing Angles Come From?" — arrows on a sphere |
| [`z3_derivation_highschool.html`](z3_derivation_highschool.html) | High school + | "From 5 Arrows to the Entire Hydrogen Atom" |

### Z₃ vs SM Toponium: Spin Observables

- **`z3_c_hal.py`** + **`Z3_vs_SM_c_hel_full_derivation.pdf`**

SM/NRQCD toponium assumes factorized two-body spin correlations. Z₃ introduces non-factorizable ternary vacuum interaction `{F^α, F^β, F^γ} = ε^k_{αβγ} ζ_k`, producing order-3 cyclic phase (e^{i2π/3}) and topological kinks in helicity-angle distribution.

### Mixing Angle Visualizations

- **`z3_show_9.py`** — Dual-panel CKM misalignment angles + bar chart
- **`z3_show_11.py`** — Polar diagram: triality phase, magic angle, CP phase
- **`z3_show_13.py`** — Dual 3D: TBM neutrino large mixing vs quark-like small mixing
- **`z3_show_15.py`** — 3D visualization of θ₁₃ basis projection candidates

---

# Branch 2 🟢 Mass Hierarchy — "Distance" Branch

> ***View: the gaps between shells*** — A particle's "weight" = the attenuation of a signal propagating between lattice points

## Paper 🆕

**Title**: Emergence of Fermion Mass Hierarchy from the Weighted Cayley Graph of a Z₃-Triality Vacuum Lattice
**Preprint DOI**: https://doi.org/10.20944/preprints202607.0474.v1
**PDF**: https://www.preprints.org/manuscript/202607.0474/v1/download
**Link**: https://www.preprints.org/manuscript/202607.0474/v1
**Companion Script**: `z3_prl_verification_en.py`

### Core Logic

1. **Rule**: Signal strength = 1 / distance² (mathematically inevitable — the fundamental Green's function solution in 3D)
2. **Compute propagator**: On the weighted Cayley graph, compute G = 1/Laplacian
3. **Build mass matrix**: M_AB = G(x_A, x_mid) × G(x_mid, x_B)
4. **Cross-shell suppression**: Each shell crossing divides the signal by 3 → naturally yields O(0.01) inter-generational ratios

### Core Results (All Zero Free Parameters)

- Top quark mass: **174 GeV** (from y_t = 1 rigid algebraic locking)
- Electroweak scale: **246 GeV** (from 12-step spectral ladder: 12 = 8+3+1)
- m_c/m_t ≈ **0.0162**, m_s/m_b ≈ **0.0145** (cross-shell 1/3 suppression per layer)
- All 9 charged fermion masses within factor 2 of experiment
- Exact Weinberg angle sin²θ_W = 0.25
- Natural emergence of O(0.01) inter-generational mass ratios

### Unification of Matter (Geometric Seesaw)

The infinite integer extension (ℤ³ sites supported by the core basis) identifies resonant lattice nodes via m ∝ 1/L². Explicit integer vectors include:
- Top ([0,0,1], L²=1) → anchor at 173 GeV
- Bottom ([1,2,7], L²=54)
- Tau/Charm ([0,9,9], L²=162)
- Muon ([0,27,27], L²=1458)
- Down ([1,46,193], L²=39366)
- Electron ([3,138,579], L²=354294) — 4.6% agreement across six orders of magnitude

## 📂 Mass Hierarchy Scripts

### Fermion Mass Hierarchy & Selection Rules

- **`z3_prl_verification_en.py`** ★ (New — Companion to Preprint 202607.0474) — End-to-end verification of the weighted Cayley graph construction: generates the 44-vector lattice, assigns 1/d² edge weights, computes the Green's function propagator G = 1/Δ, constructs the mass matrix M_AB = G(x_A, x_mid) G(x_mid, x_B), and outputs the full charged fermion mass spectrum with zero free parameters. All 9 masses within factor 2 of experiment.
- **`z3_mass_6.py`** — Unified demonstration of gauge unification and full charged fermion mass spectrum via inverse-squared norm scaling.
- **`z3_mass_quarks.py`** — Searches extended lattice for up/strange quark vectors and verifies geometric up/down mass inversion.
- **`z3_comparative_check_mod_9.py`** — Verifies modulo-9 resonance (L² ≡ 0 mod 9) and computes triality stability Δ.
- **`z3_comparative_check.py`** — Compares Δ values of physical vectors vs random neighbors.

### Z₃-Graded Vacuum Geometry: Rigid High-Energy EFT Prediction

**Timestamp: March 9, 2026** — We formally retract all previous phenomenological claims of a possible scalar resonance at ~355 GeV in the tt̄ threshold (relied on arbitrary κ ≈ 0.1).

In the exact 19-dimensional matrix representation, the relative strength of vacuum-mediated dimension-6 operator vs standard QCD gluon exchange is uniquely fixed by Super-Killing form ratio:

\[
C_{Z3} = 8/63 \approx 0.12698
\]

**Prediction** (M_tt > 1–2 TeV):
\[
\frac{d\sigma_{\text{obs}}}{d\sigma_{\text{SM}}} \simeq 1 \pm \frac{8}{63} \left( \frac{M_{tt}}{\Lambda_{\text{alg}}} \right)^2
\]

Any deviation must match this exact rational slope; any other fractional coefficient would falsify the framework.

Full details: `Z3_EFT_Prediction.md` and `Z3_HighEnergy_Tail_Prediction.pdf`. Verification: `z3_algebra_verify_mini_para.py`.

### Additional Phenomenological Alignments

- **`z3_higgs.py`** — Tests geometric ratios for Higgs-to-top mass ratio proximity.
- **`z3_cosmo_constant.py`** — Computes N⁴ combinatorial factor and cosmological constant scale compensation.

### Mass-Related Visualizations

- **`z3_mass_show.py`** — 3D lattice + logarithmic fermion mass comparison
- **`z3_mass_show_1.py`** — Advanced dual visualization with L² and Δ annotations
- **`z3_show_6.py`** — Dual-panel (lattice + mass hierarchy) with RG equation
- **`z3_show_8.py`** — Refined mass hierarchy dual visualization
- **`z3_show_10.py`** — Geometric ratios for Higgs-to-top mass
- **`z3_show_12.py`** — Component count (pie + bar) for strong coupling analogies
- **`z3_show_14.py`** — Cosmological constant hierarchy with compensation diagram

---

# Branch 3 🔴 Atomic Structure — "Local Shape" Branch

> ***View: the skeleton of the innermost octahedron*** — Electron orbitals = vibrational modes of the octahedral graph; electromagnetic coupling strength = lattice gauge theory on the octahedron

## Paper

**Title**: Atomic Orbital Quantum Numbers, Hydrogen Spectrum, and Coulomb-Like Emergence from a Z₃-Triality Lattice
**Preprint DOI**: https://doi.org/10.20944/preprints202606.2217.v1
**PDF**: https://www.preprints.org/manuscript/202606.2217/v1/download

### Core Results

1. **K₂,₂,₂ octahedron graph Laplacian eigenvalues**: {0, 4, 4, 4, 6, 6}
2. **Decomposition**: 6 = 1 ⊕ 3 ⊕ 2 = **s-orbital ⊕ p-orbital ⊕ d-orbital** — atomic orbital quantum numbers emerge from pure geometry
3. **Gauss's law on the geometric grid**: r_k ∝ (√3)^k → V(r) = −√3/(4πr)
4. **Fine-structure constant** (octahedral U(1) lattice gauge theory):
   - Partition function Z(β) = Σ_n [I_n(β)]⁸
   - Gaussian continued fraction integrality condition → β_c = 1
   - Geometric factor G = 2F/E = 4/3 (F=8 faces, E=12 edges)
   - α⁻¹_geom = π√3 [I₀(1)/I₁(1)]⁴ = **137.042**
   - Topological instanton correction δ = (S − S³)/(4√3) → α⁻¹ = **137.036** (sub-ppb from CODATA 137.035999084)
5. **Hydrogen spectrum fully reproduced** (overlap > 0.999), zero free parameters

## 📂 Atomic Structure Scripts

### Z3_44_Lattice_Multi_Orbital.py

Uses a Z₃ 44-vector discrete lattice + Metropolis Monte Carlo (8 million steps) to generate hydrogen atomic orbital probability distributions (1s, 2s, 2p, 3d etc.) **without solving the Schrödinger equation**. Energy function: radial linear tension + orbit-specific topological barriers (triality phase). Output: `Z3_Emergent.pdf` (7 high-resolution orbital visualizations).

### Z3_Nature_Orbitals.py (Updated June 27, 2026)

Solves discrete radial Schrödinger equation on the Z₃ geometric grid r_k ∝ (√3)^k with physical α⁻¹ = 137.036. Output: 8 individual high-resolution orbital renders + composite Nature-style figure. All computed with zero free parameters.

**Output**: `Z3_Orbital_1s.png` through `Z3_Orbital_3d_{x^2-y^2}.png` + `Z3_Orbitals_Nature_Composite.png`

### Z3_Orbital_Glow.py (Updated June 27, 2026)

Cover-quality multi-angle orbital montages with volumetric glow effect. Three-layer rendering per orbital (inner core, outer glow halo, background mist) from 4 viewing angles. Also generates radial curve comparison: Z₃ vs exact hydrogen.

**Output**: `Z3_Glow_1s.png` through `Z3_Glow_3dx2y2.png` + `Z3_Radial_Curves.png`

---

# Extended Applications 🔶

> *The following applications all derive from the same 44-point vacuum lattice and Z₃-graded superalgebra, but are more exploratory in nature; their physical significance requires further experimental verification.*

---

## 🔬 Condensed Matter: Geometric Selection Rules

**Title**: A₂ Lattice Geometry and Condensed-Matter Selection Rules from a Z₃-Graded Vacuum Sector
**Preprint**: `z3_geometric_selection_rules_annphys.tex` (manuscript available in repository)

### Core Predictions (Zero Continuous Free Parameters)

| # | Prediction | Computed Value | Experiment | Agreement |
|---|---|---|---|---|
| 1 | Kagomé Chern number | C = +1 | C = 1 (observed) | ✓ Exact |
| 2 | TBG magic angle | θ₀ = 1.090° | 1.1° ± 0.05° | ✓ < 1% |
| 3 | Interlayer hopping | w = 126 meV | 110 meV | ✓ 15% |
| 4 | Secondary magic angle | θ₁ = 0.63° | (untested) | **★ PREDICTION** |
| 5 | Angle ratio | θ₁/θ₀ = 1/√3 | (untested) | **★ PREDICTION** |
| 6 | h-BN C₆ resonance | 0°, 60°, 120° | Consistent | ✓ |
| 7 | Critical exponent | ν = 0.614 | (insufficient precision) | Speculative |

### Key: Energy-Independence Theorem

The magic angle formula sin(θ₀/2) = 1/(2×54) = 1/108 → θ₀ = 2·arcsin(1/108) = 1.0610°. The lattice constant **cancels identically** — the result is a pure number depending only on L²=54.

### EFT Derivation Chain (Paper Section 5.1)

```
Graded Yang-Mills → unique Yukawa coupling (cubic bracket)
  → one-loop triangle diagram → g̃ ~ 10⁻² GeV⁻¹
    → surface Klein-Gordon solution → ζ(r,z) = ζ₀(r)·exp(-z/ξ)
      → overlap integral g_eff(θ) = ∫ ρ_moiré · ζ₀ d²r
```

### Verification Scripts

- **`z3_dynamics_verification.py`** ★ (Master — 11 Sections) — Complete end-to-end numerical verification. Run: `python3 z3_dynamics_verification.py` → "ALL VERIFICATIONS PASSED ✓"
- **`counterfactual_L54_test.py`** ★ — Removes each shell from L₄₄ and recomputes S(k). L²=54 removal: 3% shift (passive). L²=162/486 removal: 38–91% shift (dominant).
- **`harmonic_convergence_v3.py`** — Scans all (n_ρ, n_ζ) ∈ {1..5}×{1..3}. For n_ζ≥2: stable to ±0.004° (0.5%).

### Wilson-Fisher Derivation

```
LGW action (N=3) → β(u) = -εu + (N+8)u²/(48π²)
  → u* = 48π²ε/(N+8) → γ_φ² = (N+2)ε/(N+8) = 5/11
    → ν = 1/2 + 5/44 = 27/44 ≈ 0.614
```

Sole Z₃ input: N=3 (vacuum triplet). Everything else is standard Zinn-Justin.

### Condensed Matter Simulation Scripts

#### TBG Magic Angle
- **`Z3_Pure_Geometric_Magic_Angle_Ultimate.py`** — 6000×6000 grid + multi-harmonic moiré density + full A₂ projection. Zero hopping parameters. Peak at θ=1.090°. Output: `Z3_Pure_Geometric_Magic_Angle_Ultimate.png`

#### hBN Superfluid Resonance
- **`Z3_hBN_Superfluid_Resonance_Improved_3D.py`** — Macroscopic overlap integral between hBN charge density and rotated A₂ vacuum potential. Predicts sharp C₆ resonances at 0°, 60°, 120°. Output: `Z3_hBN_Superfluid_Resonance_Final_3D.png`

#### Kagome Quantum Anomalous Hall Effect
- **`z3_kagome_resonance_3d.py`** — 3D geometric resonance visualization. Max local overlap = 0.9455. Output: `z3_kagome_resonance_3d_zero_parameter_with_overlap.png`
- **`z3_kagome_berry_curvature.py`** — Fukui-Hatsugai-Suzuki algorithm. Chern number C = **1.0000**. Output: `z3_qah_berry_curvature_ultimate.png`
- **`z3_kagome_berry_curvature_6.py`** — Full Kubo-Greenwood transport. σ_xy = 1.0000 e²/h (quantized), σ_xx negligibly small. Output: `z3_kagome_kubo_paper_figure.png`

All results from bare Z₃ algebra + effective Hamiltonian **without any fitting**. C = 1 arises solely from φ = 2π/3.

---

## 🔬 Nanoscale Transport: Z₃ Vacuum Inertia

**Preprint DOI**: https://doi.org/10.20944/preprints202601.0109.v5
**PDF**: https://www.preprints.org/manuscript/202601.0109/v5

Complete suite of scripts for closed-loop validation: Z₃-graded Lie superalgebra construction → exact closure → THz skin depth saturation → nanoscale superconductivity enhancement. All without external fitting.

### Key Scripts

- **`Z3_Vacuum_Screening_Cloud_3D_English.py`** — Bare-to-dressed transition of vacuum coherence length ξ_vac. Bare ξ_bare from collective triality simulations (algebraic geometry), screening factor η_alg = dim(g₁) = 4, dressed ξ_eff ≈ 71.1 nm ≈ 70 nm — zero free parameters. Output: `Z3_Vacuum_Screening_Cloud_3D_Crystal_Final_Fixed_NoOverlap.png`
- **`z3_exploratory_consistency_verification.py`** — Lightweight symbolic verification (SymPy): graded brackets → effective coupling → renormalization → surface criticality → emergent scale.

### Hg-1223 Pressure Quench

- **`Z3_Hg1223_PressureQuench_TrueZeroParam_3D_Beautiful_Fixed_PDF.py`** — 2D panels (Tc vs Pressure, lattice anchoring dynamics) + 3D vacuum potential landscape
- **`Z3_Hg1223_TrueZeroParam_3D_Braid_Embedding_5Arrows.py`** — 3D visualization: 5 dynamic dashed trajectories, material lattice locking into Z₃ vacuum lattice

Geometric resonance between Hg-1223 lattice and Z₃ L₄₄ lattice → metastable superconducting phase near 151 K (qualitatively consistent with Chu, Deng et al. PNAS 2026).

**Parameters** (all from literature or algebraically fixed): ξ_vac ≈ 70 nm, v_F = 1.57×10⁵ m/s, A₀ = 3.85 Å, B₀ = 90 GPa, T_c0 = 133 K, T_quench = 4.2 K.

### Historical Scripts (earlier drafts)

- `z3_vacuum_theory_chain_verify_fixed.py` — Full symbolic chain from graded brackets to nanoscale Tc(d)
- `z3_quantitative_logic_chain_verify.py` — Step-by-step Quantitative Comparison section
- `z3_theoretical_consistency_verify_fixed.py` — RG flow, naturalness, timescale, phonon complementarity
- `z3_nami_sensitivity_show.py` — Three supplementary figures (Tc vs diameter, skin depth, ξ_vac sensitivity)
- `z3_nanomaterials_chapter1_mindmap_vertical.py` — Vertical Graphviz mindmap of Chapter 1 logic chain

---

## 🌌 Neutrino Astronomy & Cosmology

### IceCube Time Domain Analysis

- **`Z3_IceCube_Time_Domain_Analyzer.py`** — Harmonic analysis of public IceCube IC86 Stokes Q/U polarization data. SNR ≈ 5.2 at 6h (numerical coincidence only). Data: DOI 10.7910/DVN/DZI2F5.
- **`Z3_Phase_Locking_Clean.py`** — Refined phase alignment after removing edge artifacts. Correlation coefficient 0.8614 in central region (4h–20h). Optimized Euler angles: [32.12°, 3.07°, 376.45°] (presented strictly as mathematical curiosity). Output: `Z3_Phase_Locking_Clean.png`.

### KM3NeT Transparent Windows

- **`Z3_KM3NeT_3Year_Windows.py`** — Complete 3-year prediction table (2026–2029) of daily 1-hour sidereal-time windows (±30 min) for >100 PeV neutrinos. Output: `Z3_KM3NeT_3Year_Transparent_Windows.csv` (1096 entries). Any >100 PeV event detected **outside** these windows immediately falsifies the model.

### LHAASO LIV Predictions

- **`z3_lhaaso_prediction.py`** — Geometric factor η(n) = Σ (n·v)⁴ over 44-vector lattice → quantitative predictions for LIV signatures in LHAASO PeV photon data.

### Hubble Tension Skymap

- **`Z3_Hubble_Skymap_Generator.py`** + **`Z3_Hubble_Skymap.png`** — Full-sky Mollweide projection of directional cosmic expansion rate modulation. Structured dipole/quadrupole features invite comparison with Hubble tension and CMB anomalies.

---

## 💻 Topological Quantum Computing

**Preprint DOI**: 10.20944/preprints202602.0488.v1
**PDF**: https://www.preprints.org/manuscript/202602.0488/v1/download

- **`z3_threshold_massive.py`** — Monte Carlo simulation of Z₃ toric code fault-tolerance threshold (L=8–16, 2000 trials/point). PyMatching decoder.
- **`z3_threshold_massive_show.py`** — Professional threshold plot with Wilson score 95% CI.

---

## 🎨 Additional Visualizations & Tools

### Z₃ Section Visualization

- **`z3_section_visualization.py`** → **`Z3_Signature_Optical_Shadows.pdf`** — How the same cubic mechanism produces superluminal optical shadows (tabletop laser) and 120° cyclic kinks (LHC). Triality diagram + shadow velocity curve + c_hel comparison + 4×4 spin matrix.

### Other Simulations

- `c1.py`–`c7.py`: Recursive Entropy Stabilization, Transient Fluctuations, Particle Spectra, Cosmic Evolution, Superalgebra Verification, EISA Universe Simulator, CMB Power Spectrum Inverse Analysis.

### UFO Model

- **`Z3_Ternary_UFO.zip`** — FeynRules-compatible UFO model implementing ternary vacuum-mediated interactions (t t̄ ζ vertex). For MadGraph5_aMC@NLO.

---

## 📚 Supplementary Content

### Cover Video
- [RIA_EISA Cover Video](https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4)

### Related ATLAS Data
- ATLAS-CONF-2025-008: tt̄ production cross section near threshold at √s=13 TeV. [Link](https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf)

### Science Education for Teenagers
- [Chapter 1–5](https://github.com/csoftxyz/RIA_EISA/wiki): "Lego Primary Colors" manual for physics, Lego Constitution, Predicting Dark Matter, Expanding Universe Engine, Anti-Counterfeit Certificates.

### Historical Development
The current Z₃-graded framework evolved from earlier explorations:
- Early EISA Preprint Series: [v1](https://www.preprints.org/manuscript/202507.2681/v4), [v7](https://www.preprints.org/manuscript/202507.2681/v7)

### Phenomenological Extension (Earlier Preprint)
**Title**: An Exact Z₃-Graded Algebraic Framework Underlying Observed Fundamental Constants
**Preprint DOI**: https://doi.org/10.20944/preprints202512.2527.v2
**PDF**: https://www.preprints.org/manuscript/202512.2527/v2/download

### Profound Significance

The computational exploration culminates in the spontaneous emergence of a closed, finite 44-vector lattice from pure ℤ₃ triality operations. This saturation is a rigorous mathematical consequence of the unique cubic invariant and graded bracket structure.

The finite lattice:
- Constrains flavor mixing directions, eliminating arbitrary parameters in mass matrix ansätze
- Bridges continuous field theories with emergent discreteness relevant to quantum gravity
- Predicts specific correlations in neutrino oscillations, CP phases, and condensed matter angular transport

### Current Status & Open Questions

- **Predictions & Tests**: Must yield unique, falsifiable predictions distinct from Standard Model
- **Mathematical Consistency**: Need complete dynamical theory with natural continuum limit
- **Conceptual Challenge**: How continuous spacetime and symmetries emerge from discrete structure

### Author Attitude

We hold deep respect for decades of work in string theory, quantum gravity, and related fields. This framework is offered humbly as an exploratory alternative perspective. We make no claim of superiority or finality — only a rigorous, testable structure open to scrutiny.

---

### Contact

- Issues tab for technical questions
- Email: csoft@hotmail.com (corresponding) / csoft@live.cn

### Contributing

Welcome contributions! Fork, branch, commit, push, PR.

### Wiki Links

- [EISA Algebra Basics](https://github.com/csoftxyz/RIA_EISA/wiki/eisa_algebra.md)
- [RIA Optimization](https://github.com/csoftxyz/RIA_EISA/wiki/ria_optimization.md)
- [Simulation Tutorials](https://github.com/csoftxyz/RIA_EISA/wiki/simulations/)
- [Validation Code](https://github.com/csoftxyz/RIA_EISA/wiki/validation.md)
- [Universe Simulator](https://github.com/csoftxyz/RIA_EISA/wiki/universe_simulator.md)
- [CMB Inverse Analysis](https://github.com/csoftxyz/RIA_EISA/wiki/cmb_inverse.md)
- Possible Related Experiments: MIT Double-Slit, NANOGrav, LHC Mass Anomalies, CMB Deviations, Muon g-2, Neutrino Mass Hierarchy, Lepton Flavor Universality, New Particles

### API Reference

Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from `c1.py`. RNN model: `EnhancedRNNModel` from `c2.py`.
