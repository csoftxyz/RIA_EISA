# RIA-EISA Simulation Repository

**Yuxuan Zhang**<sup>a,b</sup>, **Weitong Hu**<sup>c,\*</sup>  
<sup>a</sup> College of Communication Engineering, Jilin University, Changchun, China  
<sup>b</sup> csoft@live.cn  
<sup>c</sup> Aviation University of Air Force, Changchun, China (Corresponding Author)  
<sup>c</sup> csoft@hotmail.com

---

## Overview

This repository contains the complete simulation and verification suite for the **Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality** series. The framework is a finite-dimensional (19D: 12+4+3) ℤ₃-graded algebraic structure from which Standard Model parameters, gravitational constant, cosmological constant, black-hole entropy scaling, and vacuum entanglement properties emerge as representation-theoretic invariants — with **zero free parameters**.

## Z₃ Discrete Vacuum Geometry: A Computational Framework Exploring Unification of Force, Matter, and Algebra

This repository presents a computational exploration of a discrete algebraic model based on ℤ₃ triality symmetry. The framework constructs a self-consistent "virtual universe" from pure mathematical operations on a graded Lie superalgebra vacuum sector. This model spontaneously produces structures with striking numerical alignments to observed particle physics parameters—gauge unification, fermion mass hierarchies, and emergent discrete geometry.

Whether these alignments reflect deep properties of the actual universe is an open scientific question, to be tested through further theoretical development, new predictions, and experimental verification. The results so far are highly suggestive and warrant continued investigation.

### Core Achievements of the Model

1. **Unification of Forces**  
   The finite 44-vector core lattice (ground state under triality saturation) naturally yields  
   **sin²θ_W = 11/44 = 0.25 exactly**  
   — reproducing the tree-level GUT prediction without free parameters.

2. **Unification of Matter**  
   The infinite integer extension (ℤ³ sites supported by the core basis) identifies resonant lattice nodes corresponding to the charged fermion mass scales via a geometric seesaw mechanism (m ∝ 1/L²). Explicit integer vectors include:
   - Top ([0,0,1], L²=1)
   - Bottom ([1,2,7], L²=54)
   - Tau/Charm ([0,9,9], L²=162)
   - Muon ([0,27,27], L²=1458)
   - Down ([1,46,193], L²=39366)
   - Electron ([3,138,579], L²=354294) — 4.6% agreement across six orders of magnitude.

3. **Unification of Algebra and Geometry**  
   Abstract ℤ₃-graded algebraic operations on the vacuum spontaneously saturate into a closed, finite 44-vector discrete lattice—bridging pure algebra with concrete geometric structure in a parameter-free way.

### Publications & Preprints

Our Z₃-graded algebraic framework spans high-energy unification, particle physics, cosmology, and low-energy condensed matter phenomena.

1. **Algebraic Foundation (Published)**  
   **Title**: A Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality  
   **Journal**: Symmetry 2026, 18(1), 54  
   **DOI**: https://doi.org/10.3390/sym18010054  
   **PDF**: https://www.mdpi.com/2073-8994/18/1/54/pdf

2. **Phenomenological Extension: Fundamental Constants & Predictions**  
   **Title**: An Exact Z₃-Graded Algebraic Framework Underlying Observed Fundamental Constants  
   **Preprint DOI**: https://doi.org/10.20944/preprints202512.2527.v2  
   **PDF**: https://www.preprints.org/manuscript/202512.2527/v2/download

3. **Low-Energy Application: Nanoscale Transport**  
   **Title**: Z₃ Vacuum Inertia in Nanoscale Transport  
   **Preprint DOI**: https://doi.org/10.20944/preprints202601.0109.v3  
   **PDF**: https://www.preprints.org/manuscript/202601.0109/v3

4. **Flavor Sector Application**  
   **Title**: Discrete Vacuum Geometry Predicts the Hierarchical Mass Spectrum of Standard Model Fermions  
   **Preprint ID**: https://www.preprints.org/manuscript/202601.0914  
   Using a formal geometric scaling \(m \propto L^{-2}\) (where \(L\) is the Euclidean norm of selected lattice vectors) anchored to the top-quark mass (173 GeV), the framework yields the following **curious numerical proximities**:
   - Electron: ~0.49 MeV (4.6% agreement)
   - Muon: ~118 MeV (12% agreement)
   - Qualitative up/down quark mass inversion (\(m_u < m_d\))
   - Exact Weinberg angle \(\sin^2 \theta_W = 0.25\)
   - Higgs-related scale ratio ~0.727 (0.3%)
   - Strong/weak coupling ratio ≈0.95 (near equipartition)
   - CKM CP phase ≈65.3° (5% agreement)
   - Neutrino mixing: exact 45° (maximal atmospheric) and \(\cos^2 \theta_{12} = 1/3\) (exact tri-bimaximal solar)

5. **A Z₃-Graded Topological Quantum Computing Architecture Based on the Discrete 44-Vector Vacuum Lattice**  
   **Preprint DOI**: 10.20944/preprints202602.0488.v1  
   **PDF**: https://www.preprints.org/manuscript/202602.0488/v1

**Project Highlight**:  
With the algebraic foundation published in *Symmetry* (2026, 18, 54) and ongoing preprints/submissions extending the framework across scales—from high-energy unification (fundamental constants, gauge couplings, cosmological constant) through low-energy quantum coherence in nanoscale systems to fault-tolerant **Z₃-graded topological quantum computing architectures** based on the discrete 44-vector vacuum lattice—our Z₃-graded framework is forming a preliminary closed loop that connects fundamental physics to emergent phenomena in condensed matter and quantum information processing.

### Core Verification Scripts (Self-Contained & Reproducible)

All scripts are designed for immediate execution (Python 3 + NumPy/SymPy). They rigorously validate the algebraic closure, emergent lattice, and quantitative predictions. The full repository includes 3D visualizations and interactive notebooks.

### Z₃-Graded Algebraic Framework: Core Scripts & Standard Model Predictions

## 📂 Repository Structure & Script Categories

### 1. Foundational Algebra Verification

- **`z3_algebra_5.py`** — High-precision numerical verification of graded Jacobi identity closure across the full 19-dimensional algebra (residuals ∼10⁻¹⁶ over millions of random tests). Establishes mathematical closure of the Z₃-graded superalgebra.
- **`z3_grade_1.py`** — Exact symbolic verification (SymPy rational arithmetic) of Jacobi identities in critical mixing sectors, confirming residuals identically zero.
- **`z3_algebra_verify_19D_short.py`** — the 19-dimensional \(\mathbb{Z}_3\)-graded Lie superalgebra verification code. Test cycles: 10,000 random Jacobi identity checks.
- **`z3_algebra_verify_mini.py`** — the 19-dimensional \(\mathbb{Z}_3\)-graded Lie superalgebra verification code. Test cycles: 10,000,000 random Jacobi identity checks.
- **`z3_entanglement.py`** — SVD decomposition proof that the cubic vacuum invariant corresponds to a maximally entangled GHZ-class state.

### 2. Core 44-Vector Lattice & Gauge Unification

- **`z3_lattice_1.py`** (Core – Newly Added) — Refined ground-state pruning and geometric derivation of sin²θ_W = 11/44 = 0.25, exactly matching SU(5) GUT tree-level prediction.
- **`z3_lattice.py`** (Core) — Generation and analysis of the emergent finite 44-vector ℤ₃-invariant lattice from vacuum triality.
- **`z3_mass_6.py`** (Core Script) — Unified demonstration of gauge unification and full charged fermion mass spectrum via inverse-squared norm scaling.
- **`z3_strong_coupling.py`** — Classifies vectors into weak/strong-type components and predicts strong/weak coupling ratio analogies.

### 3. Fermion Mass Hierarchy & Selection Rules

- **`z3_mass_quarks.py`** — Searches extended lattice for up/strange quark vectors and verifies geometric up/down mass inversion.
- **`z3_comparative_check_mod_9.py`** — Verifies modulo-9 resonance (L² ≡ 0 mod 9) and computes triality stability Δ for fermion vectors.
- **`z3_comparative_check.py`** — Compares Δ values of physical vectors vs random neighbors to support selection rules.

### 4. Quark Mixing & CP Violation

- **`z3_ckm_angles.py`** — Derives CKM magnitudes (V_us, V_cb, V_ub) via integer vector misalignments to democratic direction.
- **`z3_cp_phase.py`** — Explores triality rotations and projective phase difference (120° − magic angle) for CKM CP phase approximation.

---

### Z₃-Graded Vacuum Geometry: Rigid High-Energy EFT Prediction

**Timestamp: March 9, 2026**

We formally retract all previous phenomenological claims of a possible scalar resonance at ~355 GeV in the tt̄ threshold, which relied on an arbitrary coupling κ ≈ 0.1 and lacked algebraic justification. Current ATLAS-CONF-2025-008 data firmly anchor the peak at ~345 GeV (χ²/dof ≈ 1.05), consistent with NRQCD.

We now restrict physical predictions of the Z₃ framework to the decoupled high-energy EFT regime (M_tt ≫ 2 m_t). In the exact 19-dimensional matrix representation, the relative strength of the vacuum-mediated dimension-6 operator versus standard QCD gluon exchange is uniquely fixed by the ratio of Super-Killing forms (invariant trace norms) between the vacuum generators ζᵏ and gauge generators Bᵃ. Direct computation yields the rigid algebraic constant C_Z3 = 8/63 ≈ 0.12698.

We predict that in the high-mass tail (M_tt > 1–2 TeV), the differential cross-section ratio must asymptotically follow  

\[
\frac{d\sigma_{\text{obs}}}{d\sigma_{\text{SM}}} \simeq 1 \pm \frac{8}{63} \left( \frac{M_{tt}}{\Lambda_{\text{alg}}} \right)^2
\]

with zero free parameters. Any deviation in future ATLAS/CMS global SMEFT fits of the high-energy tail must match this exact rational slope to be consistent with the Z₃ vacuum geometry; any other fractional coefficient would falsify the framework.

Full details, code, and verification are given in `Z3_EFT_Prediction.md` and `Z3_HighEnergy_Tail_Prediction.pdf`.

- **`(z3_algebra_verify_mini_para.py)`** — verification script.

### Z₃ vs SM Toponium: Spin Observables Comparison

- **`z3_c_hal.py`** – Python script that generates the visualizations  
- **`Z3_vs_SM_c_hel_full_derivation.pdf`** – Final 2-page PDF output (curve + full mathematical derivation)

**Key Theoretical Difference**:  
The Standard Model / NRQCD toponium prediction assumes **factorized two-body spin correlations** (purely real, symmetric spin-singlet matrix).  
In contrast, the **Z₃ graded Lie superalgebra** introduces a **non-factorizable ternary vacuum interaction** through the cubic bracket  
`{F^α, F^β, F^γ} = ε^k_{αβγ} ζ_k`,  
producing a characteristic **order-3 cyclic phase** (`e^{i2π/3}`) and topological kinks in the helicity-angle (`c_hel`) distribution.  
This visualization directly compares the two frameworks and demonstrates how Z₃ naturally generates observable spin asymmetries that cannot be reproduced by any adjustment of NRQCD higher-order terms.

---

### Z3_44_Lattice_Multi_Orbital.py (Updated April 10, 2026)

This script uses a Z₃ 44-vector discrete lattice together with a Metropolis Monte Carlo random walk (8 million steps) to statistically generate probability distributions of hydrogen atomic orbitals (1s, 2s, 2pₓ/2pᵧ/2p_z, 3d etc.) without solving the Schrödinger equation or employing continuous wave functions. The energy function combines a radial linear tension term with orbit-specific topological barriers, motivated by triality phase considerations. It serves as a numerical demonstration of emergent quantum orbital shapes from discrete vacuum geometry. Outputs include 7 high-resolution orbital visualizations compiled in `Z3_Emergent.pdf`. This is a phenomenological numerical exploration within the Z₃ Cubic Vacuum Triality framework.

### 5. Neutrino Mixing Parameters

This directory contains tools for exploring the geometric origins of PMNS mixing angles and neutrino mass ratios within the Z₃ vacuum framework. The scripts perform large-scale lattice searches for integer vectors that yield mixing parameters close to experimental values, with particular emphasis on the observed θ₁₃ (1/sin²θ₁₃ ≈ 44.64) emerging in the "valley" between the two natural geometric anchors at 44 (lattice-aligned) and 45 (vacuum singlet).

- **`z3_pmns.py`**  
  Computes exact tri-bimaximal neutrino mixing using symmetric projections onto the Z₃-graded structure. Reproduces the classic values sin²θ₂₃ = 0.5, cos²θ₁₂ = 1/3, and θ₁₃ = 0 analytically.

- **`Z3_Neutrino_Hunter.py`**  
  Large-scale parallel search (L² ≤ 5000) for candidate vectors yielding θ₁₃ and neutrino mass hierarchy ratios. Uses multiprocessing to scan the fundamental domain of the integer lattice.

- **`Z3_Neutrino_Hybrid_Hunter.py`**  
  Extended search (L² ≤ 20000) focused on projections near the hybrid axis [-2, 1, 1]/√6, which provides refined approximations to the observed θ₁₃.

- **`Z3_Neutrino_Hybrid_Hunter_one_shot.py`**  
  Rapid brute-force one-shot scan optimised for near-integer values of 1/sin²θ₁₃ around 44–45. Designed for quick exploration and verification of the dual-peak structure reported in the published works.

- **`Z3_Universe_Solver.py`** (main solver)  
  Full multi-task parallel framework that simultaneously searches neutrino, gauge, Higgs, and flavour sectors. Designed for high-memory environments (tested on a 768 GB RAM server with MAX_L_SQ_HUGE = 100000, generating ~2.8 million lattice points). Outputs detailed logs of geometric matches, including hundreds of near-matches for θ₁₃. The neutrino task alone identifies the characteristic bimodal distribution in 1/sin²θ₁₃.

- **`Z3_Universe_Solver_output_analysis.py`**  
  Post-processing script that parses the solver log file (`Z3_Universe_Solver_output.txt`), extracts all reported 1/sin²θ₁₃ values, and generates the key diagnostic histogram showing dual peaks at ~44 (lattice anchor) and ~45 (vacuum singlet), with the experimental value (44.64) in the intermediate valley. Example output (from a full 768 GB run) is included in the repository as `Z3_Universe_Solver_output_analysis_1.png`.

### 6. Additional Phenomenological Alignments

- **`z3_higgs.py`** — Tests geometric ratios for Higgs-to-top mass ratio proximity.
- **`z3_cosmo_constant.py`** — Computes N⁴ combinatorial factor and demonstrates cosmological constant scale compensation.

### 7. Visualizations and Lattice Renderings

- **`z3_mass_show.py`** — Standard dual-panel visualization: 3D lattice + logarithmic fermion mass comparison.
- **`z3_mass_show_1.py`** — Advanced dual visualization with L² and Δ annotations, updated for strange quark and mod-9.
- **`z3_crystal_44_schematic.py`** — Schematic crystal-style 3D rendering with classification and connections.
- **`z3_44_vector_crystal_visualizer.py`** — High-resolution crystal visualization with customizable thresholds.
- **`z3_vacuum_lattice_crystal_44.py`** — Crystal rendering emphasizing type classification and norm levels.
- **`z3_show_4.py`** — Early dual visualization highlighting weak sector and sin²θ_W = 0.25.
- **`z3_show_5.py`** — Network graph of 44-vector lattice with Tr(A⁴) combinatorial factor.
- **`z3_show_6.py`** — Comprehensive dual-panel (lattice + mass hierarchy) with RG equation.
- **`z3_show_8.py`** — Refined mass hierarchy dual visualization with RG annotation.
- **`z3_show_9.py`** — Dual-panel CKM misalignment angles + bar chart comparison.
- **`z3_show_10.py`** — Horizontal bar chart of geometric ratios for Higgs-to-top mass.
- **`z3_show_11.py`** — Polar diagram of triality phase, magic angle, and CP phase difference.
- **`z3_show_12.py`** — Dual-panel component count (pie + bar chart) for strong coupling analogies.
- **`z3_show_13.py`** — Dual 3D contrasting TBM neutrino large mixing vs quark-like small mixing.
- **`z3_show_14.py`** — Dual-panel cosmological constant hierarchy with compensation diagram.
- **`z3_show_15.py`** — 3D visualization of θ₁₃ basis projection candidates colored by integer score.
- **`z3_show_16.py`** — General-purpose high-quality crystal lattice rendering with classification.
- **`z3_show_17.py`** — Lattice visualization highlighting physical fermion vectors with L²/Δ annotations.
- **`z3_speculative_extensions_flowchart.py`** — Directed flowchart of formal algebraic extensions and analogies.
- **`z3_show_6_b.py`**  
  Generates a 3D visualization of the Z₃-graded vacuum lattice produced by iterative triality rotations and graded bracket closures from the orthonormal basis and democratic vectors (±[1,1,1]/√3); the structure spontaneously saturates at exactly 44 unique vectors, forming a rigid, self-interlocking topology analogous to the classical Chinese Luban mortise-and-tenon lock, with vectors colour-coded by norm class (democratic core ≈√3 in vivid magenta #D81B60, root-like ≈√2 in deep blue #1E88E5, hybrid tenons in deep green #43A047, residual basis in dark grey #546E7A).

### Z₃ Section Visualization

**Files**:
- `z3_section_visualization.py` – Python script that generates the visualization
- `Z3_Signature_Optical_Shadows.pdf` – 2-page PDF containing the full Section with embedded figure

**Description**:  
This visualization summarizes the core idea of the \(\mathbb{Z}_3\)-graded vacuum triality in a single clear image. It shows how the same underlying cubic mechanism produces:
- Superluminal optical shadows (lattice refresh kinks) in tabletop laser experiments
- 120° cyclic kinks in \(c_{\rm hel}\) distributions and non-factorizable spin density matrix at the LHC

The figure combines the geometric triality diagram, predicted shadow velocity curve, \(c_{\rm hel}\) comparison, and 4×4 spin matrix — providing an intuitive bridge between abstract \(\mathbb{Z}_3\) algebra and observable physics.

### 8. Z3_IceCube_Time_Domain_Analyzer.py

```
Z3_IceCube_Time_Domain_Analyzer.py
====================================
Purpose:
    Direct harmonic analysis of public IceCube IC86 Stokes Q/U polarization data
    to search for 6-hour sidereal modulation predicted by the 44-vector Z3 lattice.
Key Output:
    - Power spectrum showing excess at 4th harmonic (6-hour period)
    - SNR ≈ 5.2 at 6h (presented as numerical coincidence only)
Data:
    IceCube ic-cra2024 dataset (pre-processed Q/U maps)
    DOI: 10.7910/DVN/DZI2F5
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DZI2F5
Note:
    This is a suggestive numerical coincidence only.
    Definitive confirmation requires raw event-level data with precise arrival times.
    No physical mechanism or discovery claim is made.
```

### Supporting Scripts

- **`Z3_Isotropy_Proof.py`**  
  Generates the strictly closed 44-vector lattice from triality operations and performs isotropy test (Rank-2 and Rank-4 tensor response). Used to demonstrate the lattice's geometric properties.

- **`z3_lhaaso_prediction.py`**  
  Computes the geometric factor η(n) = Σ (n·v)⁴ over the 44-vector lattice and derives quantitative predictions for possible LIV signatures in LHAASO PeV photon data.

- **`Z3_Phase_Locking_Clean.py`**  
  Performs refined phase alignment analysis on the IceCube IC86 public Stokes \(Q/U\) polarization data after removing edge artifacts from filtering. It generates theoretical modulation curves from the 44-vector lattice and optimizes the lattice orientation to maximize correlation in the clean central region (4h–20h).  
  Key result: The correlation coefficient in the central region reaches 0.8614, showing strong visual and quantitative alignment between the observed modulation and the Z3 theoretical prediction. Output: `Z3_Phase_Locking_Clean.png` — clean comparison plot of IceCube data versus Z3 prediction.  
  The optimized Euler angles obtained from this analysis are approximately \([32.12^\circ, 3.07^\circ, 376.45^\circ]\). These angles represent a formal orientation of the 44-vector lattice relative to the celestial frame in the model. While they are derived purely from numerical optimization and carry no asserted physical meaning at present, they serve as a geometric parameter that could, in principle, be tested or compared with future analyses using independent datasets (e.g., from LHAASO or other observatories). This is presented strictly as a mathematical curiosity.

### Z3_KM3NeT_3Year_Windows.py – 3-Year Transparent Sidereal Windows for KM3NeT >100 PeV Neutrinos

This script generates the complete 3-year prediction table (2026–2029) of daily **Z3 Transparent Windows** for the KM3NeT detector. Based on the 44-vector discrete vacuum lattice geometry, it calculates the precise 1-hour sidereal-time interval (±30 min) during which >100 PeV neutrinos are allowed to reach Earth without being blocked by Rank-4 anisotropy.

The output CSV (`Z3_KM3NeT_3Year_Transparent_Windows.csv`) contains 1096 daily entries with UTC start/end/center times. Any future >100 PeV event detected **outside** these narrow windows immediately falsifies the Z3 geometric channeling model, while repeated detections confined exclusively to the predicted windows would rule out isotropic sterile-neutrino and Earth-matter resonance explanations at high statistical significance.

This provides a clean, model-independent, and highly falsifiable test for the discrete vacuum geometry hypothesis. The script uses Astropy for accurate LST computation and is fully reproducible.

🌌 **Z3 Hubble Skymap Generator**

- **`Z3_Hubble_Skymap_Generator.py`** + **`Z3_Hubble_Skymap.png`**

This script takes the optimized orientation of the 44-vector Z3 lattice and generates a full-sky prediction map of the directional dependence of the cosmic expansion rate.  
**What you are looking at**: The resulting Mollweide projection (`Z3_Hubble_Skymap.png`) reveals a striking, large-scale anisotropic texture — not random noise, but clear red-blue clusters representing regions of formally higher and lower geometric transparency in the abstract lattice.  
- Red regions: Directions where the lattice alignment predicts higher geometric factor η (mathematically “high transparency”).  
- Blue regions: Directions of lower η (“higher resistance”).  

This is one of the most visually compelling outputs from the Z3 framework — turning a 19-dimensional algebraic structure into a concrete, full-sky map that can be directly compared with cosmological observations.  
**Why it matters**: Even though presented strictly as a mathematical curiosity, the map displays structured dipole- and quadrupole-like features that invite comparison with real-world large-scale anomalies (Hubble tension, CMB low-multipole alignments, and concentrations of large-scale structure).

---

# Vacuum Inertia in Nanoscale Transport

This repository provides a complete, self-contained suite of reproducible Python scripts (using only NumPy, SymPy, Matplotlib, and Graphviz) for closed-loop symbolic and numerical validation of the Z₃ Vacuum Inertia framework. The scripts rigorously verify the full logic chain—from Z₃-graded Lie superalgebra construction and exact closure to ab initio quantitative predictions for THz skin depth saturation and nanoscale superconductivity enhancement—without external fitting parameters or unverified steps. Key features include symbolic derivations of core formulas, numerical Jacobi closure checks (residuals ≤ 10⁻¹³), reproducible experimental overlay figures, mindmap visualizations of the logic flow, and comprehensive demonstration of algebraic self-consistency, naturalness, quantitative validation, discriminating signatures, and theoretical constraints.

## Current Recommended Script

#### 1. `Z3_Vacuum_Screening_Cloud_3D_English.py`

**Purpose**: Visual demonstration of the bare-to-dressed transition of the vacuum coherence length.
- Computes the bare scale \(\xi_{\rm bare}\) from collective triality simulation of the 44-vector lattice.
- Applies phenomenological screening \(\eta = 4\) (consistent with typical metallic surface enhancement).
- Renders a high-resolution side-by-side 3D figure showing:
  - Left: Bare vacuum lattice (\(\xi_{\rm bare} \approx 284.42\) nm)
  - Right: Compressed by fermion cloud (\(\xi_{\rm eff} \approx 71.105\) nm)
- Features: real-time calculated values, orange compression arrows, crystal-like point cloud, perfect English labels.

**Output**: `Z3_Vacuum_Screening_Cloud_3D_Crystal_Final_Fixed_NoOverlap.png` (used in the paper)

#### 2. `Z3_Pure_Geometric_Magic_Angle_Ultimate.py`

**Purpose**: Purely geometric prediction of the magic angle in twisted bilayer graphene (zero hopping parameters).
- Uses 6000×6000 grid + multi-harmonic moiré density + full \(A_2\) projection of the 44-vector lattice.
- No Fermi velocity, no interlayer hopping \(w\), no fitting constants.
- Scans twist angle \(\theta\) and finds absolute maximum overlap at \(\theta = 1.090^\circ\).

**Output**: `Z3_Pure_Geometric_Magic_Angle_Ultimate.png` + CSV data

#### 3. `Z3_hBN_Superfluid_Resonance_Improved_3D.py`

**Purpose**: Quantitative simulation of vacuum-induced superfluid density suppression in hBN-cavity devices (Nature 2026 experiment).
- Macroscopic overlap integral between hBN charge density and rotated \(A_2\) vacuum potential.
- Predicts sharp \(C_6\) resonances at \(0^\circ, 60^\circ, 120^\circ\).
- Includes 3D rendering of the vacuum potential surface and comparison with experimental range.

**Output**: `Z3_hBN_Superfluid_Resonance_Final_3D.png` + `Z3_hBN_Suppression_Data.csv`

---

# Z₃ Vacuum Inertia Simulation — Hg-1223 Pressure Quench

**Purpose**  
These two Python programs perform an **illustrative zero-parameter Monte Carlo simulation** of the Z₃ vacuum inertia locking mechanism in Hg-1223 under pressure quench. The simulation demonstrates how geometric resonance between the material lattice and the discrete Z₃ L₄₄ vacuum lattice can naturally produce a metastable superconducting phase near 151 K, qualitatively consistent with the 2026 PNAS experimental results by Chu, Deng et al.

**Programs included**
- `Z3_Hg1223_PressureQuench_TrueZeroParam_3D_Beautiful_Fixed_PDF.py`  
  Generates clean 2D panels (Tc vs Pressure and lattice anchoring dynamics) together with a 3D vacuum potential landscape, exported as high-resolution PDF and PNG.
- `Z3_Hg1223_TrueZeroParam_3D_Braid_Embedding_5Arrows.py`  
  Produces a high-impact 3D visualization showing **five dynamic dashed trajectories with arrows**, clearly illustrating the “solder” (material lattice) being deeply locked into the “desoldering braid” (Z₃ vacuum lattice).

**Core Principle**  
The model relies on geometric resonance computed from the Z₃ L₄₄ lattice projection and a vacuum-inertia energy scale derived purely from dimensional analysis (δ_E = ħ v_F / ξ_vac k_B). The Metropolis Monte Carlo quench protocol then demonstrates robust lattice anchoring once the material enters the resonance window.

**Key Input Parameters** (all taken from literature or algebraically fixed values, no fitting)
- ξ_vac ≈ 70 nm (Z₃ coherence length)
- v_F = 1.57×10⁵ m/s (Hg-1223 Fermi velocity)
- A₀ = 3.85 Å, B₀ = 90 GPa (material constants)
- T_c0 = 133 K, T_quench = 4.2 K, pressure window 15–25 GPa

**Output**
- Publication-ready PDF and PNG figures
- Tc(P) data file (`Z3_Tc_vs_P_TrueZeroParam_Final.csv`)

---

### `z3_exploratory_consistency_verification.py`

**Purpose**: Lightweight symbolic verification of the logical chain (graded brackets → effective coupling → renormalization → surface criticality → emergent scale).  
**Style**: Fully aligned with the final exploratory and phenomenological tone of the paper.  
**Key features**: Purely symbolic (SymPy), no numerical predictions, no figure generation, uses cautious exploratory language.  
**When to use**: For current verification and manuscript preparation.  
**Main difference from previous scripts**: This is a simplified, tone-consistent version specifically designed for the current version of the paper. It avoids strong verification language (“fully verified”, “closed-loop”, “ab initio”) and focuses only on internal symbolic consistency.

### Previous Scripts (Kept for Historical Reproducibility)

These scripts were used in earlier drafts when the paper still contained stronger claims. They are retained for completeness:

- **`z3_vacuum_theory_chain_verify_fixed.py`**  
  Full symbolic derivation of the theoretical chain from graded brackets to nanoscale Tc(d) enhancement. Generates a closed-loop prediction plot from algebraic τ_vac. (Used in strong-claim versions.)

- **`z3_quantitative_logic_chain_verify.py`**  
  Step-by-step symbolic derivation and validation of the Quantitative Comparison section. Generates tables and THz skin depth overlay plot.

- **`z3_theoretical_consistency_verify_fixed.py`**  
  Verification of the Theoretical Consistency section (RG flow, naturalness, timescale, phonon complementarity, discriminating signatures).

- **`z3_nami_sensitivity_show.py`**  
  Generates the three supplementary figures (Tc vs diameter, skin depth saturation, sensitivity of ξ_vac to η).

- **`z3_nanomaterials_chapter1_mindmap_vertical.py`**  
  Generates a vertical Graphviz mindmap of the complete Chapter 1 logic chain (used internally during development).

---

# Z₃-Graded Topological Quantum Computing Architecture

### Key Features

- Monte Carlo simulation of Z₃ toric code fault-tolerance threshold (L=8–16 lattices)
- Identification of threshold crossing in the low-p regime (∼1.8%)
- High-resolution visualization with statistical confidence intervals
- Reproducible ab initio lattice construction and PyMatching decoder

- **`z3_threshold_massive.py`**  
  Performs low-p threshold scan via Monte Carlo (2000 trials per point, L=8,12,16). Constructs triangular toric lattice, injects noise, computes syndromes, and uses PyMatching for decoding. Outputs logical error rates and identifies threshold region.

- **`z3_threshold_massive_show.py`**  
  Generates professional threshold plot from simulation data, including Wilson score 95% confidence intervals, no-correction reference line, and shaded threshold region. Saves as high-resolution PDF/PNG (Nature Communications style).

---

### Profound Significance of This Work

The computational exploration culminates in the spontaneous emergence of a closed, finite 44-vector lattice from pure ℤ₃ triality operations on the vacuum sector. This saturation is not an artifact but a rigorous mathematical consequence of the unique cubic invariant and graded bracket structure.

This finite lattice resolves long-standing foundational issues in theoretical physics:
- It naturally constrains flavor mixing directions, eliminating arbitrary parameters in mass matrix ansätze.
- It offers a prototype for discrete spacetime or vacuum symmetry, bridging continuous field theories with emergent discreteness relevant to quantum gravity.
- Its triangular (A₂-like) symmetry enhanced by democratic deformations predicts specific correlations in neutrino oscillations, CP violation phases, lepton flavor violation ratios, and angular transport modulations in condensed matter systems.

**Example: Z₃-Graded Lie Superalgebra Numerical Verifier (z3_algebra_4.py / z3_algebra_5.py)**  
A Python implementation for verifying the algebraic closure of a 15-dimensional Z₃-graded Lie superalgebra with cubic vacuum triality.

- **Overview**: Numerical verification of closure between gauge, fermionic, and vacuum sectors. Demonstrates exact Jacobi identities with machine-precision residuals (~10⁻¹⁶).
- **Key Features**:
  - 15-dimensional representation (9 gauge + 3 fermionic + 3 vacuum generators)
  - Z₃-graded bracket operations with commutation factor ω = e^(2πi/3)
  - U(3) gauge sector using Gell-Mann matrices
  - Unique mixing term [F, ζ] = -TᵃBᵃ fixed by representation invariance
  - Zero-parameter construction—all coefficients fixed by theory
- **Installation & Usage**:
  ```bash
  pip install numpy
  python z3_algebra_5.py # or z3_algebra_4.py
  ```
- **Expected Output**:
  ```
  ----------------------------------------
  FINAL RESIDUAL: 3.2456e-16
  ----------------------------------------
  [VICTORY] The Z3 Vacuum Coupling is Mathematically Exact.
  Structure: [F, Z] = - T^a B^a
  ```
- **Mathematical Background**: Verifies structure from the published paper in Symmetry (doi:10.3390/sym18010054). Files: z3_algebra_5.py (updated high-precision version), requirements.txt.

### UFO Model (Phenomenological Implementation)

- **`Z3_Ternary_UFO.zip`** (in directory)  
  Complete FeynRules-compatible UFO model implementing ternary vacuum-mediated interactions (t t̄ ζ vertex).  
  Enables direct Monte Carlo simulation of predicted signatures (e.g., top-pair threshold enhancement) in MadGraph5_aMC@NLO.  
  Usage example provided in `UFO1.txt`.

### Simulations Overview

This repo includes seven PyTorch-based simulations validating theoretical predictions (e.g., entropy minimization, curvature feedback, particle hierarchies). Each is self-contained; run with `python c1.py` etc. For details, see wiki links below.

- `c1.py`: Recursive Entropy Stabilization
- `c2.py`: Transient Fluctuations and Curvature Feedback
- `c3.py`: Particle Spectra and Constant Freezing
- `c4.py`: Cosmic Evolution and Multi-Messenger Predictions
- `c5.py`: Superalgebra Verification and Bayesian Analysis
- `c6.py`: EISA Universe Simulator
- `c7.py`: CMB Power Spectrum Inverse Analysis

### Detailed Guides (Wiki Links)

- [EISA Algebra Basics](https://github.com/csoftxyz/RIA_EISA/wiki/eisa_algebra.md)
- [RIA Optimization](https://github.com/csoftxyz/RIA_EISA/wiki/ria_optimization.md)
- [Simulation Tutorials](https://github.com/csoftxyz/RIA_EISA/wiki/simulations/)
- [Validation Code](https://github.com/csoftxyz/RIA_EISA/wiki/validation.md)
- [Universe Simulator](https://github.com/csoftxyz/RIA_EISA/wiki/universe_simulator.md)
- [CMB Inverse Analysis](https://github.com/csoftxyz/RIA_EISA/wiki/cmb_inverse.md)
- [Equation Self-Consistency](https://github.com/csoftxyz/RIA_EISA/wiki/equation_self_consistency.md)
- [Fun Interpretations of Equations](https://github.com/csoftxyz/RIA_EISA/wiki/Fun-Interpretations-of-Equations-in-the-Manuscript)

### Possible Related Experiments (Wiki Links)

- [MIT Double-Slit Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/MIT_Double_Slit_Experiment.md)
- [NANOGrav GW Background](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background.md)
- [NANOGrav GW Frequency Range & Amplitude](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Frequency_Range_Amplitude.md)
- [NANOGrav GW Polarization Modes](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Polarization_Modes.md)
- [NANOGrav GW Non-Gaussianity & Transients](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Non_Gaussianity_Transients.md)
- [NANOGrav GW Multi-Messenger Correlations](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Multi_Messenger_Correlations_Features.md)
- [NANOGrav GW Cosmological Integration](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Cosmological_Integration_Features.md)
- [LHC Mass Anomalies](https://github.com/csoftxyz/RIA_EISA/wiki/LHC_Mass_Anomalies.md)
- [CMB Deviations](https://github.com/csoftxyz/RIA_EISA/wiki/CMB_Deviations.md)
- [SLAC/Brookhaven Breit-Wheeler Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/SLAC_Brookhaven.md)
- [Muon g-2 Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/Muon_g_2.md)
- [Neutrino Mass Hierarchy and CP Violation](https://github.com/csoftxyz/RIA_EISA/wiki/Neutrino_Mass.md)
- [Lepton Flavor Universality Violation (LHCb)](https://github.com/csoftxyz/RIA_EISA/wiki/LHCb_Legacy_Issue.md)
- [EISA-RIA Predictions for New Particles](https://github.com/csoftxyz/RIA_EISA/wiki/New_Particles_at_High_Energies.md)

### Related ATLAS Data

- ATLAS data. (2025). Measurement of the \(t\bar{t}\) production cross section near threshold in pp collisions at √s = 13 TeV with the ATLAS detector. ATLAS-CONF-2025-008. Available at: https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf.

### Cover Video

- RIA_EISA Cover Video: https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4

### Science Education for Teenagers (Wiki Links)

- [Chapter 1: The "Lego Primary Colors" Manual for Physics](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter1.md)
- [Chapter 2: Setting Rules for Cosmic Lego—Physics’ "Lego Constitution"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter2.md)
- [Chapter 3: Weighing Cosmic Lego—Predicting Dark Matter with the "Lego Scale"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter3.md)
- [Chapter 4: The Lego Engine of an Expanding Universe—Stepping on the Gas for Cosmic Acceleration](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter4.md)
- [Chapter 5: Final Appendix: Issuing "Anti-Counterfeit Certificates" for Cosmic Lego](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter5.md)

### API Reference

- Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from `c1.py`.
- RNN model: `EnhancedRNNModel` from `c2.py`.

### Contributing

Welcome contributions! Fork, branch, commit, push, PR. Code of Conduct: Open-source ethics; no conflicts.

### Author Attitude

We hold deep respect for decades of work in string theory, quantum gravity, and related fields. This framework is offered humbly as an exploratory alternative perspective from independent researchers. We make no claim of superiority or finality — only a rigorous, testable structure open to scrutiny. Feedback, criticism, and collaboration are sincerely welcomed.

### Historical Development and Early Works

The current Z₃-graded framework evolved from earlier explorations of integrated symmetry algebras and transient quantum dynamics. These foundational ideas are documented in the following preprints and proceedings, demonstrating that the theory is not an isolated speculation but the result of systematic refinement over several years:

- **Early EISA Preprint Series** (initial concepts of Extended Integrated Symmetry Algebra):  
  v1: https://www.preprints.org/manuscript/202507.2681/v4  
  v7 (major refinement): https://www.preprints.org/manuscript/202507.2681/v7

### On the Discrete Geometric Framework for Fundamental Constants

This document provides a condensed overview of a proposed theoretical framework that attempts to derive patterns of the Standard Model of particle physics from a finite, discrete algebraic structure. The following points outline its core propositions and significance, stated with necessary scientific caution and a clear acknowledgment of its exploratory status.

1. **From Measuring to Deriving Constants: An Attempt**  
   A fundamental open question in physics is the origin of the numerical values of fundamental constants (e.g., the fine-structure constant, the Weinberg angle θ_W), which are empirically measured but theoretically unexplained. This framework makes a core attempt to suggest that some of these values (e.g., yielding sin²θ_W ≈ 0.25) may originate from integer ratios and symmetries within an underlying mathematical structure (such as the cited 11/44 configuration). It aims to provide a potential, non-arbitrary geometric perspective for "why these constants have these values." The validity and universality of this derivation require rigorous testing.

2. **A Geometric Exploration of Unifying Forces and Flavor**  
   The unification of fundamental interactions and the explanation of the fermion mass hierarchy ("flavor" problem) are often separate challenges. This framework, through a single discrete geometric setup (e.g., a 44-vector lattice), attempts to simultaneously describe gauge interactions (e.g., deriving the Weinberg angle via a substructure) and fermion mixing patterns (e.g., the CKM matrix) on a common basis. This approach of tracing both "force" and "matter" textures to a geometric origin offers a new direction distinct from introducing ad hoc fields. Its complete realization and comprehensive match with experimental data remain under investigation.

3. **Discreteness as an Ontological Hypothesis**  
   The work rests on a more foundational, philosophical proposition: the continuity of spacetime and physical laws might be a macroscopic approximation, with a discrete, algebraic nature at the microscopic foundation. Should the physical predictions of this framework be verified in the future, it would not only support a specific model but also strengthen conjectures like "the universe as a computation or discrete mathematical structure." This touches on deep questions about the nature of reality and currently remains in the realm of speculation.

### Current Status & Open Questions

It is crucial to state explicitly that this is a developing theoretical proposal, not an established conclusion. Its key open questions include:

- **Predictions & Tests**: The framework must yield unique, falsifiable predictions distinct from the Standard Model, testable by experiment (e.g., colliders, precision measurements).
- **Mathematical Consistency**: A complete dynamical theory needs to be built on a rigorous mathematical foundation, demonstrating a natural continuum limit that connects seamlessly to successful low-energy existing physics.
- **Conceptual Challenge**: A fundamental conceptual hurdle is explaining how the continuous spacetime and symmetries we observe emerge naturally from an absolute discrete structure.

### Summary

This work proposes a new pathway based on discrete geometry for understanding the origin and potential unification of physical constants. It opens new possibilities, but its ultimate validity depends entirely on future theoretical development and its ability to withstand rigorous experimental verification.

---

### Contact

- Issues tab for technical questions
- Email: csoft@hotmail.com (corresponding) / csoft@live.cn

Wiki pages are continuously updated with detailed guides and interpretations. Contributions welcome.
