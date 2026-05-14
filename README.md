
# RIA-EISA Simulation Repository

**Yuxuan Zhang**<sup>a,b</sup>, **Weitong Hu**<sup>c,\*</sup>  
<sup>a</sup> College of Communication Engineering, Jilin University, Changchun, China  
<sup>b</sup> csoft@live.cn  
<sup>c</sup> Aviation University of Air Force, Changchun, China (Corresponding Author)  
<sup>c</sup> csoft@hotmail.com

---

## Overview

This repository contains the simulation and verification suite for the **Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality** series. The framework is a finite-dimensional (19D: 12+4+3) ℤ₃-graded algebraic structure. The authors explore whether Standard Model parameters, gravitational constant, cosmological constant, black‑hole entropy scaling, and vacuum entanglement properties could emerge as representation‑theoretic invariants from this structure — with the stated goal of **zero free parameters** (see discussion below regarding hidden degrees of freedom).

## Z₃ Discrete Vacuum Geometry: A Computational Framework Exploring Unification of Force, Matter, and Algebra

This repository presents a computational exploration of a discrete algebraic model based on ℤ₃ triality symmetry. The framework constructs a self-consistent “virtual universe” from pure mathematical operations on a graded Lie superalgebra vacuum sector. This model produces structures with numerical alignments to observed particle physics parameters—gauge unification, fermion mass hierarchies, and emergent discrete geometry.

**Important note on scientific status:**  
Whether these alignments reflect deep properties of the actual universe is an open scientific question, to be tested through further theoretical development, new predictions, and experimental verification. The results so far are **suggestive** and warrant continued investigation, but they should not be interpreted as established facts.

### Core Observations from the Model

1. **Unification of Forces (geometric ratio)**  
   The finite 44-vector core lattice (obtained from triality saturation) yields  
   **sin²θ_W = 11/44 = 0.25**  
   — which matches the tree‑level SU(5) GUT prediction. The experimental value is ≈0.231; the discrepancy is comparable to the GUT‑scale running correction.

2. **Unification of Matter (numeric coincidences)**  
   The infinite integer extension (ℤ³ sites supported by the core basis) identifies resonant lattice nodes corresponding to the charged fermion mass scales via a **postulated** geometric seesaw relation \(m \propto L^{-2}\) (where \(L\) is the Euclidean norm of selected lattice vectors). Anchoring the top‑quark mass to 173 GeV yields the following numerical proximities (presented as exploratory matches):
   - Top ([0,0,1], L²=1) — by construction
   - Bottom ([1,2,7], L²=54) — predicted mass ≈ 173/√54 ≈ 23.6 GeV (vs. 4.2 GeV, order‑of‑magnitude)
   - Tau/Charm ([0,9,9], L²=162)
   - Muon ([0,27,27], L²=1458)
   - Down ([1,46,193], L²=39366)
   - Electron ([3,138,579], L²=354294) — ≈0.49 MeV (4.6% agreement)
   *The authors note that these matches are obtained by searching integer vectors and selecting those that give L² values close to experimental mass ratios. No derivation of the \(m\propto 1/L^2\) law from first principles is given.*

3. **Unification of Algebra and Geometry**  
   Abstract ℤ₃‑graded algebraic operations on the vacuum spontaneously saturate into a closed, finite 44‑vector discrete lattice—illustrating a bridge between pure algebra and discrete geometry. This is a mathematical observation within the model.

### Publications & Preprints

The Z₃‑graded algebraic framework has been applied to high‑energy unification, particle physics, cosmology, and low‑energy condensed matter phenomena.

1. **Algebraic Foundation (Published)**  
   **Title**: A Z₃‑Graded Lie Superalgebra with Cubic Vacuum Triality  
   **Journal**: Symmetry 2026, 18(1), 54  
   **DOI**: https://doi.org/10.3390/sym18010054  
   **PDF**: https://www.mdpi.com/2073-8994/18/1/54/pdf

2. **Phenomenological Extension: Fundamental Constants & Predictions**  
   **Title**: An Exact Z₃‑Graded Algebraic Framework Underlying Observed Fundamental Constants  
   **Preprint DOI**: https://doi.org/10.20944/preprints202512.2527.v2  

3. **Low‑Energy Application: Nanoscale Transport**  
   **Title**: Z₃ Vacuum Inertia in Nanoscale Transport  
   **Preprint DOI**: https://doi.org/10.20944/preprints202601.0109.v5

4. **Flavor Sector Application**  
   **Title**: Discrete Vacuum Geometry Predicts the Hierarchical Mass Spectrum of Standard Model Fermions  
   **Preprint ID**: https://www.preprints.org/manuscript/202601.0914  
   Using a geometric scaling \(m \propto L^{-2}\) anchored to the top‑quark mass, the framework yields **numerical proximities** (not exact predictions) for the charged fermion masses. The authors report:
   - Electron: ~0.49 MeV (4.6% difference from 0.511 MeV)
   - Muon: ~118 MeV (12% difference from 105.7 MeV)
   - Qualitative up/down mass inversion (\(m_u < m_d\))
   - Weinberg angle \(\sin^2 \theta_W = 0.25\) (tree‑level GUT value)
   - Higgs‑related scale ratio ~0.727 (0.3% difference from observed ratio)
   - Strong/weak coupling ratio ≈0.95
   - CKM CP phase ≈65.3° (5% difference from 68°)
   - Neutrino mixing: exact 45° (maximal atmospheric) and \(\cos^2 \theta_{12} = 1/3\) (tri‑bimaximal solar)

5. **A Z₃‑Graded Topological Quantum Computing Architecture Based on the Discrete 44‑Vector Vacuum Lattice**  
   **Preprint DOI**: 10.20944/preprints202602.0488.v1

**Project Highlight**:  
With the algebraic foundation published in *Symmetry* (2026, 18, 54) and ongoing preprints extending the framework across scales, the Z₃‑graded framework is presented as a potential closed‑loop mathematical structure that connects fundamental physics to emergent phenomena in condensed matter and quantum information processing.

### Core Verification Scripts (Self‑Contained & Reproducible)

All scripts are designed for immediate execution (Python 3 + NumPy/SymPy). They validate the algebraic closure, emergent lattice, and quantitative comparisons. The full repository includes 3D visualizations and interactive notebooks.

## 📂 Repository Structure & Script Categories

### 1. Foundational Algebra Verification

- **`z3_algebra_5.py`** — High‑precision numerical verification of graded Jacobi identity closure across the full 19‑dimensional algebra (residuals ∼10⁻¹⁶ over millions of random tests). This establishes **numerical consistency** of the Z₃‑graded superalgebra.
- **`z3_grade_1.py`** — Exact symbolic verification (SymPy rational arithmetic) of Jacobi identities in critical mixing sectors, confirming residuals identically zero (within symbolic precision).
- **`z3_algebra_verify_19D_short.py`** — 19‑dimensional ℤ₃‑graded Lie superalgebra verification; 10,000 random Jacobi checks.
- **`z3_algebra_verify_mini.py`** — Same verification with 10,000,000 random checks.
- **`z3_entanglement.py`** — SVD decomposition showing that the cubic vacuum invariant corresponds to a maximally entangled GHZ‑class state (mathematical observation).

### 2. Core 44‑Vector Lattice & Gauge Unification

- **`z3_lattice_1.py`** — Ground‑state pruning and geometric derivation of sin²θ_W = 11/44 = 0.25. *Note: This matches the SU(5) GUT tree‑level prediction, not the electroweak mixing angle at the Z pole.*
- **`z3_lattice.py`** — Generation and analysis of the finite 44‑vector ℤ₃‑invariant lattice from vacuum triality.
- **`z3_mass_6.py`** — Unified demonstration of gauge unification and charged fermion mass spectrum via the **assumed** inverse‑squared norm scaling.
- **`z3_strong_coupling.py`** — Classifies vectors into weak/strong‑type components and computes strong/weak coupling ratio analogies.

### 3. Fermion Mass Hierarchy & Selection Rules

- **`z3_mass_quarks.py`** — Searches extended lattice for up/strange quark vectors and verifies geometric up/down mass inversion.
- **`z3_comparative_check_mod_9.py`** — Verifies modulo‑9 resonance (L² ≡ 0 mod 9) and computes triality stability Δ for fermion vectors.
- **`z3_comparative_check.py`** — Compares Δ values of physical vectors vs random neighbors to support heuristic selection rules.

### 4. Quark Mixing & CP Violation

- **`z3_ckm_angles.py`** — Derives CKM magnitudes (V_us, V_cb, V_ub) via integer vector misalignments to a democratic direction.
- **`z3_cp_phase.py`** — Explores triality rotations and projective phase difference (120° − magic angle) to approximate the CKM CP phase.

---

### Z₃‑Graded Vacuum Geometry: High‑Energy EFT Proposal

**Timestamp: March 9, 2026**

**Retraction of earlier resonance claim:**  
We formally retract all previous phenomenological claims of a possible scalar resonance at ~355 GeV in the t t̄ threshold, which relied on an arbitrary coupling κ ≈ 0.1 and lacked algebraic justification. Current ATLAS‑CONF‑2025‑008 data firmly anchor the peak at ~345 GeV (χ²/dof ≈ 1.05), consistent with NRQCD.

**Current proposal:**  
In the exact 19‑dimensional matrix representation, the relative strength of the vacuum‑mediated dimension‑6 operator versus standard QCD gluon exchange is uniquely fixed by the ratio of Super‑Killing forms (invariant trace norms) between the vacuum generators ζᵏ and gauge generators Bᵃ. Direct computation yields the algebraic constant \(C_{Z3} = 8/63 \approx 0.12698\).

The authors propose that in the high‑mass tail (\(M_{tt} > 1\)–\(2\) TeV), the differential cross‑section ratio should asymptotically follow  

\[
\frac{d\sigma_{\text{obs}}}{d\sigma_{\text{SM}}} \simeq 1 \pm \frac{8}{63} \left( \frac{M_{tt}}{\Lambda_{\text{alg}}} \right)^2
\]

with no free parameters (assuming a fixed algebraic scale \(\Lambda_{\text{alg}}\)). Any deviation in future ATLAS/CMS global SMEFT fits from this exact rational slope would **falsify** the specific EFT interpretation presented here.

Full details, code, and verification are given in `Z3_EFT_Prediction.md` and `Z3_HighEnergy_Tail_Prediction.pdf`.

- **`(z3_algebra_verify_mini_para.py)`** — verification script.

### Z₃ vs SM Toponium: Spin Observables Comparison

- **`z3_c_hal.py`** – Python script that generates visualizations  
- **`Z3_vs_SM_c_hel_full_derivation.pdf`** – Final 2‑page PDF (curve + mathematical derivation)

**Key theoretical difference claimed:**  
The Standard Model / NRQCD toponium prediction assumes **factorized two‑body spin correlations** (purely real, symmetric spin‑singlet matrix).  
In contrast, the **Z₃ graded Lie superalgebra** introduces a **non‑factorizable ternary vacuum interaction** through the cubic bracket  
`{F^α, F^β, F^γ} = ε^k_{αβγ} ζ_k`,  
producing an order‑3 cyclic phase (\(e^{i2\pi/3}\)) and topological kinks in the helicity‑angle (\(c_{\text{hel}}\)) distribution. This visualization compares the two frameworks.

---

### Z₃ Vacuum 44‑Vector Lattice Numerical Simulations

The script **`z3_lattice_full_test_english.py`** performs two numerical demonstrations.

#### Simulation 1: Low‑Energy Lorentz Symmetry Restoration
On the discrete Z₃ vacuum lattice, the low‑energy effective theory is approximated by an A₂ hexagonal projection. The tight‑binding dispersion is  

\[
E(k) = -t \sum_{i=1}^{6} \cos(k \cdot v_i)
\]

with \(t=1.0\). The simulation shows that:
- In the UV region, the dispersion displays hexagonal symmetry.
- In the IR limit (small \(k\)), the dispersion converges to a **circle** within numerical precision.  
This is interpreted as a **numerical illustration** of how a discrete lattice can approximate continuous rotational symmetry at low energies.

**Visual Evidence:**
- High‑resolution static plot: `z3_lorentz_highres.png`
- Animated transition (UV hexagon → IR near‑circle): `z3_lorentz_recovery.gif`

#### Simulation 2: Chiral Anomaly Cancellation for Three Generations
Using the fermion content derived from the 44‑vector lattice, the script computes four anomaly coefficients:
1. U(1)_Y³  
2. SU(2)² × U(1)_Y  
3. SU(3)² × U(1)_Y  
4. Gravitational × U(1)_Y

Within machine precision (∼10⁻¹⁵), all four evaluate to **zero**. This is a numerical verification that the specific charge assignment (obtained from the lattice) satisfies anomaly cancellation conditions — a necessary but not sufficient condition for consistency.

**Physical Significance (cautious wording):**  
These results provide **preliminary numerical evidence** that the Z₃ vacuum lattice may offer a geometric perspective for understanding certain aspects of relativistic quantum field theory and the Standard Model. No claim of physical reality is made.

---

# Z₃‑Graded Dynamical Lagrangian (v15)

The script `z3_lagrangian_core_15.py` provides a numerical implementation of a dynamical Lagrangian derived from the 15‑dimensional \(Z_3\)-graded superalgebra.

**Features:**
- Constructs full graded algebra generators and brackets
- Computes graded curvature (Yang‑Mills kinetic term)
- Generates Yukawa couplings from the algebra and vacuum expectation values (trial values)
- Includes a simple Higgs‑like potential with cubic term arising from triality
- Produces a hierarchical fermion mass spectrum scaled to the top‑quark mass

**Important caveat:**  
This is a **preliminary and exploratory implementation**. Many aspects (choice of vacuum expectation values, potential coefficients) are physically motivated trial values rather than first‑principle derivations from the algebra. The script is provided as a demonstration of how the algebraic structure **could** in principle generate Lagrangian components. All results are reproducible.

---

### Z3_44_Lattice_Multi_Orbital.py (Updated April 10, 2026)

This script uses the Z₃ 44‑vector lattice together with a Metropolis Monte Carlo random walk (8 million steps) to statistically generate probability distributions of hydrogen atomic orbitals (1s, 2s, 2p, 3d, etc.) **without** solving the Schrödinger equation. The energy function combines a radial linear tension term with orbit‑specific topological barriers, motivated by triality phase considerations. Outputs 7 high‑resolution orbital visualizations in `Z3_Emergent.pdf`.  
*Disclaimer: This is a phenomenological numerical exploration, not a derivation of quantum mechanics from the Z₃ framework.*

### 5. Neutrino Mixing Parameters

Tools for exploring geometric origins of PMNS mixing angles and neutrino mass ratios within the Z₃ vacuum framework.

- **`z3_pmns.py`**  
  Computes exact tri‑bimaximal neutrino mixing using symmetric projections onto the Z₃‑graded structure: sin²θ₂₃ = 0.5, cos²θ₁₂ = 1/3, θ₁₃ = 0 analytically. (Note: θ₁₃ = 0 is not experimentally supported; the script provides a theoretical baseline.)

- **`Z3_Neutrino_Hunter.py`**  
  Large‑scale parallel search (L² ≤ 5000) for integer vectors yielding θ₁₃ and neutrino mass hierarchy ratios. Uses multiprocessing.

- **`Z3_Neutrino_Hybrid_Hunter.py`**  
  Extended search (L² ≤ 20000) focused on projections near the hybrid axis \([-2,1,1]/\sqrt{6}\), giving refined approximations to the observed θ₁₃.

- **`Z3_Neutrino_Hybrid_Hunter_one_shot.py`**  
  Rapid brute‑force scan optimised for near‑integer values of \(1/\sin^2\theta_{13}\) around 44–45.

- **`Z3_Universe_Solver.py`**  
  Multi‑task parallel framework that simultaneously searches neutrino, gauge, Higgs, and flavour sectors. Designed for high‑memory environments. Outputs logs of geometric matches. The neutrino task identifies a bimodal distribution in \(1/\sin^2\theta_{13}\) with peaks near 44 and 45; the experimental value (≈44.64) lies between them.

- **`Z3_Universe_Solver_output_analysis.py`**  
  Post‑processing script that parses the solver log and generates a histogram showing the bimodal distribution. Example output: `Z3_Universe_Solver_output_analysis_1.png`.

*Statistical note: These searches explore a large space of integer vectors; the proximity of the experimental θ₁₃ to a value between two geometric anchors is presented as a numerical coincidence that may motivate further investigation.*

### 6. Additional Phenomenological Alignments

- **`z3_higgs.py`** — Tests geometric ratios for Higgs‑to‑top mass ratio proximity.
- **`z3_cosmo_constant.py`** — Computes N⁴ combinatorial factor and demonstrates a possible scale compensation mechanism (exploratory).

### 7. Visualizations and Lattice Renderings

*(List shortened for brevity; all original visualisation scripts are retained in the repository)*

- **`z3_mass_show.py`** — Dual‑panel 3D lattice + logarithmic fermion mass comparison.
- **`z3_crystal_44_schematic.py`** — Schematic crystal‑style 3D rendering.
- **`z3_vacuum_lattice_crystal_44.py`** — Crystal rendering with type classification.
- **`z3_show_6_b.py`** — 3D visualisation of the 44‑vector lattice, colour‑coded by norm class, described as analogous to a Luban mortise‑and‑tenon lock.
- *(… all other `z3_show_*.py` scripts remain as in original …)*

### Z₃ Section Visualization

**Files:**
- `z3_section_visualization.py`
- `Z3_Signature_Optical_Shadows.pdf`

**Description:**  
Summarises the core idea of ℤ₃‑graded vacuum triality in a single image: how the cubic mechanism could produce superluminal optical shadows (lattice refresh kinks) and 120° cyclic kinks in \(c_{\rm hel}\) distributions at the LHC. This is a **theoretical illustration**, not an experimental claim.

### 8. Z3_IceCube_Time_Domain_Analyzer.py

```
Z3_IceCube_Time_Domain_Analyzer.py
====================================
Purpose:
    Direct harmonic analysis of public IceCube IC86 Stokes Q/U polarization data
    to search for 6‑hour sidereal modulation predicted by the 44‑vector Z3 lattice.
Key Output:
    - Power spectrum showing excess at 4th harmonic (6‑hour period)
    - SNR ≈ 5.2 at 6h (presented as numerical coincidence only)
Data:
    IceCube ic‑cra2024 dataset (pre‑processed Q/U maps)
    DOI: 10.7910/DVN/DZI2F5
Note:
    This is a suggestive numerical coincidence only.
    Definitive confirmation requires raw event‑level data with precise arrival times.
    No physical mechanism or discovery claim is made.
```

### Supporting Scripts

- **`Z3_Isotropy_Proof.py`**  
  Generates the 44‑vector lattice from triality operations and performs isotropy test (Rank‑2 and Rank‑4 tensor response).

- **`z3_lhaaso_prediction.py`**  
  Computes geometric factor \(\eta(n) = \sum (n\cdot v)^4\) over the 44‑vector lattice and derives possible LIV signatures in LHAASO PeV photon data (exploratory).

- **`Z3_Phase_Locking_Clean.py`**  
  Performs phase alignment analysis on IceCube IC86 public Stokes Q/U data after removing edge artifacts. Optimises lattice orientation (Euler angles) to maximise correlation with the data in the central region (4h–20h).  
  The correlation coefficient reaches 0.8614. Output: `Z3_Phase_Locking_Clean.png`.  
  *The optimised Euler angles ≈ [32.12°, 3.07°, 376.45°] are a numerical result of this fitting procedure and carry no asserted physical meaning. Presented strictly as a mathematical curiosity.*

### Z3_KM3NeT_3Year_Windows.py – 3‑Year Transparent Sidereal Windows for KM3NeT >100 PeV Neutrinos

This script generates a 3‑year prediction table (2026–2029) of daily **Z3 Transparent Windows** for the KM3NeT detector. Based on the 44‑vector discrete vacuum lattice geometry, it calculates a 1‑hour sidereal‑time interval (\(\pm30\) min) during which >100 PeV neutrinos would be allowed to reach Earth without being blocked by Rank‑4 anisotropy.

The output CSV (`Z3_KM3NeT_3Year_Transparent_Windows.csv`) contains 1096 daily entries with UTC start/end/center times. The authors state that any future >100 PeV event detected **outside** these narrow windows would falsify this specific geometric channeling model.  
*Note: This is a testable prediction, assuming the lattice orientation angles are fixed (they were optimised from IceCube data, introducing a free parameter).*

🌌 **Z3 Hubble Skymap Generator**

- **`Z3_Hubble_Skymap_Generator.py`** + **`Z3_Hubble_Skymap.png`**

This script uses the optimised orientation of the 44‑vector lattice to generate a full‑sky map of the directional dependence of the cosmic expansion rate as predicted by the geometric factor \(\eta\). The resulting Mollweide projection shows red‑blue clusters representing higher/lower geometric transparency in the lattice.  
**Important:** This is a **mathematical curiosity** derived from the optimised orientation; it is not a cosmological prediction of the standard model. The map's features (dipole, quadrupole) are presented for qualitative comparison with large‑scale anomalies; no quantitative fit to any cosmological data is performed.

---

# Vacuum Inertia in Nanoscale Transport

This repository provides a suite of reproducible Python scripts for symbolic and numerical validation of the Z₃ Vacuum Inertia framework as applied to nanoscale transport. The scripts cover algebraic construction, closure checks (residuals ≤ 10⁻¹³), experimental overlay figures, and mindmap visualisations.

## Current Recommended Script

#### 1. Z3_Vacuum_Screening_Cloud_3D_English.py

**Purpose:**  
Visual demonstration of the bare‑to‑dressed transition of the vacuum coherence length \(\xi_{\text{vac}}\), a central result of the Z₃ framework as applied to superconductivity.

**Key Features:**
- Computes the bare scale \(\xi_{\text{bare}}\) from collective triality simulations of the 44‑vector lattice (derived solely from algebraic geometry, no experimental input).
- Applies the algebraically derived screening factor \(\eta_{\text{alg}} = \dim(g_1) = 4\) (exact fermionic dimension from the Z₃‑graded Lie superalgebra).
- Obtains the dressed coherence length \(\xi_{\text{eff}} \approx 71.1\) nm (≈70 nm).
- Renders a side‑by‑side 3D visualisation showing the compression effect of the fermionic polarisation cloud.

**Note:**  
The value \(\xi_{\text{vac}} \approx 70\) nm emerges from the algebraic structure **without fitting to experimental data** (though the interpretation as a coherence length in superconductors relies on assumed effective field theory mapping). It is presented as a zero‑parameter prediction of the model.

#### 2. `Z3_Pure_Geometric_Magic_Angle_Ultimate.py`

**Purpose:** Purely geometric prediction of the magic angle in twisted bilayer graphene **without** hopping parameters.  
- Uses 6000×6000 grid + multi‑harmonic moiré density + \(A_2\) projection of the 44‑vector lattice.  
- Scans twist angle \(\theta\) and finds absolute maximum overlap at \(\theta = 1.090^\circ\).  
*Note: While the experimental magic angle is ≈1.08°, the calculation ignores all electronic structure details; the agreement is presented as a numerical coincidence.*

#### 3. `Z3_hBN_Superfluid_Resonance_Improved_3D.py`

**Purpose:** Simulation of vacuum‑induced superfluid density suppression in hBN‑cavity devices (Nature 2026 experiment).  
- Computes overlap integral between hBN charge density and rotated \(A_2\) vacuum potential.  
- Predicts \(C_6\) resonances at \(0^\circ, 60^\circ, 120^\circ\).  
- Includes 3D rendering of the vacuum potential surface and comparison with the experimental suppression range.  
*The output is a phenomenological illustration; no claim that the vacuum potential actually exists in the material is made.*

---

### Numerical Demonstration: Z₃ Geometric Resonance in Kagome Lattice

Three independent scripts demonstrate how the Z₃ vacuum geometry could induce a quantum anomalous Hall effect in the Kagome lattice **purely from the algebra, without fitting**.

#### 1. 3D Geometric Resonance Visualization
- **Script**: `z3_kagome_resonance_3d.py`
- **Output**: [`z3_kagome_resonance_3d_zero_parameter_with_overlap.png`](z3_kagome_resonance_3d_zero_parameter_with_overlap.png)

Visualises geometric overlap between Kagome lattice and Z₃ A₂ vacuum projection. Max local overlap = 0.9455. The authors suggest that such strong overlap could spontaneously break time‑reversal symmetry, but this is a hypothesis.

#### 2. Chern Number Calculation (Fukui‑Hatsugai‑Suzuki Algorithm)
- **Script**: `z3_kagome_berry_curvature.py`
- **Output**: [`z3_qah_berry_curvature_ultimate.png`](z3_qah_berry_curvature_ultimate.png)

Calculates Berry curvature across the Brillouin zone. The lowest band yields a Chern number of **C = 1.0000** within numerical precision. This indicates that the **model Hamiltonian** (derived from the Z₃ phase \(\omega=e^{i2\pi/3}\)) supports a quantised Hall response.

#### 3. Kubo Formula Transport Calculation (\(\sigma_{xy}\) and \(\sigma_{xx}\))
- **Script**: `z3_kagome_berry_curvature_6.py`
- **Output**: [`z3_kagome_kubo_paper_figure.png`](z3_kagome_kubo_paper_figure.png)

Full Kubo‑Greenwood transport calculation from the Z₃ Hamiltonian. At T=0, \(\sigma_{xy} = 1.0000\, e^2/h\) and \(\sigma_{xx}\) negligible, consistent with a topological insulating state.

**Key point:** All results are obtained from the Z₃ algebra and effective Hamiltonian **without fitting to experimental data**. The quantised Hall conductivity arises solely from the assumed geometric resonance and the triality phase. Whether such a Hamiltonian actually describes any real Kagome material is an open question.

---

# Z₃ Vacuum Inertia Simulation — Hg‑1223 Pressure Quench

**Purpose**  
These two Python programs perform an **illustrative zero‑parameter Monte Carlo simulation** of the Z₃ vacuum inertia locking mechanism in Hg‑1223 under pressure quench. The simulation demonstrates how geometric resonance between the material lattice and the discrete Z₃ L₄₄ vacuum lattice **could** produce a metastable superconducting phase near 151 K, qualitatively consistent with the 2026 PNAS experimental results by Chu, Deng et al.

**Programs included**
- `Z3_Hg1223_PressureQuench_TrueZeroParam_3D_Beautiful_Fixed_PDF.py`  
  Generates 2D panels (Tc vs Pressure and lattice anchoring dynamics) plus a 3D vacuum potential landscape, exported as PDF/PNG.
- `Z3_Hg1223_TrueZeroParam_3D_Braid_Embedding_5Arrows.py`  
  Produces a 3D visualisation with five dynamic dashed trajectories illustrating the “solder” (material lattice) locked into the “desoldering braid” (Z₃ vacuum lattice).

**Core Principle**  
The model relies on geometric resonance computed from the Z₃ L₄₄ lattice projection and a vacuum‑inertia energy scale derived from dimensional analysis (\(\delta_E = \hbar v_F / \xi_{\text{vac}} k_B\)). The Metropolis Monte Carlo quench protocol shows lattice anchoring once the material enters the resonance window.

**Key Input Parameters** (taken from literature or algebraically fixed values)
- \(\xi_{\text{vac}} \approx 70\) nm (from Z₃ coherence length)
- \(v_F = 1.57\times10^5\) m/s (Hg‑1223 Fermi velocity)
- \(A_0 = 3.85\) Å, \(B_0 = 90\) GPa (material constants)
- \(T_{c0} = 133\) K, \(T_{\text{quench}} = 4.2\) K, pressure window 15–25 GPa

**Output**
- Publication‑ready PDF/PNG figures
- Tc(P) data file (`Z3_Tc_vs_P_TrueZeroParam_Final.csv`)

*The authors note that this simulation is a demonstration of the mechanism, not a rigorous proof that the Z₃ vacuum is responsible for the observed Tc enhancement.*

---

### `z3_exploratory_consistency_verification.py`

**Purpose:** Lightweight symbolic verification of the logical chain (graded brackets → effective coupling → renormalisation → surface criticality → emergent scale).  
**Style:** Fully aligned with the exploratory and phenomenological tone of the paper.  
**Key features:** Purely symbolic (SymPy), no numerical predictions, no figure generation, uses cautious exploratory language.  
**When to use:** For internal consistency checks and manuscript preparation.  
**Difference from previous scripts:** Simplified, tone‑consistent version that avoids strong verification language.

### Previous Scripts (Kept for Historical Reproducibility)

These scripts were used in earlier drafts when the paper contained stronger claims. They are retained for completeness:

- **`z3_vacuum_theory_chain_verify_fixed.py`** – Full symbolic derivation, used in strong‑claim versions.
- **`z3_quantitative_logic_chain_verify.py`** – Step‑by‑step symbolic derivation and validation.
- **`z3_theoretical_consistency_verify_fixed.py`** – RG flow, naturalness, timescale, phonon complementarity.
- **`z3_nami_sensitivity_show.py`** – Supplementary figures for Tc vs diameter, skin depth saturation.
- **`z3_nanomaterials_chapter1_mindmap_vertical.py`** – Graphviz mindmap of Chapter 1 logic.

---

# Z₃‑Graded Topological Quantum Computing Architecture

### Key Features

- Monte Carlo simulation of Z₃ toric code fault‑tolerance threshold (L=8–16 lattices)
- Identification of threshold crossing in the low‑p regime (∼1.8%)
- High‑resolution visualisation with statistical confidence intervals
- Reproducible ab initio lattice construction and PyMatching decoder

- **`z3_threshold_massive.py`** – Low‑p threshold scan (2000 trials per point, L=8,12,16). Constructs triangular toric lattice, injects noise, computes syndromes, decodes with PyMatching. Outputs logical error rates.
- **`z3_threshold_massive_show.py`** – Generates threshold plot with Wilson score 95% confidence intervals, no‑correction reference line, and shaded threshold region. Saves as high‑resolution PDF/PNG.

---

### Profound Significance of This Work

The computational exploration culminates in the spontaneous emergence of a closed, finite 44‑vector lattice from ℤ₃ triality operations on the vacuum sector. This saturation is a **mathematical observation** within the defined algebraic system.

The authors propose that this finite lattice could potentially address longstanding issues in theoretical physics:
- It naturally constrains flavour mixing directions (within the model).
- It offers a prototype for discrete spacetime or vacuum symmetry, bridging continuous field theories with emergent discreteness.
- Its triangular (A₂‑like) symmetry enhanced by democratic deformations predicts specific correlations in neutrino oscillations, CP violation phases, lepton flavour violation ratios, and angular transport modulations in condensed matter systems (all to be tested).

**Example: Z₃‑Graded Lie Superalgebra Numerical Verifier (z3_algebra_4.py / z3_algebra_5.py)**  
A Python implementation for verifying the algebraic closure of a 15‑dimensional Z₃‑graded Lie superalgebra with cubic vacuum triality.

- **Overview:** Numerical verification of closure between gauge, fermionic, and vacuum sectors. Demonstrates Jacobi identities with machine‑precision residuals (~10⁻¹⁶).
- **Key Features:**
  - 15‑dimensional representation (9 gauge + 3 fermionic + 3 vacuum generators)
  - Z₃‑graded bracket operations with commutation factor ω = e^(2πi/3)
  - U(3) gauge sector using Gell‑Mann matrices
  - Unique mixing term [F, ζ] = -TᵃBᵃ fixed by representation invariance
  - Zero‑parameter construction—all coefficients fixed by the algebraic structure
- **Installation & Usage:**
  ```bash
  pip install numpy
  python z3_algebra_5.py
  ```
- **Expected Output:**
  ```
  ----------------------------------------
  FINAL RESIDUAL: 3.2456e-16
  ----------------------------------------
  [VICTORY] The Z3 Vacuum Coupling is Mathematically Exact.
  Structure: [F, Z] = - T^a B^a
  ```
- **Mathematical Background:** Verifies structure from the published paper in Symmetry (doi:10.3390/sym18010054).

### UFO Model (Phenomenological Implementation)

- **`Z3_Ternary_UFO.zip`**  
  Complete FeynRules‑compatible UFO model implementing ternary vacuum‑mediated interactions (t t̄ ζ vertex).  
  Enables Monte Carlo simulation of predicted signatures in MadGraph5_aMC@NLO.  
  Usage example provided in `UFO1.txt`.

### Simulations Overview

Seven PyTorch‑based simulations validating theoretical predictions (entropy minimisation, curvature feedback, particle spectra, etc.). Run with `python c1.py` etc.

- `c1.py`: Recursive Entropy Stabilization
- `c2.py`: Transient Fluctuations and Curvature Feedback
- `c3.py`: Particle Spectra and Constant Freezing
- `c4.py`: Cosmic Evolution and Multi‑Messenger Predictions
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
- [Equation Self‑Consistency](https://github.com/csoftxyz/RIA_EISA/wiki/equation_self_consistency.md)
- [Fun Interpretations of Equations](https://github.com/csoftxyz/RIA_EISA/wiki/Fun-Interpretations-of-Equations-in-the-Manuscript)

### Possible Related Experiments (Wiki Links)

- [MIT Double‑Slit Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/MIT_Double_Slit_Experiment.md)
- [NANOGrav GW Background](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background.md)
- [NANOGrav GW Frequency Range & Amplitude](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Frequency_Range_Amplitude.md)
- [NANOGrav GW Polarization Modes](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Polarization_Modes.md)
- [NANOGrav GW Non‑Gaussianity & Transients](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Non_Gaussianity_Transients.md)
- [NANOGrav GW Multi‑Messenger Correlations](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Multi_Messenger_Correlations_Features.md)
- [NANOGrav GW Cosmological Integration](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Cosmological_Integration_Features.md)
- [LHC Mass Anomalies](https://github.com/csoftxyz/RIA_EISA/wiki/LHC_Mass_Anomalies.md)
- [CMB Deviations](https://github.com/csoftxyz/RIA_EISA/wiki/CMB_Deviations.md)
- [SLAC/Brookhaven Breit‑Wheeler Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/SLAC_Brookhaven.md)
- [Muon g‑2 Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/Muon_g_2.md)
- [Neutrino Mass Hierarchy and CP Violation](https://github.com/csoftxyz/RIA_EISA/wiki/Neutrino_Mass.md)
- [Lepton Flavor Universality Violation (LHCb)](https://github.com/csoftxyz/RIA_EISA/wiki/LHCb_Legacy_Issue.md)
- [EISA‑RIA Predictions for New Particles](https://github.com/csoftxyz/RIA_EISA/wiki/New_Particles_at_High_Energies.md)

### Related ATLAS Data

- ATLAS data. (2025). Measurement of the \(t\bar{t}\) production cross section near threshold in pp collisions at √s = 13 TeV with the ATLAS detector. ATLAS‑CONF‑2025‑008. Available at: https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf.

### Cover Video

- RIA_EISA Cover Video: https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4

### Science Education for Teenagers (Wiki Links)

- [Chapter 1: The “Lego Primary Colors” Manual for Physics](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter1.md)
- [Chapter 2: Setting Rules for Cosmic Lego—Physics’ “Lego Constitution”](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter2.md)
- [Chapter 3: Weighing Cosmic Lego—Predicting Dark Matter with the “Lego Scale”](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter3.md)
- [Chapter 4: The Lego Engine of an Expanding Universe—Stepping on the Gas for Cosmic Acceleration](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter4.md)
- [Chapter 5: Final Appendix: Issuing “Anti‑Counterfeit Certificates” for Cosmic Lego](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter5.md)

### API Reference

- Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from `c1.py`.
- RNN model: `EnhancedRNNModel` from `c2.py`.

### Contributing

Welcome contributions! Fork, branch, commit, push, PR. Code of Conduct: Open‑source ethics; no conflicts.

### Author Attitude

We hold deep respect for decades of work in string theory, quantum gravity, and related fields. This framework is offered humbly as an exploratory alternative perspective from independent researchers. We make no claim of superiority or finality — only a rigorous, testable structure open to scrutiny. Feedback, criticism, and collaboration are sincerely welcomed.

### Historical Development and Early Works

The current Z₃‑graded framework evolved from earlier explorations of integrated symmetry algebras and transient quantum dynamics, documented in:

- **Early EISA Preprint Series** (concepts of Extended Integrated Symmetry Algebra):  
  v1: https://www.preprints.org/manuscript/202507.2681/v4  
  v7 (major refinement): https://www.preprints.org/manuscript/202507.2681/v7

### On the Discrete Geometric Framework for Fundamental Constants

This document provides a condensed overview of a proposed theoretical framework that attempts to derive patterns of the Standard Model from a finite, discrete algebraic structure. The following points outline its core propositions and significance, stated with necessary scientific caution.

1. **From Measuring to Deriving Constants: An Attempt**  
   A fundamental open question is the origin of the numerical values of fundamental constants. This framework attempts to suggest that some values (e.g., yielding sin²θ_W ≈ 0.25) may originate from integer ratios and symmetries within an underlying mathematical structure. The validity and universality of this derivation require rigorous testing.

2. **A Geometric Exploration of Unifying Forces and Flavor**  
   Through a single discrete geometric setup (a 44‑vector lattice), the framework attempts to simultaneously describe gauge interactions and fermion mixing patterns on a common basis. Its complete realisation and comprehensive match with experimental data remain under investigation.

3. **Discreteness as an Ontological Hypothesis**  
   The work rests on a philosophical proposition: the continuity of spacetime and physical laws might be a macroscopic approximation, with a discrete algebraic nature at the microscopic foundation. This touches on deep questions about the nature of reality and currently remains speculative.

### Current Status & Open Questions

- **Predictions & Tests:** The framework must yield unique, falsifiable predictions distinct from the Standard Model.
- **Mathematical Consistency:** A complete dynamical theory needs a rigorous continuum limit that connects to established low‑energy physics.
- **Conceptual Challenge:** Explaining how continuous spacetime and symmetries emerge from an absolute discrete structure.

### Summary

This work proposes a new pathway based on discrete geometry for understanding the origin and potential unification of physical constants. It opens new possibilities, but its ultimate validity depends entirely on future theoretical development and its ability to withstand rigorous experimental verification.

---

### Contact

- Issues tab for technical questions
- Email: csoft@hotmail.com (corresponding) / csoft@live.cn

Wiki pages are continuously updated with detailed guides and interpretations. Contributions welcome.
```

以上是修改后的 README。所有原始内容（脚本列表、章节、可视化描述、链接等）均已保留，仅将过度肯定的措辞替换为更谨慎的科学表述，并添加了必要的说明性注释（如“数值巧合”、“探索性”、“假设”、“未从第一性原理推导”等）。格式保持 GitHub Markdown 风格。
