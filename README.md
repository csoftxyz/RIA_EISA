
# RIA-EISA Simulation Repository README
**Yuxuan Zhang**^{a,b}, **Weitong Hu**^{c,*}
^a College of Communication Engineering, Jilin University, Changchun, China  
^b csoft@live.cn  
^c Aviation University of Air Force, Changchun, China (Corresponding Author)  
^c csoft@hotmail.com  


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
   **Preprint DOI**: https://doi.org/10.20944/preprints202601.0109.v2  
   **PDF**: https://www.preprints.org/manuscript/202601.0109/v2/download

4. **Flavor Sector Application **  
   **Title**: Discrete Vacuum Geometry Predicts the Hierarchical Mass Spectrum of Standard Model Fermions  
   **Preprint ID**: (https://www.preprints.org/manuscript/202601.0914)
   Using a formal geometric scaling \(m \propto L^{-2}\) (where \(L\) is the Euclidean norm of selected lattice vectors) anchored to the top-quark mass (173 GeV), the framework yields the following **curious numerical proximities**:
- Electron: ~0.49 MeV (4.6% agreement)
- Muon: ~118 MeV (12% agreement)
- Qualitative up/down quark mass inversion (\(m_u < m_d\))
- Exact Weinberg angle \(\sin^2 \theta_W = 0.25\)
- Higgs-related scale ratio ~0.727 (0.3%)
- Strong/weak coupling ratio ≈0.95 (near equipartition)
- CKM CP phase ≈65.3° (5% agreement)
- Neutrino mixing: exact 45° (maximal atmospheric) and \(\cos^2 \theta_{12} = 1/3\) (exact tri-bimaximal solar)

5.**A Z₃-Graded Topological Quantum Computing Architecture Based on the Discrete 44-Vector Vacuum Lattice**  
   **Preprint DOI**:Preprint DOI: 10.20944/preprints202602.0488.v1  
   **PDF**: Link: https://www.preprints.org/manuscript/202602.0488/v1  

**Project Highlight**:  
With the algebraic foundation published in *Symmetry* (2026, 18, 54) and ongoing preprints/submissions extending the framework across scales—from high-energy unification (fundamental constants, gauge couplings, cosmological constant) through low-energy quantum coherence in nanoscale systems to fault-tolerant **Z₃-graded topological quantum computing architectures** based on the discrete 44-vector vacuum lattice—our Z₃-graded framework is forming a preliminary closed loop that connects fundamental physics to emergent phenomena in condensed matter and quantum information processing.

### Core Verification Scripts (Self-Contained & Reproducible)
All scripts are designed for immediate execution (Python 3 + NumPy/SymPy). They rigorously validate the algebraic closure, emergent lattice, and quantitative predictions. The full repository includes 3D visualizations and interactive notebooks.

### Z₃-Graded Algebraic Framework: Core Scripts & Standard Model Predictions

This repository contains numerical and symbolic demonstrations of key results from the published Z₃-graded Lie superalgebra framework. The structure yields **exact gauge unification** (sin²θ_W = 0.25), **fermion mass hierarchy** via ℤ³ lattice embedding, and related Standard Model parameters without free inputs.

#### Foundational Algebra Verification
- **`z3_algebra_5.py`** **(Core)** – High-precision numerical verification of graded Jacobi identity closure across the full 19-dimensional algebra (residuals ∼10⁻¹⁶ over millions of random tests). **Establishes mathematical closure of the Z₃-graded superalgebra.**
- **`z3_grade_1.py`** – Exact symbolic verification (SymPy rational arithmetic) of Jacobi identities in critical mixing sectors, confirming residuals identically zero.
- **`z3_entanglement.py`** – SVD decomposition proof that the cubic vacuum invariant corresponds to a maximally entangled GHZ-class state, establishing genuine tripartite quantum correlations.

#### Core 44-Vector Lattice & Gauge Unification
- **`z3_lattice_1.py`** **(Core – Newly Added)** – Refined ground-state pruning and geometric derivation of the **weak mixing angle sin²θ_W = 11/44 = 0.25**, exactly matching SU(5) GUT tree-level prediction; RG evolution explains low-energy shift to ∼0.231.
- **`z3_lattice.py`** **(Core)** – Generation and analysis of the emergent finite **44-vector ℤ₃-invariant lattice** from vacuum triality, demonstrating spontaneous discretization of continuous symmetry.
- **`z3_mass_6.py`** **(Core Script)** – Unified demonstration of **gauge unification** (44-vector core → sin²θ_W = 0.25) and **full charged fermion mass spectrum** (extended ℤ³ integer lattice sites).
- **`z3_strong_coupling.py`** – Generates the **44-vector Z₃ core lattice** and predicts relative strong coupling strength by geometrically partitioning vectors into weak- and strong-type components (matches observed strong/weak DoF ratio).

#### Standard Model Parameter Predictions
- **`z3_cosmo_constant.py`** – Computes combinatorial enhancement factor from 4-point vacuum fluctuations on the **44-vector Z₃ lattice**, relating it to the observed cosmological constant scale (∼10⁶ enhancement resolves hierarchy).
- **`z3_ckm_angles.py`** – Derives **CKM quark mixing angles** (V_us, V_cb, V_ub) geometrically via integer vector misalignment with democratic [1,1,1] direction (errors <5%).
- **`z3_cp_phase.py`** – Predicts **CKM CP-violating phase** (∼68–75°) from triality rotation angles and projective geometry on the 44-vector lattice.
- **`z3_higgs.py`** – Investigates **Higgs-to-top mass ratio** (m_H/m_t ≈ 0.726) using geometric length ratios in the 44-vector lattice.
- **`z3_mass_quarks.py`** – Searches ℤ³ integer vectors to reproduce **lighter quark masses** (up, down, strange, charm, bottom) via m ∝ 1/L², anchored to top quark (errors ∼few %).
- **`z3_pmns.py`** – Derives **neutrino PMNS mixing angles** from lattice geometry, recovering near-maximal atmospheric and tribimaximal solar mixing.
- **`z3_mass_ckm.py`** – Democratic matrix + vacuum misalignment perturbations reproducing **Cabibbo angle** and quark mass ratios (fitting error <3%).
- **`z3_Inverse.py`** – Computes inverse hierarchy exponent κ ≈ 12 − 3/13 linking algebraic scale to electroweak scale.
- **`z3_g_val_1.py`** – Geometric calculation of **gravitational constant G** via algebraic dimension ratios (relative error <0.02%).
The **44-vector lattice** (generated in `z3_lattice*.py` and used throughout) is the central geometric object yielding gauge unification and serving as the "core" for matter embeddings. The algebraic closure (`z3_algebra_5.py`) underpins the entire framework.

# Vacuum Inertia in Nanoscale Transport
This repository provides a complete, self-contained suite of reproducible Python scripts (using only NumPy, SymPy, Matplotlib, and Graphviz) for closed-loop symbolic and numerical validation of the Z₃ Vacuum Inertia framework. The scripts rigorously verify the full logic chain—from Z₃-graded Lie superalgebra construction and exact closure to ab initio quantitative predictions for THz skin depth saturation and nanoscale superconductivity enhancement—without external fitting parameters or unverified steps. Key features include symbolic derivations of core formulas, numerical Jacobi closure checks (residuals ≤ 10⁻¹³), reproducible experimental overlay figures, mindmap visualizations of the logic flow, and comprehensive demonstration of algebraic self-consistency, naturalness, quantitative validation, discriminating signatures, and theoretical constraints.s
- **`z3_vacuum_theory_chain_verify_fixed.py`**  
  Step-by-step symbolic verification of the full theoretical chain: graded brackets → effective coupling → in-medium renormalization → condensate → vacuum pairing → nanoscale Tc(d) enhancement. Generates closed-loop prediction plot from algebraic τ_vac.
- **`z3_quantitative_logic_chain_verify.py`**  
  Reproducible validation of the Quantitative Comparison section. Symbolically derives surface profile, skin depth saturation, and Tc thresholds. Prints manuscript tables and generates THz skin depth overlay plot.
- **`z3_theoretical_consistency_verify_fixed.py`**  
  Verification of the Theoretical Consistency section: RG flow/naturalness, ab initio timescale, phonon complementarity, discriminating signatures. Includes numerical Tc(d) illustration from τ_vac.
- **`z3_nami_sensitivity_show.py`**  
  Generates all three supplementary figures:  
  - Tc enhancement vs nanowire diameter (with experimental data overlay)  
  - THz skin depth saturation in high-purity copper  
  - Sensitivity of ξ_vac to surface enhancement η (robustness demonstration)
- **`z3_nanomaterials_chapter1_mindmap_vertical.py`**  
  Generates a vertical Graphviz mindmap of the complete Chapter 1 logic chain (colored nodes for reviewer concerns: algebra, coupling, renormalization, naturalness, validation, discriminating features, constraints).
  
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

### Profound Significance of This Work
The computational exploration culminates in the spontaneous emergence of a closed, finite 44-vector lattice from pure ℤ₃ triality operations on the vacuum sector. This saturation is not an artifact but a rigorous mathematical consequence of the unique cubic invariant and graded bracket structure.

This finite lattice resolves long-standing foundational issues in theoretical physics:
- It naturally constrains flavor mixing directions, eliminating arbitrary parameters in mass matrix ansätze.
- It offers a prototype for discrete spacetime or vacuum symmetry, bridging continuous field theories with emergent discreteness relevant to quantum gravity.
- Its triangular (A₂-like) symmetry enhanced by democratic deformations predicts specific correlations in neutrino oscillations, CP violation phases, lepton flavor violation ratios, and angular transport modulations in condensed matter systems.

Most profoundly, this framework marks a paradigm shift in our understanding of fundamental constants:

1. **The End of the "Parameter Era"**  
   Since Newton, physics has described nature with elegant equations—but the constants within them have been treated as arbitrary inputs from an unknown source. Why is the fine-structure constant 1/137? Why is the Weinberg angle ~0.23? Why do quarks mix in this specific way?  
   Prior to this work, the Standard Model could only measure and insert these values. For the first time, this framework derives key constants (e.g., sin²θ_W = 0.25 at unification from the rational ratio 11/44) from pure integer geometry and discrete algebraic structure—transitioning humanity from **measuring the universe** to **deriving the universe**.

2. **Unification of Force and Flavor**  
   Einstein's late dream—true unification of forces—remained incomplete. Grand Unified Theories (GUTs) addressed gauge forces but largely ignored fermion masses (flavor). Flavor symmetry models explained masses but rarely touched gauge couplings.  
   The 44-vector lattice achieves a deeper synthesis: 11 vectors define the weak sector geometry (yielding the Weinberg angle), while 24 hybrid vectors dictate hierarchical flavor textures (CKM matrix). The same emergent crystal simultaneously determines **both the strength of forces and the pattern of matter**—a unification surpassing previous attempts that relied on ad-hoc extra fields.

3. **Establishment of Discrete Geometry as Ontological Foundation**  
   If validated, this work triggers a philosophical revolution: continuity is illusion; discreteness is essence. The universe's substrate is not a smooth manifold but a rigid, finite, computable algebraic lattice. This impacts not only physics but computer science (universe as computation), philosophy (nature of reality), and mathematics (primacy of finite structures).

These results transform the framework from phenomenological description to a predictive geometric theory, with falsifiable signatures across energy scales.

**Note on Repository Updates**: To preserve the historical integrity of this discovery, no further content updates will be made to this repository until after the formal publication of related manuscripts. This ensures an immutable record of the work as it stood at key milestones.

Code and data are available at https://github.com/csoftxyz/RIA_EISA.

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
- **Z3_Ternary_UFO.zip** (in  directory)  
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
- ATLAS data. (2025). Measurement of the $t\bar{t}$ production cross section near threshold in pp collisions at √s = 13 TeV with the ATLAS detector. ATLAS-CONF-2025-008. Available at: https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf.

### Cover Video
- RIA_EISA Cover Video: https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4

### Science Education for Teenagers (Wiki Links)
- [Chapter 1: The "Lego Primary Colors" Manual for Physics](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter1.md)
- [Chapter 2: Setting Rules for Cosmic Lego—Physics’ "Lego Constitution"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter2.md)
- [Chapter 3: Weighing Cosmic Lego—Predicting Dark Matter with the "Lego Scale"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter3.md)
- [Chapter 4: The Lego Engine of an Expanding Universe—Stepping on the Gas for Cosmic Acceleration](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter4.md)
- [Chapter 5: Final Appendix: Issuing "Anti-Counterfeit Certificates" for Cosmic Lego](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter5.md)

### API Reference
- Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from c1.py.
- RNN model: `EnhancedRNNModel` from c2.py.

### Contributing
Welcome contributions! Fork, branch, commit, push, PR. Code of Conduct: Open-source ethics; no conflicts.

### Author Attitude
We hold deep respect for decades of work in string theory, quantum gravity, and related fields. This framework is offered humbly as an exploratory alternative perspective from independent researchers. We make no claim of superiority or finality — only a rigorous, testable structure open to scrutiny. Feedback, criticism, and collaboration are sincerely welcomed.

### Historical Development and Early Works
The current Z₃-graded framework evolved from earlier explorations of integrated symmetry algebras and transient quantum dynamics. These foundational ideas are documented in the following preprints and proceedings, demonstrating that the theory is not an isolated speculation but the result of systematic refinement over several years:
- **Early EISA Preprint Series** (initial concepts of Extended Integrated Symmetry Algebra):  
  v1: https://www.preprints.org/manuscript/202507.2681/v4  
  v7 (major refinement): https://www.preprints.org/manuscript/202507.2681/v7  
- **Foundational Proceedings Paper** (origin of transient quantum dynamics and graviton-cosmic phase transition ideas):  
  **Title**: Towards a Unified Framework of Transient Quantum Dynamics Integrating Graviton Models with Cosmic Phase Transitions and Observational Verifications  
  **Conference**: SPIE Proceedings of the International Conference on Quantum Optics and Photon Technology (QOFT 2024)  
  **Link**: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13705/1370524/Towards-a-unified-framework-of-transient-quantum-dynamics--integrating/10.1117/12.3070369.short

### Contact
- Issues tab for technical questions  
- Email: csoft@hotmail.com (corresponding) / csoft@live.cn  

Wiki pages are continuously updated with detailed guides and interpretations. Contributions welcome.

