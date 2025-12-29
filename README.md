
# RIA-EISA Simulation Repository README
**Yuxuan Zhang**^{a,b}, **Weitong Hu**^{c,*}, **Wei Zhang**^{d}  
^a College of Communication Engineering, Jilin University, Changchun, China  
^b csoft@live.cn  
^c Aviation University of Air Force, Changchun, China (Corresponding Author)  
^c csoft@hotmail.com  
^d College of Computer Science and Technology, Jilin University, Changchun, China  

## Overview
This repository contains the complete simulation and verification suite for the **Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality** series. The framework is a finite-dimensional (19D: 12+4+3) ℤ₃-graded algebraic structure from which Standard Model parameters, gravitational constant, cosmological constant, black-hole entropy scaling, and vacuum entanglement properties emerge as representation-theoretic invariants — with **zero free parameters**.

### Published & Preprinted Papers
1. **Published (Part 1 – Algebraic Foundation)**  
   **Title**: A Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality  
   **Journal**: Symmetry 2026, 18(1), 54  
   **DOI**: https://doi.org/10.3390/sym18010054  
   **PDF**: https://www.mdpi.com/2073-8994/18/1/54/pdf  

2. **Preprinted & Submitted (Part 2 – Phenomenological Extension)**  
   **Title**: An Exact Z₃-Graded Algebraic Framework Underlying Observed Fundamental Constants  
   **Preprint DOI**: https://doi.org/10.20944/preprints202512.2527.v1  
   **Link**: https://www.preprints.org/manuscript/202512.2527/v1  
   **Submitted**: Universe (MDPI), Manuscript ID: universe-4095403 (Under Review, Dec 2025)

### Core Verification Scripts (Self-Contained)
- `z3_algebra_5.py` – High-precision numerical Jacobi closure (residual ~10⁻¹⁶).
- `z3_grade_1.py` – Exact symbolic verification (SymPy rational arithmetic, residual identically zero).
- `z3_entanglement.py` – SVD proof of GHZ-class maximal entanglement for the cubic invariant.
- `z3_g_val_1.py` – Inverse geometric matching of gravitational constant G (relative error <0.02%).
- `z3_mass_ckm.py` – Democratic matrix + vacuum perturbations reproducing Cabibbo angle (error <3%).
- `z3_Inverse.py` – Full inverse hierarchy calculation confirming κ ≈ 12 − 3/13.

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

