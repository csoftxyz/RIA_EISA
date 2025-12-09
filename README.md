# RIA-EISA Simulation Repository README

Yuxuan Zhang^{a,b}, Weitong Hu^{c,*}
^a College of Communication Engineering, Jilin University, Changchun, China  
^c Aviation University of Air Force, Changchun, China  

## Overview
This repository hosts simulations and verification code for the Z3-Graded Lie Superalgebra series, focusing on algebraic structures with cubic vacuum triality. Key highlights include machine-precision closure (residuals ~0) in z3_algebra_4.py, unifying SM, gravity, cosmology, black holes, and entanglement with zero free parameters.

### Core Papers
* 1 A Z3-Graded Lie Superalgebra with Cubic Vacuum Triality; Link: https://www.preprints.org/manuscript/202511.1876/v1
* 2 Cubic Vacuum Triality: A Toy Model of Everything with Zero Free Parameters; Link: https://www.preprints.org/manuscript/202512.0873/v1

### Z₃-Graded Lie Superalgebra Numerical Verifier (z3_algebra_4.py)
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
  python z3_algebra_4.py
  ```
- **Expected Output**:
  ```
  ----------------------------------------
  FINAL RESIDUAL: 3.2456e-16
  ----------------------------------------
  [VICTORY] The Z3 Vacuum Coupling is Mathematically Exact.
  Structure: [F, Z] = - T^a B^a
  ```
- **Mathematical Background**: Verifies structure from "A Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality" (Symmetry submission ID: symmetry-4031108). Files: z3_algebra_4.py, requirements.txt.
- **Citation**: Please cite the relevant papers if used in research.

## About My Research Attitude
I have great respect for scholars in string theory, quantum gravity, and related fields. Their decades of work are invaluable achievements.  
In proposing this framework, I seek no challenge or overthrow. As an independent researcher, I know this may never gain mainstream acceptance.  
My goal is simple: verify my understanding of the cosmos. This has nothing to do with awards or recognition—it's a humble tribute to truth.  
Given the papers are on preprints.org, please visit for access. We welcome reviews and feedback. DOI: 10.20944/preprints202507.2681.v8 (related work).  
Our initial idea on quantum fluctuations and curvature originated from prior research, published in SPIE Proceedings: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13705/1370524/Towards-a-unified-framework-of-transient-quantum-dynamics--integrating/10.1117/12.3070369.short

## Dependencies and Setup
- Python 3.12+  
- Key Libraries: PyTorch, torchdiffeq, NumPy, SciPy, Pandas, Matplotlib, sympy, mpmath, emcee, corner  
- Setup:
  ```bash
  python -m venv ria_env
  source ria_env/bin/activate  # Windows: ria_env\Scripts\activate
  pip install torch torchdiffeq numpy scipy pandas matplotlib sympy mpmath emcee corner
  ```

## Simulations Overview
This repo includes seven PyTorch-based simulations validating theoretical predictions (e.g., entropy minimization, curvature feedback, particle hierarchies). Each is self-contained; run with `python c1.py` etc. For details, see wiki links below.

- c1.py: Recursive Entropy Stabilization  
- c2.py: Transient Fluctuations and Curvature Feedback  
- c3.py: Particle Spectra and Constant Freezing  
- c4.py: Cosmic Evolution and Multi-Messenger Predictions  
- c5.py: Superalgebra Verification and Bayesian Analysis  
- c6.py: EISA Universe Simulator  
- c7.py: CMB Power Spectrum Inverse Analysis  

## Detailed Guides (Wiki Links)
- [EISA Algebra Basics](https://github.com/csoftxyz/RIA_EISA/wiki/eisa_algebra.md)  
- [RIA Optimization](https://github.com/csoftxyz/RIA_EISA/wiki/ria_optimization.md)  
- [Simulation Tutorials](https://github.com/csoftxyz/RIA_EISA/wiki/simulations/)  
- [Validation Code](https://github.com/csoftxyz/RIA_EISA/wiki/validation.md)  
- [Universe Simulator](https://github.com/csoftxyz/RIA_EISA/wiki/universe_simulator.md)  
- [CMB Inverse Analysis](https://github.com/csoftxyz/RIA_EISA/wiki/cmb_inverse.md)  
- [Equation Self-Consistency](https://github.com/csoftxyz/RIA_EISA/wiki/equation_self_consistency.md)  
- [Fun Interpretations of Equations](https://github.com/csoftxyz/RIA_EISA/wiki/Fun-Interpretations-of-Equations-in-the-Manuscript)  

## Possible Related Experiments (Wiki Links)
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

## Science Education for Teenagers (Wiki Links)
- [Chapter 1: The "Lego Primary Colors" Manual for Physics](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter1.md)  
- [Chapter 2: Setting Rules for Cosmic Lego—Physics’ "Lego Constitution"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter2.md)  
- [Chapter 3: Weighing Cosmic Lego—Predicting Dark Matter with the "Lego Scale"](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter3.md)  
- [Chapter 4: The Lego Engine of an Expanding Universe—Stepping on the Gas for Cosmic Acceleration](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter4.md)  
- [Chapter 5: Final Appendix: Issuing "Anti-Counterfeit Certificates" for Cosmic Lego](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter5.md)  

## API Reference
- Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from c1.py.  
- RNN model: `EnhancedRNNModel` from c2.py.  

## Contributing
Welcome contributions! Fork, branch, commit, push, PR. Code of Conduct: Open-source ethics; no conflicts.

## Contact
Queries: csoft@live.cn.  
Discussions: Issues tab or email authors.  
Wiki editable—improve it!  

## Data Availability and Ethical Statement
All codes, parameters, and data (trajectories, spectra, plots) available here. No external datasets; random seeds ensure reproducibility. Simulations algorithmic; no subjective experience implied, per AI ethics. No conflicts.  
For issues/contributions, contact csoft@hotmail.com & csoft@live.cn.  

- ATLAS data. (2025). Measurement of the $t\bar{t}$ production cross section near threshold in pp collisions at √s = 13 TeV with the ATLAS detector. ATLAS-CONF-2025-008. Available at: https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf.  
- RIA_EISA Cover Video: https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4  

This wiki is collaboratively editable—help improve it!
