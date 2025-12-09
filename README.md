# RIA-EISA Simulation Repository README

* 1 A Z3-Graded Lie Superalgebra with Cubic Vacuum Triality; Link: https://www.preprints.org/manuscript/202511.1876/v1
* 2 Cubic Vacuum Triality: A Toy Model of Everything with Zero Free Parameters; Link: https://www.preprints.org/manuscript/202512.0873/v1
* 3 Categorical Formalization of Recursive String-Inspired Symmetries: A First-Principles Approach to Quantum Field Dynamics 
* 4 Recursive Algebra in Extended Integrated Symmetry: An Effective Framework for Quantum Field Dynamics

Yuxuan Zhang^{a,b}, Weitong Hu^{c,*}
^a College of Communication Engineering, Jilin University, Changchun, China  
^c Aviation University of Air Force, Changchun, China  

* csoft@live.cn &  csoft@hotmail.com 
* https://github.com/csoftxyz/RIA_EISA/wiki  
* == About My Research Attitude ==  
I have great respect for scholars in string theory, quantum gravity and related fields. The theoretical systems built through decades of work represent invaluable achievements of human intellect.  
In proposing the EISA-RIA theory, I don't seek to challenge anyone or overthrow existing frameworks. As an independent researcher, I'm fully aware this work may never gain mainstream acceptance.  
But understanding reality's essence is my personal quest. Even if future experimental evidence fully supports this theory, I won't engage in academic debates—because I sincerely respect the dedication every scholar has poured into their models.  
My goal is simple: to verify my understanding of the cosmos in my own way. This has nothing to do with awards, positions, or recognition. It's just one individual's humble tribute to truth.  
Given that the paper is available as a preprint on preprints.org, please visit the URL: https://www.preprints.org/manuscript/202507.2681/v8  https://www.preprints.org/manuscript/202511.1876/v1 for access. We warmly welcome reviews and feedback from global peers. DOI: 10.20944/preprints202507.2681.v8 (registered DOI).  
Our initial idea, that quantum fluctuation lifting/dropping dynamics couple with curvature to generate gravitational effects, originated from our previous research. A related paper has now been published in the Proceedings of SPIE.: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13705/1370524/Towards-a-unified-framework-of-transient-quantum-dynamics--integrating/10.1117/12.3070369.short
* [Parameter Fitting in the EISA-RIA Framework](https://github.com/csoftxyz/RIA_EISA/wiki/Parameter_Fitting.md): Overview of Parameter Fitting.
* [Formula Self-Consistency ](https://github.com/csoftxyz/RIA_EISA/wiki/Formula_Self_Consistency.md): RIA-EISA Formula Self-Consistency Verification Report .
* ATLAS data. (2025). Measurement of top-antitop quark pair production near threshold in pp collisions at √s = 13 TeV with the ATLAS detector. ATLAS-CONF-2025-008. Available at: https://cds.cern.ch/record/2937636/files/ATLAS-CONF-2025-008.pdf.
* https://github.com/csoftxyz/RIA_EISA/blob/main/RIA_EISA%20Cover%20Video.mp4
## About My Research Attitude
## Overview  
This repository provides seven PyTorch-based numerical simulations supporting the Extended Integrated Symmetry Algebra (EISA) augmented by Recursive Info-Algebra (RIA) framework, as detailed in the manuscript "Recursive Algebra in Extended Integrated Symmetry: An Effective Framework for Quantum Field Dynamics" (LaTeX source: 71a8b.tex), submitted for consideration to Entropy as part of a PhD dissertation. These simulations validate key theoretical predictions, including self-organization from chaos via entropy minimization, transient fluctuations inducing curvature feedback and phase transitions, particle mass hierarchies and fundamental constants from irrep branching, cosmological evolution resolving Hubble tension with multi-messenger signatures, superalgebra closure, universe modeling on grids, and inverse CMB fitting.  
Each script is self-contained with comments for reproducibility and integrates EISA algebraic elements (e.g., generators \(F_i\), \(B_k\) for bosonic/fermionic sectors) with RIA recursion (VQC-optimized loss involving Von Neumann entropy \(S_{vn}\) and fidelity). Simulations approximate EFT dynamics below the Planck scale, with uncertainties ~10-20% due to simplifications (e.g., 4x4-64x64 matrices). Outputs quantify observables like entropy reduction (~40.2% average, std <1% across runs), GW frequencies (10^{17} to 10^{-16} Hz), mass ratios (~10^5), and H_0 values (~73 km/s/Mpc vs. Planck 67.4).  
To ensure numerical precision for PhD-level scrutiny and peer review, simulations were executed on CPU platforms with ECC memory to mitigate bit errors, maintaining floating-point consistency. Execution times are under 1 hour per 1000 iterations on standard hardware (e.g., Intel i7, 32GB RAM). Outputs encompass console metrics, plots, and logs, all preserved for peer review.  
**Repository Structure**:  
- **c1.py**: Recursive Entropy Stabilization  
- **c2.py**: Transient Fluctuations and Curvature Feedback  
- **c3.py**: Particle Spectra and Constant Freezing  
- **c4.py**: Cosmic Evolution and Multi-Messenger Predictions  
- **c5.py**: Superalgebra Verification and Bayesian Analysis  
- **c6.py**: EISA Universe Simulator  
- **c7.py**: CMB Power Spectrum Inverse Analysis
- **z3_algebra_4.py**: Z₃-Graded Lie Superalgebra Numerical Verifier
  
Z₃-Graded Lie Superalgebra Numerical Verifier

A Python implementation for verifying the algebraic closure of a 15-dimensional Z₃-graded Lie superalgebra with cubic vacuum triality.

Overview

This code provides numerical verification of a finite-dimensional Z₃-graded Lie superalgebra structure that unifies gauge fields, fermionic matter, and vacuum sectors. The implementation focuses on the critical mixing sector between these components and demonstrates exact closure of the graded Jacobi identities.

Key Features

• 15-dimensional representation (9 gauge + 3 fermionic + 3 vacuum generators)

• Z₃-graded bracket operations with commutation factor ω = e^(2πi/3)

• U(3) gauge sector implementation using Gell-Mann matrices

• Exact Jacobi identity verification with machine-precision residuals (~10⁻¹⁶)

• Unique mixing term [F, ζ] = -TᵃBᵃ fixed by representation invariance

Installation & Usage

# Install dependencies
pip install numpy

# Run verification
python z3_algebra_4.py


Expected Output

Successful execution will display:

----------------------------------------
FINAL RESIDUAL: 3.2456e-16
----------------------------------------
[VICTORY] The Z3 Vacuum Coupling is Mathematically Exact.
Structure: [F, Z] = - T^a B^a


Mathematical Background

The code verifies the algebraic structure proposed in "A Z₃-Graded Lie Superalgebra with Cubic Vacuum Triality" (Symmetry submission ID: symmetry-4031108), demonstrating:

• Graded Jacobi identities with Z₃ commutation factors

• Faithful matrix representation of the algebra

• Zero-parameter construction - all coefficients fixed by representation theory

Files

• z3_algebra_4.py - Main verification script

• requirements.txt - Python dependencies

Citation

This code accompanies the theoretical work on Z₃-graded algebraic structures. Please cite the relevant papers if used in research.
- 
For detailed methodology and results, refer to Sections IV (Computational Methods) and V (Results), and Appendix A (Supplementary Information) in the manuscript.  
## Dependencies and Setup  
- **Python Version**: 3.12+ (tested on 3.12.3)  
- **Key Libraries** (install via pip or conda):  
  - PyTorch: `pip install torch`  
  - torchdiffeq: `pip install torchdiffeq` (for ODE in c4.py)  
  - NumPy, SciPy, Pandas, Matplotlib: `pip install numpy scipy pandas matplotlib`  
  - Additional for c3.py: mpmath, sympy (optional for extensions)  
  - For c7.py: emcee, corner (MCMC and plots)  
- **Environment Setup**:  
  ```bash
  python -m venv ria_env  
  source ria_env/bin/activate # On Windows: ria_env\Scripts\activate  
  pip install -r requirements.txt # Create requirements.txt with listed packages  
  ```  
- **Execution Notes**:  
  - Run scripts with: `python c1.py`  
  - Set `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` to suppress library warnings.  
  - For reproducibility, fix random seeds (e.g., `torch.manual_seed(42)`), critical for PhD validation and peer review.  
  - Hardware: CPU recommended for precision; tested on dual Intel Xeon with ECC RAM to ensure consistency under scrutiny.  
## Simulations and Evaluation Against Manuscript  
### c1.py: Recursive Entropy Stabilization  
- **Description**: Evolves EISA-perturbed density matrices (4x4, using generators like Pauli matrices) via VQC (RX, RY, CNOT gates) with structured noise (\(\eta \sim 0.005\)), minimizing loss \(S_{vn} + (1 - Fid) + 0.5(1 - P)\) to achieve an ordered state. Projects to PSD for stability; tracks entropy, fidelity, and parameter trajectories.  
- **Key Parameters**: \(\eta = 0.005\), learning_rate=0.0005, num_iterations=2000, num_layers=8.  
- **Execution Results vs. Manuscript**: Aligns with Section V.A and Appendix S1. Manuscript reports entropy reduction from ~0.1453 to ~0.0869 (40.2% average, std <1% over 10 runs with varying seeds), fidelity up to ~0.8 (mean ~0.8 ±0.05 over 10 runs, uncertainties ~20-30%). Outputs match initial entropy ~0.1453, final ~0.0869, fidelity threshold (Fid>0.8) reached at ~1000 iters. Visualizations confirm convergence.  
- **Outputs**: Console: final entropy/fidelity/reduction; Plots: entropy/fidelity/loss trajectories, histograms, density matrix heatmaps.  
### c2.py: Transient Fluctuations and Curvature Feedback  
- **Description**: Models scalar \(\phi(t)\) evolution via RNN with deformation \(\epsilon(t) = e^{-t / \tau_P}\), non-local coupling, and Laplacian for curvature \(R \propto \kappa \langle \phi^\dagger \nabla^2 \phi \rangle\). Detects order parameter jumps for GW emission; incorporates noise for vacuum fluctuations.  
- **Key Parameters**: \(\tau_P = 10^{-18}\) s, grid_size=100, num_steps=5000, kappa=0.5, dt=1 s.  
- **Execution Results vs. Manuscript**: Matches Section V.B and Appendix S2. Manuscript: curvature std ~5%, GW 10^{17} to 10^{-16} Hz, CMB ~10^{-7}, uncertainties ~20-30%. Outputs: std dev ~5%, peak time ~10^{-8} s, GW freq range matches, CMB dev ~1e-7. Visualizations include trajectories and histograms.  
- **Outputs**: Console: std deviation, peak time, GW freq, CMB dev; Plots: curvature/order trajectories, GW histogram, phase space.  
### c3.py: Particle Spectra and Constant Freezing  
- **Description**: Optimizes vacuum potential \(V(\Phi)\) via gradient descent with VQC rotations; computes Casimir-scaled hierarchies (fund vs. adj irreps), fractal ratios (~1.618), and constants (e.g., \(\alpha = 1/(137 + \mathcal{N}(0,0.001))\)); generates electron clouds with non-local terms. Extended to dim=64 for higher-dimensional consistency.  
- **Key Parameters**: \(\mu = -1.0\), lambda=0.1, kappa=0.5, num_iterations=500, learning_rate=0.01.  
- **Execution Results vs. Manuscript**: Consistent with Section V.C and Appendix S3. Manuscript: hierarchies ~10^5, constants <1% CODATA error (e.g., \(\alpha\) ~0.00735), std ~0.05% over 10 runs, uncertainties ~20-30%. Outputs: VEV/potential decline, fractal ratio ~1.618, constants errors <1%. Visualizations include evolutions and slices.  
- **Outputs**: Console: iteration metrics, constants table; Plots: VEV/potential/ratio trajectories, error bars, Phi scatters, phase spaces, charts, potential surface.  
### c4.py: Cosmic Evolution and Multi-Messenger Predictions  
- **Description**: Solves Friedmann ODE with RIA densities (\(\Omega_m, \Omega_r, \Omega_\Lambda, \Omega_v\)) using torchdiffeq; adds crackling perturbations for solitons; computes H(tau), densities, CMB power deviations, GW spectra.  
- **Key Parameters**: Omega_m=0.315, Omega_Lambda=0.685, tau_decay=1e-9/2.18e-18, tau_span=1e-10 to 10.0, atol=1e-10.  
- **Execution Results vs. Manuscript**: Agrees with Section V.D and Appendix S4. Manuscript: late H ~0.8-1.0, H0 tension (~73 vs. 67.4 km/s/Mpc), densities ±5%, CMB dev ~10^{-8}, GW peak ~10^{-8} Hz (std<3%). Outputs: H ~0.8-1.0, GW peak ~10^{-8} Hz, CMB ~10^{-8}. Visualizations show evolutions and spectra.  
- **Outputs**: Console: simulated H, GW peak; Plots: a(tau)/H(tau)/Omega evolutions, CMB/GW loglogs, Hubble bar, phase spaces.  
### c5.py: Superalgebra Verification and Bayesian Analysis  
- **Description**: Verifies mathematical completeness: SymPy for Super-Jacobi identities (algebraic closure) and NumPy for Bayesian evidence ratio (Hubble tension resolution). Extended to 64x64 matrices.  
- **Key Parameters**: n_samples=500000 for Monte Carlo; data=71.0 (H0 intermediate).  
- **Execution Results vs. Manuscript**: Matches Appendices A & B. Super-Jacobi: all low-dim true, 64x64 residuals <1e-10. Bayesian: log-difference ~2.3 favoring RIA. Supports "near-perfect coherence".  
- **Outputs**: Console: verification results; Plots: residuals heatmap, posterior scatterplot.  
### c6.py: EISA Universe Simulator  
- **Description**: Models RG flow and particle generation on 64x64x64 grid with EISA algebra; evolves fields b, ϕ, ζ over early universe (10^{-36} to 10^{-32} s); computes α, G, c from commutators.  
- **Key Parameters**: grid_size=64, dt=10^{-36}, t_end=10^{-32}.  
- **Execution Results vs. Manuscript**: Aligns with Section V.F. Manuscript: avg α ~0.0073 ±0.0000 (<1% CODATA), particle densities proportional to |ϕ|^2 + |ζ|^2. Outputs match with uncertainties ~10-20%.  
- **Outputs**: Console: alpha, densities; Plots: densities over time, RG flow, field slices, constants.  
### c7.py: CMB Power Spectrum Inverse Analysis  
- **Description**: Fits Planck 2018 TT data via MCMC (emcee) to recover θ=[κ, n, A_v]; uses VQC-optimized forward model for D_ℓ; compares to ΛCDM.  
- **Key Parameters**: walkers=32, steps=5000, bounds=[(0.1,0.5),(5,10),(1e-10,2.5e-9)].  
- **Execution Results vs. Manuscript**: Matches Section V.G. Manuscript: κ≈0.31, n≈7, A_v≈2.1×10^{-9}, χ²/dof~1.1 (vs. ΛCDM 1.0-1.03). Outputs: recovered params within bounds, χ²/dof~1.1.  
- **Outputs**: Console: ML point, χ²; Plots: fit/residuals, corner plot.  
## Detailed Guides  
- [EISA Algebra Basics](https://github.com/csoftxyz/RIA_EISA/wiki/eisa_algebra.md): Explains superalgebra structure and generators.  
- [RIA Optimization](https://github.com/csoftxyz/RIA_EISA/wiki/ria_optimization.md): VQC setup and loss minimization.  
- [Simulation Tutorials](https://github.com/csoftxyz/RIA_EISA/wiki/simulations/): Step-by-step for each script (c1.py etc.).  
- [Validation Code (c5.py)](https://github.com/csoftxyz/RIA_EISA/wiki/validation.md): SymPy for Super-Jacobi and Bayesian analysis.  
- [Universe Simulator (c6.py)](https://github.com/csoftxyz/RIA_EISA/wiki/universe_simulator.md): Grid-based RG flow and particle generation.  
- [CMB Inverse Analysis (c7.py)](https://github.com/csoftxyz/RIA_EISA/wiki/cmb_inverse.md): MCMC fitting to Planck data.  
- [Equation Self-Consistency in the Manuscript](https://github.com/csoftxyz/RIA_EISA/wiki/equation_self_consistency.md): Overview of how equations in the manuscript are internally consistent.  
- [Fun Interpretations of Equations in the Manuscript](https://github.com/csoftxyz/RIA_EISA/wiki/Fun-Interpretations-of-Equations-in-the-Manuscript): Offers playful and accessible explanations of the equations in the manuscript, designed to engage young learners with colorful analogies and examples.  
## Possible Related Experiments  
- [MIT Double-Slit Experiment](https://github.com/csoftxyz/RIA_EISA/wiki/MIT_Double_Slit_Experiment.md): MIT Double-Slit Experiment with Single-Atom Wave Packets and EISA-RIA Interpretation.  
- [NANOGrav GW Background](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background.md): NANOGrav GW Background Power Spectrum Features and EISA-RIA Interpretation.  
- [NANOGrav GW Background Frequency Range & Amplitude](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Frequency_Range_Amplitude.md): NANOGrav GW Background Frequency Range & Amplitude Features and EISA-RIA Interpretation.  
- [NANOGrav GW Background Polarization Modes](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Polarization_Modes.md): NANOGrav GW Background Polarization Modes Features and EISA-RIA Interpretation.  
- [NANOGrav GW Background Non-Gaussianity & Transients](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Non_Gaussianity_Transients.md): NANOGrav GW Background Non-Gaussianity & Transients Features and EISA-RIA Interpretation.  
- [NANOGrav GW Background Multi-Messenger Correlations Features](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Multi_Messenger_Correlations_Features.md): NANOGrav GW Background Multi-Messenger Correlations Features and EISA-RIA Interpretation.  
- [NANOGrav GW Background Cosmological Integration Features](https://github.com/csoftxyz/RIA_EISA/wiki/NANOGrav_GW_Background_Cosmological_Integration_Features.md): NANOGrav GW Background Cosmological Integration Features and EISA-RIA Interpretation.  
- [LHC Mass Anomalies](https://github.com/csoftxyz/RIA_EISA/wiki/LHC_Mass_Anomalies.md): LHC Mass Anomalies and EISA-RIA Interpretation.  
- [CMB Deviations](https://github.com/csoftxyz/RIA_EISA/wiki/CMB_Deviations.md): CMB Deviations and EISA-RIA Interpretation.  
- [SLAC/Brookhaven Breit-Wheeler Experiment with Photon-Photon Collisions](https://github.com/csoftxyz/RIA_EISA/wiki/SLAC_Brookhaven.md): SLAC/Brookhaven Breit-Wheeler Experiment with Photon-Photon Collisions and EISA-RIA Interpretation.  
- [Muon g-2 Experiment with Anomaly Resolution ](https://github.com/csoftxyz/RIA_EISA/wiki/Muon_g_2.md): Muon g-2 Experiment with Anomaly Resolution and EISA-RIA Interpretation.  
- [Neutrino Mass Hierarchy and CP Violation with JUNO Prospects ](https://github.com/csoftxyz/RIA_EISA/wiki/Neutrino_Mass.md): Neutrino Mass Hierarchy and CP Violation with JUNO Prospects and EISA-RIA Interpretation.  
- [Lepton Flavor Universality Violation (LHCb Legacy Issue) ](https://github.com/csoftxyz/RIA_EISA/wiki/LHCb_Legacy_Issue.md): Lepton Flavor Universality Violation (LHCb Legacy Issue) and EISA-RIA Interpretation.  
- [EISA-RIA Predictions for New Particles at High Energies](https://github.com/csoftxyz/RIA_EISA/wiki/New_Particles_at_High_Energies.md): EISA-RIA Predictions for New Particles at High Energies.  
## Science Education for Teenagers  
- [Chapter 1 ](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter1.md): The "Lego Primary Colors" Manual for Physics  
- [Chapter 2 ](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter2.md): Setting Rules for Cosmic Lego—Physics’ "Lego Constitution"  
- [Chapter 3 ](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter3.md): Weighing Cosmic Lego—Predicting Dark Matter with the "Lego Scale"  
- [Chapter 4 ](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter4.md): The Lego Engine of an Expanding Universe—Stepping on the Gas for Cosmic Acceleration  
- [Chapter 5 ](https://github.com/csoftxyz/RIA_EISA/wiki/Chapter5.md): Final Appendix: Issuing "Anti-Counterfeit Certificates" for Cosmic Lego  
### API Reference  
- Core functions: `project_to_psd()`, `von_neumann_entropy()`, `fidelity()` from c1.py.  
- RNN model: `EnhancedRNNModel` from c2.py.  
## Contributing  
We welcome contributions! Follow these steps:  
1. Fork the repo.  
2. Create a branch: `git checkout -b feature/new-sim`.  
3. Commit changes: `git commit -m "Add higher-dim extension"`.  
4. Push and open a PR.  
Code of Conduct: Adhere to open-source ethics; no conflicts of interest.  
## Contact  
For queries: csoft@hotmail.com.  
Join discussions on [Issues](https://github.com/csoftxyz/RIA_EISA/issues) or email authors.  
This wiki is collaboratively editable—help improve it!  
## Data Availability and Ethical Statement  
All codes, parameters, and generated data (e.g., trajectories, spectra, plots) are available in this repository, submitted as supplementary material for Entropy review. No external datasets used; random seeds ensure reproducibility (e.g., `torch.manual_seed(42)`). Monte Carlo analyses (10-20 runs per simulation) confirm robustness, with reported std devs.  
**Computational Integrity**: Simulations are algorithmic; no subjective experience implied, adhering to AI ethics. Open-source principles followed; no conflicts of interest.  
For issues or contributions, contact csoft@hotmail.com & csoft@live.cn.
