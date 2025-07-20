# RIA-EISA Simulation Repository README

Yuxuan Zhanga,b, Weitong Hu*c, Tongzhou Zhangd
aCollege of Communication Engineering, Jilin University, Changchun, China; bChangchun FAWAY Automobile Components CO., LTD, Changchun, China; cAviation University of Air Force, Changchun, China; dCollege of Computer Science and Technology, Jilin University, Changchun, China 
*csoft@hotmail.com

## Overview

This repository contains five PyTorch-based numerical simulations supporting the Extended Integrated Symmetry Algebra (EISA) with Recursive Info-Algebra (RIA) framework, as described in the manuscript "Recursive Algebra in Extended Integrated Symmetry: An Effective Framework for Quantum Field Dynamics" submitted to *XXXXXXXXXX*. These simulations validate key theoretical predictions, including self-organization, transient fluctuations, particle hierarchies, cosmological evolution, and emergent information dynamics. Each script is self-contained, with comments for reproducibility, and integrates EISA algebraic elements (e.g., generators \(F_i\), \(B_k\)) with RIA recursion (VQC-optimized loss \(S_{vn} + (1 - Fid)\)).

The simulations demonstrate the framework's coherence under constrained computation, providing falsifiable outputs for empirical testing. For manuscript context, refer to the Supplementary Information section.

**Repository Structure**:
# RIA_EISA Simulations

This repository contains five PyTorch-based simulations supporting the EISA-RIA framework for quantum field dynamics.

## c1b.py: Recursive Entropy Stabilization
Evolves noisy density matrices via RIA recursion with EISA seeding to quantify self-organization, verifying entropy reduction and fidelity thresholds.

## c2a.py: Transient Fluctuations and Curvature Feedback
Models φ(t) dynamics under ε(t), computing curvature and GW predictions to test phase transition criticality.

## c3a1.py: Particle Spectra and Constant Freezing
Computes branching and constants via potential minimization, verifying hierarchies and norms.

## c4a.py: Cosmic Evolution and Multi-Messenger Predictions
Solves Friedmann with densities, forecasting signals and tension resolution.

## c5c.py: Quantum Information Dynamics
Simulates dynamics under constraints with network data.

For details, refer to the paper.

**Data Availability**: All codes and generated data (e.g., trajectories, spectra) are openly available here. No external datasets required; simulations use random seeds for reproducibility (set via `torch.manual_seed(42)` if needed).

## Dependencies and Setup

- **Python Version**: 3.12+ (tested on 3.12.3).
- **Key Libraries** (install via pip or conda):
  - PyTorch (>=2.0): `pip install torch`
  - torchdiffeq: `pip install torchdiffeq` (for ODE in c4a.py)
  - NumPy, Matplotlib: `pip install numpy matplotlib`
  - Additional for c5c.py: aiohttp, tqdm, tensorboard: `pip install aiohttp nest-asyncio tqdm tensorboard`
  - SciPy (for sph_harm in c3a1.py): `pip install scipy`
- **Environment Setup**: Create a virtual environment: `python -m venv ria_env; source ria_env/bin/activate; pip install -r requirements.txt` (provide requirements.txt with above).
- **Hardware**: CPU sufficient; GPU accelerates VQC (enable via `torch.cuda.is_available()`). Tested on Intel i7/32GB RAM/RTX 3060.
- **Running**: Execute each script individually (e.g., `python c1b.py`). Outputs include console metrics, plots in `visualizations/`, and logs. For c5c.py, ensure internet for network access.


