import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.special import sph_harm

# Configuration
DIM = 64  # Matrix dimension for EISA representations
MU_SQ = -1.0  # Negative for spontaneous breaking
LAMBDA = 0.1  # Self-interaction
KAPPA = 0.5  # Curvature coupling
NUM_ITERATIONS = 500
LEARNING_RATE = 0.01

# Generate EISA generators (bosonic B_k, fermionic F_i for SU(3)-like)
def generate_eisa_generators(dim=DIM):
    # Structure constants for SU(3)
    f_abc = np.zeros((8, 8, 8))  # Gell-Mann lambda/2 generators implied
    # Simplified: Use random Hermitian for B (bosonic), anti-Hermitian for F (fermionic)
    B = [torch.randn(dim, dim, dtype=torch.complex128) for _ in range(8)]
    for i in range(8):
        B[i] = (B[i] + B[i].conj().T) / 2  # Hermitian

    F = [1j * torch.randn(dim, dim, dtype=torch.complex128) for _ in range(8)]
    for i in range(8):
        F[i] = (F[i] - F[i].conj().T) / 2  # Anti-Hermitian

    return B, F

# Effective potential V(Phi) from EISA (trace over representations)
def effective_potential(Phi, mu_sq, lambda_param, kappa, B, F):
    # ||Phi||_F^2
    norm_sq = torch.norm(Phi, p='fro')**2

    # Trace term with dynamic R ~ commutators (from A_Grav)
    comm = sum(B[i] @ Phi - Phi @ B[i] for i in range(len(B))) / len(B)  # Avg commutator
    anticomm = sum(F[i] @ Phi + Phi @ F[i] for i in range(len(F))) / len(F)
    R_dynamic = kappa * torch.real(torch.trace(comm @ anticomm.conj().T))  # Approx curvature from algebra

    # V = mu^2 Tr(Phi^dag Phi) + lambda (Tr(Phi^dag Phi))^2 + kappa R Tr(Phi^dag Phi)
    trace_term = torch.real(torch.trace(Phi.conj().T @ Phi))
    V = mu_sq * trace_term + lambda_param * trace_term**2 + kappa * R_dynamic * trace_term
    return V

# Compute masses from eigenvalues of mass matrix ~ d^2 V / d Phi^2 (approx Hessian)
def compute_masses(Phi, lambda_param):
    # Simplified mass matrix ~ 2 lambda Phi^dag Phi (for Higgs-like)
    mass_matrix = 2 * lambda_param * (Phi.conj().T @ Phi)
    eigenvalues = torch.linalg.eigvals(mass_matrix).real
    masses = torch.sqrt(eigenvalues.clamp(min=0))  # Positive sqrt for physical masses
    return masses.sort(descending=True)[0]  # Sorted hierarchy

# Compute alpha from Tr(Q^2) / ||Phi_VEV||_F (Q ~ charge operator from A_SM)
def compute_alpha(Phi_vev):
    # Q ~ diag(2/3, -1/3, -1/3) for quarks, extended to dim
    Q_diag = torch.tensor([2/3, -1/3, -1/3] + [0]*(DIM-3), dtype=torch.complex128)
    Q = torch.diag(Q_diag)
    tr_Q2 = torch.real(torch.trace(Q @ Q))
    vev_norm = torch.norm(Phi_vev, p='fro')
    alpha = (1 / (4 * np.pi)) * tr_Q2 / (vev_norm + 1e-8)
    return alpha.item()

# Compute G from 1 / (16 pi vev^2) approx (Planck scale from vev)
def compute_G(Phi_vev):
    vev_norm = torch.norm(Phi_vev, p='fro')
    G = 1 / (16 * np.pi * vev_norm**2)
    return G.item()

# Fractal ratio from mass hierarchy (ratio of consecutive masses)
def fractal_mass_ratio(masses):
    if len(masses) < 2:
        return 1.0
    ratios = masses[:-1] / masses[1:]
    return torch.mean(ratios).item()  # Avg ratio, emerges from hierarchy

# Electron cloud from Phi (project to spherical coords)
def electron_cloud_from_phi(Phi, r, theta, phi_mesh):
    # Project Phi eigenvalues to radial/angular
    eig = torch.linalg.eigvals(Phi).real.numpy()
    radial = np.exp(-r) / np.sqrt(np.pi)  # Bohr-like
    angular = np.mean(eig) * sph_harm(0, 0, phi_mesh, theta).real  # s-orbital avg
    non_local = np.trapz(eig, axis=0)  # Integral over "modes"
    density = radial[:, None, None] * angular + non_local
    return density**2

# Main simulation
B, F = generate_eisa_generators()

# Phi as optimizable parameter (direct matrix optimization)
Phi = torch.nn.Parameter(torch.complex(torch.randn(DIM, DIM), torch.randn(DIM, DIM)))
optimizer = optim.Adam([Phi], lr=LEARNING_RATE)

vev_history = []
potential_history = []
mass_ratio_history = []

for iter in range(NUM_ITERATIONS):
    optimizer.zero_grad()
    V = effective_potential(Phi, MU_SQ, LAMBDA, KAPPA, B, F)
    V.backward()
    optimizer.step()

    vev = torch.norm(Phi, p='fro').item()
    vev_history.append(vev)
    potential_history.append(V.item())

    masses = compute_masses(Phi, LAMBDA)
    fractal_ratio = fractal_mass_ratio(masses)
    mass_ratio_history.append(fractal_ratio)

    if iter % 100 == 0:
        print(f"Iter {iter}: VEV {vev:.4f}, Potential {V.item():.4f}, Fractal Ratio {fractal_ratio:.4f}")

# Final computations
Phi_vev = Phi.detach()
alpha_theory = compute_alpha(Phi_vev)
G_theory = compute_G(Phi_vev)

alpha_codata = 1 / 137.035999
G_codata = 6.67430e-11

error_alpha = abs((alpha_theory - alpha_codata) / alpha_codata) * 100
error_G = abs((G_theory - G_codata) / G_codata) * 100

# Constants table
constants_data = {
    'Constant': ['alpha', 'G'],
    'Theory': [alpha_theory, G_theory],
    'CODATA': [alpha_codata, G_codata],
    'Error (%)': [error_alpha, error_G]
}
df_constants = pd.DataFrame(constants_data)
print(df_constants)

# Electron cloud
r = np.linspace(0, 10, 100)
theta, phi_mesh = np.meshgrid(np.linspace(0, np.pi, 50), np.linspace(0, 2*np.pi, 50))
rho_e = electron_cloud_from_phi(Phi_vev, r, theta, phi_mesh)

# Visualizations (combined PDF)
with PdfPages('c3a3_mass_hierarchy.pdf') as pdf:
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0,0].plot(vev_history)
    axs[0,0].set_title('VEV Evolution')
    axs[0,0].grid(True)

    axs[0,1].plot(potential_history)
    axs[0,1].set_title('Vacuum Potential Trajectory')
    axs[0,1].grid(True)

    axs[0,2].plot(mass_ratio_history)
    axs[0,2].set_title('Fractal Mass Ratio Trajectory')
    axs[0,2].grid(True)

    axs[1,0].bar(df_constants['Constant'], df_constants['Error (%)'])
    axs[1,0].set_title('Error (%) vs CODATA')
    axs[1,0].grid(True)

    eig = torch.linalg.eigvals(Phi_vev)
    axs[1,1].scatter(eig.real.numpy(), eig.imag.numpy())
    axs[1,1].set_title('Final Phi Eigenvalues')
    axs[1,1].grid(True)

    axs[1,2].scatter(vev_history, potential_history, c=range(len(vev_history)))
    axs[1,2].set_title('VEV vs Potential')
    axs[1,2].grid(True)

    axs[2,0].hist(mass_ratio_history, bins=20)
    axs[2,0].set_title('Histogram of Fractal Ratios')
    axs[2,0].grid(True)

    axs[2,1].contourf(r[50] * np.sin(theta) * np.cos(phi_mesh), r[50] * np.sin(theta) * np.sin(phi_mesh), rho_e[50])
    axs[2,1].set_title('Electron Cloud 2D Density (Slice)')

    pdf.savefig(fig)
    plt.close(fig)

print("Simulation complete. Visualizations in 'c3a3_mass_hierarchy.pdf'.")