
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Configuration for matrix dimension - set to 64 as per paper; reduce for testing if needed
DIM = 64  # Full 64x64 for consistency with EISA triple superalgebra representation
EPSILON = 1e-8  # Reduced regularization for more physical behavior

# Project to PSD using symmetric eigenvalue decomposition (more stable, less aggressive clamping)
def project_to_psd(rho, epsilon=EPSILON):
    rho = (rho + rho.conj().T) / 2  # Ensure Hermitian
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = eigenvalues.real.clamp(min=0.0)  # Clamp to non-negative (physical)
    trace = eigenvalues.sum() + epsilon  # Small epsilon only for trace stability
    eigenvalues = eigenvalues / trace  # Normalize trace to 1
    return eigenvectors @ torch.diag(eigenvalues.to(dtype=torch.complex128)) @ eigenvectors.conj().T

# Improved matrix square root using Schur decomposition for better stability
def matrix_sqrt_schur(A, epsilon=EPSILON):
    A = A + epsilon * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)  # Minimal regularization
    schur_form, unitary = torch.linalg.schur(A)
    sqrt_diag = torch.sqrt(torch.diag(schur_form))  # Element-wise sqrt on diagonal
    sqrt_schur = torch.diag(sqrt_diag.to(dtype=torch.complex128))
    return unitary @ sqrt_schur @ unitary.conj().T

# Von Neumann entropy with stability
def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigh(rho)[0].real.clamp(min=1e-10)
    return -torch.sum(eigenvalues * torch.log(eigenvalues))

# Fidelity using improved sqrt
def fidelity(rho, sigma):
    sqrt_rho = matrix_sqrt_schur(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = matrix_sqrt_schur(inner)
    return torch.trace(sqrt_inner).real ** 2

# Quantum gates (extended for higher dim if needed, but here for qubit-like ops on subspaces)
def rx(theta):
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    return torch.tensor([[cos + 0j, -1j * sin], [-1j * sin, cos + 0j]], dtype=torch.complex128)

def ry(theta):
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    return torch.tensor([[cos + 0j, -sin + 0j], [sin + 0j, cos + 0j]], dtype=torch.complex128)

def cnot():
    return torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=torch.complex128)

# Generate EISA-like generators for higher dim
# Simulate triple superalgebra: SM (e.g., simplified SU(2)), Grav (curvature-like), Vac (Grassmann-like anticommuting)
def generate_eisa_generators(dim):
    # Bosonic generators (B_k): Hermitian matrices simulating commutators
    B_k = torch.randn(dim, dim, dtype=torch.complex128)
    B_k = (B_k + B_k.conj().T) / 2  # Hermitian

    # Fermionic generators (F_i): Anti-Hermitian for odd grading
    F_i = 1j * torch.randn(dim, dim, dtype=torch.complex128)
    F_i = (F_i - F_i.conj().T) / 2  # Anti-Hermitian

    # Vacuum zeta^k: Simulate Grassmann with Clifford-like matrices
    zeta = torch.zeros(dim, dim, dtype=torch.complex128)
    for i in range(dim // 2):
        zeta[2*i, 2*i+1] = 1.0  # Simple off-diagonal for anticommuting property approximation

    return B_k, F_i, zeta

# Create EISA-perturbed density matrix from algebraic structure
def eisa_perturbed_density_matrix(dim, B_k, F_i, zeta):
    # Start with vacuum-like state: rho_vac = exp(-sum zeta zeta^dagger)
    zeta_dag = zeta.conj().T
    vac_exponent = - (zeta @ zeta_dag + zeta_dag @ zeta) / 2  # Symmetric for exp
    rho = torch.matrix_exp(vac_exponent)
    rho = rho / torch.trace(rho).real  # Normalize

    # Perturb with SM-like (bosonic) and Grav-like operations
    rho = torch.linalg.matrix_exp(1j * B_k) @ rho @ torch.linalg.matrix_exp(-1j * B_k.conj().T)  # Unitary from B_k
    rho = F_i @ rho @ F_i.conj().T  # Fermionic perturbation (non-unitary but graded)

    return project_to_psd(rho)  # Ensure PSD

# Apply VQC: Extended to multi-qubit like for higher dim (apply on subspaces)
def apply_vqc(rho, params, num_layers=4, subspace_size=4):  # Reduced layers for high dim efficiency
    num_subspaces = DIM // subspace_size
    theta_rx, theta_ry = params[0], params[1]
    for layer in range(num_layers):
        for s in range(num_subspaces):
            # Apply RX, RY, CNOT on subspace
            start = s * subspace_size
            U_rx = torch.eye(DIM, dtype=torch.complex128)
            U_rx[start:start+2, start:start+2] = rx(theta_rx)
            rho = U_rx @ rho @ U_rx.conj().T

            U_ry = torch.eye(DIM, dtype=torch.complex128)
            U_ry[start+2:start+4, start+2:start+4] = ry(theta_ry)  # Offset for entanglement
            rho = U_ry @ rho @ U_ry.conj().T

            U_cnot = torch.eye(DIM, dtype=torch.complex128)
            U_cnot[start:start+4, start:start+4] = cnot()
            rho = U_cnot @ rho @ U_cnot.conj().T
    return rho

# Target state: Derived from EISA vacuum rho_vac
def get_target_rho(dim, zeta):
    zeta_dag = zeta.conj().T
    exponent = - (zeta @ zeta_dag + zeta_dag @ zeta) / 2
    target = torch.matrix_exp(exponent)
    return target / torch.trace(target).real

# Noise from EISA algebra: Commutators/anticommutators
def add_eisa_noise(rho, B_k, F_i, eta=0.01):
    # Noise from [B_k, rho] + {F_i, rho}
    comm_B = B_k @ rho - rho @ B_k
    anticomm_F = F_i @ rho + rho @ F_i
    noise = eta * (comm_B + anticomm_F)
    rho_noisy = rho + noise
    return project_to_psd(rho_noisy)  # Minimal projection

# Main simulation
B_k, F_i, zeta = generate_eisa_generators(DIM)
rho_init = eisa_perturbed_density_matrix(DIM, B_k, F_i, zeta)
target_rho = get_target_rho(DIM, zeta)

params = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
optimizer = optim.Adam([params], lr=0.001)  # Adjusted LR
num_iterations = 1000  # Reduced for high dim feasibility

entropies, fidelities, losses = [], [], []

for iter in range(num_iterations):
    optimizer.zero_grad()
    rho_evolved = apply_vqc(rho_init, params)
    rho_evolved = add_eisa_noise(rho_evolved, B_k, F_i)
    
    S_vn = von_neumann_entropy(rho_evolved)
    Fid = fidelity(rho_evolved, target_rho)
    purity = torch.trace(rho_evolved @ rho_evolved).real
    loss = S_vn + (1 - Fid) + 0.5 * (1 - purity)  # Weight 0.5 motivated by equal emphasis on purity as half of entropy-fidelity balance
    
    loss.backward()
    optimizer.step()
    
    entropies.append(S_vn.item())
    fidelities.append(Fid.item())
    losses.append(loss.item())
    
    if iter % 100 == 0:
        print(f"Iter {iter}: Entropy {S_vn.item():.4f}, Fidelity {Fid.item():.4f}, Loss {loss.item():.4f}")

# Plotting (simplified for high dim - plot trajectories only)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(entropies); plt.title('Entropy')
plt.subplot(1, 3, 2); plt.plot(fidelities); plt.title('Fidelity')
plt.subplot(1, 3, 3); plt.plot(losses); plt.title('Loss')
plt.show()

print(f"Final Entropy: {entropies[-1]:.4f}, Fidelity: {fidelities[-1]:.4f}")
print("Note: Code uses 64x64 dim with EISA-derived vacuum target and algebra-based noise.")
