import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Project to PSD with enhanced regularization
def project_to_psd(rho, epsilon=1e-6):
    # Ensure Hermitian and add regularization before decomposition
    rho = (rho + rho.conj().T) / 2  # Force Hermitian
    rho += epsilon * torch.eye(rho.shape[0], dtype=rho.dtype, device=rho.device)  # Stronger regularization
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = eigenvalues.real.clamp(min=epsilon)  # Clamp to avoid zero eigenvalues
    eigenvalues = eigenvalues / eigenvalues.sum()  # Renormalize to trace 1
    diag = torch.diag(eigenvalues.to(dtype=torch.complex128))
    return eigenvectors @ diag @ eigenvectors.conj().T

# Denman-Beavers iteration for differentiable matrix square root with regularization
def matrix_sqrt_db(A, num_iters=30, epsilon=1e-6):
    Y = A.clone() + epsilon * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    Z = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    for _ in range(num_iters):
        inv_Z = torch.linalg.inv(Z + epsilon * torch.eye(Z.shape[-1], dtype=Z.dtype, device=Z.device))
        inv_Y = torch.linalg.inv(Y + epsilon * torch.eye(Y.shape[-1], dtype=Y.dtype, device=Y.device))
        Y_next = (Y + inv_Z) / 2
        Z_next = (Z + inv_Y) / 2
        Y = Y_next
        Z = Z_next
    return Y

# Von Neumann entropy
def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigh(rho)[0].real  # Use real part of eigenvalues
    eigenvalues = eigenvalues.clamp(min=1e-8)
    return -torch.sum(eigenvalues * torch.log(eigenvalues))

# Fidelity calculation
def fidelity(rho, sigma):
    sqrt_rho = matrix_sqrt_db(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = matrix_sqrt_db(inner)
    tr = torch.trace(sqrt_inner).real
    return tr ** 2

# Quantum gates with gradient support
def rx(theta):
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    r00 = cos + 0j
    r01 = -1j * sin
    r10 = -1j * sin
    r11 = cos + 0j
    row1 = torch.stack([r00, r01])
    row2 = torch.stack([r10, r11])
    return torch.stack([row1, row2])

def ry(theta):
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    r00 = cos + 0j
    r01 = -sin + 0j
    r10 = sin + 0j
    r11 = cos + 0j
    row1 = torch.stack([r00, r01])
    row2 = torch.stack([r10, r11])
    return torch.stack([row1, row2])

def cnot():
    return torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=torch.complex128)

# EISA generators (Pauli matrices as examples)
F_i = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)  # Fermionic (Pauli Z)
B_k = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)  # Bosonic (Pauli X)

# Create EISA-perturbed density matrix
def eisa_perturbed_density_matrix(dim=4):
    # Create random pure state
    psi = torch.randn(dim, dtype=torch.complex128)
    psi = psi / torch.norm(psi)
    rho = torch.outer(psi, psi.conj())
    
    # Apply fermionic generator perturbation (F_i ⊗ I)
    F_extended = torch.kron(F_i, torch.eye(2, dtype=torch.complex128))
    rho = F_extended @ rho @ F_extended.conj().T
    
    # Apply bosonic generator perturbation (I ⊗ B_k)
    B_extended = torch.kron(torch.eye(2, dtype=torch.complex128), B_k)
    rho = B_extended @ rho @ B_extended.conj().T
    
    # Normalize and return
    return rho / torch.trace(rho).real

# Apply VQC circuit with increased layers
def apply_vqc(rho, params, num_layers=4):
    theta_rx, theta_ry = params[0], params[1]
    for _ in range(num_layers):
        U_rx = torch.kron(rx(theta_rx), torch.eye(2, dtype=torch.complex128))
        rho = U_rx @ rho @ U_rx.conj().T
        U_ry = torch.kron(torch.eye(2, dtype=torch.complex128), ry(theta_ry))
        rho = U_ry @ rho @ U_ry.conj().T
        U_cnot = cnot()
        rho = U_cnot @ rho @ U_cnot.conj().T
    return rho

# Target ordered state |00⟩⟨00|
target_rho = torch.zeros(4, 4, dtype=torch.complex128)
target_rho[0, 0] = 1.0

# Simulation parameters
eta = 0.01  # Adjusted noise parameter
learning_rate = 0.001
num_iterations = 1000  # Number of iterations

# Initialize EISA-perturbed density matrix
rho_init = eisa_perturbed_density_matrix()

# VQC parameters to optimize
params = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))

optimizer = optim.Adam([params], lr=learning_rate)

# Lists to store trajectories
entropies = []
fidelities = []
losses = []
params_history = []

# Structured EISA noise generators
F_extended = torch.kron(F_i, torch.eye(2, dtype=torch.complex128))
B_extended = torch.kron(torch.eye(2, dtype=torch.complex128), B_k)
eisa_generators = [F_extended, B_extended]

for iter in range(num_iterations):
    optimizer.zero_grad()
    
    # Evolve using VQC
    rho_evolved = apply_vqc(rho_init, params)
    
    # Add structured EISA noise
    noise = torch.zeros_like(rho_evolved)
    for gen in eisa_generators:
        coeff_real = eta * torch.randn(1).item()
        coeff_imag = eta * torch.randn(1).item()
        noise_term = coeff_real * gen.real + 1j * coeff_imag * gen.imag
        noise += noise_term
    
    rho_evolved = rho_evolved + noise
    trace = torch.trace(rho_evolved).real
    rho_evolved = rho_evolved / trace if trace != 0 else rho_evolved
    
    # Project to PSD for stability
    rho_evolved = project_to_psd(rho_evolved)
    
    # Compute loss components
    S_vn = von_neumann_entropy(rho_evolved)
    Fid = fidelity(rho_evolved, target_rho)
    purity = torch.trace(rho_evolved @ rho_evolved).real
    loss = S_vn + (1 - Fid) + 0.5 * (1 - purity)
    
    # Backprop and update
    loss.backward()
    optimizer.step()
    
    # Store values
    entropies.append(S_vn.item())
    fidelities.append(Fid.item())
    losses.append(loss.item())
    params_history.append(params.detach().clone().numpy())
    
    if iter % 100 == 0:
        print(f"Iter {iter}: Entropy {S_vn.item():.4f}, Fidelity {Fid.item():.4f}, Loss {loss.item():.4f}")

# Convert to arrays for plotting
params_history = np.array(params_history)
theta_rx_hist = params_history[:, 0]
theta_ry_hist = params_history[:, 1]

# Visualization 1: Entropy Trajectory
plt.figure(figsize=(15, 12))
plt.subplot(3, 3, 1)
plt.plot(entropies)
plt.title('Von Neumann Entropy Trajectory')
plt.ylabel('Entropy')
plt.grid(True)

# Visualization 2: Fidelity Trajectory
plt.subplot(3, 3, 2)
plt.plot(fidelities)
plt.title('Fidelity Trajectory')
plt.ylabel('Fidelity')
plt.grid(True)

# Visualization 3: Loss Trajectory
plt.subplot(3, 3, 3)
plt.plot(losses)
plt.title('Loss Trajectory')
plt.ylabel('Loss')
plt.grid(True)

# Visualization 4: Entropy vs Fidelity Phase Space
plt.subplot(3, 3, 4)
plt.scatter(entropies, fidelities, c=np.arange(len(entropies)), cmap='viridis')
plt.colorbar(label='Iteration')
plt.title('Entropy vs Fidelity')
plt.ylabel('Fidelity')
plt.grid(True)

# Visualization 5: Theta RX Evolution
plt.subplot(3, 3, 5)
plt.plot(theta_rx_hist)
plt.title('Theta RX Parameter Evolution')
plt.ylabel('Theta RX')
plt.grid(True)

# Visualization 6: Theta RY Evolution
plt.subplot(3, 3, 6)
plt.plot(theta_ry_hist)
plt.title('Theta RY Parameter Evolution')
plt.ylabel('Theta RY')
plt.grid(True)

# Visualization 7: Initial Density Matrix (Real Part)
plt.subplot(3, 3, 7)
plt.imshow(rho_init.real.numpy(), cmap='hot')
plt.title('Initial Rho (Real)')
plt.colorbar()

# Visualization 8: Final Density Matrix (Real Part)
final_rho = apply_vqc(rho_init, params)
plt.subplot(3, 3, 8)
plt.imshow(final_rho.real.detach().numpy(), cmap='hot')
plt.title('Final Rho (Real)')
plt.colorbar()

# Visualization 9: Initial Density Matrix (Imag Part)
plt.subplot(3, 3, 9)
plt.imshow(rho_init.imag.numpy(), cmap='coolwarm')
plt.title('Initial Rho (Imag)')
plt.colorbar()

# Additional Visualization 10: Final Density Matrix (Imag Part) - Separate figure for more
plt.figure(figsize=(6, 5))
plt.imshow(final_rho.imag.detach().numpy(), cmap='coolwarm')
plt.title('Final Rho (Imag)')
plt.colorbar()
plt.show()

# Additional Visualization 11: 3D Trajectory of Parameters and Loss
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(theta_rx_hist, theta_ry_hist, losses)
ax.set_xlabel('Theta RX')
ax.set_ylabel('Theta RY')
ax.set_zlabel('Loss')
ax.set_title('3D Parameter-Loss Trajectory')
plt.show()

# Additional Visualization 12: Histogram of Entropy Values
plt.figure(figsize=(6, 4))
plt.hist(entropies, bins=20, color='skyblue')
plt.title('Histogram of Entropy Values')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Additional Visualization 13: Histogram of Fidelity Values
plt.figure(figsize=(6, 4))
plt.hist(fidelities, bins=20, color='lightgreen')
plt.title('Histogram of Fidelity Values')
plt.xlabel('Fidelity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Detailed visualization for 0-200 iter
plt.figure(figsize=(15, 5))

# Zoomed Entropy Trajectory (0-200 iter)
plt.subplot(1, 3, 1)
plt.plot(entropies[:201])
plt.title('Entropy Trajectory (0-200 Iter)')
plt.ylabel('Entropy')
plt.grid(True)

# Zoomed Fidelity Trajectory (0-200 iter)
plt.subplot(1, 3, 2)
plt.plot(fidelities[:201])
plt.title('Fidelity Trajectory (0-200 Iter)')
plt.ylabel('Fidelity')
plt.grid(True)

# Zoomed Loss Trajectory (0-200 iter)
plt.subplot(1, 3, 3)
plt.plot(losses[:201])
plt.title('Loss Trajectory (0-200 Iter)')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# Fidelity threshold analysis
threshold_iter = next((i for i, fid in enumerate(fidelities) if fid > 0.8), 'Not reached')
print(f"Fidelity threshold (Fid > 0.8) reached at iteration: {threshold_iter}")

# Final values
print(f"Initial Entropy: {entropies[0]:.4f}")
print(f"Final Entropy: {entropies[-1]:.4f}")
print(f"Final Fidelity: {fidelities[-1]:.4f}")
print(f"Entropy reduction: {entropies[0]-entropies[-1]:.4f} ({100*(entropies[0]-entropies[-1])/entropies[0]:.1f}%)")

# EISA verification note
print("\nNote: Density matrix initialized with EISA generators (F_i ⊗ I and I ⊗ B_k)")
print("Structured noise constructed from EISA algebra elements")