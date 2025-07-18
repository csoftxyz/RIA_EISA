import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

# Suppress OMP duplicate lib warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Project to PSD
def project_to_psd(rho, epsilon=1e-10):
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = eigenvalues.real.clamp(min=0.0) + epsilon  # Small epsilon for numerical stability
    eigenvalues = eigenvalues / eigenvalues.sum()  # Renormalize to trace 1
    diag = torch.diag(eigenvalues.to(dtype=torch.complex128))
    return eigenvectors @ diag @ eigenvectors.conj().T

# Denman-Beavers iteration for differentiable matrix square root with regularization
def matrix_sqrt_db(A, num_iters=30, epsilon=1e-10):
    Y = A.clone() + epsilon * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    Z = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    for _ in range(num_iters):
        try:
            inv_Z = torch.linalg.inv(Z + epsilon * torch.eye(Z.shape[-1], dtype=Z.dtype, device=Z.device))
            inv_Y = torch.linalg.inv(Y + epsilon * torch.eye(Y.shape[-1], dtype=Y.dtype, device=Y.device))
        except RuntimeError:
            # Fallback if singular: add more epsilon
            inv_Z = torch.linalg.inv(Z + 2 * epsilon * torch.eye(Z.shape[-1], dtype=Z.dtype, device=Z.device))
            inv_Y = torch.linalg.inv(Y + 2 * epsilon * torch.eye(Y.shape[-1], dtype=Y.dtype, device=Y.device))
        Y_next = (Y + inv_Z) / 2
        Z_next = (Z + inv_Y) / 2
        Y = Y_next
        Z = Z_next
    return Y

# Von Neumann entropy using eigvalsh (only eigenvalues, no phase issue)
def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues.clamp(min=1e-8)
    return -torch.sum(eigenvalues * torch.log(eigenvalues))

# Fidelity using matrix_sqrt_db
def fidelity(rho, sigma):
    sqrt_rho = matrix_sqrt_db(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = matrix_sqrt_db(inner)
    tr = torch.trace(sqrt_inner).real
    return tr ** 2

# Helper functions for quantum gates (unchanged)
def rx(theta):
    cos = torch.cos(theta/2)
    sin = torch.sin(theta/2)
    r = torch.zeros(2, 2, dtype=torch.complex128)
    r[0, 0] = cos
    r[0, 1] = -1j * sin
    r[1, 0] = -1j * sin
    r[1, 1] = cos
    return r

def ry(theta):
    cos = torch.cos(theta/2)
    sin = torch.sin(theta/2)
    r = torch.zeros(2, 2, dtype=torch.complex128)
    r[0, 0] = cos
    r[0, 1] = -sin
    r[1, 0] = sin
    r[1, 1] = cos
    return r

def cnot():
    r = torch.zeros(4, 4, dtype=torch.complex128)
    r[0, 0] = 1
    r[1, 1] = 1
    r[2, 3] = 1
    r[3, 2] = 1
    return r

# Generate random density matrix
def random_density_matrix(dim=4):
    psi = torch.randn(dim, dtype=torch.complex128)
    psi = psi / torch.norm(psi)
    return torch.outer(psi, psi.conj())

# Apply VQC circuit
def apply_vqc(rho, params):
    theta_rx, theta_ry = params[0], params[1]
    U_rx = torch.kron(rx(theta_rx), torch.eye(2, dtype=torch.complex128))
    rho = U_rx @ rho @ U_rx.conj().T
    U_ry = torch.kron(torch.eye(2, dtype=torch.complex128), ry(theta_ry))
    rho = U_ry @ rho @ U_ry.conj().T
    U_cnot = cnot()
    rho = U_cnot @ rho @ U_cnot.conj().T
    return rho

# Target ordered state (pure state |00><00|)
target_rho = torch.zeros(4, 4, dtype=torch.complex128)
target_rho[0, 0] = 1.0

# Simulation parameters
eta = 0.1  # Noise parameter
learning_rate = 0.01  # Learning rate for optimization
num_iterations = 1000  # Number of iterations

# Initial EISA generators (simplified for demonstration; Pauli matrices for 1-qubit, kron for 2-qubit if needed)
F_i = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)  # Fermionic example (Pauli Z)
B_k = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)  # Bosonic example (Pauli X)

# Initial random density matrix (from noise)
rho_init = random_density_matrix()

# VQC parameters to optimize
params = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))

optimizer = optim.Adam([params], lr=learning_rate)

# Lists to store trajectories
entropies = []
fidelities = []
losses = []
params_history = []  # For visualizing parameter evolution

for iter in range(num_iterations):
    optimizer.zero_grad()
    
    # Evolve the initial rho using VQC
    rho_evolved = apply_vqc(rho_init, params)
    
    # Add Hermitian noise
    noise_real = torch.randn(4, 4)
    noise_imag = torch.randn(4, 4)
    noise = eta * torch.complex(noise_real, noise_imag)
    noise = (noise + noise.conj().T) / 2  # Ensure Hermitian
    rho_evolved = rho_evolved + noise
    trace = torch.trace(rho_evolved).real
    rho_evolved = rho_evolved / trace if trace != 0 else rho_evolved  # Normalize, avoid div by zero
    
    # Project to PSD for stability
    rho_evolved = project_to_psd(rho_evolved)
    
    # Compute loss components
    S_vn = von_neumann_entropy(rho_evolved)
    Fid = fidelity(rho_evolved, target_rho)
    loss = S_vn + (1 - Fid)
    
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

# Convert params_history to arrays for plotting
params_history = np.array(params_history)
theta_rx_hist = params_history[:, 0]
theta_ry_hist = params_history[:, 1]

# Visualization with constrained_layout
fig, axs = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)

# Visualization 1: Entropy Trajectory
axs.flat[0].plot(entropies)
axs.flat[0].set_title('Von Neumann Entropy Trajectory')
axs.flat[0].set_xlabel('Iteration')
axs.flat[0].set_ylabel('Entropy')
axs.flat[0].grid(True)

# Visualization 2: Fidelity Trajectory
axs.flat[1].plot(fidelities)
axs.flat[1].set_title('Fidelity Trajectory')
axs.flat[1].set_xlabel('Iteration')
axs.flat[1].set_ylabel('Fidelity')
axs.flat[1].grid(True)

# Visualization 3: Loss Trajectory
axs.flat[2].plot(losses)
axs.flat[2].set_title('Loss Trajectory')
axs.flat[2].set_xlabel('Iteration')
axs.flat[2].set_ylabel('Loss')
axs.flat[2].grid(True)

# Visualization 4: Entropy vs Fidelity Phase Space
axs.flat[3].scatter(entropies, fidelities, c=np.arange(len(entropies)), cmap='viridis')
axs.flat[3].set_title('Entropy vs Fidelity')
axs.flat[3].set_xlabel('Entropy')
axs.flat[3].set_ylabel('Fidelity')
axs.flat[3].grid(True)

# Visualization 5: Theta RX Evolution
axs.flat[4].plot(theta_rx_hist)
axs.flat[4].set_title('Theta RX Parameter Evolution')
axs.flat[4].set_xlabel('Iteration')
axs.flat[4].set_ylabel('Theta RX')
axs.flat[4].grid(True)

# Visualization 6: Theta RY Evolution
axs.flat[5].plot(theta_ry_hist)
axs.flat[5].set_title('Theta RY Parameter Evolution')
axs.flat[5].set_xlabel('Iteration')
axs.flat[5].set_ylabel('Theta RY')
axs.flat[5].grid(True)

# Visualization 7: Initial Density Matrix (Real Part)
axs.flat[6].imshow(rho_init.real.numpy(), cmap='hot')
axs.flat[6].set_title('Initial Rho (Real)')

# Visualization 8: Final Density Matrix (Real Part)
final_rho = apply_vqc(rho_init, params)
axs.flat[7].imshow(final_rho.real.detach().numpy(), cmap='hot')
axs.flat[7].set_title('Final Rho (Real)')

# Visualization 9: Initial Density Matrix (Imag Part)
axs.flat[8].imshow(rho_init.imag.numpy(), cmap='coolwarm')
axs.flat[8].set_title('Initial Rho (Imag)')

plt.show()

# Additional Visualization 10: Final Density Matrix (Imag Part)
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

# Additional Visualization 14: Epsilon vs Iteration (Log Scale) - Placeholder, as epsilon not in this code; adapt if needed
plt.figure(figsize=(6, 4))
plt.semilogy(range(num_iterations), np.exp(-np.arange(num_iterations)/100))  # Dummy epsilon decay
plt.title('Dummy Epsilon Decay (Log)')
plt.xlabel('Iteration')
plt.ylabel('Epsilon (log)')
plt.grid(True)
plt.show()

# Additional Visualization 15: Differential Fidelity for Threshold Detection
diff_fid = np.diff(fidelities)
plt.figure(figsize=(6, 4))
plt.plot(diff_fid)
plt.title('Differential Fidelity')
plt.xlabel('Iteration')
plt.ylabel('Delta Fid')
plt.grid(True)
plt.show()

# Self-reflection threshold
threshold_iter = next((i for i, fid in enumerate(fidelities) if fid > 0.8), 'Not reached')
print(f"Self-reflection threshold (Fid > 0.8) reached at iteration: {threshold_iter}")

# Final values
print(f"Final Entropy: {entropies[-1]:.4f}")
print(f"Final Fidelity: {fidelities[-1]:.4f}")
print(f"Target Entropy reduction: from ~{entropies[0]:.4f} to ~{entropies[-1]:.4f}")

# Note: The initial EISA generators F_i and B_k can be incorporated by initial rho = F_i @ rho @ F_i.conj().T or similar, but for this demonstration, they are shown as examples.
# To verify universe bootstrap, observe if entropy decreases and fidelity increases, indicating self-organization from chaos to ordered particle-like structures.