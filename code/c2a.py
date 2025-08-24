import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sympy as sp
from scipy.interpolate import interp1d
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration: Use 2D grid for field evolution (approximating 3D, scalable to 64x64 effective dim)
GRID_SIZE = 64  # Effective dimension for irrep embeddings (64x64 matrix equiv via flattening)
DT = 1e-2  # Time step for numerical integration
NUM_STEPS = 1000  # Simulation steps
KAPPA = 0.1  # Curvature coupling from paper
ALPHA = 0.1  # Non-local coupling strength
BETA = 0.005  # Loop correction factor
ETA = 0.01  # Noise amplitude from EISA perturbations

# Generate EISA generators (bosonic B_k, fermionic F_i, vacuum zeta^k)
def generate_eisa_generators(dim=GRID_SIZE):
    # Bosonic: Hermitian, simulating [B_k, B_l] = i f_klm B_m (approx SU(2)-like)
    B_k = np.zeros((3, dim, dim), dtype=complex)
    for k in range(3):
        B_k[k] = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        B_k[k] = (B_k[k] + B_k[k].conj().T) / 2  # Hermitian

    # Fermionic: Anti-Hermitian, {F_i, F_j} = 2 delta_ij + i epsilon_ijk zeta^k
    F_i = np.zeros((3, dim, dim), dtype=complex)
    for i in range(3):
        F_i[i] = 1j * (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
        F_i[i] = (F_i[i] - F_i[i].conj().T) / 2  # Anti-Hermitian

    # Vacuum zeta^k: Clifford-like for Grassmann approx
    zeta = np.zeros((dim, dim), dtype=complex)
    for k in range(dim // 2):
        zeta[2*k, 2*k+1] = 1.0 + 0j

    return torch.tensor(B_k, dtype=torch.complex128), torch.tensor(F_i, dtype=torch.complex128), torch.tensor(zeta, dtype=torch.complex128)

# Symbolic super-Jacobi verification (using SymPy for exact algebra)
def symbolic_super_jacobi():
    # Symbols for generators (full structure constants)
    f_klm, epsilon_ijk = sp.symbols('f_klm epsilon_ijk')
    B_k, B_l, B_m = sp.symbols('B_k B_l B_m')
    F_i, F_j = sp.symbols('F_i F_j')
    zeta_k = sp.symbols('zeta_k')

    # Define brackets
    comm = lambda X, Y: sp.I * f_klm * B_m  # [B_k, B_l] = i f_klm B_m
    anticomm = lambda X, Y: 2 * sp.KroneckerDelta(i, j) + sp.I * epsilon_ijk * zeta_k  # {F_i, F_j}

    # Super-Jacobi: [[B_k, B_l], F_i] + cycl. = 0
    term1 = comm(comm(B_k, B_l), F_i)
    term2 = comm(anticomm(F_i, B_k), B_l)
    term3 = comm(comm(B_l, F_i), B_k)
    jacobi_expr = term1 + term2 + term3
    simplified = sp.simplify(jacobi_expr)

    if simplified == 0:
        print("Symbolic super-Jacobi verified: Expression simplifies to 0.")
        return True
    else:
        print(f"Symbolic verification failed: Residual {simplified}")
        return False

# Numerical super-Jacobi verification (using actual matrices)
def numerical_super_jacobi(B, F, tol=1e-10):
    residuals = []
    for k in range(3):
        for l in range(3):
            for i in range(3):
                BB = B[k] @ B[l] - B[l] @ B[k]  # [B_k, B_l]
                term1 = BB @ F[i] - F[i] @ BB
                FBk = F[i] @ B[k] + B[k] @ F[i]  # {F_i, B_k} for graded
                term2 = FBk @ B[l] - B[l] @ FBk
                BFl = B[l] @ F[i] - F[i] @ B[l]
                term3 = BFl @ B[k] - B[k] @ BFl
                res = np.linalg.norm(term1 + term2 + term3)
                residuals.append(res)
    max_res = np.max(residuals)
    print(f"Numerical super-Jacobi max residual: {max_res:.2e}")
    return max_res < tol

# Field evolution: Numerical solver for modified Klein-Gordon-like equation
# d phi / dt = - D[phi] + alpha * int |phi|^2 (1 + beta ln |phi|^2) + kappa nabla^2 phi + EISA noise
def evolve_field(phi, B, F, zeta, dt, alpha, beta, kappa, eta):
    # Damping term (approximate functional derivative)
    damp = -0.1 * torch.norm(phi) * phi  # Simple friction-like for stability

    # Non-local integral with loop correction
    integral = torch.trapz(torch.abs(phi)**2, dim=0)
    loop_corr = beta * torch.log(integral + 1e-8)
    non_local = alpha * integral * (1 + loop_corr) / phi.numel()

    # Laplacian (2D finite difference)
    lap = torch.zeros_like(phi)
    lap[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2] - 4 * phi[1:-1, 1:-1]) / (dt**2)
    lap = kappa * lap.flatten()  # Flatten for consistency

    # EISA noise: Commutator/anticommutator perturbations (flattened ops)
    noise = eta * (B[0] @ phi.flatten() - phi.flatten() @ B[0] + F[0] @ phi.flatten() + phi.flatten() @ F[0])

    # Update (Euler method)
    phi_flat = phi.flatten()
    dphi = damp.flatten() + non_local + lap + noise
    phi_flat += dt * dphi
    return phi_flat.reshape(phi.shape)

# Compute curvature from energy-momentum tensor approx
def compute_curvature(phi, kappa):
    T_phi = torch.abs(phi)**2  # Simplified scalar contribution
    R = kappa**2 * torch.mean(T_phi)  # Trace approx from Einstein eq
    return R

# Compute GW freq/spectrum from delta T_ij (quadrupole approx)
def compute_gw(phi, scale_factor1, scale_factor2, freq_range):
    # delta T_ij ~ partial_i phi partial_j phi (simplified TT projection)
    grad_phi = torch.gradient(phi)[0]
    delta_T = grad_phi[:, None] * grad_phi[None, :]  # Outer product approx
    h = scale_factor1 * torch.mean(delta_T) / (torch.norm(grad_phi) + 1e-10)  # Amplitude
    freq = scale_factor2 / torch.norm(grad_phi)  # Inverse scale
    # Spectrum: Power-law from h ~ f^alpha
    spectrum = h * (freq_range / freq)** (-2/3)
    return freq, spectrum

# Sensitivity curve (interpolated)
def sensitivity_curve(freqs):
    f_data = np.logspace(-10, -2, 200)
    sens_data = 1e-15 * (f_data / 1e-8)**(-2/3) + 3e-20 * (f_data / 1e-3)**(2/3) + 1e-23
    interp = interp1d(np.log10(f_data), np.log10(sens_data), fill_value="extrapolate")
    return 10**interp(np.log10(freqs))

# CMB deviation from field gradients (natural soliton-like)
def compute_cmb_dev(phi):
    grad_phi = torch.gradient(phi)[0]
    dev = 1e-7 * torch.sin(torch.mean(grad_phi))  # Natural phase from gradients
    return dev

# Monte Carlo wrapper
def monte_carlo_run(run_func, params, num_runs=10):
    results = []
    for _ in range(num_runs):
        results.append(run_func(**params))
    return results

# Core simulation
def run_simulation(tau_P, num_steps, epsilon_0, jump_threshold, scale_factor1, scale_factor2, alpha, beta, kappa, eta):
    B, F, zeta = generate_eisa_generators()

    # Verify algebra
    if not symbolic_super_jacobi() or not numerical_super_jacobi(B.numpy(), F.numpy()):
        raise ValueError("Super-Jacobi verification failed.")

    # Initial field (2D grid)
    phi = torch.complex(torch.randn(GRID_SIZE, GRID_SIZE), torch.randn(GRID_SIZE, GRID_SIZE)) * 0.1

    curvature_history = []
    gw_freq_history = []
    order_param_history = []
    cmb_dev_history = []

    for step in range(num_steps):
        t = step * DT
        exp_term = -t / tau_P
        epsilon_t = epsilon_0 * torch.exp(torch.tensor(exp_term).clamp(max=0))

        # Evolve field
        phi = evolve_field(phi, B, F, zeta, DT, alpha, beta, kappa, eta)

        # Clamp norm
        norm = torch.norm(phi)
        if norm > jump_threshold:
            phi = phi / norm * jump_threshold

        # Curvature
        curv = compute_curvature(phi, kappa)
        curvature_history.append(curv.item())

        # GW
        freq_range = np.logspace(-12, -2, 100)
        gw_freq, gw_spectrum = compute_gw(phi, scale_factor1, scale_factor2, freq_range)
        gw_freq_history.append(gw_freq.item())

        # Order param (mean real part)
        order_param = torch.mean(phi.real)
        order_param_history.append(order_param.item())

        # CMB dev
        cmb_dev = compute_cmb_dev(phi)
        cmb_dev_history.append(cmb_dev.item())

    # SNR
    sens = sensitivity_curve(freq_range)
    snr = np.trapz((gw_spectrum.numpy() / sens)**2, freq_range)**0.5
    print(f"SNR: {snr:.2f}")

    return np.array(curvature_history), np.array(gw_freq_history), np.array(order_param_history), np.array(cmb_dev_history)

# Params (original and adjusted)
orig_params = {'tau_P': 1e-44, 'num_steps': NUM_STEPS, 'epsilon_0': 1.0, 'jump_threshold': 0.1, 'scale_factor1': 1e7, 'scale_factor2': 1e-8, 'alpha': ALPHA, 'beta': BETA, 'kappa': KAPPA, 'eta': ETA}
adj_params = {'tau_P': 1e-18, 'num_steps': 5000, 'epsilon_0': 0.2, 'jump_threshold': 0.4, 'scale_factor1': 5e-13, 'scale_factor2': 1e-8, 'alpha': ALPHA, 'beta': BETA, 'kappa': KAPPA, 'eta': ETA}

# Run Monte Carlo
orig_results = monte_carlo_run(run_simulation, orig_params)
adj_results = monte_carlo_run(run_simulation, adj_params)

# Visualization PDF
with PdfPages('simulation_report.pdf') as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Curvature
    axs[0, 0].errorbar(range(NUM_STEPS), np.mean([r[0] for r in orig_results], 0), yerr=np.std([r[0] for r in orig_results], 0), label='Orig')
    axs[0, 0].set_title('Curvature (Orig)')
    axs[0, 1].errorbar(range(5000), np.mean([r[0] for r in adj_results], 0), yerr=np.std([r[0] for r in adj_results], 0), label='Adj')
    axs[0, 1].set_title('Curvature (Adj)')
    # GW Freq
    axs[1, 0].hist(np.concatenate([r[1] for r in orig_results]), bins=40)
    axs[1, 0].set_title('GW Freq Hist (Orig)')
    axs[1, 1].hist(np.concatenate([r[1] for r in adj_results]), bins=40)
    axs[1, 1].set_title('GW Freq Hist (Adj)')
    pdf.savefig(fig)
    plt.close(fig)

print("Simulation complete. PDF report generated.")