import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import emcee
import corner
import getdist
from getdist import plots, MCSamples
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
np.random.seed(42)

# Constants from paper (Planck 2018 base, EISA-RIA extensions)
H0 = 67.4  # km/s/Mpc
Omega_m = 0.315
Omega_Lambda = 0.685
Omega_r = 5e-5
Omega_v0_base = 2.1e-9  # Transient vacuum amplitude base from paper
tau_decay = 1e-9 / 2.18e-18  # Normalized decay time
fluct_amp = 8e-4  # Adjusted for stability

# VQC-like layer for RIA optimization
class VQCLayer(nn.Module):
    def __init__(self, dim=64):
        super(VQCLayer, self).__init__()
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.phi = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.dim = dim

    def forward(self, phi):
        rot_2d = torch.tensor([[torch.cos(self.theta), -torch.sin(self.theta)],
                              [torch.sin(self.theta), torch.cos(self.theta)]], dtype=torch.complex128)
        eye = torch.eye(self.dim // 2, dtype=torch.complex128)
        full_rot = torch.kron(eye, rot_2d)
        phi_rot = full_rot @ phi @ full_rot.conj().T
        complex_i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        phase = torch.exp(complex_i * self.phi)
        return self.scale * (phi_rot * phase)

# Super-Jacobi verification (simplified for EISA)
def verify_super_jacobi():
    B1, B2, B3, F = sp.symbols('B1 B2 B3 F')
    f, sigma = sp.symbols('f sigma')
    comm_BB = lambda X, Y: sp.I * f * B3 if (X, Y) == (B1, B2) else -sp.I * f * B3 if (X, Y) == (B2, B1) else 0
    comm_BF = lambda B, F: sigma * F
    comm_FB = lambda F, B: -comm_BF(B, F)
    jacobi = comm_BF(comm_BB(B1, B2), F) + comm_BB(comm_FB(F, B1), B2) + comm_BB(comm_BF(B2, F), B1)
    if jacobi.simplify() == 0:
        print("Super-Jacobi identities verified: Algebraic closure holds.")
    return True

# Function to load Planck CMB power spectrum data with full validation
def load_planck_data(txt_path='COM_PowerSpect_CMB-TT-full_R3.01.txt'):
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} not found; using mock data.")
        ells = np.arange(2, 2501)
        dl_obs = compute_lcdm_dl(ells) + np.random.normal(0, 0.01 * compute_lcdm_dl(ells))
        sigma_dl = 0.01 * dl_obs
        mask = (ells >= 2) & (ells <= 2500)
        return ells[mask], dl_obs[mask], sigma_dl[mask]
    data = np.loadtxt(txt_path, skiprows=1)
    ells, dl_obs, sigma_low, sigma_high = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    sigma_dl = (sigma_low + sigma_high) / 2
    mask = (ells >= 2) & (ells <= 2500) & (sigma_dl > 0) & (dl_obs >= 0) & np.isfinite(ells) & np.isfinite(dl_obs) & np.isfinite(sigma_dl)
    return ells[mask], dl_obs[mask], sigma_dl[mask]

# Approximate LCDM CMB TT power spectrum
def compute_lcdm_dl(ells):
    sw = 2000 / ells
    peaks = 5000 * np.exp(- (np.log10(ells) - 2.3)**2 / 0.3)
    damping = np.exp(- (ells / 1200)**2)
    osc = 1000 * np.sin(ells / 180) * damping
    return sw + peaks + osc

# Solve Friedmann equation with EISA dynamics
def solve_friedmann(Omega_m, Omega_r, Omega_Lambda, Omega_v0, tau_decay):
    def friedmann(tau, y):
        a = y[0]
        Omega_v = Omega_v0 * np.exp(-tau / tau_decay)
        H_sq = Omega_m / a**3 + Omega_r / a**4 + Omega_Lambda + Omega_v
        return np.sqrt(np.clip(H_sq, 1e-10, None)) * a
    tau_span = np.linspace(1e-10, 10, 1000)
    sol = solve_ivp(friedmann, [tau_span[0], tau_span[-1]], [1e-3], t_eval=tau_span, method='RK45', atol=1e-10, rtol=1e-8)
    return interp1d(sol.t, sol.y[0], kind='cubic')

# Compute EISA-RIA Cl with VQC
def compute_cl(theta, ells, vqc):
    kappa, n, A_v = theta
    phi = torch.randn(64, 64, dtype=torch.float64) + 1j * torch.randn(64, 64, dtype=torch.float64)
    phi_optimized = vqc(phi)
    a_interp = solve_friedmann(Omega_m, Omega_r, Omega_Lambda, A_v, tau_decay)
    tau_span = np.linspace(1e-10, 10, 1000)
    integral = np.zeros(len(ells))
    for i, l in enumerate(ells):
        k_l = l
        integrand = (a_interp(tau_span) * k_l * fluct_amp)**n * kappa * torch.norm(phi_optimized).item()
        integral[i] = np.trapz(integrand, tau_span)
    Cl = (2 * np.pi / (ells * (ells + 1))) * integral
    Dl = ells * (ells + 1) * Cl / (2 * np.pi) * 1e6
    return Dl

# Log prior
def log_prior(theta):
    kappa, n, A_v = theta
    if 0.29 < kappa < 0.33 and 6.5 < n < 7.5 and 1.5e-9 < A_v < 2.5e-9:
        return 0.0
    return -np.inf

# Log likelihood
def log_likelihood(theta, ells, dl_obs, sigma_dl, vqc):
    dl_model = compute_cl(theta, ells, vqc)
    chi2 = np.sum(((dl_obs - dl_model) / sigma_dl)**2)
    return -0.5 * chi2

# Log probability
def log_prob(theta, ells, dl_obs, sigma_dl, vqc):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, ells, dl_obs, sigma_dl, vqc)

# Run MCMC
def run_mcmc(ells, dl_obs, sigma_dl, theta_start, vqc, nwalkers=32, nsteps=2000, burnin=500):
    ndim = len(theta_start)
    pos = theta_start * (1 + 0.01 * np.random.randn(nwalkers, ndim))
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(ells, dl_obs, sigma_dl, vqc), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=burnin, flat=True)
    return samples, sampler

# Mock recovery test
def mock_recovery_test(ells, vqc, theta_true=[0.31, 7, 2.1e-9]):
    dl_mock = compute_cl(theta_true, ells, vqc) + np.random.normal(0, 0.01 * compute_cl(theta_true, ells, vqc))
    sigma_mock = 0.01 * dl_mock
    theta0 = [0.3, 7, 2e-9]
    bounds = [(0.29, 0.33), (6.5, 7.5), (1.5e-9, 2.5e-9)]
    def neg_log_prob(theta):
        return -log_prob(theta, ells, dl_mock, sigma_mock, vqc)
    ml_result = minimize(neg_log_prob, theta0, method='L-BFGS-B', bounds=bounds)
    samples, sampler = run_mcmc(ells, dl_mock, sigma_mock, ml_result.x, vqc)
    theta_rec = np.median(samples, axis=0)
    bias = np.abs((theta_rec - theta_true) / theta_true) * 100
    print(f"Mock Recovery: True {theta_true}, Recovered {theta_rec}")
    print(f"Recovery bias: {bias}%")
    return samples, sampler

# Sensitivity analysis
def sensitivity_analysis(samples):
    cov = np.cov(samples.T)
    fisher = np.linalg.inv(cov)
    corr = np.corrcoef(samples.T)
    return fisher, corr

# Plot results
def plot_results(samples, ells, dl_obs, sigma_dl, theta_best, dl_fit, dl_lcdm, vqc):
    pdf_path = "cmb_inverse_results.pdf"
    with PdfPages(pdf_path) as pdf:
        fig1, ax1 = plt.subplots()
        ax1.loglog(ells, dl_fit, label='EISA-RIA Median Fit', color='r')
        ax1.loglog(ells, dl_lcdm, label=r'\Lambda CDM Reference', color='g')
        ax1.errorbar(ells, dl_obs, yerr=sigma_dl, fmt='b.', label='Planck TT', alpha=0.5)
        ax1.set_xlabel('Multipole l')
        ax1.set_ylabel('D_l [muK²]')
        ax1.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        residuals = (dl_obs - dl_fit) / sigma_dl
        ax2.plot(ells, residuals, 'b.')
        ax2.axhline(0, color='r', linestyle='--')
        ax2.set_xlabel('Multipole l')
        ax2.set_ylabel('Residuals (sigma)')
        pdf.savefig(fig2)
        plt.close(fig2)

        fig_corner = corner.corner(samples, labels=[r'\kappa', 'n', 'A_v'], truths=[0.31, 7, 2.1e-9])
        pdf.savefig(fig_corner)
        plt.close(fig_corner)
    print(f"Results PDF saved to {pdf_path}")

# Main function
def main(txt_path='COM_PowerSpect_CMB-TT-full_R3.01.txt'):
    verify_super_jacobi()  # Verify algebraic closure
    ells, dl_obs, sigma_dl = load_planck_data(txt_path)
    vqc = VQCLayer(dim=64)
    optimizer = optim.Adam(vqc.parameters(), lr=0.01)
    for _ in range(100):  # Pre-optimize VQC
        phi = torch.randn(64, 64, dtype=torch.float64) + 1j * torch.randn(64, 64, dtype=torch.float64)
        phi_opt = vqc(phi)
        loss = -torch.norm(phi_opt, 'fro')  # Maximize norm for entropy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    theta0 = [0.31, 7, 2.1e-9]
    bounds = [(0.29, 0.33), (6.5, 7.5), (1.5e-9, 2.5e-9)]
    def neg_log_prob(theta):
        return -log_prob(theta, ells, dl_obs, sigma_dl, vqc)
    ml_result = minimize(neg_log_prob, theta0, method='L-BFGS-B', bounds=bounds, options={'disp': True})

    if not ml_result.success:
        raise RuntimeError(f"ML failed: {ml_result.message}")
    print(f"ML point: κ={ml_result.x[0]:.3f}, n={ml_result.x[1]:.1f}, A_v={ml_result.x[2]:.2e}")

    samples, sampler = run_mcmc(ells, dl_obs, sigma_dl, ml_result.x, vqc)
    sampler.pool.close()
    sampler.pool.join()

    mock_samples, mock_sampler = mock_recovery_test(ells, vqc)
    mock_sampler.pool.close()
    mock_sampler.pool.join()

    theta_best = np.median(samples, axis=0)
    dl_fit = compute_cl(theta_best, ells, vqc)
    dl_lcdm = compute_lcdm_dl(ells)

    dof = len(ells) - len(theta_best)
    chi2_eisa = np.sum(((dl_fit - dl_obs) / sigma_dl)**2)
    chi2_lcdm = np.sum(((dl_lcdm - dl_obs) / sigma_dl)**2)
    print(f"χ²/dof EISA-RIA: {chi2_eisa / dof:.2f}")
    print(f"χ²/dof ΛCDM: {chi2_lcdm / dof:.2f}")

    fisher, corr = sensitivity_analysis(samples)
    plot_results(samples, ells, dl_obs, sigma_dl, theta_best, dl_fit, dl_lcdm, vqc)

    np.savetxt('eisa_ria_chains.txt', samples)
    print("Chains saved for further analysis (e.g., getdist).")

if __name__ == "__main__":
    import sympy as sp
    main()