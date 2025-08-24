import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import emcee
import corner
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sympy as sp

# Constants (Planck 2018)
H0_LCDM = 67.4  # km/s/Mpc
OMEGA_M = 0.315
OMEGA_R = 5e-5
OMEGA_LAMBDA = 0.685

# EISA generators (simplified for parameter modulation)
def eisa_generators(dim=64):
    # Bosonic (even), Fermionic (odd)
    B = [np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim) for _ in range(3)]
    for i in range(3):
        B[i] = (B[i] + B[i].conj().T)/2

    F = [1j*(np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)) for _ in range(3)]
    for i in range(3):
        F[i] = (F[i] - F[i].conj().T)/2

    zeta = np.zeros((3, dim, dim), dtype=complex)
    for k in range(3):
        zeta[k] = np.diag(np.random.rand(dim))  # Vacuum modes

    return B, F, zeta

# Derived Omega_v(tau) from vacuum trace
def omega_v(tau, A_v, tau_decay, zeta):
    exponent = -(zeta[0] @ zeta[0].conj().T + zeta[0].conj().T @ zeta[0])/2
    trace_rho = np.real(np.trace(np.exp(exponent))) / DIM
    return A_v * np.exp(-tau / tau_decay) * trace_rho

# Solve Friedmann with EISA vacuum
def solve_friedmann(theta, tau_span, zeta):
    kappa, n, A_v = theta
    def friedmann_eq(tau, y):
        a = y[0]
        Omega_v_tau = omega_v(tau, A_v, 1.0, zeta)  # tau_decay=1 normalized
        H_sq = OMEGA_M / a**3 + OMEGA_R / a**4 + OMEGA_LAMBDA + kappa * (Omega_v_tau)**n
        return np.sqrt(np.maximum(H_sq, 1e-10)) * a
    
    sol = solve_ivp(friedmann_eq, [tau_span[0], tau_span[-1]], [1e-3], t_eval=tau_span, method='RK45')
    return interp1d(sol.t, sol.y[0], kind='cubic')

# Theoretical Cl from integral
def compute_cl(theta, ells, zeta):
    kappa, n, A_v = theta
    tau_span = np.linspace(1e-10, 10, 1000)
    a_interp = solve_friedmann(theta, tau_span, zeta)
    Cl = np.zeros(len(ells))
    for i, l in enumerate(ells):
        k_l = l  # Approx k~l
        integrand = a_interp(tau_span)**2 * omega_v(tau_span, A_v, 1.0, zeta) * np.cos(k_l * tau_span)
        Cl[i] = np.trapz(integrand, tau_span)
    Dl = (ells * (ells + 1) / (2 * np.pi)) * np.abs(Cl)  # Positive abs
    return Dl

# LCDM reference (simplified CAMB-like)
def compute_lcdm_dl(ells):
    # Approx from Planck fit
    return 2500 * np.exp(-(np.log(ells/220)**2 / 0.5)) + 1000 * np.sin(ells/180) * np.exp(-(ells/1200)**2)

# Load Planck data
def load_planck_data():
    # Mock or load real; assume mock for now
    ells = np.arange(2, 2501)
    dl_obs = compute_lcdm_dl(ells) + np.random.normal(0, 0.01 * compute_lcdm_dl(ells))
    sigma_dl = 0.01 * dl_obs
    return ells, dl_obs, sigma_dl

# Log likelihood
def log_likelihood(theta, ells, dl_obs, sigma_dl, zeta):
    dl_model = compute_cl(theta, ells, zeta)
    chi2 = np.sum(((dl_obs - dl_model) / sigma_dl)**2)
    return -0.5 * chi2

# Gaussian prior (Planck constraints)
def log_prior(theta):
    kappa, n, A_v = theta
    if not (0.1 < kappa < 1.0 and 1 < n < 10 and 1e-10 < A_v < 1e-8):
        return -np.inf
    return -0.5 * ((kappa - 0.31)/0.01)**2 -0.5 * ((n - 7)/1)**2 -0.5 * ((np.log10(A_v) +9)/0.5)**2

# Log prob
def log_prob(theta, ells, dl_obs, sigma_dl, zeta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, ells, dl_obs, sigma_dl, zeta)

# MCMC
def run_mcmc(ells, dl_obs, sigma_dl, theta0, zeta, n_walkers=32, n_steps=2000):
    ndim = len(theta0)
    pos = theta0 + 1e-4 * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob, args=(ells, dl_obs, sigma_dl, zeta))
    sampler.run_mcmc(pos, n_steps, progress=True)
    return sampler.get_chain(discard=500, thin=15, flat=True), sampler

# Main
if __name__ == "__main__":
    verify_super_jacobi()  # Algebraic verification
    B, F, zeta = eisa_generators()  # Generators for vacuum
    ells, dl_obs, sigma_dl = load_planck_data()
    
    theta0 = [0.31, 7, 2.1e-9]
    neg_log = lambda theta: -log_prob(theta, ells, dl_obs, sigma_dl, zeta)
    ml_result = minimize(neg_log, theta0, bounds=[(0.1,1),(1,10),(1e-10,1e-8)])
    
    samples, sampler = run_mcmc(ells, dl_obs, sigma_dl, ml_result.x, zeta)
    
    theta_best = np.median(samples, axis=0)
    dl_fit = compute_cl(theta_best, ells, zeta)
    dl_lcdm = compute_lcdm_dl(ells)
    
    chi2_eisa = np.sum(((dl_fit - dl_obs)/sigma_dl)**2)
    chi2_lcdm = np.sum(((dl_lcdm - dl_obs)/sigma_dl)**2)
    dof = len(ells) - 3
    print(f"χ²/dof EISA: {chi2_eisa/dof:.2f}, ΛCDM: {chi2_lcdm/dof:.2f}")
    
    # Plots
    fig, ax = plt.subplots()
    ax.errorbar(ells, dl_obs, yerr=sigma_dl, fmt='.', label='Planck')
    ax.loglog(ells, dl_fit, label='EISA')
    ax.loglog(ells, dl_lcdm, label='ΛCDM')
    ax.legend()
    plt.savefig('cmb_fit.pdf')
    
    corner.corner(samples, labels=['κ','n','A_v'])
    plt.savefig('posterior.pdf')