import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.integrate import odeint  # Use scipy ODE solver (available in environment)
from scipy.special import legendre  # For CMB power spectrum approx

# Configuration
OMEGA_M = 0.315  # Scalar parameters from Planck
OMEGA_R = 5e-5
OMEGA_LAMBDA = 0.685
OMEGA_V0 = 0.01  # Initial vacuum density
TAU_DECAY = 1.0  # Normalized decay time
TAU_SPAN = np.linspace(1e-10, 10.0, 1000)  # Normalized time
DIM = 64  # For EISA generators
ETA = 0.01  # Noise amplitude
NUM_MC = 10  # Monte Carlo runs

# Generate EISA generators (simplified SU(3)-like for vacuum fluctuations)
def generate_eisa_generators(dim=DIM):
    B = np.random.randn(8, dim, dim) + 1j * np.random.randn(8, dim, dim)
    for i in range(8):
        B[i] = (B[i] + B[i].conj().T) / 2  # Bosonic Hermitian

    F = 1j * (np.random.randn(8, dim, dim) + 1j * np.random.randn(8, dim, dim))
    for i in range(8):
        F[i] = (F[i] - F[i].conj().T) / 2  # Fermionic anti-Hermitian

    zeta = np.zeros((dim, dim), dtype=complex)
    for k in range(dim // 2):
        zeta[2*k, 2*k+1] = 1.0

    return B, F, zeta

# Compute dynamic Omega_v(tau) from vacuum algebra (trace of exp(-zeta zeta^dag))
def compute_omega_v(tau, zeta, omega_v0, tau_decay):
    zeta_dag = zeta.conj().T
    exponent = - (zeta @ zeta_dag + zeta_dag @ zeta) / 2
    rho_vac = np.trace(np.exp(exponent)) / DIM  # Normalized trace
    return omega_v0 * np.exp(-tau / tau_decay) * rho_vac  # Decay modulated by algebra

# Compute dynamic curvature R from commutators/anticommutators
def compute_dynamic_r(B, F, phi=1.0):  # phi as placeholder field
    comm_avg = np.mean([B[i] @ phi - phi @ B[i] for i in range(8)], axis=0)
    anticomm_avg = np.mean([F[i] @ phi + phi @ F[i] for i in range(8)], axis=0)
    R = np.real(np.trace(comm_avg @ anticomm_avg.conj().T))
    return R

# EISA noise from algebra (commutator-based perturbation)
def eisa_noise(tau, B, F, eta):
    noise = eta * np.sin(2 * np.pi * tau) * compute_dynamic_r(B, F)  # Derived from algebra
    return noise

# Friedmann equation as ODE (scalar a(tau))
def friedmann_ode(y, tau, omega_m, omega_r, omega_lambda, omega_v0, tau_decay, zeta, B, F, eta):
    a = y
    omega_v = compute_omega_v(tau, zeta, omega_v0, tau_decay)
    perturbation = eisa_noise(tau, B, F, eta)
    H_sq = omega_m / a**3 + omega_r / a**4 + omega_lambda + omega_v + perturbation
    if H_sq < 0:
        H_sq = 1e-10  # Clamp
    da_dtau = a * np.sqrt(H_sq)
    return da_dtau

# Run simulation (scalar ODE)
def run_simulation(omega_m, omega_r, omega_lambda, omega_v0, tau_decay, zeta, B, F, eta):
    a_init = 1e-3
    a_tau = odeint(friedmann_ode, a_init, TAU_SPAN, args=(omega_m, omega_r, omega_lambda, omega_v0, tau_decay, zeta, B, F, eta))
    return a_tau.flatten()

# Monte Carlo over noise seeds
def monte_carlo_run(omega_m, omega_r, omega_lambda, omega_v0, tau_decay, eta, num_mc=NUM_MC):
    results = []
    for _ in range(num_mc):
        B, F, zeta = generate_eisa_generators()
        a_tau = run_simulation(omega_m, omega_r, omega_lambda, omega_v0, tau_decay, zeta, B, F, eta)
        results.append(a_tau)
    mean_a = np.mean(results, axis=0)
    std_a = np.std(results, axis=0)
    return mean_a, std_a

# Params for CMB (Planck-like) and local (adjusted for tension via vacuum)
cmb_params = (OMEGA_M, OMEGA_R, OMEGA_LAMBDA, OMEGA_V0, TAU_DECAY, ETA)
local_params = (OMEGA_M - 0.015, OMEGA_R, OMEGA_LAMBDA + 0.015, OMEGA_V0 * 1.1, TAU_DECAY * 0.9, ETA * 1.2)  # Motivated by vacuum variation

mean_a_cmb, std_a_cmb = monte_carlo_run(*cmb_params)
mean_a_local, std_a_local = monte_carlo_run(*local_params[0:5], local_params[5])  # Adjust for tuple

# Compute H(tau) = da/dtau / a
da_dtau_cmb = np.diff(mean_a_cmb) / np.diff(TAU_SPAN)
H_cmb = da_dtau_cmb / mean_a_cmb[:-1]
da_dtau_local = np.diff(mean_a_local) / np.diff(TAU_SPAN)
H_local = da_dtau_local / mean_a_local[:-1]

# Densities (using mean a)
omega_v_tau = [compute_omega_v(t, generate_eisa_generators()[2], OMEGA_V0, TAU_DECAY) for t in TAU_SPAN]
omega_m_tau = OMEGA_M / mean_a_cmb**3
omega_lambda_tau = np.full_like(TAU_SPAN, OMEGA_LAMBDA)

# CMB power spectrum deviations from delta a (curvature perturbations ~ da/a)
ell = np.linspace(2, 2500, 500)
delta_a = std_a_cmb / mean_a_cmb  # Relative fluctuation
Cl_standard = 1e-10 / ell  # Simplified LambdaCDM
soliton_dev = 1e-8 * np.sin(np.pi * ell / 500) * np.mean(delta_a)  # Scaled by fluctuations
Cl_deviated = Cl_standard * (1 + soliton_dev)

# GW spectrum from energy fluctuations (h_c ~ sqrt(Omega_GW) / f)
gw_freq = np.logspace(-12, 3, 500)
delta_rho = np.diff(omega_v_tau) / np.diff(TAU_SPAN)
gw_power = 1e-15 * np.abs(np.mean(delta_rho)) * (gw_freq / 1e-8)**(-2/3)  # Derived from vacuum fluctuations

# Sensitivity curve
def sensitivity_curve(freqs):
    interp = interp1d(np.log10(np.logspace(-10, -2, 200)), np.log10(1e-15 * (np.logspace(-10, -2, 200) / 1e-8)**(-2/3) + 3e-20 * (np.logspace(-10, -2, 200) / 1e-3)**(2/3) + 1e-23), fill_value="extrapolate")
    return 10**interp(np.log10(freqs))

sens = sensitivity_curve(gw_freq)

# Inflation validation (n_s from Cl tilt, r from tensor/scalar ~ Omega_GW / Delta^2)
n_s = 1 - np.gradient(np.log(Cl_standard), np.log(ell))[0]  # Approx tilt
r = np.mean(gw_power) / np.mean(Cl_standard**2) * 1e9  # Scaled ratio
print(f"Inflation: n_s ≈ {n_s:.3f}, r ≈ {r:.3f}")

# Late H uncertainty
std_H_late = np.std(H_cmb[-10:])
print(f"Late H std: {std_H_late:.4f}")

# Visualizations in PDF
with PdfPages('simulation_report.pdf') as pdf:
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    axs[0,0].errorbar(TAU_SPAN, mean_a_cmb, yerr=std_a_cmb, label='CMB')
    axs[0,0].errorbar(TAU_SPAN, mean_a_local, yerr=std_a_local, label='Local')
    axs[0,0].set_title('Scale Factor a(tau)')
    axs[0,0].set_yscale('log')
    axs[0,0].legend()
    axs[0,0].grid(True)

    axs[0,1].plot(TAU_SPAN[:-1], H_cmb, label='CMB')
    axs[0,1].plot(TAU_SPAN[:-1], H_local, label='Local')
    axs[0,1].set_title('Hubble H(tau)')
    axs[0,1].set_yscale('log')
    axs[0,1].legend()
    axs[0,1].grid(True)

    axs[0,2].plot(TAU_SPAN, omega_m_tau, label='Ω_m')
    axs[0,2].plot(TAU_SPAN, omega_v_tau, label='Ω_v')
    axs[0,2].plot(TAU_SPAN, omega_lambda_tau, label='Ω_Λ')
    axs[0,2].set_title('Density Parameters')
    axs[0,2].legend()
    axs[0,2].grid(True)

    axs[1,0].loglog(ell, Cl_standard, label='Standard')
    axs[1,0].loglog(ell, Cl_deviated, label='Deviated')
    axs[1,0].set_title('CMB Power Spectrum')
    axs[1,0].legend()
    axs[1,0].grid(True)

    axs[1,1].loglog(gw_freq, gw_power, label='GW Power')
    axs[1,1].loglog(gw_freq, sens, label='Sensitivity')
    axs[1,1].set_title('GW Spectrum vs Sensitivity')
    axs[1,1].legend()
    axs[1,1].grid(True)

    # GW-Neutrino (placeholder cross-power from fluctuations)
    cross_power = gw_power * np.mean(omega_v_tau)
    axs[1,2].loglog(gw_freq, cross_power)
    axs[1,2].set_title('GW-Neutrino Cross-Power')
    axs[1,2].grid(True)

    axs[2,0].bar(['Planck', 'Local'], [planck_H0, local_H0])
    axs[2,0].set_title('Hubble Tension')
    axs[2,0].grid(True)

    q_tau = - (omega_m_tau / 2 + np.array(omega_v_tau) / 2 - omega_lambda_tau)  # Deceleration
    axs[2,1].plot(TAU_SPAN, q_tau)
    axs[2,1].set_title('Deceleration q(tau)')
    axs[2,1].grid(True)

    axs[2,2].plot(mean_a_cmb[:-1], da_dtau_cmb)
    axs[2,2].set_title('Phase Space a vs da/dtau')
    axs[2,2].set_xscale('log')
    axs[2,2].set_yscale('log')
    axs[2,2].grid(True)

    pdf.savefig(fig)
    plt.close(fig)

# Classical RNN comparison (mimic a(tau) fitting)
class ClassicalRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, tau):
        tau = tau.unsqueeze(-1).unsqueeze(-1)  # Shape for RNN
        out, _ = self.rnn(tau)
        return self.fc(out.squeeze(1))

model_classic = ClassicalRNN().double()
optimizer_classic = optim.Adam(model_classic.parameters(), lr=0.01)
a_target = torch.tensor(mean_a_cmb, dtype=torch.float64).unsqueeze(1)

for iter in range(500):
    a_pred = model_classic(torch.tensor(TAU_SPAN, dtype=torch.float64))
    loss = nn.MSELoss()(a_pred, a_target)
    loss.backward()
    optimizer_classic.step()
    if iter % 100 == 0:
        print(f"Classic Iter {iter}: Loss {loss.item():.4f}")

plt.plot(mean_a_cmb, label='EISA ODE')
plt.plot(model_classic(torch.tensor(TAU_SPAN, dtype=torch.float64)).detach().numpy(), label='RNN')
plt.title('a(tau): EISA vs Classical RNN')
plt.legend()
plt.grid(True)
plt.show()

print("Simulation complete. PDF report generated.")
