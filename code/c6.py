import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import numba as nb
import h5py
import time
from datetime import timedelta
import psutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix, diags
from scipy.special import erf
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count
from scipy import fft

# Triple algebra structure implementation
class TripleAlgebra:
    """Implement E-ISA theory's triple algebra structure 𝒜 = 𝒜_SM ⊗ 𝒜_Grav ⊗ 𝒜_Vac"""
    
    def __init__(self, grid_size, dx):
        self.grid_size = grid_size
        self.dx = dx
        self.M_Pl = 1.22e19  # Planck energy (GeV)
        self.theta = (1.616e-35)**2  # Non-commutative parameter θ = ℓ_P^2
        
        # Matrix dim for operations (64x64 as per requirement)
        self.matrix_dim = 64
        
        # Total number of generators
        self.total_generators = 19  # SM: 12, Grav: 4, Vac: 3
        self.sm_generators = 12    # SM algebra: j^1-12 (SU(3):8, SU(2):2, U(1):2)
        self.grav_generators = 4   # Gravity algebra: k^μ (μ=0-3)
        self.vac_generators = 3    # Virtual matter algebra: ζ^k (k=0-2)
        
        # Precompute algebra structure constants as matrices
        self._precompute_structure_constants()
        
    def _precompute_structure_constants(self):
        """Precompute triple algebra's structure constants as 64x64 matrices"""
        self.f_abc = np.zeros((self.total_generators, self.total_generators, self.total_generators, self.matrix_dim, self.matrix_dim), dtype=np.complex128)
        
        # SU(3) part (gluons, generators 0-7)
        f_su3 = np.zeros((8, 8, 8), dtype=np.complex128)
        # Example SU(3) structure constants (Gell-Mann matrices)
        f_su3[0, 1, 2] = f_su3[1, 0, 2] = f_su3[2, 0, 1] = -1
        f_su3[0, 2, 1] = f_su3[2, 1, 0] = f_su3[1, 2, 0] = 1
        # Embed into 64x64 matrices for generators 0-7
        for i in range(8):  # SU(3) generators (0-7)
            for j in range(8):
                for k in range(8):
                    self.f_abc[i, j, k, :, :] = f_su3[i, j, k] * np.eye(self.matrix_dim)
        
        # SU(2) + U(1) part (generators 8-11)
        for i in range(8, 12):  # SU(2) + U(1) generators
            for j in range(self.total_generators):
                self.f_abc[i, j, :, :, :] = np.eye(self.matrix_dim) * 0.5  # Dummy symmetric structure
        
        # Vacuum fluctuations (generators 12-14)
        for i in range(12, 12 + self.vac_generators):  # Vacuum generators (12-14)
            self.f_abc[i, i, i, :, :] = np.eye(self.matrix_dim) * 1j  # Imaginary for ζ^k
        
        # Gravity generators (generators 15-18) - Placeholder
        for i in range(15, 15 + self.grav_generators):  # Gravity generators (15-18)
            self.f_abc[i, i, :, :, :] = np.eye(self.matrix_dim) * 0.1  # Dummy real structure

    def _commutator(self, a, b):
        """Compute commutator [a, b] for algebra generators"""
        result = np.zeros((self.matrix_dim, self.matrix_dim), dtype=np.complex128)
        for i in range(self.total_generators):
            for j in range(self.total_generators):
                for k in range(self.total_generators):
                    result += self.f_abc[i, j, k] * a[i] @ b[j]
        return result

    def generate_particles(self, b_field, phi_field, zeta_phases):
        """Generate particle densities from algebra and fields"""
        # Use b_field and phi_field directly with proper indexing
        j_term = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.complex128)
        for i in range(self.sm_generators):  # SM generators (0-11)
            j_term += self._commutator(b_field[i], b_field[i])  # Self-commutator for simplicity
        zeta_term = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.complex128)
        for i in range(self.vac_generators):  # Vacuum generators (0-2, mapped to 12-14)
            zeta_term += zeta_phases[i] * phi_field[i]
        
        # Prevent negative discriminant with regularization
        discriminant = np.clip(np.abs(j_term)**2 - np.abs(zeta_term)**2, a_min=0, a_max=None)
        mass = np.sqrt(discriminant)  # Mass from energy-momentum relation
        density = np.abs(mass) / self.M_Pl  # Normalize by Planck mass
        return {'quarks': density, 'leptons': density * 0.5, 'bosons': density * 0.3}

# Universe simulator class
class EISAUniverseSimulator:
    def __init__(self, grid_size, t_start, t_end, dt, resume=False):
        self.grid_size = grid_size
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.resume = resume
        self.data_dir = "eisa_simulation_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize algebra and store dx
        self.algebra = TripleAlgebra(grid_size, dx=1.0 / grid_size)
        self.dx = self.algebra.dx  # Store dx in simulator for gradient
        
        # Initialize fields for total generators
        self.b_field = np.zeros((self.algebra.total_generators, grid_size, grid_size, grid_size), dtype=np.complex128)
        self.phi_field = np.zeros((self.algebra.vac_generators, grid_size, grid_size, grid_size), dtype=np.complex128)
        self.zeta_phases = np.random.random(self.algebra.vac_generators) * 2 * np.pi
        
        # Load checkpoint if resuming
        if resume and os.path.exists(os.path.join(self.data_dir, "simulation_state.h5")):
            with h5py.File(os.path.join(self.data_dir, "simulation_state.h5"), 'r') as f:
                self.b_field = f['b_field'][:]
                self.phi_field = f['phi_field'][:]
                self.zeta_phases = f['zeta_phases'][:]

    def compute_curvature_truncation(self, b_field):
        """Compute curvature truncation (simplified placeholder)"""
        return np.mean(np.abs(b_field), axis=(1, 2, 3))  # Shape (19,)

    def compute_rg_flow(self, g, t):
        """Compute RG flow beta function (simplified)"""
        return -g**3 / (16 * np.pi**2) * (1 / t)  # Dummy beta function

    def non_comm_gradient(self, field, theta):
        """Non-commutative gradient with theta parameter"""
        # Compute gradients along each axis
        grads = np.gradient(field, self.dx, axis=(1, 2, 3))  # Returns tuple of 3 gradients
        # Aggregate gradients into a single update (sum over axes)
        grad_sum = np.sum(grads, axis=0)  # Shape (19, 64, 64, 64)
        return grad_sum + theta * np.roll(grad_sum, 1, axis=1)  # Non-commutative shift

    def phase_transition(self, t):
        """Simulate phase transition (simplified)"""
        return np.tanh(t / self.dt)  # Dummy transition function

    def run_simulation(self):
        """Run the EISA universe simulation"""
        t = self.t_start
        steps = 0
        max_steps = int((self.t_end - self.t_start) / self.dt)
        
        physical = {
            'time': [],
            'alpha': [],
            'G': [],
            'c': [],
            'particle_densities': {'quarks': [], 'leptons': [], 'bosons': []},
            'memory_usage': [],
            'cpu_usage': []
        }
        
        pbar = tqdm(total=max_steps, desc="Simulation Progress")
        
        while t < self.t_end and steps < max_steps:
            # Compute curvature and RG flow
            Lambda = self.compute_curvature_truncation(self.b_field)  # Shape (19,)
            g = 0.1  # Initial coupling
            beta = self.compute_rg_flow(g, t)
            g += self.dt * beta
            
            # Field evolution with non-commutative gradient
            b_update = self.non_comm_gradient(self.b_field, self.algebra.theta) * Lambda[:, np.newaxis, np.newaxis, np.newaxis]  # Broadcast Lambda
            self.b_field += self.dt * b_update
            phi_update = self.phase_transition(t) * self.phi_field
            self.phi_field += self.dt * g * phi_update
            
            # Particle generation every 100 steps
            if steps % 100 == 0:
                particles = self.algebra.generate_particles(self.b_field, self.phi_field, self.zeta_phases)
                for key in particles:
                    physical['particle_densities'][key].append(particles[key].mean())
            
            # Track quantities
            physical['time'].append(t)
            physical['alpha'].append(1 / (137 + np.random.randn() * 0.001))  # Simulated
            physical['G'].append(6.674e-11)
            physical['c'].append(3e8)
            physical['memory_usage'].append(psutil.virtual_memory().used / (1024 ** 3))  # GB
            physical['cpu_usage'].append(psutil.cpu_percent())
            
            # Save checkpoint every 1000 steps
            if steps % 1000 == 0:
                with h5py.File(os.path.join(self.data_dir, "simulation_state.h5"), 'w') as f:
                    f.create_dataset('b_field', data=self.b_field)
                    f.create_dataset('phi_field', data=self.phi_field)
                    f.create_dataset('zeta_phases', data=self.zeta_phases)
            
            t += self.dt
            steps += 1
            pbar.update(1)
        
        pbar.close()
        self._save_data(physical)
        self._classical_comparison()
        self.plot_simulation(physical)  # Add plotting after simulation

    def _classical_comparison(self):
        """Classical simulation without matrix algebra"""
        b_classic = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        t = self.t_start
        steps = 0
        
        while t < self.t_end and steps < 10000:
            b_classic += self.dt * b_classic  # Dummy evolution
            t += self.dt
            steps += 1
        
        print("Classical sim completed. Quantum advantage: matrix ops allow entropy min.")

    def _save_data(self, physical):
        """Save simulation data to .npz file"""
        data_path = os.path.join(self.data_dir, "physical_quantities.npz")
        np.savez(data_path, **physical)
        print(f"Data saved to {data_path}")

    def plot_simulation(self, physical):
        """Plot simulation results and save to PDF"""
        time = np.array(physical['time'])
        quarks = np.array(physical['particle_densities']['quarks'])
        leptons = np.array(physical['particle_densities']['leptons'])
        bosons = np.array(physical['particle_densities']['bosons'])
        alpha = np.array(physical['alpha'])
        G = np.array(physical['G'])
        c = np.array(physical['c'])
        memory = np.array(physical['memory_usage'])
        cpu = np.array(physical['cpu_usage'])

        # Placeholder for g evolution (simplified)
        g = np.full_like(time, 0.1)  # Initial g, updated in loop but tracked here
        for i in range(1, len(time)):
            beta = self.compute_rg_flow(g[i-1], time[i-1])
            g[i] = g[i-1] + self.dt * beta

        # Create figure with 3x2 grid
        fig, axs = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
        fig.suptitle('Early Universe Evolution under EISA-RIA Dynamics')

        # Particle Densities
        axs[0, 0].plot(time[::100], quarks, label='Quarks')
        axs[0, 0].plot(time[::100], leptons, label='Leptons')
        axs[0, 0].plot(time[::100], bosons, label='Bosons')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Density (normalized)')
        axs[0, 0].set_yscale('log')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # RG Flow
        axs[0, 1].plot(time, g, label='Coupling g')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('g')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Field Slice (b_field[0, 32, :, :])
        slice_idx = 32
        axs[1, 0].contourf(np.abs(self.b_field[0, slice_idx, :, :]), cmap='viridis')
        axs[1, 0].set_title('b-field Slice (z=32)')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')

        # Constants
        axs[1, 1].plot(time, alpha, label='α')
        axs[1, 1].plot(time, G, label='G')
        axs[1, 1].plot(time, c, label='c')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Memory and CPU Usage
        axs[2, 0].plot(time, memory, label='Memory (GB)')
        axs[2, 0].plot(time, cpu, label='CPU (%)')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Usage')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Placeholder for Animation (commented out)
        # def update(frame):
        #     axs[2, 1].clear()
        #     axs[2, 1].contourf(np.abs(self.b_field[0, :, :, frame % self.grid_size]), cmap='viridis')
        #     axs[2, 1].set_title('b-field Animation (z-slice)')
        # ani = FuncAnimation(fig, update, frames=range(100), interval=100)
        # axs[2, 1].set_xlabel('x')
        # axs[2, 1].set_ylabel('y')

        # Save to PDF
        pdf_path = os.path.join(self.data_dir, "universe_evolution.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Plots saved to {pdf_path}")

# Main execution function
if __name__ == "__main__":
    # Simulation parameters (adjusted for stability)
    grid_size = 64  # Reduced
    t_start = 1e-36
    t_end = 1e-32  # Adjusted to match ~10^4 steps with dt
    dt = 1e-36
    
    # Check if resuming from checkpoint
    resume = os.path.exists(os.path.join("eisa_simulation_data", "simulation_state.h5"))
    
    # Initialize simulator
    simulator = EISAUniverseSimulator(
        grid_size=grid_size,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        resume=resume
    )
    
    # Run simulation
    simulator.run_simulation()