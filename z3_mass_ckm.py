import numpy as np

def calculate_final_precision_mixing():
    print("--- Z3 Algebra Final Precision CKM Simulation ---")
    
    # 1. Base Structure: Democratic Matrix
    # The "Unit" of mass in this toy model
    M_demo = np.ones((3, 3)) / np.sqrt(3)
    
    # 2. High-Contrast Parameters (To break degeneracy and force mixing)
    
    # Up Sector: Needs m_u << m_c
    # We use a "Cancelation Pattern": 
    # Large positive vs Large negative perturbation to split light generations
    u_a =  0.020
    u_b = -0.018  # Diff is small, sum is small
    # To keep m_t stable, the third element balances the trace? 
    # No, we let m_t absorb the trace. 
    # Key is u_a approx -u_b to make m1 small.
    
    # Down Sector: Needs m_d << m_s AND Large Mixing
    # We use larger parameters because Down quarks are lighter relative to Top,
    # but in this normalized matrix they need larger 'geometry' to rotate.
    d_a =  0.075
    d_b = -0.065 # Strong splitting to separate d and s
    
    # [CRITICAL] Vacuum Misalignment
    # Based on previous error (38%), we need to nearly double the mixing term.
    # 0.0185 * (0.226/0.14) ~ 0.03
    delta_d_mix = 0.033 

    # 3. Construct Matrices
    # Up: Diagonal perturbation
    # We add (u_a, u_b, -(u_a+u_b)) to keep trace constant-ish
    M_u = M_demo.copy() + np.diag([u_a, u_b, -(u_a+u_b)])
    
    # Down: Diagonal + Off-Diagonal
    M_d = M_demo.copy() + np.diag([d_a, d_b, -(d_a+d_b)])
    
    # Injecting the geometric misalignment
    # This represents the "twist" of the vacuum in the isospin -1/2 sector
    M_d[0, 1] += delta_d_mix
    M_d[1, 0] += delta_d_mix
    
    # Stabilization terms (coupling to heavy generation)
    M_d[0, 2] += delta_d_mix * 0.3
    M_d[2, 0] += delta_d_mix * 0.3
    M_d[1, 2] += delta_d_mix * 0.1
    M_d[2, 1] += delta_d_mix * 0.1

    # 4. Diagonalization
    val_u, vec_u = np.linalg.eigh(M_u)
    val_d, vec_d = np.linalg.eigh(M_d)
    
    # Sort Light -> Heavy
    idx_u = np.argsort(np.abs(val_u))
    idx_d = np.argsort(np.abs(val_d))
    
    vec_u = vec_u[:, idx_u]
    vec_d = vec_d[:, idx_d]
    val_u = np.abs(val_u[idx_u])
    val_d = np.abs(val_d[idx_d])

    # 5. Check Mass Hierarchies
    # We want [small, medium, 1]
    print(f"\n[Mass Hierarchies (Normalized)]")
    # Up quarks usually have very large hierarchy m_u/m_t ~ 1e-5
    # In this perturbative toy model, we aim for "distinct" not "exact SM values"
    print(f"Up   : [{val_u[0]/val_u[2]:.5f}, {val_u[1]/val_u[2]:.5f}, 1.0]")
    print(f"Down : [{val_d[0]/val_d[2]:.5f}, {val_d[1]/val_d[2]:.5f}, 1.0]")

    # 6. CKM Matrix
    V_ckm = np.dot(vec_u.conj().T, vec_d)
    V_abs = np.abs(V_ckm)

    print("\n[Derived CKM Matrix |V_ij|]")
    print(f"{V_abs[0,0]:.4f}  {V_abs[0,1]:.4f}  {V_abs[0,2]:.4f}")
    print(f"{V_abs[1,0]:.4f}  {V_abs[1,1]:.4f}  {V_abs[1,2]:.4f}")
    print(f"{V_abs[2,0]:.4f}  {V_abs[2,1]:.4f}  {V_abs[2,2]:.4f}")

    # 7. Validation
    cabibbo = V_abs[0,1]
    target = 0.2265
    error = abs(cabibbo - target)/target
    
    print(f"\nCalculated Cabibbo |V_us|: {cabibbo:.4f}")
    print(f"Observed Target:           {target}")
    print(f"Error:                     {error:.2%}")
    
    if error < 0.05:
        print("\n[SUCCESS] The model reproduces the Cabibbo angle with < 5% error.")
        print(f"Required Vacuum Misalignment: {delta_d_mix:.3f} (~3.3%)")

calculate_final_precision_mixing()