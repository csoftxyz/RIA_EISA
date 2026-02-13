import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import combinations_with_replacement
import time
import os

# ==============================================================================
# Z3 Vacuum Inertia Framework: Neutrino Parameter Geometric Search
# Hardware Target: 768GB RAM Server
# Goal: Find exact integer vector origins for theta_13 and mass ratios
# ==============================================================================

# --- Configuration ---
MAX_L_SQ = 5000      # L^2 limit. 5000 is huge for discrete geometry.
                     # Allows components up to sqrt(5000) ~ 70.
                     # Memory usage scales with this. 768G allows pushing this to 10000+
TOLERANCE = 0.001    # Search tolerance for numerical coincidence

# --- Experimental Targets (NuFIT 5.2) ---
# sin^2(theta_13)
TARGET_THETA_13 = 0.0224 
TARGET_THETA_13_ERR = 0.0006

# Mass Ratio r = dm^2_21 / dm^2_31
# dm^2_21 ~ 7.42e-5, dm^2_31 ~ 2.51e-3
TARGET_MASS_RATIO = 0.0295
TARGET_MASS_RATIO_ERR = 0.001

def generate_lattice_vectors(max_l_sq):
    """
    Generates all unique integer vectors [x, y, z] with x^2+y^2+z^2 <= max_l_sq.
    Uses symmetry (x >= y >= z >= 0) to save memory, then expands.
    """
    print(f"[*] Generating Lattice Vectors up to L^2 = {max_l_sq}...")
    limit = int(np.sqrt(max_l_sq)) + 1
    vectors = []
    
    # Nested loops optimized for symmetry
    for x in range(limit):
        for y in range(x + 1): # y <= x
            for z in range(y + 1): # z <= y
                l2 = x*x + y*y + z*z
                if 0 < l2 <= max_l_sq:
                    vectors.append((x, y, z, l2))
    
    # Convert to structured array for fast access
    # Columns: x, y, z, L^2, Norm
    arr = np.array(vectors, dtype=np.float64)
    print(f"[*] Base unique geometric shapes found: {len(arr)}")
    return arr

def search_theta13(vectors):
    """
    Searches for vector projections that yield sin^2(theta) ~ 0.0224.
    Model: theta is angle between a 'Perturbation Vector' v and the 'Vacuum Axis' (Democratic [1,1,1]).
    Or angle between v and Basis [1,0,0].
    """
    print("\n[Phase 1] Searching for Theta_13 (Reactor Angle)...")
    
    # Reference 1: Democratic Vector (Vacuum VEV direction)
    v_dem = np.array([1, 1, 1]) / np.sqrt(3)
    
    # Reference 2: Basis Vector (Flavor Axis)
    v_basis = np.array([1, 0, 0])
    
    hits = []
    
    # Vectorized calculation
    # Reconstruct full 3D vectors from symmetry reduced set for projection checking
    # (Simplified: we just check the fundamental shapes first)
    
    vecs = vectors[:, 0:3]
    norms = np.sqrt(vectors[:, 3])
    
    # Projection onto Democratic Axis
    # cos_theta = (v . dem) / |v|
    # dot product: (x+y+z)/sqrt(3)
    dots_dem = np.sum(vecs, axis=1) / np.sqrt(3)
    cos_sq_dem = (dots_dem / norms) ** 2
    sin_sq_dem = 1.0 - cos_sq_dem
    
    # Find matches
    diff_dem = np.abs(sin_sq_dem - TARGET_THETA_13)
    matches_dem = np.where(diff_dem < TOLERANCE)[0]
    
    print(f"[*] Checking {len(vecs)} vector classes against Democratic axis...")
    
    results = []
    for idx in matches_dem:
        v = vecs[idx]
        val = sin_sq_dem[idx]
        # Check rational approximations
        # Is it 1/N?
        inv = 1.0 / val
        
        results.append({
            "Vector": v.astype(int),
            "L^2": int(vectors[idx, 3]),
            "sin^2": val,
            "1/sin^2": inv,
            "Type": "Angle to Democratic"
        })
        
    # Projection onto Basis Axis (if vacuum aligns differently)
    # dot product: x
    dots_basis = vecs[:, 0]
    cos_sq_basis = (dots_basis / norms) ** 2
    sin_sq_basis = 1.0 - cos_sq_basis
    
    diff_basis = np.abs(sin_sq_basis - TARGET_THETA_13)
    matches_basis = np.where(diff_basis < TOLERANCE)[0]
    
    for idx in matches_basis:
        v = vecs[idx]
        val = sin_sq_basis[idx]
        inv = 1.0 / val
        results.append({
            "Vector": v.astype(int),
            "L^2": int(vectors[idx, 3]),
            "sin^2": val,
            "1/sin^2": inv,
            "Type": "Angle to Basis [1,0,0]"
        })

    return pd.DataFrame(results).sort_values("L^2")

def search_mass_ratio(vectors):
    """
    Searches for mass squared difference ratios.
    Ansatz: m ~ 1/L^2 (Geometric Seesaw) => m^2 ~ 1/L^4
    Ratio = (L_2^-4 - L_1^-4) / (L_3^-4 - L_1^-4)
    We search for triplet of integers (L1, L2, L3) that satisfy this.
    """
    print("\n[Phase 2] Searching for Mass Squared Ratio ~ 0.03...")
    
    # We only care about the Lengths (L^2)
    # Extract unique L^2 integers
    l_sq_values = np.unique(vectors[:, 3]).astype(int)
    # Remove 0 if present
    l_sq_values = l_sq_values[l_sq_values > 0]
    
    print(f"[*] Unique geometrical norms found: {len(l_sq_values)}")
    print("[*] Building Lookup Table for Triplet Ratios (This consumes RAM)...")
    
    # This is O(N^3) complexity. With 768GB RAM, we can vectorise this.
    # But N=2000 is still 8 billion. We need to be smart.
    # Hierarchical assumption: m1 < m2 << m3 implies L1 > L2 >> L3
    # Or L1 mass is 0 (infinite length). Let's try Normal Hierarchy: m1 ~ 0.
    # Ratio simplifies to: m2^2 / m3^2 ~ (1/L2^4) / (1/L3^4) = (L3/L2)^4
    
    # Simplified ansatz: Ratio = (L_sol / L_atm)^4 or (L_sol / L_atm)^2 ?
    # Let's check pure power ratios first.
    
    candidates = []
    
    # Limit search to reasonable range to finish in minutes
    # L_small (corresponding to heavy mass m3)
    # L_large (corresponding to light mass m2)
    
    # Assume m ~ 1/L^n. Check n=2 and n=4.
    
    # Case n=2 (Mass ~ 1/L^2): Ratio = (L3/L2)^4
    # Case n=1 (Mass ~ 1/L): Ratio = (L3/L2)^2
    
    # Let's do a brute force on n=4 (Prompt implies m ~ 1/L^2, so m^2 ~ 1/L^4)
    # We look for (L_atm / L_sol)^4 ~ 0.03
    # L_atm / L_sol ~ 0.03^(1/4) ~ 0.416
    
    l_sq_list = l_sq_values[l_sq_values < 2000] # Optimization
    
    for l2_heavy in l_sq_list: # Corresponds to m3 (Atmospheric)
        for l2_light in l_sq_list: # Corresponds to m2 (Solar)
            if l2_light <= l2_heavy: continue
            
            ratio_n4 = (l2_heavy / l2_light)**2 # (L^2_heavy / L^2_light)^2 = L^4 / L^4
            
            if abs(ratio_n4 - TARGET_MASS_RATIO) < TARGET_MASS_RATIO_ERR:
                candidates.append({
                    "L^2_heavy (m3)": l2_heavy,
                    "L^2_light (m2)": l2_light,
                    "Ratio": ratio_n4,
                    "Model": "m ~ 1/L^2",
                    "Notes": f"Sqrt Ratio = {np.sqrt(ratio_n4):.4f}"
                })
                
    return pd.DataFrame(candidates).sort_values("L^2_heavy (m3)")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    print(f"=== Z3 Neutrino Parameter Hunter ===")
    print(f"Hardware: Assuming Large Memory Environment")
    print(f"Target theta_13: {TARGET_THETA_13}")
    print(f"Target Mass Ratio: {TARGET_MASS_RATIO}")
    
    # 1. Generate Database
    vectors = generate_lattice_vectors(MAX_L_SQ)
    
    # 2. Search Theta13
    df_theta = search_theta13(vectors)
    
    print("\n>>> Top Candidates for Theta_13 (Look for Integrality):")
    if not df_theta.empty:
        # Filter for "Nice" fractions (Inv is close to integer)
        df_theta['Integer_Score'] = abs(df_theta['1/sin^2'] - df_theta['1/sin^2'].round())
        print(df_theta.sort_values('Integer_Score').head(10).to_string(index=False))
    else:
        print("No matches found in this range.")

    # 3. Search Mass Ratio
    df_mass = search_mass_ratio(vectors)
    
    print("\n>>> Top Candidates for Mass Ratio (Geometric Hierarchy):")
    if not df_mass.empty:
        print(df_mass.head(10).to_string(index=False))
    else:
        print("No matches found.")
        
    end_time = time.time()
    print(f"\n[Done] Execution Time: {end_time - start_time:.2f} seconds")