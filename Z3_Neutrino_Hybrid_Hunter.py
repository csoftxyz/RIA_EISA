import numpy as np
import pandas as pd
import time
import os

# ==============================================================================
# Z3 Vacuum Inertia Framework: Neutrino Parameter Geometric Search (Hybrid Edition)
# Hardware Target: 768GB RAM Server
# Key Objective: Check projections onto the Hybrid Axis [-2, 1, 1]
# ==============================================================================

# --- Configuration ---
# With 768GB RAM, we can push this high. 
# 20,000 covers a massive geometric space (vectors with components up to ~140).
MAX_L_SQ = 20000      

# Search Tolerances
TOLERANCE_THETA = 0.002    # Slightly wider to catch "near integers" like 44.6
TARGET_THETA_13 = 0.0224   # Experimental value (NuFIT 5.2)

# Mass Ratio Targets
TARGET_MASS_RATIO = 0.0295
TARGET_MASS_RATIO_ERR = 0.001

def generate_lattice_vectors(max_l_sq):
    """
    Generates all unique integer vectors [x, y, z] with x^2+y^2+z^2 <= max_l_sq.
    Uses symmetry (x >= y >= z >= 0) to save memory, then expands for projection.
    """
    print(f"[*] Generating Lattice Vectors up to L^2 = {max_l_sq}...")
    limit = int(np.sqrt(max_l_sq)) + 1
    vectors = []
    
    # 1. Generate Fundamental Domain (x >= y >= z >= 0)
    for x in range(limit):
        for y in range(x + 1): 
            for z in range(y + 1): 
                l2 = x*x + y*y + z*z
                if 0 < l2 <= max_l_sq:
                    vectors.append((x, y, z, l2))
    
    # Convert to structured array
    arr = np.array(vectors, dtype=np.float64)
    print(f"[*] Fundamental geometric shapes found: {len(arr)}")
    
    # 2. Expand to Full Permutations for Hybrid Projection
    # Because [-2, 1, 1] is NOT symmetric under x,y,z permutation,
    # we need to check permutations like [1, -2, 1] etc. explicitly or 
    # generate full set.
    # Strategy: For Hybrid check, we need specific alignments.
    # The hybrid vector breaks S3 symmetry. 
    # We will expand the set for the projection phase.
    
    return arr

def search_theta13_expanded(fundamental_vectors):
    """
    Searches for vector projections onto:
    1. Basis [1,0,0]
    2. Democratic [1,1,1]
    3. Hybrid [-2,1,1] (NEW!)
    """
    print("\n[Phase 1] Searching for Theta_13 (Reactor Angle)...")
    
    # Expand fundamental vectors to handle permutations for Hybrid projection
    # [-2, 1, 1] is sensitive to which component is the "-2".
    # We iterate over fundamental shapes and permute them logically.
    
    results = []
    
    # A. Pre-calculate norms
    vecs_fund = fundamental_vectors[:, 0:3]
    norms = np.sqrt(fundamental_vectors[:, 3])
    l2_vals = fundamental_vectors[:, 3]
    
    total_checks = 0
    
    for i in range(len(vecs_fund)):
        v = vecs_fund[i]
        n = norms[i]
        l2 = int(l2_vals[i])
        
        # Generate permutations/sign flips that are geometrically distinct
        # relative to the projection axes.
        # Since we look for sin^2, signs matter less for Demo/Basis (squared),
        # but matter for Hybrid [-2, 1, 1] dot product.
        
        # For efficiency, we construct specific "worst case" alignments
        # The Hybrid axis v_h = [-2, 1, 1] / sqrt(6)
        
        # Permutations of v components to test against Hybrid
        # v could be [x, y, z]. Permutations: [x,y,z], [y,z,x], [z,x,y]...
        # We need to test the dot product with [-2, 1, 1].
        # Dot = -2*a + 1*b + 1*c. 
        
        x, y, z = v[0], v[1], v[2]
        
        # Possible dot products (up to sign of components)
        # We assume vacuum can pick the sign to minimize energy/align.
        # So we test all sign combinations for the max alignment (or specific angle).
        
        # 1. Basis Projection [1,0,0]
        # cos = x / n
        sin2_basis = 1.0 - (x/n)**2
        if abs(sin2_basis - TARGET_THETA_13) < TOLERANCE_THETA:
            results.append({
                "Vector": f"[{int(x)},{int(y)},{int(z)}]", 
                "L^2": l2, 
                "sin^2": sin2_basis, 
                "1/sin^2": 1.0/sin2_basis if sin2_basis>0 else 0,
                "Axis": "Basis [1,0,0]"
            })

        # 2. Democratic Projection [1,1,1]
        # cos = (x+y+z) / (n*sqrt(3))
        # Note: signs of x,y,z matter. We usually assume positive quadrant for fundamental,
        # but vacuum might select signs. Max alignment = (|x|+|y|+|z|).
        sum_xyz = x + y + z
        sin2_demo = 1.0 - (sum_xyz**2 / (3 * l2))
        if abs(sin2_demo - TARGET_THETA_13) < TOLERANCE_THETA:
            results.append({
                "Vector": f"[{int(x)},{int(y)},{int(z)}]", 
                "L^2": l2, 
                "sin^2": sin2_demo, 
                "1/sin^2": 1.0/sin2_demo if sin2_demo>0 else 0,
                "Axis": "Democratic [1,1,1]"
            })

        # 3. Hybrid Projection [-2, 1, 1] / sqrt(6)  <-- THE NEW CORE
        # Dot = (-2*a + b + c) / sqrt(6)
        # We permute x,y,z to find "best fit" or "characteristic fit"
        # Combinations of -2 coefficient:
        d1 = -2*x + y + z
        d2 = -2*y + x + z
        d3 = -2*z + x + y
        
        for dot_val in [d1, d2, d3]:
            # cos^2 = dot^2 / (6 * L^2)
            cos2_hyb = (dot_val**2) / (6 * l2)
            sin2_hyb = 1.0 - cos2_hyb
            
            if abs(sin2_hyb - TARGET_THETA_13) < TOLERANCE_THETA:
                # Calculate integer score (distance to nearest integer inverse)
                inv = 1.0/sin2_hyb
                score = abs(inv - round(inv))
                
                results.append({
                    "Vector": f"[{int(x)},{int(y)},{int(z)}]", 
                    "L^2": l2, 
                    "sin^2": sin2_hyb, 
                    "1/sin^2": inv,
                    "Axis": "Hybrid [-2,1,1]",
                    "Int_Score": score
                })
        
        total_checks += 1

    print(f"[*] Processed {total_checks} fundamental geometries against 3 axes.")
    return pd.DataFrame(results)

def search_mass_ratio(vectors):
    print("\n[Phase 2] Searching for Mass Squared Ratio ~ 0.03...")
    l_sq_values = np.unique(vectors[:, 3]).astype(int)
    l_sq_values = l_sq_values[l_sq_values > 0]
    
    # Optimization: Only check L^2 < 3000 to save time, unless 768G allows massive parallel.
    # For a quick Python script, N^2 loop on 3000 items is fast. 
    # N=20000 is 400 million checks, takes ~1 min in C++, slower in Python.
    # Let's subset.
    l_sq_subset = l_sq_values[l_sq_values < 3000]
    
    candidates = []
    for l2_heavy in l_sq_subset: 
        for l2_light in l_sq_subset: 
            if l2_light <= l2_heavy: continue
            
            # Model: m ~ 1/L^2 (Geometric Seesaw)
            ratio = (l2_heavy / l2_light)**2
            
            if abs(ratio - TARGET_MASS_RATIO) < TARGET_MASS_RATIO_ERR:
                candidates.append({
                    "L^2(m3)": l2_heavy,
                    "L^2(m2)": l2_light,
                    "Ratio": ratio,
                    "Error": abs(ratio - TARGET_MASS_RATIO)
                })
                
    return pd.DataFrame(candidates).sort_values("Error")

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print(f"=== Z3 Neutrino Hunter: Hybrid Axis Edition ===")
    print(f"Search Space: L^2 <= {MAX_L_SQ}")
    print(f"Targeting: 1/sin^2(theta_13) ~ 44.6 (looking for 44 or 45)")
    
    start = time.time()
    
    # 1. Generate
    vecs = generate_lattice_vectors(MAX_L_SQ)
    
    # 2. Theta 13 Search
    df_theta = search_theta13_expanded(vecs)
    
    if not df_theta.empty:
        # Filter for Hybrid axis specifically to answer the user's question
        df_hyb = df_theta[df_theta['Axis'].str.contains("Hybrid")]
        
        print("\n>>> Top Candidates on HYBRID AXIS [-2, 1, 1] (Sorted by Integer Purity):")
        # Sort by how close 1/sin^2 is to an integer
        if not df_hyb.empty:
            print(df_hyb.sort_values("Int_Score").head(15).to_string(columns=["Vector", "L^2", "sin^2", "1/sin^2", "Axis"], index=False))
        else:
            print("No matches on Hybrid axis in this range.")
            
        print("\n>>> Top Candidates on Other Axes:")
        df_other = df_theta[~df_theta['Axis'].str.contains("Hybrid")]
        df_other['Int_Score'] = abs(df_other['1/sin^2'] - df_other['1/sin^2'].round())
        print(df_other.sort_values("Int_Score").head(10).to_string(columns=["Vector", "L^2", "sin^2", "1/sin^2", "Axis"], index=False))

    # 3. Mass Ratio
    df_mass = search_mass_ratio(vecs)
    print("\n>>> Top Candidates for Mass Ratio (m ~ 1/L^2):")
    if not df_mass.empty:
        print(df_mass.head(10).to_string(index=False))
        
    print(f"\n[Done] Time elapsed: {time.time() - start:.2f}s")